import os
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple

from .base_dynamics import BaseDynamics
from ..utils.act_obs_adapter import ACTObservationAdapter
from ..utils.logger import Logger


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        terminal_fn: Callable,
        env,
        obs_adapter: ACTObservationAdapter,
        cfg,
        gamma: float = 0.99,
        lamda: float = 0.95,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
    ) -> None:
        super().__init__(model, optim)
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.env = env
        self.obs_adapter = obs_adapter
        self.cfg = cfg
        self.n_obs_steps = cfg.n_obs_steps
        self.predict_delta = cfg.dynamics.predict_delta
        self.predict_r = cfg.predict_r
        self.lamda = lamda
        self.gamma = gamma
        self.device = obs_adapter.device

    def _model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _prepare_action(self, action: torch.Tensor) -> torch.Tensor:
        model = self._model()
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        batch_size = action.shape[0]

        if getattr(self.cfg, "use_conv_action_embed", False):
            action_chunk = action.reshape(batch_size, -1, model._single_action_dim)
            z = model._conv_action_encoder(action_chunk)
            return model._conv_action_layer_norm(z.reshape(batch_size, -1))

        action = action.reshape(batch_size, -1)
        if getattr(self.cfg, "use_action_embed", False):
            return model._action_encoder(action)
        if getattr(model, "use_action_scale_norm", False):
            action = model._action_scale_layer_norm(action)
        return action

    @torch.no_grad()
    def step(
        self,
        nobs_features: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        model = self._model()
        nobs_features = torch.as_tensor(nobs_features, device=self.device, dtype=torch.float32)
        action = self._prepare_action(action)

        nobs_np = nobs_features.cpu().numpy()
        action_np = action.cpu().numpy()
        obs_act = np.concatenate([nobs_np, action_np], axis=-1)

        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()

        if self.predict_delta:
            if self.predict_r:
                mean[..., :-1] += nobs_np
            else:
                mean += nobs_np

        std = np.sqrt(np.exp(logvar))
        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
        _, batch_size, _ = ensemble_samples.shape
        model_idxs = model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]

        if self.predict_r:
            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
        else:
            next_obs = samples
            reward = np.zeros((batch_size, 1), dtype=np.float32)

        terminal = self.terminal_fn(nobs_np, action_np, next_obs, self.env)
        info = {"raw_reward": reward}

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1] if self.predict_r else mean
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1] if self.predict_r else mean
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError(
                    f"Unsupported uncertainty mode: {self._uncertainty_mode}"
                )

            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty

        return next_obs, reward, terminal, info

    @torch.no_grad()
    def multi_step(
        self,
        nobs_features: torch.Tensor,
        nactions: torch.Tensor,
        reward_strategy: str = "sum",
        discount: float = 1.0,
        Return: float = 0.0,
        Qs: List[torch.Tensor] = None,
        Q: Callable = None,
    ):
        n_step_actions = nactions.shape[1]
        rewards = 0
        batch_size = nactions.shape[0]
        nobs_features = nobs_features.reshape(batch_size, -1)

        for i in range(n_step_actions):
            if Q is not None:
                Qs.append(Q(nobs_features, nactions[:, i]))
            next_obs, reward, terminal, info = self.step(
                nobs_features,
                nactions[:, i],
            )
            if reward_strategy == "sum":
                rewards += reward
            Return += discount * reward
            discount *= self.gamma
            nobs_features = torch.from_numpy(next_obs).to(self.device)

        return next_obs, rewards, terminal, info, Return, Qs, discount

    @torch.no_grad()
    def multi_step_evaluation(
        self,
        nobs_features: torch.Tensor,
        nactions: torch.Tensor,
        Q: Callable,
        state_dict: Dict = None,
        use_gae: bool = False,
    ):
        n_step_actions = nactions.shape[1]
        all_obs_features, rewards, terminals, infos = [], [], [], []
        gae_advantages, Qs = [], []
        G, discount = 0, 1

        if self.cfg.online:
            if self.cfg.ppo.iql_ft:
                iql_input = state_dict
            else:
                iql_input = nobs_features
        else:
            iql_input = nobs_features

        if n_step_actions > 1:
            raise NotImplementedError

        for i in range(n_step_actions):
            Qs.append(Q(iql_input, nactions[:, i]))
            all_obs_features.append(nobs_features)
            next_obs, reward, terminal, info = self.step(
                nobs_features,
                nactions[:, i],
            )
            G += discount * reward
            discount *= self.gamma
            rewards.append(reward)
            terminals.append(terminal)
            infos.append(info)
            nobs_features = torch.from_numpy(next_obs).to(self.device)

        if use_gae:
            deltas, gae = Qs, 0
            for delta in reversed(deltas):
                gae = delta + self.gamma * self.lamda * gae
                gae_advantages.insert(0, gae)

        G += discount * Qs[-1].cpu().numpy()

        if use_gae:
            return (
                all_obs_features,
                rewards,
                terminals,
                infos,
                G.squeeze(),
                torch.stack(gae_advantages).squeeze(1),
            )
        return (
            all_obs_features,
            rewards,
            terminals,
            infos,
            torch.from_numpy(G).squeeze(),
            None,
        )

    def obs2latent(self, nobs) -> torch.Tensor:
        features = self.obs_adapter.encode(
            nobs,
            start=0,
            track_grad=not self.obs_adapter.fix_encoder,
        )
        return features.reshape(features.shape[0], -1)

    @torch.no_grad()
    def compute_model_uncertainty(
        self,
        obs,
        action,
        uncertainty_mode: str = "aleatoric",
    ) -> np.ndarray:
        if self.obs_adapter.fix_encoder:
            nobs_features = torch.as_tensor(
                obs,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            nobs_features = self.obs2latent(obs)

        action = self._prepare_action(action)
        obs_act = torch.cat((nobs_features, action), dim=-1)
        mean, logvar = self.model(obs_act)

        if self.predict_delta:
            if self.predict_r:
                mean[..., :-1] += nobs_features
            else:
                mean += nobs_features

        std = torch.sqrt(torch.exp(logvar))
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()

        if uncertainty_mode == "aleatoric":
            penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        elif uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean[..., :-1] if self.predict_r else mean
            next_obs_mean = np.mean(next_obses_mean, axis=0)
            diff = next_obses_mean - next_obs_mean
            penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        else:
            raise ValueError(f"Unsupported uncertainty mode: {uncertainty_mode}")

        penalty = np.expand_dims(penalty, 1).astype(np.float32)
        return self._penalty_coef * penalty

    @torch.no_grad()
    def predict_next_obs(
        self,
        obs,
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if self.obs_adapter.fix_encoder:
            nobs_features = torch.as_tensor(
                obs,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            nobs_features = self.obs2latent(obs)

        action = self._prepare_action(action)
        obs_act = torch.cat((nobs_features, action), dim=-1)
        mean, logvar = self.model(obs_act)

        if self.predict_delta:
            if self.predict_r:
                mean[..., :-1] += nobs_features
            else:
                mean += nobs_features

        std = torch.sqrt(torch.exp(logvar))
        model = self._model()
        elite_idx = model.elites.long()
        mean = mean[elite_idx]
        std = std[elite_idx]

        samples = torch.stack(
            [mean + torch.randn_like(std) * std for _ in range(num_samples)],
            dim=0,
        )
        return samples[..., :-1] if self.predict_r else samples

    def format_samples_for_training(
        self,
        data: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nobs_features = self.obs2latent(data["obs"]).detach()
        next_nobs_features = self.obs2latent(data["next_obs"]).detach()

        if self.predict_delta:
            targets = next_nobs_features - nobs_features
        else:
            targets = next_nobs_features

        actions = self.obs_adapter.normalize_action(data["action"])
        index = self.n_obs_steps - 1
        action = actions[:, index]
        action = self._prepare_action(action)
        inputs = torch.cat((nobs_features, action), dim=-1)

        if self.predict_r:
            rewards = data["reward"][:, index].to(self.device)
            rewards = rewards.reshape(rewards.shape[0], -1)
            targets = torch.cat((targets, rewards), dim=-1)

        return inputs, targets


    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 15,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        model = self._model()
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 100)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(
            range(data_size),
            (train_size, holdout_size),
        )
        train_inputs = inputs[train_splits.indices]
        train_targets = targets[train_splits.indices]
        holdout_inputs = inputs[holdout_splits.indices]
        holdout_targets = targets[holdout_splits.indices]
        holdout_losses = [1e10 for _ in range(model.num_ensemble)]
        data_idxes = torch.from_numpy(
            np.random.randint(
                train_size,
                size=[model.num_ensemble, train_size],
            )
        )

        def shuffle_rows(arr):
            shape = arr.size()
            idxes = torch.argsort(torch.rand(shape), dim=-1)
            row_indices = torch.arange(shape[0]).unsqueeze(1)
            return arr[row_indices, idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")

        while True:
            epoch += 1
            print(f"epoch {epoch}")
            train_loss = self.learn(
                train_inputs[data_idxes],
                train_targets[data_idxes],
                batch_size,
                logvar_loss_coef,
            )
            new_holdout_losses = self.validate(
                holdout_inputs,
                holdout_targets,
            )
            holdout_loss = np.sort(
                new_holdout_losses
            )[:model.num_elites].mean()

            logger.logkv(
                "loss/dynamics_train_loss",
                train_loss,
            )
            logger.logkv(
                "loss/dynamics_holdout_loss",
                holdout_loss,
            )
            logger.set_timestep(epoch)
            logger.dumpkvs(
                exclude=["policy_training_progress"]
            )

            data_idxes = shuffle_rows(data_idxes)
            indexes = []

            for i, new_loss, old_loss in zip(
                range(len(holdout_losses)),
                new_holdout_losses,
                holdout_losses,
            ):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss

            if indexes:
                model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if cnt >= max_epochs_since_update:
                break
            if max_epochs and epoch >= max_epochs:
                break

        indexes = self.select_elites(holdout_losses)
        model.set_elites(indexes)
        model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log(
            "elites:{} , holdout loss: {}".format(
                indexes,
                np.sort(holdout_losses)[:model.num_elites].mean(),
            )
        )

    def learn(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01,
    ) -> float:
        self.model.train()
        model = self._model()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            start = batch_num * batch_size
            end = (batch_num + 1) * batch_size
            inputs_batch = inputs[:, start:end]
            targets_batch = targets[:, start:end].to(self.device)

            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            mse_loss_inv = (
                (mean - targets_batch).pow(2) * inv_var
            ).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss += model.get_decay_loss()
            loss += (
                logvar_loss_coef * model.max_logvar.sum()
                - logvar_loss_coef * model.min_logvar.sum()
            )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())

        return np.mean(losses)

    @torch.no_grad()
    def rollout(
        self,
        policy: nn.Module,
        Q: nn.Module,
        iql: nn.Module,
        batch: dict,
        rollout_length: int,
        is_iql: bool = True,
        use_gae: bool = False,
        first_action: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        if use_gae:
            raise NotImplementedError(
                "ACT V1 rollout does not support recursive GAE because "
                "ACT cannot consume predicted dynamics latent."
            )

        q_eval = iql.minQ if is_iql else Q
        policy_batch = self.obs_adapter.normalize_obs(batch["obs"])
        actions, _, _ = policy.sample_action_chunk(policy_batch)

        if rollout_length < 1 or rollout_length > actions.shape[1]:
            raise ValueError(
                f"ACT V1 dynamics requires rollout_length in "
                f"[1, {actions.shape[1]}], got {rollout_length}."
            )

        nobs_features = self.obs2latent(batch["obs"])
        Return, discount, Qs = 0, 1, []
        next_nobs_features, rewards, _, _, Return, Qs, discount = self.multi_step(
            nobs_features,
            actions[:, :rollout_length],
            discount=discount,
            Return=Return,
            Qs=Qs,
            Q=q_eval,
        )

        q_evaluation = torch.mean(torch.stack(Qs))
        return q_evaluation, np.asarray(rewards).mean()

    @torch.no_grad()
    def validate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(
            targets,
            device=self.device,
        )
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        return list(loss.cpu().numpy())

    def select_elites(self, metrics: List) -> List[int]:
        model = self._model()
        pairs = [
            (metric, index)
            for metric, index in zip(
                metrics,
                range(len(metrics)),
            )
        ]
        pairs = sorted(pairs, key=lambda x: x[0])
        return [
            pairs[i][1]
            for i in range(model.num_elites)
        ]

    def save(self, save_path: str) -> None:
        model = self._model()
        torch.save(
            model.state_dict(),
            os.path.join(save_path, "dynamics.pth"),
        )

        if not self.obs_adapter.fix_encoder:
            torch.save(
                self.obs_adapter.encoder.state_dict(),
                os.path.join(
                    save_path,
                    "dynamics_encoder.pth",
                ),
            )

        print(f"dynamics model saved in {save_path}")

    def load(self, load_path: str) -> None:
        model = self._model()
        model.load_state_dict(
            torch.load(
                os.path.join(load_path, "dynamics.pth"),
                map_location=self.device,
            )
        )

        if not self.obs_adapter.fix_encoder:
            self.obs_adapter.encoder.load_state_dict(
                torch.load(
                    os.path.join(
                        load_path,
                        "dynamics_encoder.pth",
                    ),
                    map_location=self.device,
                )
            )

        print(f"dynamics model loaded from {load_path}")


 