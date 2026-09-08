import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base_dynamics import BaseDynamics
from ..utils.act_obs_adapter import ACTObservationAdapter


class EnsembleDynamics_batch(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        terminal_fn: Callable,
        env,
        obs_adapter: ACTObservationAdapter,
        cfg,
        action_dim: int,
        gamma: float = 0.99,
        lamda: float = 0.95,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        device: str = "cpu",
        chunk_as_single_action: bool = False,
        n_action_steps: int = 1,
        prediction_mode: str = "last",
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
        self.use_conv_action_embed = getattr(cfg, "use_conv_action_embed", False)
        self.chunk_as_single_action = chunk_as_single_action
        self.n_action_steps = n_action_steps
        self.prediction_mode = prediction_mode

        model = self.model.module if hasattr(self.model, "module") else self.model
        self.holdout_losses = [1e10 for _ in range(model.num_ensemble)]

        self.lamda = lamda
        self.gamma = gamma
        self.cnt = 0
        self.action_dim = action_dim
        self.predict_r = cfg.predict_r
        self.dynamics_type = cfg.dynamics_type
        self.device = device

    def set_logger(self, logger) -> None:
        self.logger = logger
        self.logger.log("Training dynamics:")

    @staticmethod
    def _q_requires_raw_obs(Q: Callable) -> bool:
        q_owner = getattr(Q, "__self__", None)
        if q_owner is not None and getattr(q_owner, "eval_with_raw_obs", False):
            return True
        if q_owner is not None and hasattr(q_owner, "is_share_encoder"):
            return not bool(q_owner.is_share_encoder)
        q_module = getattr(q_owner, "_Q", None) if q_owner is not None else Q
        return getattr(q_module, "_obs_encoder", None) is not None

    @staticmethod
    def _as_column_tensor(value, ref: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
        tensor = tensor.reshape(ref.shape[0], -1)
        if tensor.shape[1] != 1:
            tensor = tensor[:, :1]
        return tensor

    @staticmethod
    def _as_chunk_reward_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor[..., 0]
        return tensor

    def _discounted_chunk_rewards(
        self,
        reward_chunk: torch.Tensor,
        not_done_chunk: Optional[torch.Tensor] = None,
        done_chunk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reward_chunk = self._as_chunk_reward_tensor(reward_chunk)
        gamma_weights = torch.pow(
            torch.tensor(
                self.gamma,
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            ),
            torch.arange(
                reward_chunk.shape[1],
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            ),
        )

        if not_done_chunk is None and done_chunk is not None:
            done_chunk = self._as_chunk_reward_tensor(done_chunk).to(
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            )
            not_done_chunk = 1.0 - done_chunk

        if not_done_chunk is not None:
            not_done_chunk = self._as_chunk_reward_tensor(not_done_chunk).to(
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            )
            prior_not_done = torch.ones_like(reward_chunk)
            if reward_chunk.shape[1] > 1:
                prior_not_done[:, 1:] = torch.cumprod(
                    not_done_chunk[:, :-1],
                    dim=1,
                )
            reward_chunk = reward_chunk * prior_not_done

        return torch.sum(
            reward_chunk * gamma_weights,
            dim=-1,
            keepdim=True,
        )

    def _return_and_gae(
        self,
        rewards,
        terminals,
        q_values=None,
        final_bootstrap: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
    ):
        if gamma is None:
            gamma = self.gamma

        if q_values is not None and len(q_values) > 0:
            if len(q_values) != len(rewards):
                raise ValueError(
                    f"Expected one q_value per reward, got "
                    f"{len(q_values)} q_values and {len(rewards)} rewards."
                )
            ref = self._as_column_tensor(q_values[0], q_values[0])
            q_values = [
                self._as_column_tensor(q_value, ref)
                for q_value in q_values
            ]
        elif final_bootstrap is not None:
            ref = self._as_column_tensor(final_bootstrap, final_bootstrap)
        else:
            raise ValueError(
                "Need q_values or final_bootstrap to infer rollout shape."
            )

        if final_bootstrap is not None:
            final_bootstrap = self._as_column_tensor(final_bootstrap, ref)

        returns = torch.zeros_like(ref)
        discount = torch.ones_like(ref)
        alive = torch.ones_like(ref)
        reward_tensors, alive_tensors, nonterminals = [], [], []

        for reward, terminal in zip(rewards, terminals):
            reward_t = self._as_column_tensor(reward, ref)
            terminal_t = self._as_column_tensor(
                terminal,
                ref,
            ).clamp(0.0, 1.0)
            alive_before = alive
            nonterminal = alive_before * (1.0 - terminal_t)
            masked_reward = alive_before * reward_t
            returns = returns + discount * masked_reward

            reward_tensors.append(masked_reward)
            alive_tensors.append(alive_before)
            nonterminals.append(nonterminal)

            discount = discount * gamma
            alive = nonterminal

        if final_bootstrap is not None:
            returns = returns + discount * alive * final_bootstrap

        gae_advantages = None
        if q_values is not None and len(q_values) > 0:
            final_q = (
                final_bootstrap
                if final_bootstrap is not None
                else torch.zeros_like(ref)
            )
            deltas = []
            for i, q_value in enumerate(q_values):
                next_q = (
                    q_values[i + 1]
                    if i < len(q_values) - 1
                    else final_q
                )
                delta = (
                    reward_tensors[i]
                    + gamma * nonterminals[i] * next_q
                    - alive_tensors[i] * q_value
                )
                deltas.append(delta)

            gae = torch.zeros_like(ref)
            gae_advantages = []
            for i in reversed(range(len(deltas))):
                gae = (
                    deltas[i]
                    + gamma
                    * self.lamda
                    * nonterminals[i]
                    * gae
                )
                gae_advantages.insert(0, gae)
            gae_advantages = torch.stack(gae_advantages).squeeze(-1)

        return returns, gae_advantages
      
    def obs2latent(self, nobs):
        return self.obs_adapter.encode(
            nobs,
            start=0,
            track_grad=not self.obs_adapter.fix_encoder,
        )
        
    def next_obs2latent(self, nobs):
        start = self.n_action_steps - 1 if self.chunk_as_single_action else 0
        return self.obs_adapter.encode(
            nobs,
            start=start,
            track_grad=not self.obs_adapter.fix_encoder,
        )

    def format_samples_for_training(
        self,
        data: Dict,
        nobs_features: torch.Tensor,
        next_nobs_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = data["action"].shape[0]

        if self.predict_delta:
            targets = next_nobs_features - nobs_features
        else:
            targets = next_nobs_features

        actions = self.obs_adapter.normalize_action(data["action"])

        start = self.n_obs_steps - 1

        if self.cfg.chunk_as_single_action:
            end = start + self.n_action_steps
            action = actions[:, start:end]

            if self.predict_r:
                reward_chunk = data["reward"][:, start:end]
                not_done_chunk = None
                done_chunk = None

                if "not_done" in data:
                    not_done_chunk = data["not_done"][:, start:end]
                elif "done" in data:
                    done_chunk = data["done"][:, start:end]

                rewards = self._discounted_chunk_rewards(
                    reward_chunk,
                    not_done_chunk,
                    done_chunk,
                )
        else:
            action = actions[:, start]
            if self.predict_r:
                rewards = data["reward"][:, start]

        model = self.model.module if hasattr(self.model, "module") else self.model

        if self.use_conv_action_embed:
            Da = model._single_action_dim
            action_chunk = action.reshape(batch_size, -1, Da)
            z = model._conv_action_encoder(action_chunk)
            action = model._conv_action_layer_norm(
                z.reshape(batch_size, -1)
            )
        elif self.cfg.use_action_embed:
            action = action.reshape(batch_size, -1)
            action = model._action_encoder(action)
        else:
            action = action.reshape(batch_size, -1)
            if getattr(model, "use_action_scale_norm", False):
                action = model._action_scale_layer_norm(action)

        inputs = torch.cat(
            (nobs_features, action.reshape(batch_size, -1)),
            dim=-1,
        )

        if self.predict_r:
            targets = torch.cat(
                (targets, rewards.reshape(batch_size, -1)),
                dim=-1,
            )

        return inputs, targets
  

    @torch.no_grad()
    def step(
        self,
        nobs_features: torch.Tensor,
        action: torch.Tensor,
        policy_features: torch.Tensor = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        batch_size = nobs_features.shape[0]
        model = self.model.module if hasattr(self.model, "module") else self.model

        if self.use_conv_action_embed:
            action = torch.as_tensor(
                action,
                dtype=torch.float32,
                device=nobs_features.device,
            )
            Da = model._single_action_dim
            action_chunk = action.reshape(batch_size, -1, Da)
            z = model._conv_action_encoder(action_chunk)
            action = model._conv_action_layer_norm(z.reshape(batch_size, -1))
        elif self.cfg.use_action_embed:
            action = action.reshape(batch_size, -1)
            action = model._action_encoder(action)
        else:
            action = torch.as_tensor(
                action,
                dtype=torch.float32,
                device=nobs_features.device,
            ).reshape(batch_size, -1)
            if getattr(model, "use_action_scale_norm", False):
                action = model._action_scale_layer_norm(action)

        action = action.reshape(batch_size, -1)

        if self.prediction_mode == "full" and policy_features is not None:
            input_features = policy_features.reshape(batch_size, -1)
        else:
            input_features = nobs_features

        input_features_np = input_features.cpu().numpy()
        action_np = action.cpu().numpy()
        obs_act = np.concatenate([input_features_np, action_np], axis=-1)

        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        is_diffusion_dynamics = bool(
            getattr(model, "is_diffusion_dynamics", False)
        )

        if self.predict_delta:
            if self.predict_r:
                mean[..., :-1] += input_features_np
            else:
                mean += input_features_np

        std = np.sqrt(np.exp(logvar))
        if is_diffusion_dynamics:
            ensemble_samples = mean.astype(np.float32)
        else:
            ensemble_samples = (
                mean + np.random.normal(size=mean.shape) * std
            ).astype(np.float32)

        _, batch_size, _ = ensemble_samples.shape
        model_idxs = model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]

        if self.predict_r:
            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
        else:
            next_obs = samples
            reward = np.zeros((batch_size, 1), dtype=np.float32)

        terminal = self.terminal_fn(
            input_features_np,
            action_np,
            next_obs,
            self.env,
        )
        info = {"raw_reward": reward}

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError(
                    f"Unsupported uncertainty mode: {self._uncertainty_mode}"
                )

            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty

        if self.prediction_mode == "full":
            feature_dim = next_obs.shape[-1] // self.n_obs_steps
            next_obs = next_obs.reshape(
                batch_size,
                self.n_obs_steps,
                feature_dim,
            )

        return next_obs, reward, terminal, info

    @torch.no_grad()
    def multi_step(
        self,
        single_nob_features: torch.Tensor,
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

        for i in range(n_step_actions):
            action = nactions[:, i, :self.action_dim]

            if Q is not None:
                Qs.append(
                    Q(
                        nobs_features.reshape(batch_size, -1),
                        action,
                    )
                )

            next_obs, reward, terminal, info = self.step(
                single_nob_features,
                action,
                nobs_features,
            )

            if reward_strategy == "sum":
                rewards += reward

            Return += discount * reward
            discount *= self.gamma

            model = self.model.module if hasattr(self.model, "module") else self.model
            device = (
                model._device
                if hasattr(model, "_device")
                else next(model.parameters()).device
            )

            if self.prediction_mode == "full":
                nobs_features = torch.from_numpy(next_obs).to(device)
                single_nob_features = nobs_features[:, -1, :]
            else:
                single_nob_features = torch.from_numpy(next_obs).to(device)
                nobs_features = torch.cat(
                    (
                        nobs_features[:, 1:, :],
                        single_nob_features.unsqueeze(1),
                    ),
                    dim=1,
                )

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
        all_obs_features, rewards, terminals, infos, Qs = [], [], [], [], []
        G, discount = 0, 1
        use_state_dict_q = self._q_requires_raw_obs(Q)

        if use_state_dict_q and n_step_actions > 1:
            raise ValueError(
                "multi_step_evaluation cannot update raw observation inputs "
                "across latent dynamics rollout when Q owns its encoder."
            )

        batch_size = nactions.shape[0]
        policy_features = nobs_features.reshape(
            batch_size,
            self.n_obs_steps,
            -1,
        )
        single_nob_features = policy_features[:, -1, :]

        for i in range(n_step_actions):
            if use_state_dict_q:
                q_input = state_dict
            else:
                q_input = policy_features.reshape(batch_size, -1)

            action = nactions[:, i, :self.action_dim]
            Qs.append(Q(q_input, action))
            all_obs_features.append(policy_features)

            next_obs, reward, terminal, info = self.step(
                single_nob_features,
                action,
                policy_features,
            )

            G += discount * reward
            discount *= self.gamma
            rewards.append(reward)
            terminals.append(terminal)
            infos.append(info)

            model = self.model.module if hasattr(self.model, "module") else self.model
            device = (
                model._device
                if hasattr(model, "_device")
                else next(model.parameters()).device
            )

            if self.prediction_mode == "full":
                policy_features = torch.from_numpy(next_obs).to(device)
                single_nob_features = policy_features[:, -1, :]
            else:
                single_nob_features = torch.from_numpy(next_obs).to(device)
                policy_features = torch.cat(
                    (
                        policy_features[:, 1:, :],
                        single_nob_features.unsqueeze(1),
                    ),
                    dim=1,
                )

        if use_gae:
            G_tensor, gae_advantages = self._return_and_gae(
                rewards,
                terminals,
                q_values=Qs,
                final_bootstrap=None,
                gamma=self.gamma,
            )
            return (
                all_obs_features,
                rewards,
                terminals,
                infos,
                G_tensor.squeeze(-1),
                gae_advantages,
            )

        bootstrap_q = Qs[-1].detach().cpu()
        G_tensor = (
            torch.as_tensor(G, dtype=bootstrap_q.dtype)
            + discount * bootstrap_q
        )

        return (
            all_obs_features,
            rewards,
            terminals,
            infos,
            G_tensor.squeeze(),
            None,
        )

    @torch.no_grad()
    def chunk_evaluation(
        self,
        nobs_features: torch.Tensor,
        nactions: torch.Tensor,
        Q: Callable,
        state_dict: Dict = None,
        use_gae: bool = False,
    ):
        batch_size = nactions.shape[0]
        policy_features = nobs_features.reshape(
            batch_size,
            self.n_obs_steps,
            -1,
        )

        use_state_dict_q = self._q_requires_raw_obs(Q)

        if use_state_dict_q:
            if state_dict is None:
                raise ValueError(
                    "chunk_evaluation requires raw state_dict when Q owns "
                    "its observation encoder."
                )
            q_input = state_dict
        else:
            q_input = policy_features.reshape(batch_size, -1)

        q_owner = getattr(Q, "__self__", None)

        if q_owner is not None and hasattr(q_owner, "get_advantage"):
            return q_owner.get_advantage(q_input, nactions)

        return Q(q_input, nactions)

    def learn(
        self,
        batch: dict,
        nobs_features: torch.Tensor,
        next_nobs_features: torch.Tensor,
        logvar_loss_coef: float = 0.01,
    ) -> float:
        self.model.train()
        model = self.model.module if hasattr(self.model, "module") else self.model

        if self.use_conv_action_embed and not hasattr(model, "compute_loss"):
            batch_size = nobs_features.shape[0]

            if self.predict_delta:
                targets = next_nobs_features - nobs_features
            else:
                targets = next_nobs_features

            actions = self.obs_adapter.normalize_action(batch["action"])

            if self.cfg.chunk_as_single_action:
                start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                action = actions[:, start:end]

                if self.predict_r:
                    reward_chunk = batch["reward"][:, start:end]
                    not_done_chunk = None
                    done_chunk = None

                    if "not_done" in batch:
                        not_done_chunk = batch["not_done"][:, start:end]
                    elif "done" in batch:
                        done_chunk = batch["done"][:, start:end]

                    rewards = self._discounted_chunk_rewards(
                        reward_chunk,
                        not_done_chunk,
                        done_chunk,
                    )
            else:
                action = actions[:, self.n_obs_steps - 1]

                if self.predict_r:
                    rewards = batch["reward"][:, self.n_obs_steps - 1]

            if self.predict_r:
                targets = torch.cat(
                    (targets, rewards.reshape(batch_size, -1)),
                    dim=-1,
                )

            Da = model._single_action_dim
            action_chunk = action.reshape(batch_size, -1, Da)
            action_recon_beta = getattr(
                self.cfg,
                "action_recon_beta",
                0.5,
            )

            return self.model(
                nobs_features,
                targets=targets,
                action_chunk=action_chunk,
                logvar_loss_coef=logvar_loss_coef,
                action_recon_beta=action_recon_beta,
            )

        inputs_batch, targets_batch = self.format_samples_for_training(
            batch,
            nobs_features,
            next_nobs_features,
        )

        if hasattr(model, "compute_loss"):
            loss = model.compute_loss(
                inputs_batch,
                targets_batch,
            )
        else:
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            mse_loss_inv = (
                (mean - targets_batch).pow(2) * inv_var
            ).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))

            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + model.get_decay_loss()
            loss = (
                loss
                + logvar_loss_coef * model.max_logvar.sum()
                - logvar_loss_coef * model.min_logvar.sum()
            )

        return loss

    def optimize(self, loss: float) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
    @torch.no_grad()
    def validate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> List[float]:
        self.model.eval()
        model = self.model.module if hasattr(self.model, "module") else self.model
        device = (
            model._device
            if hasattr(model, "_device")
            else next(model.parameters()).device
        )
        targets = torch.as_tensor(targets).to(device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        return list(loss.cpu().numpy())

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [
            (metric, index)
            for metric, index in zip(metrics, range(len(metrics)))
        ]
        pairs = sorted(pairs, key=lambda x: x[0])
        model = self.model.module if hasattr(self.model, "module") else self.model
        return [pairs[i][1] for i in range(model.num_elites)]

    def post_well_learned(self) -> None:
        indexes = self.select_elites(self.holdout_losses)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.set_elites(indexes)
        model.load_save()
        self.logger.log(
            "elites:{} , holdout loss: {}".format(
                indexes,
                np.sort(self.holdout_losses)[:model.num_elites].mean(),
            )
        )
        self.save(self.logger.model_dir)
        self.model.eval()

    @torch.no_grad()
    def validation(
        self,
        holdout_data: dict,
        nobs_features: torch.Tensor,
        next_nobs_features: torch.Tensor,
        train_loss: float,
        wandb,
        epoch,
        max_epochs_since_update=5,
        max_epochs=500,
    ) -> list:
        holdout_inputs, holdout_targets = self.format_samples_for_training(
            data=holdout_data,
            nobs_features=nobs_features,
            next_nobs_features=next_nobs_features,
        )
        new_holdout_losses = self.validate(
            holdout_inputs,
            holdout_targets,
        )
        return self._update_holdout_and_log(
            new_holdout_losses,
            train_loss,
            wandb,
            epoch,
            max_epochs_since_update,
            max_epochs,
        )

    def _update_holdout_and_log(
        self,
        new_holdout_losses: list,
        train_loss: float,
        wandb,
        epoch,
        max_epochs_since_update=5,
        max_epochs=500,
    ) -> bool:
        model = self.model.module if hasattr(self.model, "module") else self.model
        holdout_loss = np.sort(new_holdout_losses)[:model.num_elites].mean()

        self.logger.logkv("loss/dynamics_train_loss", train_loss)
        self.logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
        self.logger.set_timestep(epoch)
        self.logger.dumpkvs(exclude=["policy_training_progress"])

        wandb.log({"loss/dynamics_train_loss": train_loss})
        wandb.log({"loss/dynamics_holdout_loss": holdout_loss})

        indexes = []
        for i, new_loss, old_loss in zip(
            range(len(self.holdout_losses)),
            new_holdout_losses,
            self.holdout_losses,
        ):
            improvement = (old_loss - new_loss) / old_loss
            if improvement > 0.01:
                indexes.append(i)
                self.holdout_losses[i] = new_loss

        if indexes:
            model.update_save(indexes)
            self.cnt = 0
        else:
            self.cnt += 1

        if self.cnt >= max_epochs_since_update:
            return True
        if max_epochs and epoch >= max_epochs:
            return True
        return False

    def save(self, save_path: str) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            model.state_dict(),
            os.path.join(save_path, "dynamics.pth"),
        )
        print(f"dynamics model saved in {save_path}")
        
    def load(self, load_path: str) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        device = (
            model._device
            if hasattr(model, "_device")
            else next(model.parameters()).device
        )
        model.load_state_dict(
            torch.load(
                os.path.join(load_path, "dynamics.pth"),
                map_location=device,
            )
        )
        print(f"dynamics model loaded from {load_path}")
        
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
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if use_gae:
            raise NotImplementedError(
                "ACT V1 rollout does not support recursive GAE because "
                "ACT cannot consume predicted dynamics latent."
            )

        q_eval = iql.minQ if is_iql else Q
        policy_batch = self.obs_adapter.normalize_obs(batch["obs"])
        actions, _, _ = policy.sample_action_chunk(policy_batch)

        batch_size = actions.shape[0]
        nobs_features = self.obs2latent(batch["obs"])
        policy_features = nobs_features.reshape(
            batch_size,
            self.n_obs_steps,
            -1,
        )
        single_nob_features = policy_features[:, -1, :]
        rewards_arr = np.array([])
        rollout_qs = []

        if self.chunk_as_single_action:
            if rollout_length != 1:
                raise ValueError(
                    "ACT V1 chunk dynamics supports rollout_length=1 only."
                )

            actions = actions[:, :self.n_action_steps]
            rollout_qs.append(
                q_eval(
                    policy_features.reshape(batch_size, -1),
                    actions,
                )
            )
            _, reward, _, _ = self.step(
                single_nob_features,
                actions,
                policy_features,
            )
            rewards_arr = np.append(rewards_arr, reward.flatten())
        else:
            if rollout_length < 1 or rollout_length > actions.shape[1]:
                raise ValueError(
                    f"ACT V1 single-step dynamics requires rollout_length "
                    f"in [1, {actions.shape[1]}], got {rollout_length}."
                )

            model = self.model.module if hasattr(self.model, "module") else self.model
            device = (
                model._device
                if hasattr(model, "_device")
                else next(model.parameters()).device
            )

            for i in range(rollout_length):
                action_i = actions[:, i, :self.action_dim]
                rollout_qs.append(
                    q_eval(
                        policy_features.reshape(batch_size, -1),
                        action_i,
                    )
                )
                next_obs, reward, _, _ = self.step(
                    single_nob_features,
                    action_i,
                    policy_features,
                )
                rewards_arr = np.append(rewards_arr, reward.flatten())

                if self.prediction_mode == "full":
                    policy_features = torch.from_numpy(next_obs).to(device)
                    single_nob_features = policy_features[:, -1, :]
                else:
                    single_nob_features = torch.from_numpy(next_obs).to(device)
                    policy_features = torch.cat(
                        (
                            policy_features[:, 1:, :],
                            single_nob_features.unsqueeze(1),
                        ),
                        dim=1,
                    )

        q_evaluation = torch.mean(torch.stack(rollout_qs))
        return q_evaluation, rewards_arr.mean()