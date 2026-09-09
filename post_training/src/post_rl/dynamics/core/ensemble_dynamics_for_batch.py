import os
from typing import Callable, Dict, List, Tuple

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
        prediction_mode: str = "full",
    ) -> None:
        super().__init__(model, optim)
        if not chunk_as_single_action:
            raise ValueError(
                "ACT transformer-latent dynamics requires chunk_as_single_action=true."
            )
        if obs_adapter.n_obs_steps != 1:
            raise ValueError(
                "ACT transformer-latent dynamics currently requires n_obs_steps=1."
            )
        if not getattr(self._unwrap(model), "token_dynamics", False):
            raise TypeError("Expected EnsembleTokenDynamicsModel.")

        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.env = env
        self.obs_adapter = obs_adapter
        self.cfg = cfg
        self.n_obs_steps = 1
        self.predict_delta = cfg.dynamics.predict_delta
        self.chunk_as_single_action = True
        self.n_action_steps = n_action_steps
        self.prediction_mode = "full"
        self.lamda = lamda
        self.gamma = gamma
        self.action_dim = action_dim
        self.predict_r = cfg.predict_r
        self.dynamics_type = cfg.dynamics_type
        self.device = torch.device(device)
        self.cnt = 0
        self.holdout_losses = [
            1e10 for _ in range(self._model().num_ensemble)
        ]

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _model(self) -> nn.Module:
        return self._unwrap(self.model)

    def set_logger(self, logger) -> None:
        self.logger = logger
        self.logger.log("Training dynamics:")

    def obs2latent(self, nobs) -> torch.Tensor:
        latent = self.obs_adapter.encode(
            nobs,
            start=0,
            track_grad=not self.obs_adapter.fix_encoder,
        )
        return latent[:, 0]

    def next_obs2latent(self, nobs) -> torch.Tensor:
        latent = self.obs_adapter.encode(
            nobs,
            start=self.n_action_steps - 1,
            track_grad=not self.obs_adapter.fix_encoder,
        )
        return latent[:, 0]

    def _dataset_chunk_action(self, batch: Dict) -> torch.Tensor:
        actions = self.obs_adapter.normalize_action(batch["action"])
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        if actions.shape[1] < end:
            raise ValueError(
                f"Dynamics needs action horizon >= {end}, got {actions.shape[1]}."
            )
        return actions[:, start:end]

    def _targets(
        self,
        state_tokens: torch.Tensor,
        next_state_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if state_tokens.shape != next_state_tokens.shape:
            raise ValueError(
                f"Dynamics latent shape mismatch: {tuple(state_tokens.shape)} vs "
                f"{tuple(next_state_tokens.shape)}"
            )
        return (
            next_state_tokens - state_tokens
            if self.predict_delta
            else next_state_tokens
        )

    def learn(
        self,
        batch: Dict,
        nobs_features: torch.Tensor,
        next_nobs_features: torch.Tensor,
        logvar_loss_coef: float = 0.01,
    ) -> torch.Tensor:
        if self.predict_r:
            raise NotImplementedError(
                "RL-100-style ACT latent OPE uses predict_r=false."
            )

        targets = self._targets(nobs_features, next_nobs_features)
        action = self._dataset_chunk_action(batch)
        mean, logvar = self.model(nobs_features, action)
        inv_var = torch.exp(-logvar)
        reduce_dims = tuple(range(1, mean.ndim))
        mse_loss_inv = ((mean - targets).pow(2) * inv_var).mean(dim=reduce_dims)
        var_loss = logvar.mean(dim=reduce_dims)
        model = self._model()
        loss = mse_loss_inv.sum() + var_loss.sum()
        loss = loss + model.get_decay_loss()
        loss = (
            loss
            + logvar_loss_coef * model.max_logvar.sum()
            - logvar_loss_coef * model.min_logvar.sum()
        )
        return loss

    def optimize(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    @torch.no_grad()
    def validate_batch(
        self,
        batch: Dict,
        nobs_features: torch.Tensor,
        next_nobs_features: torch.Tensor,
    ) -> List[float]:
        self.model.eval()
        targets = self._targets(nobs_features, next_nobs_features)
        action = self._dataset_chunk_action(batch)
        mean, _ = self.model(nobs_features, action)
        reduce_dims = tuple(range(1, mean.ndim))
        loss = ((mean - targets) ** 2).mean(dim=reduce_dims)
        return list(loss.cpu().numpy())

    @torch.no_grad()
    def step(
        self,
        nobs_features: torch.Tensor,
        action: torch.Tensor,
        policy_features: torch.Tensor | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        state_tokens = torch.as_tensor(
            policy_features if policy_features is not None else nobs_features,
            device=self.device,
            dtype=torch.float32,
        )
        if state_tokens.ndim == 4 and state_tokens.shape[1] == 1:
            state_tokens = state_tokens[:, 0]
        if state_tokens.ndim != 3:
            raise ValueError(
                f"Dynamics step expects ACT latent [B,S,D], got "
                f"{tuple(state_tokens.shape)}"
            )

        action = torch.as_tensor(
            action,
            device=self.device,
            dtype=torch.float32,
        )
        mean, logvar = self.model(state_tokens, action)
        if self.predict_delta:
            mean = mean + state_tokens.unsqueeze(0)

        std = torch.sqrt(torch.exp(logvar))
        samples = mean + torch.randn_like(std) * std
        model = self._model()
        batch_size = state_tokens.shape[0]
        model_idxs = model.random_elite_idxs(batch_size)
        batch_idx = torch.arange(batch_size, device=self.device)
        elite_idx = torch.as_tensor(model_idxs, device=self.device)
        next_state = samples[elite_idx, batch_idx]

        reward = np.zeros((batch_size, 1), dtype=np.float32)
        state_np = state_tokens.detach().cpu().numpy().reshape(batch_size, -1)
        next_np_flat = next_state.detach().cpu().numpy().reshape(batch_size, -1)
        action_np = action.detach().cpu().numpy().reshape(batch_size, -1)
        terminal = self.terminal_fn(
            state_np,
            action_np,
            next_np_flat,
            self.env,
        )
        info = {"raw_reward": reward}

        if self._penalty_coef:
            std_np = std.detach().cpu().numpy().reshape(
                std.shape[0],
                batch_size,
                -1,
            )
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std_np, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                mean_np = mean.detach().cpu().numpy().reshape(
                    mean.shape[0],
                    batch_size,
                    -1,
                )
                penalty = np.sqrt(mean_np.var(0).mean(1))
            else:
                raise ValueError(
                    f"Unsupported uncertainty mode: {self._uncertainty_mode}"
                )
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty

        return next_state.cpu().numpy(), reward, terminal, info

    @torch.no_grad()
    def chunk_evaluation(
        self,
        nobs_features: torch.Tensor,
        nactions: torch.Tensor,
        Q: Callable,
        state_dict: Dict = None,
        use_gae: bool = False,
    ):
        q_owner = getattr(Q, "__self__", None)
        if q_owner is not None and hasattr(q_owner, "get_advantage"):
            return q_owner.get_advantage(nobs_features, nactions)
        return Q(nobs_features, nactions)

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
                "GAE-based OPE is not part of the RL-100-aligned ACT V1 path."
            )
        if rollout_length < 1:
            raise ValueError("rollout_length must be >= 1")

        q_eval = iql.minQ if is_iql else Q
        policy_batch = self.obs_adapter.normalize_obs(batch["obs"])
        _, encoder_pos_embed = policy.encode_observation(policy_batch)
        policy_features = self.obs2latent(batch["obs"])
        rollout_qs = []
        rewards_arr = []

        for _ in range(int(rollout_length)):
            actions, _, _ = policy.sample_action_chunk_from_latent(
                policy_features,
                encoder_pos_embed,
            )
            action_chunk = actions[:, :self.n_action_steps]
            rollout_qs.append(q_eval(policy_features, action_chunk))
            next_obs, reward, _, _ = self.step(
                policy_features,
                action_chunk,
            )
            rewards_arr.append(np.asarray(reward).reshape(-1))
            policy_features = torch.as_tensor(
                next_obs,
                device=self.device,
                dtype=policy_features.dtype,
            )

        q_evaluation = torch.mean(torch.stack(rollout_qs))
        reward_mean = (
            float(np.concatenate(rewards_arr).mean())
            if rewards_arr
            else 0.0
        )
        return q_evaluation, reward_mean

    @torch.no_grad()
    def compute_model_uncertainty(
        self,
        obs,
        action,
        uncertainty_mode: str = "aleatoric",
    ) -> np.ndarray:
        state_tokens = (
            self.obs2latent(obs)
            if isinstance(obs, dict)
            else torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        )
        mean, logvar = self.model(state_tokens, action)
        if self.predict_delta:
            mean = mean + state_tokens.unsqueeze(0)
        std = torch.sqrt(torch.exp(logvar))
        flat_std = std.cpu().numpy().reshape(std.shape[0], std.shape[1], -1)
        if uncertainty_mode == "aleatoric":
            penalty = np.amax(np.linalg.norm(flat_std, axis=2), axis=0)
        elif uncertainty_mode == "ensemble_std":
            flat_mean = mean.cpu().numpy().reshape(mean.shape[0], mean.shape[1], -1)
            penalty = np.sqrt(flat_mean.var(0).mean(1))
        else:
            raise ValueError(f"Unsupported uncertainty mode: {uncertainty_mode}")
        return self._penalty_coef * np.expand_dims(penalty, 1).astype(np.float32)

    def select_elites(self, metrics: List) -> List[int]:
        pairs = sorted((metric, index) for index, metric in enumerate(metrics))
        return [pairs[i][1] for i in range(self._model().num_elites)]

    def _update_holdout_and_log(
        self,
        new_holdout_losses: list,
        train_loss: float,
        wandb,
        epoch,
        max_epochs_since_update=5,
        max_epochs=500,
    ) -> bool:
        model = self._model()
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

        return (
            self.cnt >= max_epochs_since_update
            or bool(max_epochs and epoch >= max_epochs)
        )

    def post_well_learned(self) -> None:
        indexes = self.select_elites(self.holdout_losses)
        model = self._model()
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

    def save(self, save_path: str) -> None:
        model = self._model()
        torch.save(
            model.state_dict(),
            os.path.join(save_path, "dynamics.pth"),
        )

    def load(self, load_path: str) -> None:
        model = self._model()
        model.load_state_dict(
            torch.load(
                os.path.join(load_path, "dynamics.pth"),
                map_location=self.device,
            )
        )
