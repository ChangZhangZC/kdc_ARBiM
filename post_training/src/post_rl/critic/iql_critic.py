import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .networks import ACTCriticEncoder, ValueMLP, QMLP, DoubleQMLP
from lerobot.utils.constants import OBS_STATE

RGB_BUFFER_TO_FEATURE = {
    "head_rgb": "observation.images.head_cam_h",
    "wrist_left_rgb": "observation.images.wrist_cam_l",
    "wrist_right_rgb": "observation.images.wrist_cam_r",
}

DEPTH_BUFFER_TO_FEATURE = {
    "head_depth": "observation.depth_h",
    "wrist_left_depth": "observation.depth_l",
    "wrist_right_depth": "observation.depth_r",
}

ACTION_FEATURE = "action"


class IQLCritic(nn.Module):
    def __init__(
        self,
        device: torch.device,
        obs_encoder: ACTCriticEncoder,
        stats: dict,
        action_dim: int,
        feature_dim: int,
        q_hidden_dim: int,
        q_depth: int,
        q_lr: float,
        v_hidden_dim: int,
        v_depth: int,
        v_lr: float,
        omega: float = 0.7,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_freq: int = 2,
        is_double_q: bool = True,
        is_share_encoder: bool = True,
        fix_encoder: bool = True,
        encoder_update_with: str = "value",
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        chunk_as_single_action: bool = False,
        use_action_embed: bool = False,
        use_conv_action_embed: bool = False,
        conv_hidden_dims: list[int] = [128, 256],
        conv_latent_cz: int = 32,
        conv_kernel_size: int = 5,
        conv_n_groups: int = 8,
        action_recon_beta: float = 0.5,
        q_layer_norm: bool = False,
        action_embed_layer_norm: bool = False,
        action_scale_norm: bool = False,
    ) -> None:
        super().__init__()

        if encoder_update_with not in {"value", "q", "both"}:
            raise ValueError("encoder_update_with must be 'value', 'q', or 'both'")
        if n_obs_steps < 1 or n_action_steps < 1:
            raise ValueError("n_obs_steps and n_action_steps must be >= 1")

        self._device = device
        self.stats = stats
        self._omega = omega
        self._gamma = gamma
        self._tau = tau
        self._target_update_freq = target_update_freq
        self._total_update_step = 0
        self._is_double_q = is_double_q

        self.obs_encoder = obs_encoder
        self.is_share_encoder = is_share_encoder
        self.fix_encoder = fix_encoder
        self.encoder_update_with = encoder_update_with
        self.n_obs_steps = n_obs_steps

        self.chunk_as_single_action = chunk_as_single_action
        self.n_action_steps = n_action_steps if chunk_as_single_action else 1
        self.single_action_dim = action_dim
        self.action_dim = action_dim * self.n_action_steps
        self.action_recon_beta = action_recon_beta

        self.state_dim = obs_encoder.output_dim * n_obs_steps
        
        self.use_conv_action_embed = use_conv_action_embed

        if is_share_encoder:
            q_encoder = None
            target_q_encoder = None
            v_encoder = None
        else:
            q_encoder = deepcopy(obs_encoder)
            target_q_encoder = deepcopy(obs_encoder)
            v_encoder = deepcopy(obs_encoder)

        q_kwargs = dict(
            use_action_embed=use_action_embed,
            state_dim=self.state_dim,
            feature_dim=feature_dim,
            action_dim=self.action_dim,
            hidden_dim=q_hidden_dim,
            depth=q_depth,
            fix_encoder=fix_encoder,
            use_conv_action_embed=use_conv_action_embed,
            single_action_dim=action_dim,
            n_action_steps=self.n_action_steps,
            conv_hidden_dims=conv_hidden_dims,
            conv_latent_cz=conv_latent_cz,
            conv_kernel_size=conv_kernel_size,
            conv_n_groups=conv_n_groups,
            q_layer_norm=q_layer_norm,
            action_embed_layer_norm=action_embed_layer_norm,
            action_scale_norm=action_scale_norm,
        )

        q_cls = DoubleQMLP if is_double_q else QMLP
        
        self._Q = q_cls(obs_encoder=q_encoder, **q_kwargs).to(device)
        self._target_Q = q_cls(obs_encoder=target_q_encoder, **q_kwargs).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._target_Q.requires_grad_(False)
        self._target_Q.eval()

        self._value = ValueMLP(
            obs_encoder=v_encoder,
            state_dim=self.state_dim,
            hidden_dim=v_hidden_dim,
            depth=v_depth,
            n_obs_steps=n_obs_steps,
            fix_encoder=fix_encoder,
        ).to(device)

        self._build_optimizers(q_lr, v_lr)
        self.train()
        
    def _build_optimizers(self, q_lr: float, v_lr: float) -> None:
        if self.is_share_encoder and self.fix_encoder:
            self.obs_encoder.eval()
            for param in self.obs_encoder.parameters():
                param.requires_grad = False

        if not self.is_share_encoder or self.fix_encoder:
            self._q_optimizer = torch.optim.Adam(self._Q.parameters(), lr=q_lr)
            self._v_optimizer = torch.optim.Adam(self._value.parameters(), lr=v_lr)
            return

        if self.encoder_update_with == "value":
            self._q_optimizer = torch.optim.Adam(self._Q.parameters(), lr=q_lr)
            self._v_optimizer = torch.optim.Adam(
                list(self._value.parameters()) + list(self.obs_encoder.parameters()),
                lr=v_lr,
            )
        elif self.encoder_update_with == "q":
            self._q_optimizer = torch.optim.Adam(
                list(self._Q.parameters()) + list(self.obs_encoder.parameters()),
                lr=q_lr,
            )
            self._v_optimizer = torch.optim.Adam(self._value.parameters(), lr=v_lr)
        else:
            self._q_optimizer = torch.optim.Adam(self._Q.parameters(), lr=q_lr)
            self._v_optimizer = torch.optim.Adam(self._value.parameters(), lr=v_lr)
            self._encoder_optimizer = torch.optim.Adam(
                self.obs_encoder.parameters(),
                lr=min(q_lr, v_lr),
            )
            
    def train(self, mode: bool = True):
        super().train(mode)

        # Target Q 永远只作为稳定的 Bellman / IQL target。
        self._target_Q.eval()

        if not self.fix_encoder:
            return self

        if self.is_share_encoder:
            self.obs_encoder.eval()
        else:
            for model in (self._Q, self._value):
                encoder = getattr(model, "_obs_encoder", None)
                if encoder is not None:
                    encoder.eval()

        return self
    
    def _get_stat(
        self,
        feature: str,
        name: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if feature not in self.stats:
            raise KeyError(f"Missing stats for {feature}")
        if name not in self.stats[feature]:
            raise KeyError(f"Missing {name} stats for {feature}")
        return torch.as_tensor(
            self.stats[feature][name],
            device=ref.device,
            dtype=torch.float32,
        )
        
    def _normalize_mean_std(
        self,
        x: torch.Tensor,
        feature: str,
    ) -> torch.Tensor:
        x = x.float()
        mean = self._get_stat(feature, "mean", x)
        std = self._get_stat(feature, "std", x)

        if x.ndim >= 4 and mean.ndim == 1:
            shape = [1] * x.ndim
            shape[-3] = mean.shape[0]
            mean = mean.view(*shape)
            std = std.view(*shape)

        return (x - mean) / (std + 1e-8)
    
    def _normalize_min_max(
        self,
        x: torch.Tensor,
        feature: str,
    ) -> torch.Tensor:
        x = x.float()
        min_v = self._get_stat(feature, "min", x)
        max_v = self._get_stat(feature, "max", x)

        if x.ndim >= 4 and min_v.ndim == 1:
            shape = [1] * x.ndim
            shape[-3] = min_v.shape[0]
            min_v = min_v.view(*shape)
            max_v = max_v.view(*shape)

        return 2.0 * (x - min_v) / (max_v - min_v + 1e-8) - 1.0
    
    def _normalize_obs(
        self,
        obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        normalized = {
            OBS_STATE: self._normalize_mean_std(
                obs["state"],
                OBS_STATE,
            )
        }

        for buffer_key, feature_key in RGB_BUFFER_TO_FEATURE.items():
            if buffer_key in obs:
                normalized[feature_key] = self._normalize_mean_std(
                    obs[buffer_key],
                    feature_key,
                )

        for buffer_key, feature_key in DEPTH_BUFFER_TO_FEATURE.items():
            if buffer_key in obs:
                normalized[feature_key] = self._normalize_min_max(
                    obs[buffer_key],
                    feature_key,
                )

        return normalized
    
    def _normalize_action(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self._normalize_mean_std(action, ACTION_FEATURE)
        
    def _prepare_obs(
        self,
        obs: dict[str, torch.Tensor],
        start: int = 0,
    ) -> tuple[dict[str, torch.Tensor], int]:
        obs = self._normalize_obs(obs)
        first = next(iter(obs.values()))
        batch_size = first.shape[0]
        end = start + self.n_obs_steps

        prepared = {}
        for key, value in obs.items():
            if value.shape[1] < end:
                raise ValueError(
                    f"{key} needs at least {end} steps, got {value.shape[1]}"
                )

            value = value[:, start:end]
            prepared[key] = value.reshape(
                batch_size * self.n_obs_steps,
                *value.shape[2:],
            )

        return prepared, batch_size

    
    def _encode_shared_obs(
        self,
        obs: dict[str, torch.Tensor],
        start: int = 0,
        track_grad: bool | None = None,
    ) -> torch.Tensor:
        obs, batch_size = self._prepare_obs(obs, start)
        if track_grad is None:
            track_grad = not self.fix_encoder

        if track_grad:
            features = self.obs_encoder(obs)
        else:
            with torch.no_grad():
                features = self.obs_encoder(obs)

        return features.reshape(batch_size, self.state_dim)
    
    def _prepare_action(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = action.reshape(action.shape[0], -1)

        if action.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, "
                f"got {action.shape[1]}"
            )

        return action
    
    def minQ(
        self,
        s: torch.Tensor | dict[str, torch.Tensor],
        a: torch.Tensor,
    ) -> torch.Tensor:
        a = self._prepare_action(a)

        if isinstance(s, dict):
            if self.is_share_encoder:
                s = self._encode_shared_obs(s)
            else:
                s, _ = self._prepare_obs(s)

        if self._is_double_q:
            q1, q2 = self._Q(s, a)
            return torch.min(q1, q2)

        return self._Q(s, a)
    
    def target_minQ(
        self,
        s: torch.Tensor | dict[str, torch.Tensor],
        a: torch.Tensor,
    ) -> torch.Tensor:
        a = self._prepare_action(a)

        if isinstance(s, dict):
            if self.is_share_encoder:
                s = self._encode_shared_obs(s)
            else:
                s, _ = self._prepare_obs(s)

        if self._is_double_q:
            q1, q2 = self._target_Q(s, a)
            return torch.min(q1, q2)

        return self._target_Q(s, a)
    
    def expectile_loss(
        self,
        error: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.where(
            error > 0,
            self._omega,
            1.0 - self._omega,
        )
        return weight * error.pow(2)
    
    @staticmethod
    def _as_column(
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        tensor = tensor.reshape(tensor.shape[0], -1)

        if tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must be scalar per sample, "
                f"got {tuple(tensor.shape)}"
            )

        return tensor
    
    def _select_transition(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = self._normalize_action(batch["action"])
        batch_size = actions.shape[0]
        start = self.n_obs_steps - 1

        if not self.chunk_as_single_action:
            if actions.shape[1] <= start:
                raise ValueError("Action horizon is shorter than n_obs_steps")

            action = actions[:, start]
            reward = self._as_column(batch["reward"][:, start], "reward")
            not_done = self._as_column(batch["not_done"][:, start], "not_done")
            return action, reward, not_done

        end = start + self.n_action_steps
        if actions.shape[1] < end:
            raise ValueError(
                f"Chunk critic needs action horizon >= {end}, got {actions.shape[1]}"
            )

        action = actions[:, start:end]
        rewards = batch["reward"][:, start:end].reshape(
            batch_size,
            self.n_action_steps,
            -1,
        )

        if rewards.shape[-1] != 1:
            raise ValueError(
                f"Reward must be scalar per step, got {tuple(rewards.shape)}"
            )

        rewards = rewards.squeeze(-1)
        gamma = torch.pow(
            self._gamma,
            torch.arange(
                self.n_action_steps,
                device=rewards.device,
                dtype=rewards.dtype,
            ),
        )

        reward = (rewards * gamma).sum(dim=1, keepdim=True)
        not_done = self._as_column(
            batch["not_done"][:, end - 1],
            "not_done",
        )

        return action, reward, not_done
    
    def _prepare_nonshared_obs(
        self,
        batch: dict,
    ) -> tuple[dict, dict]:
        obs, _ = self._prepare_obs(batch["obs"])

        next_start = (
            self.n_action_steps - 1
            if self.chunk_as_single_action
            else 0
        )

        next_obs, _ = self._prepare_obs(
            batch["next_obs"],
            start=next_start,
        )

        return obs, next_obs
    
    def _update_value(
        self,
        batch: dict,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_share_encoder:
            update_encoder = (
                not self.fix_encoder
                and self.encoder_update_with in {"value", "both"}
            )

            state = self._encode_shared_obs(
                batch["obs"],
                track_grad=update_encoder,
            )

            with torch.no_grad():
                target_q = self.target_minQ(state, action)

            value = self._value(state)

        else:
            obs, _ = self._prepare_nonshared_obs(batch)

            with torch.no_grad():
                target_q = self.target_minQ(batch["obs"], action)

            value = self._value(obs)

        value_loss = self.expectile_loss(
            target_q - value
        ).mean()

        self._v_optimizer.zero_grad()

        if (
            self.is_share_encoder
            and not self.fix_encoder
            and self.encoder_update_with == "both"
        ):
            self._encoder_optimizer.zero_grad()

        value_loss.backward()
        self._v_optimizer.step()

        return value_loss
    
    def _update_q(
        self,
        batch: dict,
        action: torch.Tensor,
        reward: torch.Tensor,
        not_done: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_share_encoder:
            update_encoder = (
                not self.fix_encoder
                and self.encoder_update_with in {"q", "both"}
            )

            state = self._encode_shared_obs(
                batch["obs"],
                track_grad=update_encoder,
            )

            next_start = (
                self.n_action_steps - 1
                if self.chunk_as_single_action
                else 0
            )

            next_state = self._encode_shared_obs(
                batch["next_obs"],
                start=next_start,
                track_grad=False,
            )

            q_input = state

            with torch.no_grad():
                next_v = self._value(next_state)

        else:
            obs, next_obs = self._prepare_nonshared_obs(batch)
            q_input = obs

            with torch.no_grad():
                next_v = self._value(next_obs)

        target_q = (reward + not_done * (self._gamma ** self.n_action_steps) * next_v)

        action_recon_loss = None
        if self._is_double_q:
            if self.use_conv_action_embed:
                q1, q2, action_recon_loss = self._Q(q_input,action,return_action_recon_loss=True,)
            else:
                q1, q2 = self._Q(q_input, action)

            q_loss = ((q1 - target_q).pow(2) + (q2 - target_q).pow(2)).mean()

        else:
            if self.use_conv_action_embed:
                q, action_recon_loss = self._Q(
                    q_input,
                    action,
                    return_action_recon_loss=True,
                )
            else:
                q = self._Q(q_input, action)

            q_loss = F.mse_loss(q, target_q)

        if action_recon_loss is not None:
            q_loss = (
                q_loss
                + self.action_recon_beta * action_recon_loss
            )

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        if (
            self.is_share_encoder
            and not self.fix_encoder
            and self.encoder_update_with == "both"
        ):
            self._encoder_optimizer.step()

        return q_loss
    
    def _update_target_q(self) -> None:
        self._total_update_step += 1

        if self._total_update_step % self._target_update_freq != 0:
            return

        with torch.no_grad():
            for param, target_param in zip(
                self._Q.parameters(),
                self._target_Q.parameters(),
            ):
                target_param.data.mul_(1.0 - self._tau)
                target_param.data.add_(self._tau * param.data)

    def update(
        self,
        batch: dict,
    ) -> tuple[float, float]:
        action, reward, not_done = self._select_transition(batch)

        value_loss = self._update_value(
            batch,
            action,
        )

        q_loss = self._update_q(
            batch,
            action,
            reward,
            not_done,
        )

        self._update_target_q()

        return q_loss.item(), value_loss.item()

    def value(
        self,
        s: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(s, dict):
            return self._value(s)

        if self.is_share_encoder:
            s = self._encode_shared_obs(s)
            return self._value(s)

        s, _ = self._prepare_obs(s)
        return self._value(s)    
    
    @torch.no_grad()
    def get_advantage(
        self,
        s: torch.Tensor | dict[str, torch.Tensor],
        a: torch.Tensor,
    ) -> torch.Tensor:
        a = self._prepare_action(a)

        if self.is_share_encoder:
            if isinstance(s, dict):
                s = self._encode_shared_obs(s)

            if self._is_double_q:
                q1, q2 = self._Q(s, a)
                q = torch.min(q1, q2)
            else:
                q = self._Q(s, a)

            v = self._value(s)
            return q - v

        if isinstance(s, dict):
            s, _ = self._prepare_obs(s)

        if self._is_double_q:
            q1, q2 = self._Q(s, a)
            q = torch.min(q1, q2)
        else:
            q = self._Q(s, a)

        v = self._value(s)
        return q - v
    
    def save(
        self,
        q_path: str,
        v_path: str,
        encoder_path: str | None = None,
    ) -> None:
        q_model = self._Q.module if hasattr(self._Q, "module") else self._Q
        v_model = self._value.module if hasattr(self._value, "module") else self._value

        torch.save(q_model.state_dict(), q_path)
        torch.save(v_model.state_dict(), v_path)

        if self.is_share_encoder and encoder_path is not None:
            encoder = (
                self.obs_encoder.module
                if hasattr(self.obs_encoder, "module")
                else self.obs_encoder
            )
            torch.save(encoder.state_dict(), encoder_path)
            
    def load(
        self,
        q_path: str,
        v_path: str,
        encoder_path: str | None = None,
    ) -> None:
        self._Q.load_state_dict(torch.load(q_path, map_location=self._device))

        self._target_Q.load_state_dict(self._Q.state_dict())

        self._value.load_state_dict(torch.load(v_path, map_location=self._device))

        if self.is_share_encoder and encoder_path is not None:
            self.obs_encoder.load_state_dict(
                torch.load(
                    encoder_path,
                    map_location=self._device,
                )
            )

        self.train()