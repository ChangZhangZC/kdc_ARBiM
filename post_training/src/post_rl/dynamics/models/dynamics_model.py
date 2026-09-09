import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from .nets import EnsembleLinear
from ...critic.action_embedding import ActionChunkEncoder, ActionChunkDecoder


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def soft_clamp(
    x: torch.Tensor,
    min_v: Optional[torch.Tensor] = None,
    max_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if max_v is not None:
        x = max_v - F.softplus(max_v - x)
    if min_v is not None:
        x = min_v + F.softplus(x - min_v)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: type[nn.Module] = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str | torch.device = "cpu",
        cfg=None,
    ) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self._device = torch.device(device)
        self.activation = activation()
        self.token_dynamics = bool(
            getattr(getattr(cfg, "dynamics", None), "latent_mode", "compact")
            == "transformer_encoder"
        )
        self.token_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        action_embed_layer_norm = getattr(
            getattr(cfg, "dynamics", None),
            "action_embed_layer_norm",
            False,
        )
        action_scale_norm = getattr(
            getattr(cfg, "dynamics", None),
            "action_scale_norm",
            False,
        )
        self.use_conv_action_embed = getattr(cfg, "use_conv_action_embed", False)
        self.use_action_embed = getattr(cfg, "use_action_embed", False)
        self.use_action_scale_norm = bool(
            action_scale_norm
            and not self.use_conv_action_embed
            and not self.use_action_embed
        )

        action_embed_dim = action_dim
        if self.use_conv_action_embed:
            n_action_steps = getattr(cfg, "n_action_steps", 1)
            if action_dim % n_action_steps != 0:
                raise ValueError(
                    "action_dim must be divisible by n_action_steps when "
                    "use_conv_action_embed=True"
                )
            single_action_dim = action_dim // n_action_steps
            self._single_action_dim = single_action_dim
            self._n_action_steps = n_action_steps
            self._conv_action_encoder = ActionChunkEncoder(
                action_dim=single_action_dim,
                hidden_dims=list(getattr(cfg, "conv_hidden_dims", [128, 256])),
                latent_cz=getattr(cfg, "conv_latent_cz", 32),
                kernel_size=getattr(cfg, "conv_kernel_size", 5),
                n_groups=getattr(cfg, "conv_n_groups", 8),
            )
            self._conv_action_decoder = ActionChunkDecoder(
                action_dim=single_action_dim,
                hidden_dims=list(reversed(getattr(cfg, "conv_hidden_dims", [128, 256]))),
                latent_cz=getattr(cfg, "conv_latent_cz", 32),
                kernel_size=getattr(cfg, "conv_kernel_size", 5),
                n_groups=getattr(cfg, "conv_n_groups", 8),
                target_len=n_action_steps,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                action_embed_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]
            self._conv_action_layer_norm = (
                nn.LayerNorm(action_embed_dim)
                if action_embed_layer_norm
                else nn.Identity()
            )
        elif self.use_action_embed:
            embed_dim = self.token_dim if self.token_dynamics else int(obs_dim)
            self._action_encoder = nn.Sequential(
                nn.Linear(action_dim, embed_dim),
                nn.LayerNorm(embed_dim) if action_embed_layer_norm else nn.Identity(),
                nn.ReLU(),
            )
            action_embed_dim = embed_dim
        elif self.use_action_scale_norm:
            self._action_scale_layer_norm = nn.LayerNorm(action_dim)

        if self.token_dynamics:
            if with_reward:
                raise NotImplementedError(
                    "Transformer-token dynamics currently follows RL-100 OPE with predict_r=false."
                )
            dims = [2 * self.token_dim + action_embed_dim] + list(hidden_dims)
            output_dim = self.token_dim
        else:
            dims = [obs_dim + action_embed_dim] + list(hidden_dims)
            output_dim = obs_dim + int(with_reward)

        if weight_decays is None:
            weight_decays = [0.0] * len(dims)
        if len(weight_decays) != len(dims):
            raise ValueError(
                "weight_decays must have one value per hidden transition plus output layer"
            )

        self.backbones = nn.ModuleList(
            [
                EnsembleLinear(
                    in_dim,
                    out_dim,
                    num_ensemble,
                    weight_decay,
                )
                for in_dim, out_dim, weight_decay in zip(
                    dims[:-1],
                    dims[1:],
                    weight_decays[:-1],
                )
            ]
        )
        self.output_layer = EnsembleLinear(
            dims[-1],
            2 * output_dim,
            num_ensemble,
            weight_decays[-1],
        )
        self.max_logvar = nn.Parameter(torch.ones(output_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(output_dim) * -10.0)
        self.elites = nn.Parameter(
            torch.arange(num_elites),
            requires_grad=False,
        )
        self.to(self._device)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self._device,
        )
        batch_size = action.shape[0]
        if self.use_conv_action_embed:
            action_chunk = action.reshape(
                batch_size,
                -1,
                self._single_action_dim,
            )
            z = self._conv_action_encoder(action_chunk)
            return self._conv_action_layer_norm(z.reshape(batch_size, -1))
        action = action.reshape(batch_size, -1)
        if self.use_action_embed:
            return self._action_encoder(action)
        if self.use_action_scale_norm:
            action = self._action_scale_layer_norm(action)
        return action

    def _forward_token_dynamics(
        self,
        state_tokens: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_tokens = torch.as_tensor(
            state_tokens,
            dtype=torch.float32,
            device=self._device,
        )
        if state_tokens.ndim == 2:
            if state_tokens.shape[-1] % self.token_dim != 0:
                raise ValueError(
                    f"Flattened latent dim {state_tokens.shape[-1]} is not divisible "
                    f"by token_dim={self.token_dim}."
                )
            state_tokens = state_tokens.reshape(
                state_tokens.shape[0],
                -1,
                self.token_dim,
            )
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected ACT latent [B,S,{self.token_dim}], got "
                f"{tuple(state_tokens.shape)}"
            )

        batch_size, token_count, _ = state_tokens.shape
        action_embed = self.encode_action(action)
        context = state_tokens.mean(dim=1, keepdim=True).expand(-1, token_count, -1)
        action_embed = action_embed.unsqueeze(1).expand(-1, token_count, -1)
        output = torch.cat([state_tokens, context, action_embed], dim=-1)
        output = output.reshape(batch_size * token_count, -1)

        for layer in self.backbones:
            output = self.activation(layer(output))

        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        mean = mean.reshape(
            self.num_ensemble,
            batch_size,
            token_count,
            self.token_dim,
        )
        logvar = logvar.reshape_as(mean)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def forward(
        self,
        obs_action,
        action=None,
        targets=None,
        action_chunk=None,
        logvar_loss_coef=0.01,
        action_recon_beta=0.5,
    ):
        if self.token_dynamics:
            if action is None:
                raise ValueError("Token dynamics requires an explicit action chunk.")
            return self._forward_token_dynamics(obs_action, action)

        if action_chunk is not None and targets is not None:
            batch_size = action_chunk.shape[0]
            z = self._conv_action_encoder(action_chunk)
            action_embed = self._conv_action_layer_norm(z.reshape(batch_size, -1))
            output = torch.cat([obs_action, action_embed], dim=-1)

            for layer in self.backbones:
                output = self.activation(layer(output))

            mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
            logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
            inv_var = torch.exp(-logvar)
            mse_loss_inv = ((mean - targets).pow(2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss += self.get_decay_loss()
            loss += (
                logvar_loss_coef * self.max_logvar.sum()
                - logvar_loss_coef * self.min_logvar.sum()
            )
            action_recon = self._conv_action_decoder(z)
            recon_loss = F.mse_loss(action_recon, action_chunk)
            return loss + action_recon_beta * recon_loss

        obs_action = torch.as_tensor(
            obs_action,
            dtype=torch.float32,
            device=self._device,
        )
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        loss = torch.zeros((), device=self._device)
        for layer in self.backbones:
            loss += layer.get_decay_loss()
        return loss + self.output_layer.get_decay_loss()

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble
        assert max(indexes) < self.num_ensemble
        self.elites.data = torch.tensor(
            indexes,
            device=self.elites.device,
            dtype=self.elites.dtype,
        )

    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        return np.random.choice(
            self.elites.data.cpu().numpy(),
            size=batch_size,
        )
