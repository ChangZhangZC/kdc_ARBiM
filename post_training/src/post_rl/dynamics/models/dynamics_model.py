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
        self.use_conv_action_embed = getattr(
            cfg,
            "use_conv_action_embed",
            False,
        )
        self.use_action_scale_norm = bool(
            action_scale_norm
            and not self.use_conv_action_embed
            and not cfg.use_action_embed
        )

        if self.use_conv_action_embed:
            n_action_steps = getattr(cfg, "n_action_steps", 1)
            if action_dim % n_action_steps != 0:
                raise ValueError(
                    "action_dim must be divisible by n_action_steps "
                    "when use_conv_action_embed=True"
                )

            single_action_dim = action_dim // n_action_steps
            conv_latent_cz = getattr(cfg, "conv_latent_cz", 32)
            conv_hidden_dims = list(
                getattr(cfg, "conv_hidden_dims", [128, 256])
            )
            conv_kernel_size = getattr(cfg, "conv_kernel_size", 5)
            conv_n_groups = getattr(cfg, "conv_n_groups", 8)

            self._conv_action_encoder = ActionChunkEncoder(
                action_dim=single_action_dim,
                hidden_dims=conv_hidden_dims,
                latent_cz=conv_latent_cz,
                kernel_size=conv_kernel_size,
                n_groups=conv_n_groups,
            )
            self._conv_action_decoder = ActionChunkDecoder(
                action_dim=single_action_dim,
                hidden_dims=list(reversed(conv_hidden_dims)),
                latent_cz=conv_latent_cz,
                kernel_size=conv_kernel_size,
                n_groups=conv_n_groups,
                target_len=n_action_steps,
            )

            with torch.no_grad():
                dummy = torch.zeros(
                    1,
                    n_action_steps,
                    single_action_dim,
                )
                conv_out_dim = self._conv_action_encoder(
                    dummy
                ).reshape(1, -1).shape[-1]

            self._conv_action_layer_norm = (
                nn.LayerNorm(conv_out_dim)
                if action_embed_layer_norm
                else nn.Identity()
            )
            hidden_dims = [obs_dim + conv_out_dim] + list(hidden_dims)
            self._single_action_dim = single_action_dim
            self._n_action_steps = n_action_steps

        elif cfg.use_action_embed:
            action_encoder_layers = [
                nn.Linear(action_dim, int(obs_dim))
            ]
            if action_embed_layer_norm:
                action_encoder_layers.append(
                    nn.LayerNorm(int(obs_dim))
                )
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(
                *action_encoder_layers
            )
            hidden_dims = [
                obs_dim + int(obs_dim)
            ] + list(hidden_dims)

        else:
            if self.use_action_scale_norm:
                self._action_scale_layer_norm = nn.LayerNorm(
                    action_dim
                )
            hidden_dims = [
                obs_dim + action_dim
            ] + list(hidden_dims)

        if weight_decays is None:
            weight_decays = [0.0] * len(hidden_dims)

        if len(weight_decays) != len(hidden_dims):
            raise ValueError(
                "weight_decays must have len(hidden_dims) values "
                "after adding the input dimension"
            )

        self.backbones = nn.ModuleList([
            EnsembleLinear(
                in_dim,
                out_dim,
                num_ensemble,
                weight_decay,
            )
            for in_dim, out_dim, weight_decay in zip(
                hidden_dims[:-1],
                hidden_dims[1:],
                weight_decays[:-1],
            )
        ])

        output_dim = obs_dim + int(with_reward)
        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * output_dim,
            num_ensemble,
            weight_decays[-1],
        )
        self.max_logvar = nn.Parameter(
            torch.ones(output_dim) * 0.5
        )
        self.min_logvar = nn.Parameter(
            torch.ones(output_dim) * -10.0
        )
        self.elites = nn.Parameter(
            torch.arange(num_elites),
            requires_grad=False,
        )
        self.to(self._device)

    def forward(
        self,
        obs_action,
        targets=None,
        action_chunk=None,
        logvar_loss_coef=0.01,
        action_recon_beta=0.5,
    ):
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