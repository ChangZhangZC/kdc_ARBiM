import numpy as np
import torch
import torch.nn as nn

from .dynamics_model import Swish, soft_clamp
from .nets import EnsembleLinear
from ...critic.action_embedding import ActionChunkEncoder


class EnsembleTokenDynamicsModel(nn.Module):
    token_dynamics = True

    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        hidden_dims,
        num_ensemble: int = 7,
        num_elites: int = 5,
        weight_decays=None,
        device: str | torch.device = "cpu",
        cfg=None,
    ) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self.token_dim = token_dim
        self.action_dim = action_dim
        self._device = torch.device(device)
        self.activation = Swish()
        self.use_conv_action_embed = getattr(cfg, "use_conv_action_embed", False)
        self.use_action_embed = getattr(cfg, "use_action_embed", False)

        action_embed_layer_norm = getattr(
            getattr(cfg, "dynamics", None),
            "action_embed_layer_norm",
            False,
        )

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
            with torch.no_grad():
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                action_embed_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]
            self._conv_action_layer_norm = (
                nn.LayerNorm(action_embed_dim)
                if action_embed_layer_norm
                else nn.Identity()
            )
        elif self.use_action_embed:
            self._action_encoder = nn.Sequential(
                nn.Linear(action_dim, token_dim),
                nn.LayerNorm(token_dim) if action_embed_layer_norm else nn.Identity(),
                nn.ReLU(),
            )
            action_embed_dim = token_dim
        else:
            action_embed_dim = action_dim

        dims = [2 * token_dim + action_embed_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * len(dims)
        if len(weight_decays) != len(dims):
            raise ValueError(
                "dynamics_weight_decay must have one value per hidden transition "
                "plus the output layer."
            )

        self.backbones = nn.ModuleList(
            [
                EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay)
                for in_dim, out_dim, weight_decay in zip(
                    dims[:-1],
                    dims[1:],
                    weight_decays[:-1],
                )
            ]
        )
        self.output_layer = EnsembleLinear(
            dims[-1],
            2 * token_dim,
            num_ensemble,
            weight_decays[-1],
        )
        self.max_logvar = nn.Parameter(torch.ones(token_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(token_dim) * -10.0)
        self.elites = nn.Parameter(
            torch.arange(num_elites),
            requires_grad=False,
        )
        self.to(self._device)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(
            action,
            device=self._device,
            dtype=torch.float32,
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
        return action

    def forward(
        self,
        state_tokens: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_tokens = torch.as_tensor(
            state_tokens,
            device=self._device,
            dtype=torch.float32,
        )
        if state_tokens.ndim != 3:
            raise ValueError(
                f"Expected state tokens [B,S,D], got {tuple(state_tokens.shape)}"
            )
        if state_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Token dim {state_tokens.shape[-1]} does not match {self.token_dim}."
            )

        batch_size, token_count, _ = state_tokens.shape
        action_embed = self.encode_action(action)
        context = state_tokens.mean(dim=1, keepdim=True).expand(-1, token_count, -1)
        action_embed = action_embed.unsqueeze(1).expand(-1, token_count, -1)
        x = torch.cat([state_tokens, context, action_embed], dim=-1)
        x = x.reshape(batch_size * token_count, -1)

        for layer in self.backbones:
            x = self.activation(layer(x))

        mean, logvar = torch.chunk(self.output_layer(x), 2, dim=-1)
        mean = mean.reshape(self.num_ensemble, batch_size, token_count, self.token_dim)
        logvar = logvar.reshape(
            self.num_ensemble,
            batch_size,
            token_count,
            self.token_dim,
        )
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        loss = torch.zeros((), device=self._device)
        for layer in self.backbones:
            loss += layer.get_decay_loss()
        return loss + self.output_layer.get_decay_loss()

    def set_elites(self, indexes) -> None:
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
