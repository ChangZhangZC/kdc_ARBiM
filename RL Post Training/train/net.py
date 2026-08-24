import torch
import torch.nn as nn

from action_ae import ActionChunkEncoder, ActionChunkDecoder
from kuavo_train.wrapper.policy.act.ACTModelWrapper import CustomACTModelWrapper
from copy import deepcopy
from lerobot.utils.constants import OBS_STATE

def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: str = "relu",
    final_activation: str | None = None,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    if activation == "tanh":
        act = nn.Tanh
    elif activation == "relu":
        act = nn.ReLU
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers = [nn.Linear(input_dim, hidden_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(act())

    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())

    layers.append(nn.Linear(hidden_dim, output_dim))

    if final_activation == "relu":
        layers.append(nn.ReLU())
    elif final_activation == "tanh":
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class ACTCriticEncoder(nn.Module):
    def __init__(
        self,
        act_model: CustomACTModelWrapper,
        copy_model: bool = True,
    ) -> None:
        super().__init__()
        model = deepcopy(act_model) if copy_model else act_model

        self.config = model.config
        self.use_depth = bool(self.config.use_depth and self.config.depth_features)

        self.backbone = model.backbone
        self.image_proj = model.encoder_img_feat_input_proj
        self.state_proj = model.encoder_robot_state_input_proj

        if self.use_depth:
            self.depth_backbone = model.depth_backbone
            self.depth_proj = model.encoder_depth_feat_input_proj
            self.cross_modal_fusion = model.cross_modal_fusion
            self.cross_modal_fusion_proj = model.cross_modal_fusion_proj

        self.n_cameras = len(self.config.image_features)
        self.output_dim = self.config.dim_model * (self.n_cameras + int(self.config.robot_state_feature is not None))

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        rgb_features = []
        for key in self.config.image_features:
            feat = self.backbone(obs[key])["feature_map"]
            feat = self.image_proj(feat)
            rgb_features.append(feat)

        if self.use_depth:
            depth_features = []
            for key in self.config.depth_features:
                depth = obs[key].mean(dim=-3, keepdim=True)
                feat = self.depth_backbone(depth)["feature_map"]
                feat = self.depth_proj(feat)
                depth_features.append(feat)

            rgb_tokens = [x.flatten(2).permute(2, 0, 1) for x in rgb_features]
            depth_tokens = [x.flatten(2).permute(2, 0, 1) for x in depth_features]
            fused_rgb, fused_depth = self.cross_modal_fusion(rgb_tokens, depth_tokens)

            visual_features = [
                self.cross_modal_fusion_proj(torch.cat([rgb, depth], dim=-1)).mean(dim=0)
                for rgb, depth in zip(fused_rgb, fused_depth)
            ]
        else:
            visual_features = [x.mean(dim=(-2, -1)) for x in rgb_features]

        state_feature = self.state_proj(obs[OBS_STATE])
        return torch.cat([*visual_features, state_feature], dim=-1)

    def output_shape(self) -> int:
        return self.output_dim
    
    
class ValueMLP(nn.Module):
    def __init__(
        self,
        obs_encoder: ACTCriticEncoder | None,
        state_dim: int,
        hidden_dim: int,
        depth: int,
        n_obs_steps: int = 1,
        fix_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self._net = MLP(state_dim, hidden_dim, depth - 1, 1)
        self._obs_encoder = obs_encoder

        if self._obs_encoder is not None and fix_encoder:
            self._obs_encoder.eval()
            for param in self._obs_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        state: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._obs_encoder is not None and isinstance(state, dict):
            state = self._obs_encoder(state)

        state = state.reshape(-1, self.state_dim)
        return self._net(state)


class QMLP(nn.Module):
    def __init__(
        self,
        use_action_embed: bool,
        obs_encoder: ACTCriticEncoder | None,
        state_dim: int,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        fix_encoder: bool = False,
        use_conv_action_embed: bool = False,
        single_action_dim: int | None = None,
        n_action_steps: int = 16,
        conv_hidden_dims: list[int] = [128, 256],
        conv_latent_cz: int = 32,
        conv_kernel_size: int = 5,
        conv_n_groups: int = 8,
        q_layer_norm: bool = False,
        action_embed_layer_norm: bool = False,
        action_scale_norm: bool = False,
    ) -> None:
        super().__init__()

        self.use_action_embed = use_action_embed
        self.use_conv_action_embed = use_conv_action_embed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_action_scale_norm = (
            action_scale_norm
            and not use_action_embed
            and not use_conv_action_embed
        )

        if use_conv_action_embed:
            if single_action_dim is None:
                raise ValueError(
                    "single_action_dim is required when use_conv_action_embed=True"
                )

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
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                conv_out_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]

            self._conv_action_layer_norm = (
                nn.LayerNorm(conv_out_dim)
                if action_embed_layer_norm
                else nn.Identity()
            )
            q_input_dim = state_dim + conv_out_dim

        else:
            action_encoder_layers = [nn.Linear(action_dim, feature_dim)]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(feature_dim))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)

            if use_action_embed:
                q_input_dim = state_dim + feature_dim
            else:
                if self.use_action_scale_norm:
                    self._action_scale_layer_norm = nn.LayerNorm(action_dim)
                q_input_dim = state_dim + action_dim

        self._net = MLP(
            q_input_dim,
            hidden_dim,
            depth - 1,
            1,
            use_layer_norm=q_layer_norm,
        )

        self._obs_encoder = obs_encoder
        if self._obs_encoder is not None and fix_encoder:
            self._obs_encoder.eval()
            for param in self._obs_encoder.parameters():
                param.requires_grad = False

    def encode_action(
        self,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_conv_action_embed:
            batch_size = a.shape[0]
            single_action_dim = self._conv_action_encoder.action_dim
            a_chunk = a.reshape(batch_size, -1, single_action_dim)

            z = self._conv_action_encoder(a_chunk)
            recon = self._conv_action_decoder(z)

            z = self._conv_action_layer_norm(z.reshape(batch_size, -1))
            return z, recon.reshape(batch_size, -1)

        if self.use_action_embed:
            return self._action_encoder(
                a.reshape(-1, self.action_dim)
            ), None

        a = a.reshape(-1, self.action_dim)
        if self.use_action_scale_norm:
            a = self._action_scale_layer_norm(a)

        return a, None


    def compute_action_recon_loss(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_conv_action_embed:
            return torch.tensor(0.0, device=a.device)

        a_flat = a.reshape(a.shape[0], -1)
        _, a_recon = self.encode_action(a_flat)

        return torch.nn.functional.mse_loss(
            a_recon,
            a_flat,
        )

    def forward(
        self,
        s: torch.Tensor | dict[str, torch.Tensor],
        a: torch.Tensor,
        return_action_recon_loss: bool = False,
    ):
        if self._obs_encoder is not None:
            s = self._obs_encoder(s)

        s = s.reshape(-1, self.state_dim)
        a_embed, a_recon = self.encode_action(a)

        sa = torch.cat([s, a_embed], dim=1)
        q = self._net(sa)

        if return_action_recon_loss:
            if self.use_conv_action_embed and a_recon is not None:
                recon_loss = torch.nn.functional.mse_loss(
                    a_recon,
                    a.reshape(a.shape[0], -1),
                )
            else:
                recon_loss = torch.tensor(0.0, device=a.device)
            return q, recon_loss

        return q



class DoubleQMLP(nn.Module):
    def __init__(
        self,
        use_action_embed: bool,
        obs_encoder: ACTCriticEncoder | None,
        state_dim: int,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        fix_encoder: bool = False,
        use_conv_action_embed: bool = False,
        single_action_dim: int | None = None,
        n_action_steps: int = 16,
        conv_hidden_dims: list[int] = [128, 256],
        conv_latent_cz: int = 32,
        conv_kernel_size: int = 5,
        conv_n_groups: int = 8,
        q_layer_norm: bool = False,
        action_embed_layer_norm: bool = False,
        action_scale_norm: bool = False,
    ) -> None:
        super().__init__()

        self.use_action_embed = use_action_embed
        self.use_conv_action_embed = use_conv_action_embed
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_action_scale_norm = (
            action_scale_norm
            and not use_action_embed
            and not use_conv_action_embed
        )

        if use_conv_action_embed:
            if single_action_dim is None:
                raise ValueError(
                    "single_action_dim is required when use_conv_action_embed=True"
                )

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
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                conv_out_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]

            self._conv_action_layer_norm = (
                nn.LayerNorm(conv_out_dim)
                if action_embed_layer_norm
                else nn.Identity()
            )
            q_input_dim = state_dim + conv_out_dim

        else:
            action_encoder_layers = [nn.Linear(action_dim, feature_dim)]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(feature_dim))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)

            if use_action_embed:
                q_input_dim = state_dim + feature_dim
            else:
                if self.use_action_scale_norm:
                    self._action_scale_layer_norm = nn.LayerNorm(action_dim)
                q_input_dim = state_dim + action_dim

        self._net1 = MLP(
            q_input_dim,
            hidden_dim,
            depth - 1,
            1,
            use_layer_norm=q_layer_norm,
        )
        self._net2 = MLP(
            q_input_dim,
            hidden_dim,
            depth - 1,
            1,
            use_layer_norm=q_layer_norm,
        )

        self._obs_encoder = obs_encoder
        if self._obs_encoder is not None and fix_encoder:
            self._obs_encoder.eval()
            for param in self._obs_encoder.parameters():
                param.requires_grad = False
                
    def encode_action(
        self,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_conv_action_embed:
            batch_size = a.shape[0]
            single_action_dim = self._conv_action_encoder.action_dim
            a_chunk = a.reshape(batch_size, -1, single_action_dim)

            z = self._conv_action_encoder(a_chunk)
            recon = self._conv_action_decoder(z)

            z = self._conv_action_layer_norm(z.reshape(batch_size, -1))
            return z, recon.reshape(batch_size, -1)

        if self.use_action_embed:
            return self._action_encoder(
                a.reshape(-1, self.action_dim)
            ), None

        a = a.reshape(-1, self.action_dim)
        if self.use_action_scale_norm:
            a = self._action_scale_layer_norm(a)

        return a, None


    def compute_action_recon_loss(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_conv_action_embed:
            return torch.tensor(0.0, device=a.device)

        a_flat = a.reshape(a.shape[0], -1)
        _, a_recon = self.encode_action(a_flat)

        return torch.nn.functional.mse_loss(
            a_recon,
            a_flat,
        )
        
    def forward(
        self,
        s: torch.Tensor | dict[str, torch.Tensor],
        a: torch.Tensor,
        return_action_recon_loss: bool = False,
    ):
        if self._obs_encoder is not None:
            s = self._obs_encoder(s)

        s = s.reshape(-1, self.state_dim)
        a_embed, a_recon = self.encode_action(a)

        sa = torch.cat([s, a_embed], dim=1)
        q1 = self._net1(sa)
        q2 = self._net2(sa)

        if return_action_recon_loss:
            if self.use_conv_action_embed and a_recon is not None:
                recon_loss = torch.nn.functional.mse_loss(
                    a_recon,
                    a.reshape(a.shape[0], -1),
                )
            else:
                recon_loss = torch.tensor(0.0, device=a.device)
            return q1, q2, recon_loss

        return q1, q2