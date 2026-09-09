from copy import deepcopy

import einops
import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import OBS_DEPTH


def prepare_act_model_batch(config, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    model_batch = dict(batch)
    if config.image_features:
        model_batch[OBS_IMAGES] = [model_batch[key] for key in config.image_features]
    if config.use_depth and config.depth_features:
        model_batch[OBS_DEPTH] = [
            model_batch[key].mean(dim=-3, keepdim=True)
            for key in config.depth_features
        ]
    return model_batch


def encode_act_state(model: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    if OBS_IMAGES in batch:
        batch_size = batch[OBS_IMAGES][0].shape[0]
        n_cam = len(batch[OBS_IMAGES])
    elif OBS_STATE in batch:
        batch_size = batch[OBS_STATE].shape[0]
        n_cam = 0
    elif OBS_ENV_STATE in batch:
        batch_size = batch[OBS_ENV_STATE].shape[0]
        n_cam = 0
    else:
        raise KeyError("ACT latent encoder requires image, robot-state, or env-state input.")

    latent_sample = torch.zeros(
        batch_size,
        model.config.latent_dim,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    encoder_in_tokens = [model.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

    if model.config.robot_state_feature:
        encoder_in_tokens.append(model.encoder_robot_state_input_proj(batch[OBS_STATE]))
    if model.config.env_state_feature:
        encoder_in_tokens.append(model.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

    if model.config.image_features:
        imgs = torch.cat(batch[OBS_IMAGES], dim=0)
        cam_features = model.backbone(imgs)["feature_map"]
        pos_cam_input = einops.rearrange(
            cam_features,
            "(v b) c h w -> v b c h w",
            v=n_cam,
            b=batch_size,
        )
        cam_pos_embed = torch.cat(
            [
                model.encoder_cam_feat_pos_embed(pos_cam_input[i]).to(
                    dtype=cam_features.dtype
                )
                for i in range(n_cam)
            ],
            dim=0,
        )
        cam_features = model.encoder_img_feat_input_proj(cam_features)
        cam_features = cam_features.view(
            n_cam,
            batch_size,
            cam_features.size(1),
            cam_features.size(2),
            cam_features.size(3),
        )
        cam_pos_embed = cam_pos_embed.view(
            n_cam,
            1,
            cam_pos_embed.size(1),
            cam_pos_embed.size(2),
            cam_pos_embed.size(3),
        )
        cam_features = einops.rearrange(
            cam_features,
            "v b c h w -> v (h w) b c",
        )
        cam_pos_embed = einops.rearrange(
            cam_pos_embed,
            "v b c h w -> v (h w) b c",
        )
        rgb_tokens = [cam_features[v] for v in range(n_cam)]
        rgb_pos = [cam_pos_embed[v] for v in range(n_cam)]
    else:
        rgb_tokens, rgb_pos = [], []

    if model.config.use_depth and OBS_DEPTH in batch:
        depths = torch.cat(batch[OBS_DEPTH], dim=0)
        depth_features = model.depth_backbone(depths)["feature_map"]
        pos_depth_input = einops.rearrange(
            depth_features,
            "(v b) c h w -> v b c h w",
            v=n_cam,
            b=batch_size,
        )
        depth_pos_embed = torch.cat(
            [
                model.encoder_depth_feat_pos_embed(pos_depth_input[i]).to(
                    dtype=depth_features.dtype
                )
                for i in range(n_cam)
            ],
            dim=0,
        )
        depth_features = model.encoder_depth_feat_input_proj(depth_features)
        depth_features = depth_features.view(
            n_cam,
            batch_size,
            depth_features.size(1),
            depth_features.size(2),
            depth_features.size(3),
        )
        depth_pos_embed = depth_pos_embed.view(
            n_cam,
            1,
            depth_pos_embed.size(1),
            depth_pos_embed.size(2),
            depth_pos_embed.size(3),
        )
        depth_features = einops.rearrange(
            depth_features,
            "v b c h w -> v (h w) b c",
        )
        depth_pos_embed = einops.rearrange(
            depth_pos_embed,
            "v b c h w -> v (h w) b c",
        )
        depth_tokens = [depth_features[v] for v in range(n_cam)]
    else:
        depth_tokens = []

    if model.config.use_depth and model.config.depth_features:
        fused_rgb, fused_depth = model.cross_modal_fusion(rgb_tokens, depth_tokens)
        for rgb, depth in zip(fused_rgb, fused_depth):
            fused = model.cross_modal_fusion_proj(torch.cat([rgb, depth], dim=-1))
            encoder_in_tokens.extend(list(fused))
        for pos in rgb_pos:
            encoder_in_pos_embed.extend(list(pos))
    else:
        for tokens, pos in zip(rgb_tokens, rgb_pos):
            encoder_in_tokens.extend(list(tokens))
            encoder_in_pos_embed.extend(list(pos))

    encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
    encoder_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)
    encoder_out = model.encoder(
        encoder_in_tokens,
        pos_embed=encoder_pos_embed,
    )
    return encoder_out.transpose(0, 1), encoder_pos_embed


def decode_act_state(
    model: nn.Module,
    latent: Tensor,
    encoder_pos_embed: Tensor,
) -> Tensor:
    if latent.ndim != 3:
        raise ValueError(f"Expected ACT latent [B,S,D], got {tuple(latent.shape)}")
    batch_size, token_count, dim = latent.shape
    if dim != model.config.dim_model:
        raise ValueError(
            f"ACT latent dim {dim} does not match dim_model={model.config.dim_model}."
        )
    if encoder_pos_embed.shape[0] != token_count:
        raise ValueError(
            f"ACT latent has {token_count} tokens but positional embedding has "
            f"{encoder_pos_embed.shape[0]}."
        )

    encoder_out = latent.transpose(0, 1)
    encoder_pos_embed = encoder_pos_embed.to(
        device=latent.device,
        dtype=latent.dtype,
    )
    decoder_in = torch.zeros(
        model.config.chunk_size,
        batch_size,
        model.config.dim_model,
        dtype=latent.dtype,
        device=latent.device,
    )
    decoder_out = model.decoder(
        decoder_in,
        encoder_out,
        encoder_pos_embed=encoder_pos_embed,
        decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1),
    )
    return model.action_head(decoder_out.transpose(0, 1))


class ACTStateEncoder(nn.Module):
    def __init__(self, act_model: nn.Module, copy_model: bool = True) -> None:
        super().__init__()
        source = deepcopy(act_model) if copy_model else act_model
        self.config = source.config
        self.encoder_latent_input_proj = source.encoder_latent_input_proj
        self.encoder_1d_feature_pos_embed = source.encoder_1d_feature_pos_embed
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = source.encoder_robot_state_input_proj
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = source.encoder_env_state_input_proj
        if self.config.image_features:
            self.backbone = source.backbone
            self.encoder_cam_feat_pos_embed = source.encoder_cam_feat_pos_embed
            self.encoder_img_feat_input_proj = source.encoder_img_feat_input_proj
        self.encoder = source.encoder
        if self.config.use_depth and self.config.depth_features:
            self.depth_backbone = source.depth_backbone
            self.encoder_depth_feat_pos_embed = source.encoder_depth_feat_pos_embed
            self.encoder_depth_feat_input_proj = source.encoder_depth_feat_input_proj
            self.cross_modal_fusion = source.cross_modal_fusion
            self.cross_modal_fusion_proj = source.cross_modal_fusion_proj
        self.output_dim = self.config.dim_model
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        model_batch = prepare_act_model_batch(self.config, batch)
        latent, _ = encode_act_state(self, model_batch)
        return latent

    def output_shape(self) -> int:
        return self.output_dim
