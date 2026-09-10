import os
from pathlib import Path

import torch
from torch import Tensor
from torch.distributions import Normal
from safetensors.torch import load_model
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError

from .act_latent import decode_act_state, encode_act_state, prepare_act_model_batch
from .stochastic_act_config import StochasticACTConfigWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from lerobot.utils.constants import ACTION


class StochasticACTPolicyWrapper(CustomACTPolicyWrapper):
    def __init__(self, config: StochasticACTConfigWrapper):
        super().__init__(config)

        action_shape = config.action_feature.shape
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action feature, got shape={action_shape}")
        action_dim = action_shape[0]

        init_ratio = (
            (config.init_log_std - config.log_std_min)
            / (config.log_std_max - config.log_std_min)
        )
        raw_init = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.raw_log_std = torch.nn.Parameter(raw_init.repeat(action_dim))
        self.register_buffer(
            "_frozen_encoder_pos_embed",
            torch.empty(0),
            persistent=False,
        )

    def _get_log_std(self) -> Tensor:
        return self.config.log_std_min + (
            self.config.log_std_max - self.config.log_std_min
        ) * torch.sigmoid(self.raw_log_std)

    def _get_std(self) -> Tensor:
        return self._get_log_std().exp()

    def _prepare_model_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        model_batch = dict(batch)
        model_batch.pop(ACTION, None)
        model_batch.pop("action_is_pad", None)
        return prepare_act_model_batch(self.config, model_batch)

    def set_frozen_encoder_pos_embed(self, encoder_pos_embed: Tensor) -> None:
        encoder_pos_embed = torch.as_tensor(
            encoder_pos_embed,
            device=next(self.parameters()).device,
        ).detach()
        if encoder_pos_embed.ndim != 3:
            raise ValueError(
                "Frozen ACT encoder positional embedding must be [S,1,D], got "
                f"{tuple(encoder_pos_embed.shape)}"
            )
        if encoder_pos_embed.shape[-1] != self.model.config.dim_model:
            raise ValueError(
                "Frozen ACT encoder positional embedding dim "
                f"{encoder_pos_embed.shape[-1]} != dim_model={self.model.config.dim_model}"
            )
        self._frozen_encoder_pos_embed = encoder_pos_embed.clone()

    def _cached_latent_and_pos(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        latent = torch.as_tensor(
            latent,
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )
        if latent.ndim == 4:
            if latent.shape[1] != 1:
                raise ValueError(
                    "Cached ACT policy latent must contain one observation endpoint, got "
                    f"{tuple(latent.shape)}"
                )
            latent = latent[:, 0]
        if latent.ndim != 3:
            raise ValueError(
                "Cached ACT policy latent must be [B,S,D] or [B,1,S,D], got "
                f"{tuple(latent.shape)}"
            )
        if latent.shape[-1] != self.model.config.dim_model:
            raise ValueError(
                f"Cached ACT policy latent dim {latent.shape[-1]} != "
                f"dim_model={self.model.config.dim_model}"
            )
        if self._frozen_encoder_pos_embed.numel() == 0:
            raise RuntimeError(
                "Cached ACT policy latent requires frozen encoder positional embeddings. "
                "Call set_frozen_encoder_pos_embed() after building the ACT frontend."
            )
        encoder_pos_embed = self._frozen_encoder_pos_embed.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        if encoder_pos_embed.shape[0] != latent.shape[1]:
            raise ValueError(
                f"Cached ACT policy latent has {latent.shape[1]} tokens but frozen "
                f"positional embedding has {encoder_pos_embed.shape[0]} tokens"
            )
        return latent, encoder_pos_embed

    def encode_observation(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        if "latent" in batch:
            return self._cached_latent_and_pos(batch["latent"])
        model_batch = self._prepare_model_batch(batch)
        return encode_act_state(self.model, model_batch)

    def get_action_mean(self, batch: dict[str, Tensor]) -> Tensor:
        latent, encoder_pos_embed = self.encode_observation(batch)
        return self.get_action_mean_from_latent(latent, encoder_pos_embed)

    def get_action_mean_from_latent(
        self,
        latent: Tensor,
        encoder_pos_embed: Tensor,
    ) -> Tensor:
        return decode_act_state(self.model, latent, encoder_pos_embed)

    def get_distribution(self, batch: dict[str, Tensor]) -> Normal:
        mu = self.get_action_mean(batch)
        return Normal(mu, self._get_std())

    def get_distribution_from_latent(
        self,
        latent: Tensor,
        encoder_pos_embed: Tensor,
    ) -> Normal:
        mu = self.get_action_mean_from_latent(latent, encoder_pos_embed)
        return Normal(mu, self._get_std())

    def sample_action_chunk(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.get_distribution(batch)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def sample_action_chunk_from_latent(
        self,
        latent: Tensor,
        encoder_pos_embed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.get_distribution_from_latent(latent, encoder_pos_embed)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_action_chunk(
        self,
        batch: dict[str, Tensor],
        action_chunk: Tensor,
    ) -> tuple[Tensor, Tensor]:
        dist = self.get_distribution(batch)
        return dist.log_prob(action_chunk), dist.entropy()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self.get_action_mean(batch)

    @classmethod
    def from_il_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: StochasticACTConfigWrapper | None = None,
        init_log_std: float | None = None,
        log_std_min: float | None = None,
        log_std_max: float | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "StochasticACTPolicyWrapper":
        if config is not None and any(
            v is not None for v in (init_log_std, log_std_min, log_std_max)
        ):
            raise ValueError(
                "Do not pass stochastic parameters together with an explicit config."
            )
        if config is None:
            config = StochasticACTConfigWrapper.from_il_pretrained(
                pretrained_name_or_path,
                init_log_std=init_log_std,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found in {model_id}"
                ) from e

        policy = cls(config)
        missing, unexpected = load_model(
            policy,
            model_file,
            strict=False,
            device=config.device,
        )
        if set(missing) != {"raw_log_std"} or unexpected:
            raise RuntimeError(
                f"Invalid IL checkpoint migration: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: StochasticACTConfigWrapper | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "StochasticACTPolicyWrapper":
        if config is None:
            config = StochasticACTConfigWrapper.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=True,
        )
