import torch
from torch import Tensor
from torch.distributions import Normal

import os
from pathlib import Path
from safetensors.torch import load_model
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError

from kuavo_train.wrapper.policy.act.StochasticACTConfigWrapper import StochasticACTConfigWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper, OBS_DEPTH
from lerobot.utils.constants import ACTION, OBS_IMAGES


class StochasticACTPolicyWrapper(CustomACTPolicyWrapper):
    
    def __init__(self, config: StochasticACTConfigWrapper):
        super().__init__(config)
        
        action_shape = config.action_feature.shape
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action feature, got shape={action_shape}")
        action_dim = action_shape[0]
        
        init_ratio = (config.init_log_std - config.log_std_min) / (config.log_std_max - config.log_std_min)
        raw_init = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.raw_log_std = torch.nn.Parameter(raw_init.repeat(action_dim))

    def _get_log_std(self) -> Tensor:
        return self.config.log_std_min + (self.config.log_std_max - self.config.log_std_min) * torch.sigmoid(self.raw_log_std)

    def _get_std(self) -> Tensor:
        return self._get_log_std().exp()
    
    # RL 上层调用这个函数时 policy/model 必须处于 eval() mode。
    def get_action_mean(self, batch: dict[str, Tensor]) -> Tensor:
        model_batch = dict(batch)
        model_batch.pop(ACTION, None)
        model_batch.pop("action_is_pad", None)
        if self.config.image_features:
            model_batch[OBS_IMAGES] = [model_batch[key] for key in self.config.image_features]
        if self.config.use_depth and self.config.depth_features:
            model_batch[OBS_DEPTH] = [model_batch[key].mean(dim=-3, keepdim=True) for key in self.config.depth_features]
        return self.model(model_batch)[0]
    
    def get_distribution(self, batch: dict[str, Tensor]) -> Normal:
        mu = self.get_action_mean(batch)
        std = self._get_std()
        return Normal(mu, std)
    
    def sample_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.get_distribution(batch)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
    
    def evaluate_action_chunk(self, batch: dict[str, Tensor], action_chunk: Tensor) -> tuple[Tensor, Tensor]:
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
        if config is not None and any(v is not None for v in (init_log_std, log_std_min, log_std_max)):
            raise ValueError("Do not pass stochastic parameters together with an explicit config.")
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
                raise FileNotFoundError(f"{SAFETENSORS_SINGLE_FILE} not found in {model_id}") from e

        policy = cls(config)
        missing, unexpected = load_model(policy, model_file, strict=False, device=config.device)
        if set(missing) != {"raw_log_std"} or unexpected:
            raise RuntimeError(f"Invalid IL checkpoint migration: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
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