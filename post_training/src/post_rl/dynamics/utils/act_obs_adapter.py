import torch

from lerobot.utils.constants import OBS_STATE
from ...critic.networks import ACTCriticEncoder


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


class ACTObservationAdapter:
    def __init__(
        self,
        encoder: ACTCriticEncoder,
        stats: dict,
        n_obs_steps: int = 1,
        device: torch.device | str = "cpu",
        fix_encoder: bool = True,
    ) -> None:
        if n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")

        self.encoder = encoder
        self.stats = stats
        self.n_obs_steps = n_obs_steps
        self.device = torch.device(device)
        self.fix_encoder = fix_encoder
        self.feature_dim = encoder.output_dim

        self.encoder.to(self.device)

        if fix_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _to_device(self, data):
        if isinstance(data, dict):
            return {
                key: self._to_device(value)
                for key, value in data.items()
            }

        if torch.is_tensor(data):
            return data.to(
                self.device,
                non_blocking=True,
            )

        return data

    def _get_stat(
        self,
        feature: str,
        name: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if feature not in self.stats:
            raise KeyError(f"Missing stats for {feature}")
        if name not in self.stats[feature]:
            raise KeyError(
                f"Missing {name} stats for {feature}"
            )

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

        denom = max_v - min_v
        denom = torch.where(
            denom == 0,
            torch.full_like(denom, 1e-8),
            denom,
        )
        return 2.0 * (x - min_v) / denom - 1.0

    def normalize_obs(
        self,
        obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        obs = self._to_device(obs)

        normalized = {
            OBS_STATE: self._normalize_mean_std(
                obs["state"],
                OBS_STATE,
            )
        }

        for buffer_key, feature_key in (
            RGB_BUFFER_TO_FEATURE.items()
        ):
            if buffer_key in obs:
                normalized[feature_key] = (
                    self._normalize_mean_std(
                        obs[buffer_key],
                        feature_key,
                    )
                )

        for buffer_key, feature_key in (
            DEPTH_BUFFER_TO_FEATURE.items()
        ):
            if buffer_key in obs:
                normalized[feature_key] = (
                    self._normalize_min_max(
                        obs[buffer_key],
                        feature_key,
                    )
                )

        return normalized

    def normalize_action(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = action.to(
            self.device,
            non_blocking=True,
        )

        return self._normalize_mean_std(
            action,
            ACTION_FEATURE,
        )

    def prepare_obs(
        self,
        obs: dict[str, torch.Tensor],
        start: int = 0,
    ) -> tuple[dict[str, torch.Tensor], int]:
        obs = self.normalize_obs(obs)
        batch_size = next(iter(obs.values())).shape[0]
        end = start + self.n_obs_steps

        prepared = {}

        for key, value in obs.items():
            if value.shape[1] < end:
                raise ValueError(
                    f"{key} needs at least {end} steps, "
                    f"got {value.shape[1]}"
                )

            value = value[:, start:end]

            prepared[key] = value.reshape(
                batch_size * self.n_obs_steps,
                *value.shape[2:],
            )

        return prepared, batch_size

    def encode(
        self,
        obs: dict[str, torch.Tensor],
        start: int = 0,
        track_grad: bool = False,
    ) -> torch.Tensor:
        obs, batch_size = self.prepare_obs(
            obs,
            start=start,
        )

        if track_grad and not self.fix_encoder:
            features = self.encoder(obs)
        else:
            with torch.no_grad():
                features = self.encoder(obs)

        return features.reshape(
            batch_size,
            self.n_obs_steps,
            self.feature_dim,
        )