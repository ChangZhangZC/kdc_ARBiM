import torch

from .iql_critic import IQLCritic


class CachedIQLCritic(IQLCritic):
    """IQL critic variant that accepts frozen ACT token latents directly."""

    def _encode_shared_obs(
        self,
        obs: dict[str, torch.Tensor],
        start: int = 0,
        track_grad: bool | None = None,
    ) -> torch.Tensor:
        if "latent" not in obs:
            return super()._encode_shared_obs(obs, start=start, track_grad=track_grad)

        latent = obs["latent"].to(self._device, non_blocking=True).float()
        if latent.ndim == 4:
            if latent.shape[1] != 1:
                raise ValueError(
                    "Cached critic latent must contain exactly one endpoint timestep, "
                    f"got {tuple(latent.shape)}"
                )
            latent = latent[:, 0]
        if latent.ndim != 3:
            raise ValueError(
                f"Cached critic latent must be [B,S,D] or [B,1,S,D], got {tuple(latent.shape)}"
            )
        state = latent.mean(dim=1)
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Cached critic state dim {state.shape[-1]} != expected {self.state_dim}"
            )
        return state
