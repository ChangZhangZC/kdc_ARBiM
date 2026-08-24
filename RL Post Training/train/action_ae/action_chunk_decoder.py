import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_100.model.diffusion.conv1d_components import Conv1dBlock, Upsample1d


class ActionChunkDecoder(nn.Module):
    """
    Temporal Conv1d decoder: z [B, Tz, Cz] -> action_chunk_recon [B, H, Da].
    Mirror of ActionChunkEncoder.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dims: list = [256, 128],
        latent_cz: int = 32,
        kernel_size: int = 5,
        n_groups: int = 8,
        target_len: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.target_len = target_len

        dims = [latent_cz] + hidden_dims + [action_dim]
        layers = []
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layers.append(Conv1dBlock(dims[i], dims[i + 1], kernel_size, n_groups))
                layers.append(Upsample1d(dims[i + 1]))
            else:
                layers.append(nn.Conv1d(dims[i], dims[i + 1], kernel_size, padding=kernel_size // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, Tz, Cz]
        Returns:
            recon: [B, H, Da]
        """
        x = z.transpose(1, 2)  # [B, Cz, Tz]
        x = self.net(x)        # [B, Da, T']
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode='linear', align_corners=False)
        return x.transpose(1, 2)  # [B, H, Da]
