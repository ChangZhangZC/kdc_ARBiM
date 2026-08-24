import torch
import torch.nn as nn
from rl_100.model.diffusion.conv1d_components import Conv1dBlock, Downsample1d


class ActionChunkEncoder(nn.Module):
    """
    Temporal Conv1d encoder: action_chunk [B, H, Da] -> z [B, Tz, Cz].
    With default params and H=16: Tz=4, output flatten dim = Tz*Cz = 128.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dims: list = [128, 256],
        latent_cz: int = 32,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_cz = latent_cz

        dims = [action_dim] + hidden_dims + [latent_cz]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(Conv1dBlock(dims[i], dims[i + 1], kernel_size, n_groups))
            if i < len(dims) - 2:
                layers.append(Downsample1d(dims[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_chunk: [B, H, Da]
        Returns:
            z: [B, Tz, Cz]
        """
        x = action_chunk.transpose(1, 2)  # [B, Da, H]
        x = self.net(x)                    # [B, Cz, Tz]
        return x.transpose(1, 2)           # [B, Tz, Cz]
