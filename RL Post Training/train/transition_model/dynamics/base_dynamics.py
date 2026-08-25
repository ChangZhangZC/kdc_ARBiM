import numpy as np
import torch.nn as nn
from typing import Dict, Tuple


class BaseDynamics:
    def __init__(
        self,
        model: nn.Module,
        optim,
    ) -> None:
        self.model = model
        self.optim = optim

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        raise NotImplementedError