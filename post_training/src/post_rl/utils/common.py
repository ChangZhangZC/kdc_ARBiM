import torch
import torch.nn as nn
from torch.distributions import Distribution
from typing import Callable, Dict

CONST_EPS = 1e-10


def orthogonal_initWeights(net: nn.Module) -> None:
    for param in net.parameters():
        if len(param.size()) >= 2:
            nn.init.orthogonal_(param)


def log_prob_func(
    dist: Distribution,
    action: torch.Tensor,
) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    return log_prob.sum(-1, keepdim=True)


def dict_apply(
    x: Dict[str, torch.Tensor],
    func: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class AdaptiveScheduler:
    def __init__(
        self,
        kl_threshold: float,
        min_lr: float,
        max_lr: float,
        init_lr: float,
    ) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold
        self.current_lr = init_lr

    def update(self, kl_dist: float) -> float:
        lr = self.current_lr
        if kl_dist > 2.0 * self.kl_threshold:
            lr = max(self.current_lr / 1.5, self.min_lr)
        if kl_dist < 0.5 * self.kl_threshold:
            lr = min(self.current_lr * 1.5, self.max_lr)
        self.current_lr = lr
        return lr
    
