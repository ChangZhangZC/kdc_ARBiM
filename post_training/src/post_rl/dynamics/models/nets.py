import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        layers = []

        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                layers.append(nn.Dropout(p=dropout_rate))

        self.output_dim = hidden_dims[-1]

        if output_dim is not None:
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
            self.output_dim = output_dim

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.weight = nn.Parameter(
            torch.zeros(num_ensemble, input_dim, output_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_ensemble, 1, output_dim)
        )

        nn.init.trunc_normal_(
            self.weight,
            std=1 / (2 * input_dim**0.5),
        )

        self.saved_weight = nn.Parameter(
            self.weight.detach().clone(),
            requires_grad=False,
        )
        self.saved_bias = nn.Parameter(
            self.bias.detach().clone(),
            requires_grad=False,
        )

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = torch.einsum(
                "ij,bjk->bik",
                x,
                self.weight,
            )
        else:
            x = torch.einsum(
                "bij,bjk->bik",
                x,
                self.weight,
            )

        return x + self.bias

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        return self.weight_decay * 0.5 * self.weight.pow(2).sum()