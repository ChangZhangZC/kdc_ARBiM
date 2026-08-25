import os.path as path

import numpy as np
import torch


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mu) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.std * data + self.mu

    def save_scaler(self, save_path: str) -> None:
        np.save(path.join(save_path, "mu.npy"), self.mu)
        np.save(path.join(save_path, "std.npy"), self.std)

    def load_scaler(self, load_path: str) -> None:
        self.mu = np.load(path.join(load_path, "mu.npy"))
        self.std = np.load(path.join(load_path, "std.npy"))

    def transform_tensor(
        self,
        data: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        data = self.transform(data.detach().cpu().numpy())
        return torch.as_tensor(data, device=device)