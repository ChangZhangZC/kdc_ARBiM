from copy import copy
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from OfflineBuffer import OfflineBuffer
from sampler import SequenceSampler, get_val_mask, downsample_mask


class OfflineDataset(Dataset):
    RGB_KEYS = ("head_rgb", "wrist_left_rgb", "wrist_right_rgb")
    DEPTH_KEYS = ("head_depth", "wrist_left_depth", "wrist_right_depth")

    def __init__(
        self,
        buffer: OfflineBuffer,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        sequence_stride: int = 1,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        use_depth: bool = False,
    ) -> None:
        if len(buffer) == 0:
            raise ValueError("OfflineBuffer must be loaded before creating dataset.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

        self.buffer = buffer
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sequence_stride = sequence_stride
        self.use_depth = use_depth

        self.sampler_keys = [
            "state",
            "action",
            "reward",
            "next_state",
            "next_action",
            "not_done",
            "return",
            *self.RGB_KEYS,
            *(f"next_{key}" for key in self.RGB_KEYS),
        ]

        if use_depth:
            self.sampler_keys.extend(self.DEPTH_KEYS)
            self.sampler_keys.extend(f"next_{key}" for key in self.DEPTH_KEYS)

        missing = [key for key in self.sampler_keys if key not in buffer]
        if missing:
            raise KeyError(f"OfflineBuffer is missing required field(s): {missing}")

        episode_ends = buffer.episode_ends
        val_mask = get_val_mask(
            n_episodes=len(episode_ends),
            val_ratio=val_ratio,
            seed=seed,
        )

        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.train_mask = train_mask
        self.val_mask = val_mask

        self.sampler = self._build_sampler(train_mask)

    def _build_sampler(self, episode_mask: np.ndarray) -> SequenceSampler:
        return SequenceSampler(
            replay_buffer=self.buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.sampler_keys,
            episode_mask=episode_mask,
            sequence_stride=self.sequence_stride,
        )

    def _sample_to_data(self, sample: dict[str, np.ndarray]) -> dict[str, Any]:
        obs = {
            "state": sample["state"].astype(np.float32, copy=False),
            "head_rgb": sample["head_rgb"],
            "wrist_left_rgb": sample["wrist_left_rgb"],
            "wrist_right_rgb": sample["wrist_right_rgb"],
        }

        next_obs = {
            "state": sample["next_state"].astype(np.float32, copy=False),
            "head_rgb": sample["next_head_rgb"],
            "wrist_left_rgb": sample["next_wrist_left_rgb"],
            "wrist_right_rgb": sample["next_wrist_right_rgb"],
        }

        if self.use_depth:
            obs.update({
                "head_depth": sample["head_depth"],
                "wrist_left_depth": sample["wrist_left_depth"],
                "wrist_right_depth": sample["wrist_right_depth"],
            })
            next_obs.update({
                "head_depth": sample["next_head_depth"],
                "wrist_left_depth": sample["next_wrist_left_depth"],
                "wrist_right_depth": sample["next_wrist_right_depth"],
            })

        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": sample["action"].astype(np.float32, copy=False),
            "next_action": sample["next_action"].astype(np.float32, copy=False),
            "reward": sample["reward"].astype(np.float32, copy=False),
            "not_done": sample["not_done"].astype(np.float32, copy=False),
            "return": sample["return"].astype(np.float32, copy=False),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return self._to_tensor(data)

    def __len__(self) -> int:
        return len(self.sampler)

    def get_validation_dataset(self) -> "OfflineDataset":
        val_dataset = copy(self)
        val_dataset.sampler = val_dataset._build_sampler(self.val_mask)
        return val_dataset

    @staticmethod
    def _to_tensor(data: Any) -> Any:
        if isinstance(data, dict):
            return {
                key: OfflineRLDataset._to_tensor(value)
                for key, value in data.items()
            }
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data