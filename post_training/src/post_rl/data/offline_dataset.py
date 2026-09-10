from copy import copy
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .offline_buffer import OfflineBuffer
from .sampler import SequenceSampler, downsample_mask, get_val_mask


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
        endpoint_obs_only: bool = False,
        latent_cache=None,
        include_next_obs: bool = True,
    ) -> None:
        if len(buffer) == 0:
            raise ValueError("OfflineBuffer must be loaded before creating dataset.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if endpoint_obs_only and (pad_before != 0 or pad_after != 0):
            raise ValueError("endpoint_obs_only requires pad_before=pad_after=0.")

        self.buffer = buffer
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sequence_stride = sequence_stride
        self.use_depth = use_depth
        self.endpoint_obs_only = bool(endpoint_obs_only)
        self.latent_cache = latent_cache
        self.include_next_obs = bool(include_next_obs)

        if self.endpoint_obs_only:
            self.sampler_keys = ["action", "next_action", "reward", "not_done", "return"]
        else:
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

    def set_latent_cache(self, latent_cache) -> None:
        self.latent_cache = latent_cache

    def _endpoint_indices(self, idx: int) -> tuple[int, int]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        if sample_start_idx != 0 or sample_end_idx != self.horizon:
            raise RuntimeError("Endpoint sampling does not support padded sequences.")
        return int(buffer_start_idx), int(buffer_end_idx - 1)

    def _raw_endpoint_obs(self, index: int, next_obs: bool) -> dict[str, np.ndarray]:
        prefix = "next_" if next_obs else ""
        obs = {
            "state": np.asarray(self.buffer[f"{prefix}state"][index:index + 1]),
        }
        for key in self.RGB_KEYS:
            obs[key] = np.asarray(self.buffer[f"{prefix}{key}"][index:index + 1])
        if self.use_depth:
            for key in self.DEPTH_KEYS:
                obs[key] = np.asarray(self.buffer[f"{prefix}{key}"][index:index + 1])
        return obs

    def _endpoint_obs(self, index: int, next_obs: bool) -> dict[str, np.ndarray]:
        if self.latent_cache is not None:
            latent = (
                self.latent_cache.next_obs[index]
                if next_obs
                else self.latent_cache.obs[index]
            )
            return {"latent": np.asarray(latent, dtype=np.float32)[None]}
        return self._raw_endpoint_obs(index, next_obs)

    def _sample_to_data(self, sample: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
        if self.endpoint_obs_only:
            obs_idx, next_idx = self._endpoint_indices(idx)
            obs = self._endpoint_obs(obs_idx, next_obs=False)
            next_obs = self._endpoint_obs(next_idx, next_obs=True) if self.include_next_obs else None
        else:
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

        data = {
            "obs": obs,
            "action": sample["action"].astype(np.float32, copy=False),
            "next_action": sample["next_action"].astype(np.float32, copy=False),
            "reward": sample["reward"].astype(np.float32, copy=False),
            "not_done": sample["not_done"].astype(np.float32, copy=False),
            "return": sample["return"].astype(np.float32, copy=False),
        }
        if next_obs is not None:
            data["next_obs"] = next_obs
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.sampler.sample_sequence(idx)
        return self._to_tensor(self._sample_to_data(sample, idx))

    def __len__(self) -> int:
        return len(self.sampler)

    def get_validation_dataset(self) -> "OfflineDataset":
        val_dataset = copy(self)
        val_dataset.sampler = val_dataset._build_sampler(self.val_mask)
        return val_dataset

    @staticmethod
    def _to_tensor(data: Any) -> Any:
        if isinstance(data, dict):
            return {key: OfflineDataset._to_tensor(value) for key, value in data.items()}
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data
