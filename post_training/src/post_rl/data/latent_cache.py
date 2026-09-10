from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch


RGB_KEYS = ("head_rgb", "wrist_left_rgb", "wrist_right_rgb")
DEPTH_KEYS = ("head_depth", "wrist_left_depth", "wrist_right_depth")


class LazyNpyArray:
    """Process-safe read-only mmap wrapper for cached latent arrays."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        array = np.load(self.path, mmap_mode="r")
        self.shape = tuple(array.shape)
        self.dtype = np.dtype(array.dtype)
        self._array = None
        self._pid = None

    def _open(self):
        pid = os.getpid()
        if self._array is None or self._pid != pid:
            self._array = np.load(self.path, mmap_mode="r")
            self._pid = pid
        return self._array

    def __getitem__(self, index):
        return np.asarray(self._open()[index])

    def __len__(self) -> int:
        return self.shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_pid"] = None
        return state


class MappedLatentArray:
    """View next-observation latents through the transition next-index mapping."""

    def __init__(self, base: LazyNpyArray, next_indices: np.ndarray) -> None:
        self.base = base
        self.next_indices = np.asarray(next_indices, dtype=np.int64)
        self.shape = base.shape
        self.dtype = base.dtype
        if len(self.next_indices) != len(base):
            raise ValueError("next-index map length must match latent cache length")

    def __getitem__(self, index):
        mapped = self.next_indices[index]
        return self.base[mapped]

    def __len__(self) -> int:
        return len(self.base)


class FrozenLatentCache:
    def __init__(self, directory: str, expected_metadata: dict | None = None) -> None:
        self.directory = os.path.abspath(directory)
        metadata_path = os.path.join(self.directory, "metadata.json")
        obs_path = os.path.join(self.directory, "obs_latent.npy")
        next_indices_path = os.path.join(self.directory, "next_indices.npy")
        if not all(
            os.path.isfile(path)
            for path in (metadata_path, obs_path, next_indices_path)
        ):
            raise FileNotFoundError(f"Incomplete frozen latent cache: {self.directory}")
        with open(metadata_path, "r") as file:
            self.metadata = json.load(file)
        if expected_metadata is not None:
            mismatch = {
                key: (self.metadata.get(key), value)
                for key, value in expected_metadata.items()
                if self.metadata.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"Frozen latent cache contract mismatch: {mismatch}")
        self.obs = LazyNpyArray(obs_path)
        self.next_indices = np.load(next_indices_path, mmap_mode="r")
        self.next_obs = MappedLatentArray(self.obs, self.next_indices)


def _raw_obs_batch(buffer, start: int, end: int, use_depth: bool):
    obs = {
        "state": torch.from_numpy(np.asarray(buffer["state"][start:end])).unsqueeze(1),
    }
    for key in RGB_KEYS:
        obs[key] = torch.from_numpy(np.asarray(buffer[key][start:end])).unsqueeze(1)
    if use_depth:
        for key in DEPTH_KEYS:
            obs[key] = torch.from_numpy(np.asarray(buffer[key][start:end])).unsqueeze(1)
    return obs


def _encode_obs_batch(obs_adapter, buffer, start, end, use_depth):
    obs = _raw_obs_batch(buffer, start, end, use_depth)
    with torch.no_grad():
        latent = obs_adapter.encode(obs, start=0, track_grad=False)[:, 0]
    return latent.detach().float().cpu().numpy()


def _build_next_indices(buffer) -> np.ndarray:
    size = len(buffer)
    next_indices = np.arange(size, dtype=np.int64) + 1
    for episode_end in np.asarray(buffer.episode_ends, dtype=np.int64):
        terminal = int(episode_end) - 1
        next_indices[terminal] = terminal
    if np.any(next_indices < 0) or np.any(next_indices >= size):
        raise RuntimeError("Invalid transition next-index mapping")

    state = np.asarray(buffer["state"])
    next_state = np.asarray(buffer["next_state"])
    if not np.allclose(next_state, state[next_indices], rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "Frozen latent cache requires the ARBiM transition alignment contract: "
            "next_obs[i] must equal obs[i+1], with terminal self-loops."
        )
    return next_indices


def _episode_ends_sha256(buffer) -> str:
    episode_ends = np.asarray(buffer.episode_ends, dtype=np.int64)
    return hashlib.sha256(episode_ends.tobytes()).hexdigest()


def build_frozen_latent_cache(
    *,
    buffer,
    obs_adapter,
    cache_dir: str,
    metadata: dict,
    batch_size: int,
    use_depth: bool,
    progress: bool = True,
) -> FrozenLatentCache:
    cache_dir = os.path.abspath(cache_dir)
    metadata = dict(metadata)
    metadata.update(
        {
            "size": int(len(buffer)),
            "episode_ends_sha256": _episode_ends_sha256(buffer),
            "transition_alignment": "next_obs=obs[next_index]; terminal=self",
        }
    )
    if os.path.isdir(cache_dir):
        try:
            return FrozenLatentCache(cache_dir, expected_metadata=metadata)
        except (FileNotFoundError, RuntimeError):
            pass

    if batch_size < 1:
        raise ValueError("latent cache batch_size must be >= 1")

    parent = os.path.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)
    tmp_dir = f"{cache_dir}.tmp.{os.getpid()}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=False)

    size = len(buffer)
    next_indices = _build_next_indices(buffer)
    obs_memmap = None
    try:
        ranges = range(0, size, batch_size)
        if progress:
            from tqdm import tqdm

            ranges = tqdm(
                ranges,
                total=(size + batch_size - 1) // batch_size,
                desc="Caching ACT latents",
            )

        for start in ranges:
            end = min(start + batch_size, size)
            obs_latent = _encode_obs_batch(
                obs_adapter,
                buffer,
                start,
                end,
                use_depth,
            )
            if obs_memmap is None:
                latent_shape = (size,) + tuple(obs_latent.shape[1:])
                obs_memmap = np.lib.format.open_memmap(
                    os.path.join(tmp_dir, "obs_latent.npy"),
                    mode="w+",
                    dtype=np.float32,
                    shape=latent_shape,
                )
            if obs_latent.shape[1:] != obs_memmap.shape[1:]:
                raise RuntimeError("ACT latent shape changed while building cache")
            obs_memmap[start:end] = obs_latent

        if obs_memmap is None:
            raise RuntimeError("Cannot build latent cache from an empty buffer")
        obs_memmap.flush()
        metadata.update(
            {
                "latent_shape": list(obs_memmap.shape[1:]),
                "dtype": "float32",
            }
        )
        np.save(os.path.join(tmp_dir, "next_indices.npy"), next_indices)
        with open(os.path.join(tmp_dir, "metadata.json"), "w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        del obs_memmap

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.replace(tmp_dir, cache_dir)
    except Exception:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise

    return FrozenLatentCache(cache_dir, expected_metadata=metadata)


def default_latent_cache_dir(
    dataset_path: str,
    encoder_sha256: str,
    normalizer_sha256: str,
) -> str:
    root = Path(os.path.abspath(os.path.expanduser(dataset_path)))
    cache_root = Path(str(root) + ".act_latent_cache")
    key = f"{encoder_sha256[:16]}_{normalizer_sha256[:16]}"
    return str(cache_root / key)
