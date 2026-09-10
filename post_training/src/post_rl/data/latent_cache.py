from __future__ import annotations

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


class FrozenLatentCache:
    def __init__(self, directory: str, expected_metadata: dict | None = None) -> None:
        self.directory = os.path.abspath(directory)
        metadata_path = os.path.join(self.directory, "metadata.json")
        obs_path = os.path.join(self.directory, "obs_latent.npy")
        next_obs_path = os.path.join(self.directory, "next_obs_latent.npy")
        if not all(os.path.isfile(path) for path in (metadata_path, obs_path, next_obs_path)):
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
        self.next_obs = LazyNpyArray(next_obs_path)
        if self.obs.shape != self.next_obs.shape:
            raise RuntimeError(
                f"Frozen latent cache shape mismatch: {self.obs.shape} vs {self.next_obs.shape}"
            )


def _raw_endpoint_batch(buffer, start: int, end: int, use_depth: bool, next_obs: bool):
    prefix = "next_" if next_obs else ""
    obs = {
        "state": torch.from_numpy(np.asarray(buffer[f"{prefix}state"][start:end])).unsqueeze(1),
    }
    for key in RGB_KEYS:
        obs[key] = torch.from_numpy(np.asarray(buffer[f"{prefix}{key}"][start:end])).unsqueeze(1)
    if use_depth:
        for key in DEPTH_KEYS:
            obs[key] = torch.from_numpy(np.asarray(buffer[f"{prefix}{key}"][start:end])).unsqueeze(1)
    return obs


def _encode_endpoint_batch(obs_adapter, buffer, start, end, use_depth, next_obs):
    obs = _raw_endpoint_batch(buffer, start, end, use_depth, next_obs)
    with torch.no_grad():
        latent = obs_adapter.encode(obs, start=0, track_grad=False)[:, 0]
    return latent.detach().float().cpu().numpy()


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
    obs_memmap = None
    next_memmap = None
    try:
        ranges = range(0, size, batch_size)
        if progress:
            from tqdm import tqdm

            ranges = tqdm(ranges, total=(size + batch_size - 1) // batch_size, desc="Caching ACT latents")

        for start in ranges:
            end = min(start + batch_size, size)
            obs_latent = _encode_endpoint_batch(
                obs_adapter, buffer, start, end, use_depth, next_obs=False
            )
            next_latent = _encode_endpoint_batch(
                obs_adapter, buffer, start, end, use_depth, next_obs=True
            )
            if obs_memmap is None:
                latent_shape = (size,) + tuple(obs_latent.shape[1:])
                obs_memmap = np.lib.format.open_memmap(
                    os.path.join(tmp_dir, "obs_latent.npy"),
                    mode="w+",
                    dtype=np.float32,
                    shape=latent_shape,
                )
                next_memmap = np.lib.format.open_memmap(
                    os.path.join(tmp_dir, "next_obs_latent.npy"),
                    mode="w+",
                    dtype=np.float32,
                    shape=latent_shape,
                )
            if obs_latent.shape[1:] != obs_memmap.shape[1:]:
                raise RuntimeError("ACT latent shape changed while building cache")
            obs_memmap[start:end] = obs_latent
            next_memmap[start:end] = next_latent

        if obs_memmap is None or next_memmap is None:
            raise RuntimeError("Cannot build latent cache from an empty buffer")
        obs_memmap.flush()
        next_memmap.flush()
        metadata = dict(metadata)
        metadata.update(
            {
                "size": int(size),
                "latent_shape": list(obs_memmap.shape[1:]),
                "dtype": "float32",
            }
        )
        with open(os.path.join(tmp_dir, "metadata.json"), "w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        del obs_memmap, next_memmap

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
