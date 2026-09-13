import os

import cv2
import numpy as np
import torch
import zarr
from tqdm import tqdm

RGB_ZARR_STORAGE = "jpeg_bytes_v1"


class LazyZarrArray:
    """Process-safe lazy handle for one dense array under Zarr data/."""

    def __init__(self, zarr_path: str, key: str, shape, dtype) -> None:
        self.zarr_path = str(zarr_path)
        self.key = str(key)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._array = None
        self._pid = None

    def _open(self):
        pid = os.getpid()
        if self._array is None or self._pid != pid:
            root = zarr.open_group(self.zarr_path, mode="r")
            self._array = root["data"][self.key]
            self._pid = pid
        return self._array

    def __getitem__(self, index):
        array = self._open()
        try:
            return np.asarray(array[index])
        except (IndexError, TypeError):
            return np.asarray(array.oindex[index])

    def __len__(self) -> int:
        return self.shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_pid"] = None
        return state


class LazyJpegZarrArray:
    """Logical dense RGB array backed by JPEG bytes + offsets in Zarr."""

    def __init__(
        self,
        zarr_path: str,
        key: str,
        shape,
        next_view: bool = False,
    ) -> None:
        self.zarr_path = str(zarr_path)
        self.key = str(key)
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.uint8)
        self.next_view = bool(next_view)
        self._data = None
        self._offsets = None
        self._next_index = None
        self._pid = None

    def _open(self):
        pid = os.getpid()
        if self._data is None or self._pid != pid:
            root = zarr.open_group(self.zarr_path, mode="r")
            group = root["data"]
            self._data = group[f"{self.key}_jpeg_data"]
            self._offsets = group[f"{self.key}_jpeg_offsets"]
            self._next_index = group["next_index"] if self.next_view else None
            self._pid = pid

    def _normalize_indices(self, index):
        size = self.shape[0]
        scalar = isinstance(index, (int, np.integer))
        if scalar:
            value = int(index)
            if value < 0:
                value += size
            if value < 0 or value >= size:
                raise IndexError(value)
            indices = np.asarray([value], dtype=np.int64)
        elif isinstance(index, slice):
            start, stop, step = index.indices(size)
            indices = np.arange(start, stop, step, dtype=np.int64)
        else:
            values = np.asarray(index)
            if values.dtype == bool:
                if values.ndim != 1 or len(values) != size:
                    raise IndexError("boolean index must match RGB array length")
                indices = np.flatnonzero(values).astype(np.int64)
            else:
                indices = values.astype(np.int64, copy=False).reshape(-1)
                indices = np.where(indices < 0, indices + size, indices)
                if np.any((indices < 0) | (indices >= size)):
                    raise IndexError("RGB index out of bounds")
        return indices, scalar

    def _take_zarr(self, array, indices):
        if len(indices) == 0:
            return np.empty((0,), dtype=array.dtype)
        try:
            return np.asarray(array.oindex[indices])
        except (AttributeError, IndexError, TypeError):
            return np.asarray([array[int(i)] for i in indices])

    @staticmethod
    def _decode(payload):
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to decode JPEG from Offline RL Zarr")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(np.moveaxis(image, -1, 0))

    def _decode_indices(self, indices):
        if len(indices) == 0:
            return np.empty((0, *self.shape[1:]), dtype=np.uint8)
        if self.next_view:
            indices = self._take_zarr(self._next_index, indices).astype(np.int64, copy=False)

        images = []
        contiguous = len(indices) == 1 or np.all(np.diff(indices) == 1)
        if contiguous:
            first = int(indices[0])
            last = int(indices[-1])
            offsets = np.asarray(self._offsets[first:last + 2], dtype=np.int64)
            byte_start = int(offsets[0])
            byte_end = int(offsets[-1])
            raw = np.asarray(self._data[byte_start:byte_end], dtype=np.uint8)
            for i in range(len(indices)):
                start = int(offsets[i] - byte_start)
                end = int(offsets[i + 1] - byte_start)
                images.append(self._decode(raw[start:end].tobytes()))
        else:
            offsets_start = self._take_zarr(self._offsets, indices).astype(np.int64)
            offsets_end = self._take_zarr(self._offsets, indices + 1).astype(np.int64)
            for start, end in zip(offsets_start, offsets_end, strict=True):
                raw = np.asarray(self._data[int(start):int(end)], dtype=np.uint8)
                images.append(self._decode(raw.tobytes()))

        result = np.stack(images, axis=0)
        if result.shape[1:] != self.shape[1:]:
            raise ValueError(
                f"decoded {self.key} shape {result.shape[1:]} != stored shape {self.shape[1:]}"
            )
        return result

    def __getitem__(self, index):
        self._open()
        indices, scalar = self._normalize_indices(index)
        result = self._decode_indices(indices)
        return result[0] if scalar else result

    def __len__(self) -> int:
        return self.shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        state["_offsets"] = None
        state["_next_index"] = None
        state["_pid"] = None
        return state


class OfflineBuffer:
    """Offline-only replay buffer with lazy Zarr-backed image modalities."""

    RGB_KEYS = ("head_rgb", "wrist_left_rgb", "wrist_right_rgb")
    DEPTH_KEYS = ("head_depth", "wrist_left_depth", "wrist_right_depth")

    def __init__(
        self,
        device: torch.device,
        gamma: float,
        use_depth: bool = False,
    ) -> None:
        self._device = device
        self._gamma = gamma
        self._use_depth = use_depth
        self._state = np.empty(0)
        self._action = np.empty(0)
        self._reward = np.empty((0, 1), dtype=np.float32)
        self._next_state = np.empty(0)
        self._next_action = np.empty(0)
        self._done = np.empty((0, 1), dtype=np.float32)
        self._timeout = np.empty((0, 1), dtype=np.float32)
        self._not_done = np.empty((0, 1), dtype=np.float32)
        self._return = np.empty((0, 1), dtype=np.float32)
        self._episode_ends = np.empty(0, dtype=np.int64)
        self._extra_data: dict[str, object] = {}
        self._size = 0

    def _modal_keys(self) -> set[str]:
        keys = set(self.RGB_KEYS) | {f"next_{key}" for key in self.RGB_KEYS}
        if self._use_depth:
            keys |= set(self.DEPTH_KEYS) | {f"next_{key}" for key in self.DEPTH_KEYS}
        return keys

    def _validate_episode_ends(self, meta_group, size):
        if "episode_ends" not in meta_group:
            raise KeyError("Zarr meta group is missing 'episode_ends'.")
        episode_ends = np.asarray(meta_group["episode_ends"][:], dtype=np.int64)
        if episode_ends.ndim != 1 or len(episode_ends) == 0:
            raise ValueError("episode_ends must be a non-empty 1D array.")
        if np.any(np.diff(episode_ends) <= 0):
            raise ValueError("episode_ends must be strictly increasing.")
        if int(episode_ends[-1]) != size:
            raise ValueError(
                f"Last episode_end must equal dataset size {size}, got {episode_ends[-1]}."
            )
        return episode_ends

    def _load_core(self, data_group, episode_ends):
        self._state = np.asarray(data_group["state"][:])
        self._action = np.asarray(data_group["action"][:])
        self._reward = np.asarray(data_group["reward"][:], dtype=np.float32).reshape(-1, 1)
        self._next_state = np.asarray(data_group["next_state"][:])
        self._next_action = np.asarray(data_group["next_action"][:])
        self._done = np.asarray(data_group["done"][:], dtype=np.float32).reshape(-1, 1)
        self._timeout = np.asarray(data_group["timeout"][:], dtype=np.float32).reshape(-1, 1)
        self._not_done = 1.0 - self._done
        self._episode_ends = episode_ends
        self._size = len(self._reward)
        if "return" in data_group:
            self._return = np.asarray(data_group["return"][:], dtype=np.float32).reshape(-1, 1)
        else:
            self._return = np.zeros((self._size, 1), dtype=np.float32)

    def _load_jpeg_rgb(self, root, data_group, zarr_path, size):
        if "next_index" not in data_group:
            raise KeyError("JPEG RGB Zarr is missing data/next_index")
        if int(data_group["next_index"].shape[0]) != size:
            raise ValueError("data/next_index length must match dataset size")
        shapes = dict(root.attrs.get("rgb_shapes", {}))
        self._extra_data = {}
        for key in self.RGB_KEYS:
            data_key = f"{key}_jpeg_data"
            offsets_key = f"{key}_jpeg_offsets"
            missing = [name for name in (data_key, offsets_key) if name not in data_group]
            if missing:
                raise KeyError(f"JPEG RGB Zarr is missing field(s): {missing}")
            if key not in shapes:
                raise KeyError(f"JPEG RGB Zarr attrs.rgb_shapes is missing {key!r}")
            image_shape = tuple(int(value) for value in shapes[key])
            if len(image_shape) != 3:
                raise ValueError(f"RGB shape for {key} must be CHW, got {image_shape}")
            if int(data_group[offsets_key].shape[0]) != size + 1:
                raise ValueError(f"{offsets_key} length must be dataset size + 1")
            logical_shape = (size, *image_shape)
            self._extra_data[key] = LazyJpegZarrArray(
                zarr_path=zarr_path,
                key=key,
                shape=logical_shape,
                next_view=False,
            )
            self._extra_data[f"next_{key}"] = LazyJpegZarrArray(
                zarr_path=zarr_path,
                key=key,
                shape=logical_shape,
                next_view=True,
            )

        if self._use_depth:
            for key in self.DEPTH_KEYS:
                for logical_key in (key, f"next_{key}"):
                    if logical_key not in data_group:
                        raise KeyError(f"Dataset is missing required field {logical_key!r}")
                    array = data_group[logical_key]
                    if int(array.shape[0]) != size:
                        raise ValueError(f"{logical_key} length must match dataset size")
                    self._extra_data[logical_key] = LazyZarrArray(
                        zarr_path=zarr_path,
                        key=logical_key,
                        shape=array.shape,
                        dtype=array.dtype,
                    )

    def _load_dense_modalities(self, data_group, zarr_path, size):
        modal_keys = self._modal_keys()
        missing = sorted(key for key in modal_keys if key not in data_group)
        if missing:
            raise KeyError(f"Dataset is missing required field(s): {missing}")
        self._extra_data = {}
        for key in modal_keys:
            array = data_group[key]
            if int(array.shape[0]) != size:
                raise ValueError(f"{key} length must match dataset size")
            self._extra_data[key] = LazyZarrArray(
                zarr_path=zarr_path,
                key=key,
                shape=array.shape,
                dtype=array.dtype,
            )

    def load_zarr(self, zarr_path: str) -> None:
        root = zarr.open_group(zarr_path, mode="r")
        if "data" not in root or "meta" not in root:
            raise KeyError("Zarr dataset must contain 'data' and 'meta' groups.")
        data_group = root["data"]
        meta_group = root["meta"]
        core_keys = (
            "state",
            "action",
            "reward",
            "next_state",
            "next_action",
            "done",
            "timeout",
        )
        missing = sorted(key for key in core_keys if key not in data_group)
        if missing:
            raise KeyError(f"Dataset is missing required field(s): {missing}")
        size = int(data_group["reward"].shape[0])
        invalid_lengths = {
            key: int(data_group[key].shape[0])
            for key in core_keys
            if int(data_group[key].shape[0]) != size
        }
        if invalid_lengths:
            raise ValueError(
                f"Dataset fields must have equal length {size}, got {invalid_lengths}"
            )
        episode_ends = self._validate_episode_ends(meta_group, size)
        self._load_core(data_group, episode_ends)
        if root.attrs.get("rgb_storage") == RGB_ZARR_STORAGE:
            self._load_jpeg_rgb(root, data_group, zarr_path, size)
        else:
            self._load_dense_modalities(data_group, zarr_path, size)

    def load_dataset(self, dataset: dict[str, np.ndarray]) -> None:
        self._validate_dataset(dataset)
        self._state = np.asarray(dataset["state"])
        self._action = np.asarray(dataset["action"])
        self._reward = np.asarray(dataset["reward"], dtype=np.float32).reshape(-1, 1)
        self._next_state = np.asarray(dataset["next_state"])
        self._next_action = np.asarray(dataset["next_action"])
        self._done = np.asarray(dataset["done"], dtype=np.float32).reshape(-1, 1)
        self._timeout = np.asarray(dataset["timeout"], dtype=np.float32).reshape(-1, 1)
        self._not_done = 1.0 - self._done
        self._episode_ends = np.asarray(dataset["episode_ends"], dtype=np.int64)
        self._size = len(self._reward)
        self._return = np.asarray(
            dataset.get("return", np.zeros((self._size, 1), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 1)
        reserved = {
            "state",
            "action",
            "reward",
            "next_state",
            "next_action",
            "done",
            "timeout",
            "episode_ends",
            "not_done",
            "return",
        }
        if not self._use_depth:
            reserved.update(self.DEPTH_KEYS)
            reserved.update(f"next_{key}" for key in self.DEPTH_KEYS)
        self._extra_data = {
            key: np.asarray(value)
            for key, value in dataset.items()
            if key not in reserved
        }

    def compute_return(self) -> None:
        if self._size == 0:
            raise RuntimeError("Dataset must be loaded before computing returns.")
        self._return.fill(0.0)
        episode_start = 0
        for episode_end in tqdm(self._episode_ends, desc="Computing returns"):
            running_return = 0.0
            for i in reversed(range(episode_start, episode_end)):
                running_return = self._reward[i] + self._gamma * running_return * self._not_done[i]
                self._return[i] = running_return
            episode_start = episode_end

    def reward_normalize(self, scaling: str = "dynamic", fixed_scale: float = 0.1) -> None:
        if self._size == 0:
            raise RuntimeError("Dataset must be loaded before reward normalization.")
        if scaling == "dynamic":
            rewards = self._reward.copy()
            n, mean, S = 0, 0.0, 1.0
            episode_start = 0
            for episode_end in self._episode_ends:
                running_return = 0.0
                for i in range(episode_start, episode_end):
                    running_return = self._gamma * running_return + float(self._reward[i, 0])
                    n += 1
                    if n == 1:
                        mean = running_return
                    else:
                        old_mean = mean
                        mean += (running_return - old_mean) / n
                        S += (running_return - old_mean) * (running_return - mean)
                    std = np.sqrt(S / n)
                    rewards[i, 0] /= std + 1e-8
                episode_start = episode_end
            self._reward = rewards
        elif scaling == "normal":
            episode_returns = []
            episode_start = 0
            for episode_end in self._episode_ends:
                episode_returns.append(self._reward[episode_start:episode_end].sum())
                episode_start = episode_end
            reward_range = max(episode_returns) - min(episode_returns)
            if reward_range > 1e-8:
                self._reward = self._reward / reward_range * 1000.0
        elif scaling == "number":
            self._reward *= fixed_scale
        elif scaling != "none":
            raise ValueError(f"Unknown reward scaling mode: {scaling}")
        self.compute_return()

    def load_filter_dataset(self, dataset: dict[str, np.ndarray], min_return: float = 0.0) -> None:
        self.load_dataset(dataset)
        self.compute_return()
        keep_indices = []
        new_episode_ends = []
        episode_start = 0
        new_end = 0
        for episode_end in self._episode_ends:
            episode_return = float(self._return[episode_start, 0])
            if episode_return > min_return:
                keep_indices.extend(range(episode_start, episode_end))
                new_end += episode_end - episode_start
                new_episode_ends.append(new_end)
            episode_start = episode_end
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        self._state = self._state[keep_indices]
        self._action = self._action[keep_indices]
        self._reward = self._reward[keep_indices]
        self._next_state = self._next_state[keep_indices]
        self._next_action = self._next_action[keep_indices]
        self._done = self._done[keep_indices]
        self._timeout = self._timeout[keep_indices]
        self._not_done = self._not_done[keep_indices]
        self._return = self._return[keep_indices]
        for key, value in self._extra_data.items():
            if len(value) == self._size:
                self._extra_data[key] = value[keep_indices]
        self._episode_ends = np.asarray(new_episode_ends, dtype=np.int64)
        self._size = len(keep_indices)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Dataset must be loaded before sampling.")
        indices = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "state": self._state[indices],
            "action": self._action[indices],
            "reward": self._reward[indices],
            "next_state": self._next_state[indices],
            "next_action": self._next_action[indices],
            "done": self._done[indices],
            "timeout": self._timeout[indices],
            "not_done": self._not_done[indices],
            "return": self._return[indices],
        }
        for key, value in self._extra_data.items():
            if len(value) == self._size:
                batch[key] = value[indices]
        return {
            key: torch.as_tensor(value, device=self._device)
            for key, value in batch.items()
        }

    def sample_all(self) -> dict[str, np.ndarray]:
        if self._size == 0:
            raise RuntimeError("Dataset must be loaded before sampling.")
        data = {
            "state": self._state.copy(),
            "action": self._action.copy(),
            "reward": self._reward.copy(),
            "next_state": self._next_state.copy(),
            "next_action": self._next_action.copy(),
            "done": self._done.copy(),
            "timeout": self._timeout.copy(),
            "not_done": self._not_done.copy(),
            "return": self._return.copy(),
            "episode_ends": self._episode_ends.copy(),
        }
        for key, value in self._extra_data.items():
            if isinstance(value, (LazyZarrArray, LazyJpegZarrArray)):
                data[key] = np.asarray(value[:]).copy()
            else:
                data[key] = np.asarray(value).copy()
        return data

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def keys(self):
        return [
            "state",
            "action",
            "reward",
            "next_state",
            "next_action",
            "done",
            "timeout",
            "not_done",
            "return",
            *self._extra_data.keys(),
        ]

    def __getitem__(self, key: str):
        core_data = {
            "state": self._state,
            "action": self._action,
            "reward": self._reward,
            "next_state": self._next_state,
            "next_action": self._next_action,
            "done": self._done,
            "timeout": self._timeout,
            "not_done": self._not_done,
            "return": self._return,
        }
        if key in core_data:
            return core_data[key]
        if key in self._extra_data:
            return self._extra_data[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    @property
    def episode_ends(self) -> np.ndarray:
        if self._size == 0:
            raise RuntimeError("Dataset has not been loaded.")
        return self._episode_ends

    def _validate_dataset(self, dataset: dict[str, np.ndarray]) -> None:
        core_keys = (
            "state",
            "action",
            "reward",
            "next_state",
            "next_action",
            "done",
            "timeout",
            "episode_ends",
        )
        rgb_keys = self.RGB_KEYS + tuple(f"next_{key}" for key in self.RGB_KEYS)
        required = core_keys + rgb_keys
        if self._use_depth:
            depth_keys = self.DEPTH_KEYS + tuple(f"next_{key}" for key in self.DEPTH_KEYS)
            required += depth_keys
        missing = [key for key in required if key not in dataset]
        if missing:
            raise KeyError(f"Dataset is missing required field(s): {missing}")
        size = len(dataset["reward"])
        transition_keys = [key for key in required if key != "episode_ends"]
        lengths = {key: len(dataset[key]) for key in transition_keys}
        invalid = {key: length for key, length in lengths.items() if length != size}
        if invalid:
            raise ValueError(
                f"Dataset fields must have equal length {size}, got {invalid}"
            )
        episode_ends = np.asarray(dataset["episode_ends"], dtype=np.int64)
        if episode_ends.ndim != 1 or len(episode_ends) == 0:
            raise ValueError("episode_ends must be a non-empty 1D array.")
        if np.any(np.diff(episode_ends) <= 0):
            raise ValueError("episode_ends must be strictly increasing.")
        if episode_ends[-1] != size:
            raise ValueError(
                f"Last episode_end must equal dataset size {size}, got {episode_ends[-1]}."
            )
