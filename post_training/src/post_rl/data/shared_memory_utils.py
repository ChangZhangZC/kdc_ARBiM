import os
import pickle
import tempfile
from multiprocessing import shared_memory

import numpy as np
import zarr


class SharedMemoryManager:
    """Share offline Zarr arrays across DDP worker processes."""

    def __init__(self) -> None:
        self.shm_handles: dict[str, shared_memory.SharedMemory] = {}
        self.array_info: dict[str, dict] = {}
        self.meta_info: dict[str, np.ndarray] = {}
        self._owns_segments = False

    def create_from_zarr(
        self,
        zarr_path: str,
        keys: list[str] | None = None,
    ) -> dict:
        root = zarr.open_group(zarr_path, mode="r")
        if "data" not in root or "meta" not in root:
            raise KeyError("Zarr dataset must contain 'data' and 'meta' groups.")
        if "episode_ends" not in root["meta"]:
            raise KeyError("Zarr meta group is missing 'episode_ends'.")

        data_group = root["data"]
        selected_keys = list(data_group.keys()) if keys is None else list(keys)
        missing = [key for key in selected_keys if key not in data_group]
        if missing:
            raise KeyError(f"Zarr data group is missing field(s): {missing}")

        self.meta_info = {
            "episode_ends": np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        }
        self._owns_segments = True
        shared_data = {"meta": self.meta_info, "data": {}}

        for key in selected_keys:
            array = np.asarray(data_group[key][:])
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shared_array[...] = array

            self.shm_handles[key] = shm
            self.array_info[key] = {
                "shape": array.shape,
                "dtype": array.dtype.str,
                "shm_name": shm.name,
            }
            shared_data["data"][key] = shared_array

        return shared_data

    def get_from_shared_memory(self) -> dict:
        shared_data = {"meta": self.meta_info, "data": {}}
        self._owns_segments = False

        for key, info in self.array_info.items():
            shm = shared_memory.SharedMemory(name=info["shm_name"])
            array = np.ndarray(
                tuple(info["shape"]),
                dtype=np.dtype(info["dtype"]),
                buffer=shm.buf,
            )
            self.shm_handles[key] = shm
            shared_data["data"][key] = array

        return shared_data

    def save_info(self, filepath: str) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(
                {
                    "array_info": self.array_info,
                    "meta_info": self.meta_info,
                },
                file,
            )

    def load_info(self, filepath: str) -> None:
        with open(filepath, "rb") as file:
            info = pickle.load(file)
        self.array_info = info["array_info"]
        self.meta_info = info["meta_info"]

    def cleanup(self) -> None:
        for shm in self.shm_handles.values():
            try:
                shm.close()
            except FileNotFoundError:
                pass
            if self._owns_segments:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self.shm_handles.clear()


def setup_shared_memory_dataset(
    zarr_path: str,
    info_path: str | None = None,
    keys: list[str] | None = None,
) -> tuple[str, SharedMemoryManager]:
    if info_path is None:
        info_path = os.path.join(
            tempfile.gettempdir(),
            f"arbim_ddp_dataset_{os.getpid()}.pkl",
        )

    manager = SharedMemoryManager()
    manager.create_from_zarr(zarr_path, keys=keys)
    manager.save_info(info_path)
    return info_path, manager


def get_shared_memory_data(
    info_path: str,
) -> tuple[dict, SharedMemoryManager]:
    manager = SharedMemoryManager()
    manager.load_info(info_path)
    shared_data = manager.get_from_shared_memory()
    return shared_data, manager
