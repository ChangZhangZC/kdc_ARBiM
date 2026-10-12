"""Merge ARBiM Offline RL Zarr datasets without RGB decoding or NPY conversion.

Contract: existing jpeg_bytes_v1 Zarr schema, local next_index, episode_ends,
uniform reward semantics, compatible RGB/Depth shapes and dtypes.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import zarr


RGB_NAMES = ("head_rgb", "wrist_left_rgb", "wrist_right_rgb")
DEPTH_NAMES = ("head_depth", "wrist_left_depth", "wrist_right_depth")
CORE_KEYS = {
    "state", "next_state", "action", "next_action", "reward", "return",
    "done", "timeout", "next_index",
}
REWARD_MODES = {"last_gripper_release_sparse", "episode_end_shaped"}
BLOCK_FRAMES = 2048
BLOCK_BYTES = 4 * 1024 * 1024


def _reward_mode(root, path):
    mode = root.attrs.get("reward_mode")
    if not mode:
        manifests = root.attrs.get("source_manifest", [])
        modes = {item.get("reward_mode") for item in manifests}
        if len(modes) == 1:
            mode = next(iter(modes))
    if mode not in REWARD_MODES:
        raise ValueError(
            f"{path}: missing, mixed or unsupported Reward mode {mode!r}; "
            "migrate rewards first, then merge"
        )
    return mode


def _read_source(path):
    root = zarr.open_group(str(path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError(f"{path}: missing data/ or meta/")
    data, meta = root["data"], root["meta"]
    if "episode_ends" not in meta:
        raise ValueError(f"{path}: missing meta/episode_ends")
    if root.attrs.get("rgb_storage") != "jpeg_bytes_v1":
        raise ValueError(f"{path}: expected jpeg_bytes_v1 RGB storage")
    keys = set(data.keys())
    mandatory = CORE_KEYS | {
        f"{name}_jpeg_{suffix}"
        for name in RGB_NAMES for suffix in ("data", "offsets")
    }
    if not mandatory <= keys:
        raise ValueError(f"{path}: missing arrays {sorted(mandatory - keys)}")
    use_depth = any(
        key in keys for name in DEPTH_NAMES for key in (name, f"next_{name}")
    )
    expected = mandatory | (
        {key for name in DEPTH_NAMES for key in (name, f"next_{name}")}
        if use_depth else set()
    )
    if keys != expected:
        raise ValueError(
            f"{path}: unexpected or incomplete arrays {sorted(keys ^ expected)}"
        )

    count = int(data["action"].shape[0])
    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64).reshape(-1)
    if count < 1 or not len(ends) or ends[-1] != count:
        raise ValueError(f"{path}: frames / episode_ends mismatch")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError(f"{path}: invalid episode boundaries")

    for name in keys:
        array = data[name]
        axis = count + 1 if name.endswith("_jpeg_offsets") else count
        if not name.endswith("_jpeg_data") and (
            len(array.shape) == 0 or array.shape[0] != axis
        ):
            raise ValueError(f"{path}: invalid {name} first dimension")

    if data["state"].shape[1:] != data["next_state"].shape[1:]:
        raise ValueError(f"{path}: state/next_state shape mismatch")
    if data["action"].shape[1:] != data["next_action"].shape[1:]:
        raise ValueError(f"{path}: action/next_action shape mismatch")
    for name in ("reward", "return", "done", "timeout"):
        if data[name].shape != (count, 1):
            raise ValueError(f"{path}: {name} must have shape [N,1]")
    if data["next_index"].shape != (count,):
        raise ValueError(f"{path}: next_index must have shape [N]")
    for name in RGB_NAMES:
        offsets = np.asarray(data[f"{name}_jpeg_offsets"][:], dtype=np.int64)
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError(f"{path}: corrupt JPEG offsets for {name}")
        if offsets[-1] != len(data[f"{name}_jpeg_data"]):
            raise ValueError(f"{path}: JPEG byte size mismatch for {name}")
        if data[f"{name}_jpeg_data"].dtype != np.dtype("uint8"):
            raise ValueError(f"{path}: JPEG payload must be uint8")
    rgb_shapes = root.attrs.get("rgb_shapes")
    if not isinstance(rgb_shapes, dict) or set(rgb_shapes) != set(RGB_NAMES):
        raise ValueError(f"{path}: missing RGB camera shape metadata")
    for name in RGB_NAMES:
        if len(rgb_shapes[name]) != 3 or int(rgb_shapes[name][0]) != 3:
            raise ValueError(f"{path}: unexpected RGB shape for {name}")

    done = np.asarray(data["done"][:], dtype=bool).reshape(-1)
    timeout = np.asarray(data["timeout"][:], dtype=bool).reshape(-1)
    expected_done = np.zeros(count, dtype=bool)
    expected_done[ends - 1] = True
    if not np.array_equal(done, expected_done) or not np.array_equal(done, timeout):
        raise ValueError(f"{path}: done/timeout do not match episode_ends")
    next_index = np.asarray(data["next_index"][:], dtype=np.int64)
    expected_next = np.arange(count, dtype=np.int64) + 1
    expected_next[done] -= 1
    if not np.array_equal(next_index, expected_next):
        raise ValueError(f"{path}: next_index crosses episode boundaries")

    reward_mode = _reward_mode(root, path)
    rewards = np.asarray(data["reward"][:], dtype=np.float32).reshape(-1)
    if not np.isfinite(rewards).all():
        raise ValueError(f"{path}: non-finite Reward")
    if reward_mode == "last_gripper_release_sparse":
        if (
            np.count_nonzero(rewards == 1.0) != len(ends)
            or np.count_nonzero(rewards) != len(ends)
            or np.any(rewards[ends - 1] != 0)
        ):
            raise ValueError(f"{path}: Reward V2 must contain one nonterminal +1 per episode")
        # Episode-level check avoids accepting two rewards in one episode and zero in another.
        for start, end in zip(np.r_[0, ends[:-1]], ends):
            if np.count_nonzero(rewards[int(start):int(end)]) != 1:
                raise ValueError(f"{path}: invalid Reward V2 episode [{start}:{end}]")
    for name in (DEPTH_NAMES if use_depth else ()):
        if data[name].shape != data[f"next_{name}"].shape:
            raise ValueError(f"{path}: Depth current/next shape mismatch")

    return {
        "path": str(path), "root": root, "data": data,
        "frames": count, "ends": ends,
        "rgb_shapes": {key: tuple(value) for key, value in rgb_shapes.items()},
        "use_depth": use_depth, "reward_mode": reward_mode, "keys": keys,
    }


def _check_compatible(sources):
    reference = sources[0]
    for source in sources[1:]:
        if source["keys"] != reference["keys"]:
            raise ValueError(f"{source['path']}: mismatched Zarr arrays")
        for field in ("rgb_shapes", "use_depth", "reward_mode"):
            if source[field] != reference[field]:
                raise ValueError(f"{source['path']}: mismatched {field}")
        for key in reference["keys"]:
            left = reference["data"][key]
            right = source["data"][key]
            if left.dtype != right.dtype or left.shape[1:] != right.shape[1:]:
                raise ValueError(
                    f"{source['path']}: incompatible {key} dtype or trailing dimensions"
                )


def _create_arrays(out, template, total_frames, sources):
    data = out.create_group("data")
    meta = out.create_group("meta")
    total_bytes = {
        name: sum(len(source["data"][f"{name}_jpeg_data"]) for source in sources)
        for name in RGB_NAMES
    }
    for name in sorted(template["keys"]):
        original = template["data"][name]
        if name.endswith("_jpeg_data"):
            camera = name.removesuffix("_jpeg_data")
            leading = total_bytes[camera]
        elif name.endswith("_jpeg_offsets"):
            leading = total_frames + 1
        else:
            leading = total_frames
        new_shape = (leading, *original.shape[1:])
        chunks = (min(original.chunks[0], max(1, leading)), *original.chunks[1:])
        data.create_dataset(
            name, shape=new_shape, chunks=chunks,
            dtype=original.dtype, compressor=original.compressor,
            overwrite=True,
        )
    return data, meta


def merge_zarr_datasets(config):
    """Merge compatible Zarr sources into a new dataset, without touching sources."""
    sources_cfg = config.get("zarr_sources") or []
    if not isinstance(sources_cfg, list) or len(sources_cfg) < 2:
        raise ValueError("merge_zarr requires at least two zarr_sources")
    if config.get("overwrite", False):
        raise ValueError("merge_zarr refuses overwrite: choose a new zarr_output_path")
    gamma = float(config.get("return_gamma", 0.99))
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("return_gamma must be in (0,1]")
    destination = config.get("zarr_output_path")
    if not destination:
        raise ValueError("merge_zarr requires zarr_output_path")
    target = Path(destination).expanduser().resolve()
    source_paths = []
    for entry in sources_cfg:
        if isinstance(entry, dict):
            source_path = entry.get("path")
        else:
            source_path = entry
        if not source_path:
            raise ValueError(f"Zarr source missing path: {entry!r}")
        source_paths.append(Path(source_path).expanduser().resolve())
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("duplicate Zarr source paths")
    for path in source_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Zarr source directory not found: {path}")
        if target == path or path in target.parents:
            raise ValueError("output Zarr may not overwrite or nest inside an input")
    if target.exists():
        raise FileExistsError(f"output Zarr already exists: {target}")
    temp = target.with_name(f"{target.name}.merge_tmp")
    if temp.exists():
        raise FileExistsError(f"temporary merge output already exists: {temp}")

    # Validate every input *before* allocating the destination.
    sources = [_read_source(path) for path in source_paths]
    _check_compatible(sources)
    total_frames = sum(source["frames"] for source in sources)
    total_episodes = sum(len(source["ends"]) for source in sources)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = zarr.group(str(temp), overwrite=False)
        data, meta = _create_arrays(out, sources[0], total_frames, sources)
        cursor = 0
        byte_cursor = {name: 0 for name in RGB_NAMES}
        merged_ends = []
        for source in sources:
            local = source["data"]
            count = source["frames"]
            for key in sorted(source["keys"]):
                if key.endswith("_jpeg_data") or key.endswith("_jpeg_offsets") or key == "return":
                    continue
                for start in range(0, count, BLOCK_FRAMES):
                    stop = min(start + BLOCK_FRAMES, count)
                    chunk = local[key][start:stop]
                    if key == "next_index":
                        chunk = np.asarray(chunk, dtype=np.int64) + cursor
                    data[key][cursor + start:cursor + stop] = chunk
            for name in RGB_NAMES:
                src_bytes = local[f"{name}_jpeg_data"]
                length = len(src_bytes)
                offset = byte_cursor[name]
                dst_bytes = data[f"{name}_jpeg_data"]
                for start in range(0, length, BLOCK_BYTES):
                    end = min(start + BLOCK_BYTES, length)
                    dst_bytes[offset + start:offset + end] = src_bytes[start:end]
                src_offsets = local[f"{name}_jpeg_offsets"]
                dst_offsets = data[f"{name}_jpeg_offsets"]
                for start in range(0, count, BLOCK_FRAMES):
                    end = min(start + BLOCK_FRAMES, count)
                    dst_offsets[cursor + start:cursor + end] = (
                        np.asarray(src_offsets[start:end], dtype=np.int64) + offset
                    )
                byte_cursor[name] += length
            merged_ends.extend((source["ends"] + cursor).tolist())
            cursor += count

        for name in RGB_NAMES:
            data[f"{name}_jpeg_offsets"][total_frames] = byte_cursor[name]
        rewards = np.asarray(data["reward"][:], dtype=np.float32).reshape(-1)
        done = np.asarray(data["done"][:], dtype=bool).reshape(-1)
        computed_return = np.empty((total_frames, 1), dtype=np.float32)
        running = 0.0
        for index in range(total_frames - 1, -1, -1):
            running = float(rewards[index]) + gamma * running * float(not done[index])
            computed_return[index, 0] = running
        for start in range(0, total_frames, BLOCK_FRAMES):
            end = min(start + BLOCK_FRAMES, total_frames)
            data["return"][start:end] = computed_return[start:end]
        meta.create_dataset("episode_ends", data=np.asarray(merged_ends, dtype=np.int64), overwrite=True)

        first_attrs = sources[0]["root"].attrs
        out.attrs["rgb_storage"] = "jpeg_bytes_v1"
        out.attrs["rgb_shapes"] = dict(first_attrs["rgb_shapes"])
        out.attrs["rgb_dtype"] = first_attrs.get("rgb_dtype", "uint8")
        out.attrs["reward_mode"] = sources[0]["reward_mode"]
        out.attrs["return_gamma"] = gamma
        manifests = []
        merge_inputs = []
        for source in sources:
            manifest = source["root"].attrs.get("source_manifest", [])
            for entry in manifest:
                manifests.append(dict(entry))
            merge_inputs.append({
                "path": source["path"],
                "frames": source["frames"],
                "episodes": len(source["ends"]),
                "reward_mode": source["reward_mode"],
            })
        out.attrs["source_manifest"] = manifests
        out.attrs["merge_inputs"] = merge_inputs
        if target.exists():
            raise FileExistsError(f"output appeared during merge: {target}")
        os.replace(temp, target)
        return {
            "zarr_path": str(target),
            "num_frames": total_frames,
            "num_episodes": total_episodes,
            "reward_mode": sources[0]["reward_mode"],
            "jpeg_rgb_bytes": sum(byte_cursor.values()),
        }
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
