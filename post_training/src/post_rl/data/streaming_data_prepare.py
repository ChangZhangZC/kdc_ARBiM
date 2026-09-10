from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from termcolor import cprint
from tqdm import tqdm

from .data_prepare import (
    DEPTH_FEATURE_TO_BUFFER,
    RETURN_GAMMA,
    RGB_FEATURE_TO_BUFFER,
    ZARR_CHUNK_LEAD,
    compute_return,
    load_lerobot_dataset,
    safe_prepare_output_dir,
)


def _as_numpy(value, feature_name, frame_index):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 0 or array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"frame {frame_index} has an invalid {feature_name!r} value "
            f"(shape={array.shape}, dtype={array.dtype})"
        )
    return array


def _episode_id(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def _group_episode_indices(dataset):
    """Return episode -> frame-index mapping without decoding RGB when possible."""
    episodes = {}
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        try:
            episode_column = hf_dataset["episode_index"]
            for frame_index, value in enumerate(
                tqdm(episode_column, desc="Grouping frames by episode")
            ):
                episodes.setdefault(_episode_id(value), []).append(frame_index)
            if episodes:
                return episodes
        except (KeyError, TypeError, AttributeError):
            episodes.clear()

    for frame_index in tqdm(range(len(dataset)), desc="Grouping frames by episode"):
        frame = dataset[frame_index]
        if "episode_index" not in frame:
            raise KeyError(f"frame {frame_index} is missing required key 'episode_index'")
        episodes.setdefault(_episode_id(frame["episode_index"]), []).append(frame_index)
        del frame
    return episodes


def _validate_frame(frame, frame_index, use_depth):
    required = ["observation.state", "action", *RGB_FEATURE_TO_BUFFER.keys()]
    if use_depth:
        required.extend(DEPTH_FEATURE_TO_BUFFER.keys())
    missing = [key for key in required if key not in frame]
    if missing:
        raise KeyError(f"frame {frame_index} is missing required key(s): {missing}")


def _frame_arrays(frame, frame_index, use_depth):
    _validate_frame(frame, frame_index, use_depth)
    arrays = {
        "state": _as_numpy(frame["observation.state"], "observation.state", frame_index),
        "action": _as_numpy(frame["action"], "action", frame_index),
        "rgb": {
            feature_name: _as_numpy(frame[feature_name], feature_name, frame_index)
            for feature_name in RGB_FEATURE_TO_BUFFER
        },
    }
    if use_depth:
        arrays["depth"] = {
            feature_name: _as_numpy(frame[feature_name], feature_name, frame_index)
            for feature_name in DEPTH_FEATURE_TO_BUFFER
        }
    return arrays


def _make_compressor():
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=1)
    except Exception:
        return None


def _create_dataset(group, name, shape, dtype, compressor):
    chunks = (min(ZARR_CHUNK_LEAD, shape[0]), *shape[1:])
    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "overwrite": True,
    }
    if compressor is not None:
        kwargs["compressor"] = compressor
    return group.create_dataset(name, **kwargs)


def _write_block(array, start, values, dtype=None):
    block = np.stack(values, axis=0)
    if dtype is not None:
        block = block.astype(dtype, copy=False)
    array[start:start + len(values)] = block


def _flush_transition_batch(arrays, batch, start, use_depth):
    """Write one contiguous batch so each Zarr chunk is compressed only once."""
    if not batch:
        return start

    end = start + len(batch)
    _write_block(arrays["state"], start, [item["state"] for item in batch], np.float32)
    _write_block(
        arrays["next_state"], start, [item["next_state"] for item in batch], np.float32
    )
    _write_block(arrays["action"], start, [item["action"] for item in batch], np.float32)
    _write_block(
        arrays["next_action"], start, [item["next_action"] for item in batch], np.float32
    )

    arrays["reward"][start:end, 0] = np.asarray(
        [item["reward"] for item in batch], dtype=np.float32
    )
    terminal = np.asarray([item["terminal"] for item in batch], dtype=bool)
    arrays["done"][start:end, 0] = terminal
    arrays["timeout"][start:end, 0] = terminal

    for feature_name, buffer_name in RGB_FEATURE_TO_BUFFER.items():
        _write_block(
            arrays[buffer_name],
            start,
            [item["rgb"][feature_name] for item in batch],
        )
        _write_block(
            arrays[f"next_{buffer_name}"],
            start,
            [item["next_rgb"][feature_name] for item in batch],
        )

    if use_depth:
        for feature_name, buffer_name in DEPTH_FEATURE_TO_BUFFER.items():
            _write_block(
                arrays[buffer_name],
                start,
                [item["depth"][feature_name] for item in batch],
            )
            _write_block(
                arrays[f"next_{buffer_name}"],
                start,
                [item["next_depth"][feature_name] for item in batch],
            )

    batch.clear()
    return end


def build_zarr_from_lerobot(config):
    """Convert LeRobot directly to the Offline-RL Zarr contract with bounded RAM.

    Frames are decoded sequentially, buffered for one small contiguous block, then
    written as a Zarr slice. The default block size equals ZARR_CHUNK_LEAD so a
    compressed Zarr chunk is written once instead of being rewritten per frame.
    """
    lerobot_root = config.get("lerobot_root")
    output_path = config.get("zarr_output_path")
    if not lerobot_root or not output_path:
        raise ValueError("lerobot_to_zarr requires lerobot_root and zarr_output_path")

    use_depth = bool(config.get("use_depth", False))
    overwrite = bool(config.get("overwrite", True))
    max_episode_len = int(config.get("max_episode_len", 2000))
    lambda_penalty = float(config.get("lambda_penalty", 0.05))
    smooth_penalty = float(config.get("smooth_penalty", 0.01))
    batch_size = int(config.get("stream_batch_size", ZARR_CHUNK_LEAD))
    if max_episode_len <= 0:
        raise ValueError("max_episode_len must be positive")
    if batch_size <= 0:
        raise ValueError("stream_batch_size must be positive")

    dataset = load_lerobot_dataset(lerobot_root)
    n_frames = len(dataset)
    if n_frames == 0:
        raise RuntimeError("LeRobot dataset contains no frames")

    episodes = _group_episode_indices(dataset)
    if not episodes:
        raise RuntimeError("LeRobot dataset contains no episodes")

    ordered_source_indices = [
        frame_index
        for episode_frames in episodes.values()
        for frame_index in episode_frames
    ]
    if len(ordered_source_indices) != n_frames:
        raise RuntimeError(
            f"episode grouping covers {len(ordered_source_indices)} frames, expected {n_frames}"
        )

    first_index = ordered_source_indices[0]
    first_frame = dataset[first_index]
    first = _frame_arrays(first_frame, first_index, use_depth)
    del first_frame

    safe_prepare_output_dir(output_path, overwrite)
    root = zarr.group(output_path)
    data = root.create_group("data")
    meta = root.create_group("meta")
    root.attrs["source_manifest"] = [
        {
            "kind": "lerobot",
            "name": Path(lerobot_root).expanduser().resolve().name,
            "path": str(Path(lerobot_root).expanduser().resolve()),
        }
    ]

    compressor = _make_compressor()
    state_shape = first["state"].shape
    action_shape = first["action"].shape
    arrays = {
        "state": _create_dataset(data, "state", (n_frames, *state_shape), "float32", compressor),
        "next_state": _create_dataset(
            data, "next_state", (n_frames, *state_shape), "float32", compressor
        ),
        "action": _create_dataset(
            data, "action", (n_frames, *action_shape), "float32", compressor
        ),
        "next_action": _create_dataset(
            data, "next_action", (n_frames, *action_shape), "float32", compressor
        ),
        "reward": _create_dataset(data, "reward", (n_frames, 1), "float32", compressor),
        "return": _create_dataset(data, "return", (n_frames, 1), "float32", compressor),
        "done": _create_dataset(data, "done", (n_frames, 1), "bool", compressor),
        "timeout": _create_dataset(data, "timeout", (n_frames, 1), "bool", compressor),
    }

    for feature_name, buffer_name in RGB_FEATURE_TO_BUFFER.items():
        image = first["rgb"][feature_name]
        arrays[buffer_name] = _create_dataset(
            data, buffer_name, (n_frames, *image.shape), image.dtype, compressor
        )
        arrays[f"next_{buffer_name}"] = _create_dataset(
            data, f"next_{buffer_name}", (n_frames, *image.shape), image.dtype, compressor
        )

    if use_depth:
        for feature_name, buffer_name in DEPTH_FEATURE_TO_BUFFER.items():
            depth = first["depth"][feature_name]
            arrays[buffer_name] = _create_dataset(
                data, buffer_name, (n_frames, *depth.shape), depth.dtype, compressor
            )
            arrays[f"next_{buffer_name}"] = _create_dataset(
                data, f"next_{buffer_name}", (n_frames, *depth.shape), depth.dtype, compressor
            )
    del first

    rewards = np.empty((n_frames, 1), dtype=np.float32)
    not_done = np.empty((n_frames, 1), dtype=np.float32)
    episode_ends = []
    transition_batch = []
    write_index = 0
    transition_count = 0

    cprint(
        f"Streaming {n_frames} frames from {len(episodes)} episodes directly to Zarr "
        f"in batches of {batch_size}",
        "cyan",
    )

    progress = tqdm(total=n_frames, desc="Streaming frames to Zarr")
    for episode_frames in episodes.values():
        episode_length = len(episode_frames)
        if episode_length == 0:
            continue

        current_source_index = episode_frames[0]
        current_frame = dataset[current_source_index]
        current = _frame_arrays(current_frame, current_source_index, use_depth)
        del current_frame
        previous_action = None

        for t, source_index in enumerate(episode_frames):
            if source_index != current_source_index:
                raise RuntimeError(
                    f"stream cursor mismatch: expected source frame {current_source_index}, "
                    f"got {source_index}"
                )

            terminal = t == episode_length - 1
            if terminal:
                next_source_index = source_index
                next_item = current
            else:
                next_source_index = episode_frames[t + 1]
                next_frame = dataset[next_source_index]
                next_item = _frame_arrays(next_frame, next_source_index, use_depth)
                del next_frame

            if current["state"].shape != state_shape or current["action"].shape != action_shape:
                raise ValueError(
                    f"frame {source_index} state/action shape changed: "
                    f"state={current['state'].shape}, action={current['action'].shape}"
                )

            reward = float(terminal)
            if terminal:
                reward -= lambda_penalty * episode_length / max_episode_len
            if previous_action is not None:
                reward -= smooth_penalty * np.linalg.norm(
                    current["action"] - previous_action
                )

            item = {
                "state": current["state"],
                "next_state": next_item["state"],
                "action": current["action"],
                "next_action": next_item["action"],
                "rgb": current["rgb"],
                "next_rgb": next_item["rgb"],
                "reward": reward,
                "terminal": terminal,
            }
            if use_depth:
                item["depth"] = current["depth"]
                item["next_depth"] = next_item["depth"]
            transition_batch.append(item)

            rewards[transition_count, 0] = reward
            not_done[transition_count, 0] = 0.0 if terminal else 1.0
            transition_count += 1
            progress.update(1)

            previous_action = np.array(current["action"], copy=True)
            if not terminal:
                current = next_item
                current_source_index = next_source_index

            if len(transition_batch) >= batch_size:
                write_index = _flush_transition_batch(
                    arrays, transition_batch, write_index, use_depth
                )

        episode_ends.append(transition_count)
        del current

    progress.close()
    write_index = _flush_transition_batch(arrays, transition_batch, write_index, use_depth)

    if transition_count != n_frames or write_index != n_frames:
        raise RuntimeError(
            f"processed={transition_count}, wrote={write_index}, expected={n_frames}"
        )

    returns = compute_return(rewards, not_done, gamma=RETURN_GAMMA).astype(np.float32)
    arrays["return"][:] = returns
    meta.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        overwrite=True,
    )

    cprint(f"Saved streaming Zarr to {output_path}", "green")
    return {
        "num_frames": n_frames,
        "num_episodes": len(episode_ends),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
        "source_indices": np.asarray(ordered_source_indices, dtype=np.int64),
        "zarr_path": str(output_path),
    }
