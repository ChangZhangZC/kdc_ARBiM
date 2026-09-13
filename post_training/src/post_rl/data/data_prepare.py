import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import zarr
from termcolor import cprint
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

if not LEROBOT_SRC.is_dir():
    raise RuntimeError(
        "LeRobot submodule is not initialized. "
        "Run `git submodule update --init --recursive`."
    )

for path in (REPO_ROOT, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_CONFIG_PATH = str(
    REPO_ROOT / "post_training" / "configs" / "data" / "data_prepare.yaml"
)

RGB_FEATURE_TO_BUFFER = {
    "observation.images.head_cam_h": "head_rgb",
    "observation.images.wrist_cam_l": "wrist_left_rgb",
    "observation.images.wrist_cam_r": "wrist_right_rgb",
}
DEPTH_FEATURE_TO_BUFFER = {
    "observation.depth_h": "head_depth",
    "observation.depth_l": "wrist_left_depth",
    "observation.depth_r": "wrist_right_depth",
}

RETURN_GAMMA = 0.99
ZARR_CHUNK_LEAD = 50
JPEG_BYTE_CHUNK = 4 * 1024 * 1024
DEFAULT_JPEG_QUALITY = 95
STREAM_NPY_FORMAT_V1 = "arbim_processed_npy_stream_v1"
STREAM_NPY_FORMAT = "arbim_processed_npy_stream_v2"
RGB_ZARR_STORAGE = "jpeg_bytes_v1"


def load_config(path):
    with open(path, "r") as file:
        cfg = yaml.safe_load(file) or {}
    cfg.setdefault("lambda_penalty", 0.05)
    cfg.setdefault("smooth_penalty", 0.01)
    cfg.setdefault("max_episode_len", 2000)
    cfg.setdefault("use_depth", False)
    cfg.setdefault("overwrite", True)
    cfg.setdefault("teleop_sources", [])
    cfg.setdefault("stream_batch_size", ZARR_CHUNK_LEAD)
    cfg.setdefault("jpeg_quality", DEFAULT_JPEG_QUALITY)
    return cfg


def load_lerobot_dataset(root):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"LeRobot dataset root is not a directory: {root}")
    if not (root / "meta").exists():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {root / 'meta'}")
    return LeRobotDataset(repo_id=root.name, root=root)


def _as_numpy(value, feature_name, frame_index):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if (
        array.size == 0
        or array.dtype == object
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(
            f"frame {frame_index} has an invalid {feature_name!r} value "
            f"(shape={array.shape}, dtype={array.dtype})"
        )
    return array


def _to_uint8_chw(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"RGB image must be 3D, got shape={image.shape}")
    if image.shape[0] in (1, 3, 4):
        chw = image
    elif image.shape[-1] in (1, 3, 4):
        chw = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"cannot infer RGB channel axis from shape={image.shape}")
    if chw.shape[0] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape={chw.shape}")
    if chw.dtype == np.uint8:
        return np.ascontiguousarray(chw)
    if np.issubdtype(chw.dtype, np.floating):
        min_value = float(np.nanmin(chw))
        max_value = float(np.nanmax(chw))
        if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
            chw = chw * 255.0
        elif min_value < -1e-6 or max_value > 255.0 + 1e-6:
            raise ValueError(
                f"RGB float image must be in [0,1] or [0,255], got [{min_value}, {max_value}]"
            )
    chw = np.clip(np.rint(chw), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(chw)


def encode_rgb_jpeg(image, quality=DEFAULT_JPEG_QUALITY):
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg_quality must be in [1,100], got {quality}")
    chw = _to_uint8_chw(image)
    rgb_hwc = np.moveaxis(chw, 0, -1)
    bgr_hwc = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr_hwc,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    return encoded.tobytes()


def decode_rgb_jpeg(payload):
    encoded = np.frombuffer(payload, dtype=np.uint8)
    bgr_hwc = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr_hwc is None:
        raise ValueError("OpenCV JPEG decoding failed")
    rgb_hwc = cv2.cvtColor(bgr_hwc, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.moveaxis(rgb_hwc, -1, 0))


def _episode_id(value):
    return value.item() if hasattr(value, "item") else value


def _group_episode_indices(dataset):
    """Group frame indices by episode without decoding RGB whenever possible."""
    episodes = {}
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        try:
            episode_column = hf_dataset["episode_index"]
            for frame_index, value in enumerate(
                tqdm(episode_column, desc="Grouping episode indices")
            ):
                episodes.setdefault(_episode_id(value), []).append(frame_index)
            if episodes:
                return episodes
        except (KeyError, TypeError, AttributeError):
            episodes.clear()
    for frame_index in tqdm(range(len(dataset)), desc="Grouping episode indices"):
        frame = dataset[frame_index]
        if "episode_index" not in frame:
            raise KeyError(f"frame {frame_index} is missing required key 'episode_index'")
        episodes.setdefault(_episode_id(frame["episode_index"]), []).append(frame_index)
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
    item = {
        "agent_pos": _as_numpy(frame["observation.state"], "observation.state", frame_index),
        "action": _as_numpy(frame["action"], "action", frame_index),
        "rgb": {
            key: _as_numpy(frame[key], key, frame_index)
            for key in RGB_FEATURE_TO_BUFFER
        },
    }
    if use_depth:
        item["depth"] = {
            key: _as_numpy(frame[key], key, frame_index)
            for key in DEPTH_FEATURE_TO_BUFFER
        }
    return item


def _batch_column(dataset, key, indices):
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise AttributeError("LeRobot dataset does not expose hf_dataset")
    try:
        values = hf_dataset[key][indices]
    except (KeyError, TypeError, IndexError):
        values = hf_dataset[indices][key]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return values


def _stack_video_batch(frames, count, feature_name):
    if count == 1 and getattr(frames, "ndim", None) == 3:
        frames = frames.unsqueeze(0)
    if len(frames) != count:
        raise RuntimeError(
            f"batched video decode for {feature_name!r} returned {len(frames)} frames, "
            f"expected {count}"
        )
    return frames


def _load_lerobot_frame_batch(dataset, episode_id, frame_indices, use_depth):
    """Decode one episode slice in batches so RAM is bounded by stream_batch_size."""
    if not frame_indices:
        return []
    count = len(frame_indices)
    state_values = _batch_column(dataset, "observation.state", frame_indices)
    action_values = _batch_column(dataset, "action", frame_indices)
    items = [
        {
            "agent_pos": _as_numpy(state_values[i], "observation.state", frame_indices[i]),
            "action": _as_numpy(action_values[i], "action", frame_indices[i]),
            "rgb": {},
        }
        for i in range(count)
    ]
    if use_depth:
        for item in items:
            item["depth"] = {}

    video_keys = set(getattr(dataset.meta, "video_keys", []))
    requested_video_keys = [
        key
        for key in [
            *RGB_FEATURE_TO_BUFFER.keys(),
            *(DEPTH_FEATURE_TO_BUFFER.keys() if use_depth else []),
        ]
        if key in video_keys
    ]
    if requested_video_keys:
        timestamp_values = _batch_column(dataset, "timestamp", frame_indices)
        timestamps = [
            float(value.item() if hasattr(value, "item") else value)
            for value in timestamp_values
        ]
        video_batch = dataset._query_videos(
            {key: timestamps for key in requested_video_keys}, episode_id
        )
        for key in requested_video_keys:
            frames = _stack_video_batch(video_batch[key], count, key)
            target = "rgb" if key in RGB_FEATURE_TO_BUFFER else "depth"
            for i in range(count):
                items[i][target][key] = _as_numpy(frames[i], key, frame_indices[i])

    for key in [key for key in RGB_FEATURE_TO_BUFFER if key not in video_keys]:
        values = _batch_column(dataset, key, frame_indices)
        if len(values) != count:
            raise RuntimeError(
                f"batched column read for {key!r} returned {len(values)} frames, expected {count}"
            )
        for i, value in enumerate(values):
            items[i]["rgb"][key] = _as_numpy(value, key, frame_indices[i])

    if use_depth:
        for key in [key for key in DEPTH_FEATURE_TO_BUFFER if key not in video_keys]:
            values = _batch_column(dataset, key, frame_indices)
            if len(values) != count:
                raise RuntimeError(
                    f"batched column read for {key!r} returned {len(values)} frames, expected {count}"
                )
            for i, value in enumerate(values):
                items[i]["depth"][key] = _as_numpy(value, key, frame_indices[i])

    for i, item in enumerate(items):
        missing_rgb = [key for key in RGB_FEATURE_TO_BUFFER if key not in item["rgb"]]
        if missing_rgb:
            raise KeyError(f"frame {frame_indices[i]} is missing RGB feature(s): {missing_rgb}")
        if use_depth:
            missing_depth = [key for key in DEPTH_FEATURE_TO_BUFFER if key not in item["depth"]]
            if missing_depth:
                raise KeyError(
                    f"frame {frame_indices[i]} is missing depth feature(s): {missing_depth}"
                )
    return items


def _can_batch_decode(dataset):
    return getattr(dataset, "hf_dataset", None) is not None and hasattr(dataset, "_query_videos")


def _new_processed_chunk(use_depth):
    chunk = {
        "agent_pos": [],
        "action": [],
        "rgb": [],
        "reward": [],
        "done": [],
        "timeout": [],
    }
    if use_depth:
        chunk["depth"] = []
    return chunk


def _validate_processed_chunk(data, use_depth, context, require_terminal=False, rgb_storage=None):
    required = ["agent_pos", "action", "rgb", "reward", "done", "timeout"]
    if use_depth:
        required.append("depth")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{context} is missing required key(s): {missing}")
    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{context} fields must have equal lengths, got {lengths}")
    for index, (done, timeout) in enumerate(zip(data["done"], data["timeout"])):
        if bool(done) != bool(timeout):
            raise ValueError(
                f"{context} frame {index} has done != timeout ({done!r} != {timeout!r})"
            )
    if require_terminal and len(data["timeout"]) and not bool(data["timeout"][-1]):
        raise ValueError(f"{context} must end with timeout=True")
    for index, rgb in enumerate(data["rgb"]):
        if not isinstance(rgb, dict):
            raise TypeError(f"{context} frame {index} rgb must be a dict")
        missing_rgb = [key for key in RGB_FEATURE_TO_BUFFER if key not in rgb]
        if missing_rgb:
            raise KeyError(
                f"{context} frame {index} is missing RGB feature(s): {missing_rgb}"
            )
        if rgb_storage == "jpeg":
            invalid = [
                key for key in RGB_FEATURE_TO_BUFFER
                if not isinstance(rgb[key], (bytes, bytearray, memoryview))
            ]
            if invalid:
                raise TypeError(
                    f"{context} frame {index} JPEG RGB must be bytes for {invalid}"
                )
    if use_depth:
        for index, depth in enumerate(data["depth"]):
            if not isinstance(depth, dict):
                raise TypeError(f"{context} frame {index} depth must be a dict")
            missing_depth = [key for key in DEPTH_FEATURE_TO_BUFFER if key not in depth]
            if missing_depth:
                raise KeyError(
                    f"{context} frame {index} is missing depth feature(s): {missing_depth}"
                )


def _prepare_file(path, overwrite):
    path = Path(path).expanduser().resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists and overwrite=false: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_dir(path, overwrite):
    return _prepare_file(path, overwrite)


def _flush_processed_npy_chunk(handle, chunk, use_depth, chunk_index):
    if not chunk["action"]:
        return chunk_index
    _validate_processed_chunk(
        chunk,
        use_depth,
        f"processed npy chunk {chunk_index}",
        require_terminal=False,
        rgb_storage="jpeg",
    )
    np.save(handle, chunk, allow_pickle=True)
    return chunk_index + 1


def process_raw_teleop_to_npy(config):
    """raw_to_npy: LeRobot -> streamed NPY with JPEG-compressed RGB bytes."""
    lerobot_root = config.get("lerobot_root")
    output_path = config.get("processed_npy_output")
    if not lerobot_root or not output_path:
        raise ValueError("raw_to_npy requires lerobot_root and processed_npy_output")

    use_depth = bool(config.get("use_depth", False))
    max_episode_len = int(config.get("max_episode_len", 2000))
    lambda_penalty = float(config.get("lambda_penalty", 0.05))
    smooth_penalty = float(config.get("smooth_penalty", 0.01))
    batch_size = int(config.get("stream_batch_size", ZARR_CHUNK_LEAD))
    jpeg_quality = int(config.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
    overwrite = bool(config.get("overwrite", True))
    if max_episode_len <= 0:
        raise ValueError("max_episode_len must be positive")
    if batch_size <= 0:
        raise ValueError("stream_batch_size must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")

    dataset = load_lerobot_dataset(lerobot_root)
    if len(dataset) == 0:
        raise RuntimeError("LeRobot dataset contains no frames")
    episodes = _group_episode_indices(dataset)
    if not episodes:
        raise RuntimeError("LeRobot dataset contains no episodes")

    batch_decode = _can_batch_decode(dataset)
    if batch_decode:
        cprint(
            f"raw_to_npy: batched LeRobot decode enabled "
            f"(batch_size={batch_size}, backend={getattr(dataset, 'video_backend', 'unknown')})",
            "cyan",
        )
    else:
        cprint(
            "raw_to_npy: batched decode unavailable; falling back to per-frame __getitem__",
            "yellow",
        )

    first_episode_id, first_episode_frames = next(iter(episodes.items()))
    first_index = first_episode_frames[0]
    if batch_decode:
        first = _load_lerobot_frame_batch(
            dataset, first_episode_id, [first_index], use_depth
        )[0]
    else:
        first = _frame_arrays(dataset[first_index], first_index, use_depth)

    rgb_shapes = {
        key: tuple(_to_uint8_chw(first["rgb"][key]).shape)
        for key in RGB_FEATURE_TO_BUFFER
    }
    manifest = {
        "__format__": STREAM_NPY_FORMAT,
        "num_frames": int(len(dataset)),
        "num_episodes": int(len(episodes)),
        "use_depth": use_depth,
        "state_shape": tuple(first["agent_pos"].shape),
        "action_shape": tuple(first["action"].shape),
        "rgb_storage": "jpeg",
        "jpeg_quality": jpeg_quality,
        "rgb_shapes": rgb_shapes,
        "rgb_dtypes": {key: "uint8" for key in RGB_FEATURE_TO_BUFFER},
        "source_root": str(Path(lerobot_root).expanduser().resolve()),
    }
    if use_depth:
        manifest["depth_shapes"] = {
            key: tuple(first["depth"][key].shape) for key in DEPTH_FEATURE_TO_BUFFER
        }
        manifest["depth_dtypes"] = {
            key: str(first["depth"][key].dtype) for key in DEPTH_FEATURE_TO_BUFFER
        }
    del first

    output_path = _prepare_file(output_path, overwrite)
    chunk = _new_processed_chunk(use_depth)
    chunk_index = 0
    processed = 0
    with open(output_path, "wb") as handle:
        np.save(handle, manifest, allow_pickle=True)
        progress = tqdm(total=len(dataset), desc="Streaming raw_to_npy")
        for episode_id, episode_frames in episodes.items():
            episode_length = len(episode_frames)
            previous_action = None
            for batch_start in range(0, episode_length, batch_size):
                batch_indices = episode_frames[batch_start:batch_start + batch_size]
                if batch_decode:
                    batch_items = _load_lerobot_frame_batch(
                        dataset, episode_id, batch_indices, use_depth
                    )
                else:
                    batch_items = [
                        _frame_arrays(dataset[index], index, use_depth)
                        for index in batch_indices
                    ]
                for batch_offset, (frame_index, item) in enumerate(
                    zip(batch_indices, batch_items, strict=True)
                ):
                    t = batch_start + batch_offset
                    terminal = t == episode_length - 1
                    reward = float(terminal)
                    if terminal:
                        reward -= lambda_penalty * episode_length / max_episode_len
                    if previous_action is not None:
                        reward -= smooth_penalty * np.linalg.norm(
                            item["action"] - previous_action
                        )
                    encoded_rgb = {
                        key: encode_rgb_jpeg(item["rgb"][key], jpeg_quality)
                        for key in RGB_FEATURE_TO_BUFFER
                    }
                    chunk["agent_pos"].append(item["agent_pos"])
                    chunk["action"].append(item["action"])
                    chunk["rgb"].append(encoded_rgb)
                    if use_depth:
                        chunk["depth"].append(item["depth"])
                    chunk["reward"].append(float(reward))
                    chunk["done"].append(bool(terminal))
                    chunk["timeout"].append(bool(terminal))
                    previous_action = np.array(item["action"], copy=True)
                    processed += 1
                    progress.update(1)
                    if len(chunk["action"]) >= batch_size:
                        chunk_index = _flush_processed_npy_chunk(
                            handle, chunk, use_depth, chunk_index
                        )
                        chunk = _new_processed_chunk(use_depth)
                del batch_items
        chunk_index = _flush_processed_npy_chunk(
            handle, chunk, use_depth, chunk_index
        )
        progress.close()

    if processed != len(dataset):
        raise RuntimeError(f"processed {processed} frames, expected {len(dataset)}")
    size_gib = output_path.stat().st_size / (1024 ** 3)
    cprint(
        f"Saved processed NPY: {output_path} "
        f"(frames={processed}, episodes={len(episodes)}, chunks={chunk_index}, "
        f"size={size_gib:.2f} GiB)",
        "green",
    )
    return {
        "npy_path": str(output_path),
        "num_frames": processed,
        "num_episodes": len(episodes),
        "num_chunks": chunk_index,
        "size_bytes": output_path.stat().st_size,
    }


def _load_npy_record(handle):
    try:
        return np.load(handle, allow_pickle=True).item()
    except EOFError:
        return None
    except ValueError:
        if handle.tell() == os.fstat(handle.fileno()).st_size:
            return None
        raise


def _read_source_header(path, use_depth):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"processed npy source does not exist: {path}")
    with open(path, "rb") as handle:
        first = _load_npy_record(handle)
    if first is None:
        raise ValueError(f"processed npy source is empty: {path}")

    stream_format = first.get("__format__")
    if stream_format in (STREAM_NPY_FORMAT, STREAM_NPY_FORMAT_V1):
        if bool(first["use_depth"]) != bool(use_depth):
            raise ValueError(
                f"{path} use_depth={first['use_depth']} does not match config use_depth={use_depth}"
            )
        return {
            "path": str(path),
            "streamed": True,
            "stream_format": stream_format,
            "rgb_storage": first.get(
                "rgb_storage", "jpeg" if stream_format == STREAM_NPY_FORMAT else "dense"
            ),
            "jpeg_quality": int(first.get("jpeg_quality", DEFAULT_JPEG_QUALITY)),
            "num_frames": int(first["num_frames"]),
            "num_episodes": int(first["num_episodes"]),
            "state_shape": tuple(first["state_shape"]),
            "action_shape": tuple(first["action_shape"]),
            "rgb_shapes": {key: tuple(value) for key, value in first["rgb_shapes"].items()},
            "rgb_dtypes": {key: np.dtype(value) for key, value in first["rgb_dtypes"].items()},
            "depth_shapes": {
                key: tuple(value) for key, value in first.get("depth_shapes", {}).items()
            },
            "depth_dtypes": {
                key: np.dtype(value) for key, value in first.get("depth_dtypes", {}).items()
            },
        }

    _validate_processed_chunk(
        first, use_depth, f"legacy processed npy {path}", require_terminal=True
    )
    num_frames = len(first["action"])
    num_episodes = int(np.count_nonzero(np.asarray(first["timeout"], dtype=bool)))
    first_rgb = first["rgb"][0]
    info = {
        "path": str(path),
        "streamed": False,
        "stream_format": "legacy_single_record",
        "rgb_storage": "dense",
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "num_frames": num_frames,
        "num_episodes": num_episodes,
        "state_shape": tuple(np.asarray(first["agent_pos"][0]).shape),
        "action_shape": tuple(np.asarray(first["action"][0]).shape),
        "rgb_shapes": {
            key: tuple(_to_uint8_chw(first_rgb[key]).shape)
            for key in RGB_FEATURE_TO_BUFFER
        },
        "rgb_dtypes": {key: np.dtype("uint8") for key in RGB_FEATURE_TO_BUFFER},
        "depth_shapes": {},
        "depth_dtypes": {},
    }
    if use_depth:
        first_depth = first["depth"][0]
        info["depth_shapes"] = {
            key: tuple(np.asarray(first_depth[key]).shape)
            for key in DEPTH_FEATURE_TO_BUFFER
        }
        info["depth_dtypes"] = {
            key: np.asarray(first_depth[key]).dtype for key in DEPTH_FEATURE_TO_BUFFER
        }
    return info


def _iter_processed_chunks(path, use_depth):
    path = Path(path).expanduser().resolve()
    with open(path, "rb") as handle:
        first = _load_npy_record(handle)
        if first is None:
            return
        stream_format = first.get("__format__")
        if stream_format not in (STREAM_NPY_FORMAT, STREAM_NPY_FORMAT_V1):
            _validate_processed_chunk(
                first, use_depth, f"legacy processed npy {path}", require_terminal=True
            )
            yield first
            return
        rgb_storage = "jpeg" if stream_format == STREAM_NPY_FORMAT else "dense"
        chunk_index = 0
        while True:
            chunk = _load_npy_record(handle)
            if chunk is None:
                break
            _validate_processed_chunk(
                chunk,
                use_depth,
                f"processed npy {path} chunk {chunk_index}",
                rgb_storage=rgb_storage,
            )
            yield chunk
            chunk_index += 1


def _iter_processed_frames(path, use_depth):
    for chunk in _iter_processed_chunks(path, use_depth):
        for index in range(len(chunk["action"])):
            frame = {
                "agent_pos": np.asarray(chunk["agent_pos"][index]),
                "action": np.asarray(chunk["action"][index]),
                "rgb": chunk["rgb"][index],
                "reward": float(chunk["reward"][index]),
                "done": bool(chunk["done"][index]),
                "timeout": bool(chunk["timeout"][index]),
            }
            if use_depth:
                frame["depth"] = chunk["depth"][index]
            yield frame
        del chunk


def _make_compressor():
    try:
        from numcodecs import Blosc
        return Blosc(cname="zstd", clevel=3, shuffle=1)
    except Exception:
        return None


def _create_zarr_array(group, name, shape, dtype, compressor, chunks=None):
    if chunks is None:
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


def _ensure_jpeg_bytes(value, quality):
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return encode_rgb_jpeg(value, quality)


def _append_jpeg_payloads(arrays, buffer_name, payloads, start, cursors):
    byte_array = arrays[f"{buffer_name}_jpeg_data"]
    offsets = arrays[f"{buffer_name}_jpeg_offsets"]
    cursor = int(cursors[buffer_name])
    if int(offsets[start]) != cursor:
        raise RuntimeError(
            f"JPEG offset mismatch for {buffer_name}: offsets[{start}]={int(offsets[start])}, "
            f"cursor={cursor}"
        )
    lengths = np.asarray([len(payload) for payload in payloads], dtype=np.int64)
    total_bytes = int(lengths.sum())
    if total_bytes:
        payload = b"".join(payloads)
        byte_array.resize((cursor + total_bytes,))
        byte_array[cursor:cursor + total_bytes] = np.frombuffer(payload, dtype=np.uint8)
    offsets[start + 1:start + 1 + len(payloads)] = cursor + np.cumsum(lengths)
    cursors[buffer_name] = cursor + total_bytes


def _flush_transition_batch(
    arrays,
    batch,
    start,
    use_depth,
    jpeg_quality,
    jpeg_cursors,
):
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
    terminal = np.asarray([item["done"] for item in batch], dtype=bool)
    arrays["done"][start:end, 0] = terminal
    arrays["timeout"][start:end, 0] = terminal
    arrays["next_index"][start:end] = np.asarray(
        [item["next_index"] for item in batch], dtype=np.int64
    )
    for feature_name, buffer_name in RGB_FEATURE_TO_BUFFER.items():
        payloads = [
            _ensure_jpeg_bytes(item["rgb"][feature_name], jpeg_quality)
            for item in batch
        ]
        _append_jpeg_payloads(arrays, buffer_name, payloads, start, jpeg_cursors)
    if use_depth:
        for feature_name, buffer_name in DEPTH_FEATURE_TO_BUFFER.items():
            _write_block(
                arrays[buffer_name], start, [item["depth"][feature_name] for item in batch]
            )
            _write_block(
                arrays[f"next_{buffer_name}"],
                start,
                [item["next_depth"][feature_name] for item in batch],
            )
    batch.clear()
    return end


def compute_return(reward, not_done, gamma=RETURN_GAMMA):
    size = len(reward)
    returns = np.zeros((size, 1), dtype=np.float32)
    running = 0.0
    for index in tqdm(
        reversed(range(size)), total=size, desc="Computing returns"
    ):
        returns[index] = reward[index] + gamma * running * not_done[index]
        running = returns[index]
    return returns


def _validate_source_shapes(infos, use_depth):
    if not infos:
        raise ValueError("build_db requires at least one teleop_source")
    reference = infos[0]
    keys = ["state_shape", "action_shape", "rgb_shapes"]
    if use_depth:
        keys.append("depth_shapes")
    for info in infos[1:]:
        mismatches = {
            key: (reference[key], info[key])
            for key in keys
            if reference[key] != info[key]
        }
        if mismatches:
            raise ValueError(
                f"processed npy source shape mismatch for {info['path']}: {mismatches}"
            )


def _build_zarr_arrays(output_path, total_frames, info, use_depth, overwrite):
    output_path = _prepare_dir(output_path, overwrite)
    root = zarr.group(str(output_path))
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = _make_compressor()
    state_shape = info["state_shape"]
    action_shape = info["action_shape"]
    arrays = {
        "state": _create_zarr_array(
            data, "state", (total_frames, *state_shape), "float32", compressor
        ),
        "next_state": _create_zarr_array(
            data, "next_state", (total_frames, *state_shape), "float32", compressor
        ),
        "action": _create_zarr_array(
            data, "action", (total_frames, *action_shape), "float32", compressor
        ),
        "next_action": _create_zarr_array(
            data, "next_action", (total_frames, *action_shape), "float32", compressor
        ),
        "reward": _create_zarr_array(
            data, "reward", (total_frames, 1), "float32", compressor
        ),
        "return": _create_zarr_array(
            data, "return", (total_frames, 1), "float32", compressor
        ),
        "done": _create_zarr_array(
            data, "done", (total_frames, 1), "bool", compressor
        ),
        "timeout": _create_zarr_array(
            data, "timeout", (total_frames, 1), "bool", compressor
        ),
        "next_index": _create_zarr_array(
            data, "next_index", (total_frames,), "int64", compressor
        ),
    }
    for buffer_name in RGB_FEATURE_TO_BUFFER.values():
        arrays[f"{buffer_name}_jpeg_data"] = _create_zarr_array(
            data,
            f"{buffer_name}_jpeg_data",
            (0,),
            "uint8",
            None,
            chunks=(JPEG_BYTE_CHUNK,),
        )
        arrays[f"{buffer_name}_jpeg_offsets"] = _create_zarr_array(
            data,
            f"{buffer_name}_jpeg_offsets",
            (total_frames + 1,),
            "int64",
            compressor,
        )
        arrays[f"{buffer_name}_jpeg_offsets"][0] = 0
    if use_depth:
        for feature_name, buffer_name in DEPTH_FEATURE_TO_BUFFER.items():
            shape = info["depth_shapes"][feature_name]
            dtype = info["depth_dtypes"][feature_name]
            arrays[buffer_name] = _create_zarr_array(
                data, buffer_name, (total_frames, *shape), dtype, compressor
            )
            arrays[f"next_{buffer_name}"] = _create_zarr_array(
                data, f"next_{buffer_name}", (total_frames, *shape), dtype, compressor
            )
    root.attrs["rgb_storage"] = RGB_ZARR_STORAGE
    root.attrs["rgb_shapes"] = {
        buffer_name: list(info["rgb_shapes"][feature_name])
        for feature_name, buffer_name in RGB_FEATURE_TO_BUFFER.items()
    }
    root.attrs["rgb_dtype"] = "uint8"
    return output_path, root, meta, arrays


def source_id(source):
    return source.get("name") or source["path"]


def run_build_db(config):
    """build_db: merge processed NPY source(s) into compressed Offline RL Zarr."""
    output_path = config.get("zarr_output_path")
    if not output_path:
        raise ValueError("build_db requires zarr_output_path")
    sources = config.get("teleop_sources", [])
    if not sources:
        raise ValueError("build_db requires at least one teleop_source")

    use_depth = bool(config.get("use_depth", False))
    overwrite = bool(config.get("overwrite", True))
    batch_size = int(config.get("stream_batch_size", ZARR_CHUNK_LEAD))
    jpeg_quality = int(config.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
    if batch_size <= 0:
        raise ValueError("stream_batch_size must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")

    infos = []
    for source in sources:
        if "path" not in source:
            raise ValueError(f"teleop source missing path: {source}")
        info = _read_source_header(source["path"], use_depth)
        info["name"] = source_id(source)
        infos.append(info)
    _validate_source_shapes(infos, use_depth)
    total_frames = sum(info["num_frames"] for info in infos)
    total_episodes = sum(info["num_episodes"] for info in infos)
    if total_frames <= 0:
        raise RuntimeError("processed npy sources contain no frames")

    output_path, root, meta, arrays = _build_zarr_arrays(
        output_path, total_frames, infos[0], use_depth, overwrite
    )
    root.attrs["source_manifest"] = [
        {
            "kind": "teleop_npy",
            "name": info["name"],
            "path": info["path"],
            "format": info["stream_format"],
            "rgb_storage": info["rgb_storage"],
        }
        for info in infos
    ]

    rewards = np.empty((total_frames, 1), dtype=np.float32)
    not_done = np.empty((total_frames, 1), dtype=np.float32)
    episode_ends = []
    transition_batch = []
    write_index = 0
    transition_count = 0
    jpeg_cursors = {name: 0 for name in RGB_FEATURE_TO_BUFFER.values()}
    progress = tqdm(total=total_frames, desc="Streaming build_db")

    for info in infos:
        pending = None
        source_episode_count = 0
        source_frame_count = 0
        for current in _iter_processed_frames(info["path"], use_depth):
            source_frame_count += 1
            if pending is None:
                pending = current
            else:
                if pending["done"]:
                    raise ValueError(
                        f"{info['path']} contains data after terminal without resetting the transition stream"
                    )
                current_index = transition_count
                transition_batch.append(
                    {
                        "state": pending["agent_pos"],
                        "next_state": current["agent_pos"],
                        "action": pending["action"],
                        "next_action": current["action"],
                        "rgb": pending["rgb"],
                        "depth": pending.get("depth"),
                        "next_depth": current.get("depth"),
                        "reward": pending["reward"],
                        "done": False,
                        "next_index": current_index + 1,
                    }
                )
                rewards[current_index, 0] = pending["reward"]
                not_done[current_index, 0] = 1.0
                transition_count += 1
                progress.update(1)
                pending = current

            if pending is not None and pending["done"]:
                current_index = transition_count
                transition_batch.append(
                    {
                        "state": pending["agent_pos"],
                        "next_state": pending["agent_pos"],
                        "action": pending["action"],
                        "next_action": pending["action"],
                        "rgb": pending["rgb"],
                        "depth": pending.get("depth"),
                        "next_depth": pending.get("depth"),
                        "reward": pending["reward"],
                        "done": True,
                        "next_index": current_index,
                    }
                )
                rewards[current_index, 0] = pending["reward"]
                not_done[current_index, 0] = 0.0
                transition_count += 1
                progress.update(1)
                episode_ends.append(transition_count)
                source_episode_count += 1
                pending = None

            if len(transition_batch) >= batch_size:
                write_index = _flush_transition_batch(
                    arrays,
                    transition_batch,
                    write_index,
                    use_depth,
                    jpeg_quality,
                    jpeg_cursors,
                )

        if pending is not None:
            raise ValueError(f"{info['path']} ended with an unterminated episode")
        if source_frame_count != info["num_frames"]:
            raise RuntimeError(
                f"{info['path']} yielded {source_frame_count} frames, expected {info['num_frames']}"
            )
        if source_episode_count != info["num_episodes"]:
            raise RuntimeError(
                f"{info['path']} yielded {source_episode_count} episodes, expected {info['num_episodes']}"
            )

    progress.close()
    write_index = _flush_transition_batch(
        arrays,
        transition_batch,
        write_index,
        use_depth,
        jpeg_quality,
        jpeg_cursors,
    )
    if transition_count != total_frames or write_index != total_frames:
        raise RuntimeError(
            f"processed={transition_count}, wrote={write_index}, expected={total_frames}"
        )
    if len(episode_ends) != total_episodes:
        raise RuntimeError(
            f"episodes={len(episode_ends)}, expected={total_episodes}"
        )

    returns = compute_return(rewards, not_done, gamma=RETURN_GAMMA).astype(np.float32)
    arrays["return"][:] = returns
    meta.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        overwrite=True,
    )
    total_jpeg_bytes = sum(jpeg_cursors.values())
    cprint(
        f"Saved Offline RL Zarr: {output_path} "
        f"(frames={total_frames}, episodes={len(episode_ends)}, "
        f"jpeg_rgb={total_jpeg_bytes / (1024 ** 3):.2f} GiB)",
        "green",
    )
    return {
        "zarr_path": str(output_path),
        "num_frames": total_frames,
        "num_episodes": len(episode_ends),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
        "jpeg_rgb_bytes": total_jpeg_bytes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    mode = str(config.get("mode", "")).strip()
    if mode == "raw_to_npy":
        process_raw_teleop_to_npy(config)
    elif mode == "build_db":
        run_build_db(config)
    else:
        raise ValueError(
            f"unknown mode: {mode!r}. expected 'raw_to_npy' or 'build_db'."
        )


if __name__ == "__main__":
    main()
