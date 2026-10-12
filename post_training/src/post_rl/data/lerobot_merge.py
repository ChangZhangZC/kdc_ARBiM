"""Merge LeRobot datasets for ACT imitation learning, preserving episode boundaries.

Copies selected common training features (16D state/action, 3 RGB, optional 3 Depth)
using the installed LeRobot writer rather than concatenating video/Parquet files.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
for folder in (REPO_ROOT, REPO_ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import lerobot_patches.custom_patches  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from post_rl.data import data_prepare as dp


RGB_KEYS = tuple(dp.RGB_FEATURE_TO_BUFFER)
DEPTH_KEYS = tuple(dp.DEPTH_FEATURE_TO_BUFFER)
MERGE_METADATA = "arbim_merge_provenance.json"


def _schema(dataset, use_depth):
    keys = ("observation.state", "action", *RGB_KEYS)
    if use_depth:
        keys += DEPTH_KEYS
    features = {}
    for key in keys:
        if key not in dataset.meta.features:
            raise ValueError(f"{dataset.repo_id}: missing LeRobot feature {key}")
        features[key] = copy.deepcopy(dataset.meta.features[key])
    if tuple(features["observation.state"]["shape"]) != (16,):
        raise ValueError("Expected 16D observation.state")
    if tuple(features["action"]["shape"]) != (16,):
        raise ValueError("Expected 16D action")
    cameras = (*RGB_KEYS, *(DEPTH_KEYS if use_depth else ()))
    modes = {features[key]["dtype"] for key in cameras}
    if modes not in ({"video"}, {"image"}):
        raise ValueError("All camera features must share image or video dtype")
    for key in cameras:
        shape = tuple(features[key]["shape"])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"Expected CHW 3-channel camera feature {key}")
    metadata = dataset.meta.info
    return {
        "features": features,
        "fps": float(metadata["fps"]),
        "robot_type": metadata.get("robot_type", "kuavo4pro"),
        "use_videos": modes == {"video"},
    }


def _check_compatible(sources, use_depth):
    first = sources[0]["contract"]
    for source in sources[1:]:
        current = source["contract"]
        for name in ("fps", "robot_type", "use_videos"):
            if current[name] != first[name]:
                raise ValueError(f"{source['path']}: incompatible LeRobot {name}")
        for key, template in first["features"].items():
            other = current["features"][key]
            if (
                tuple(other["shape"]) != tuple(template["shape"])
                or other["dtype"] != template["dtype"]
                or other.get("names") != template.get("names")
            ):
                raise ValueError(
                    f"{source['path']}: incompatible LeRobot feature {key}"
                )


def _episode_task(dataset, first_index, fallback):
    """Use source task text when available; otherwise require a caller fallback."""
    hf = getattr(dataset, "hf_dataset", None)
    tasks = getattr(dataset.meta, "tasks", None)
    if hf is not None and tasks:
        try:
            index_value = hf[int(first_index)]["task_index"]
            index_value = int(index_value.item() if hasattr(index_value, "item") else index_value)
            if isinstance(tasks, dict):
                value = tasks.get(index_value, tasks.get(str(index_value)))
                if isinstance(value, str) and value:
                    return value
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    if not fallback:
        raise ValueError("Cannot determine source LeRobot task; set --fallback-task")
    return fallback


def _to_hwc_rgb(value, key, reference_shape):
    chw = dp._to_uint8_chw(value)
    if tuple(chw.shape) != tuple(reference_shape):
        raise ValueError(
            f"Decoded LeRobot {key} shape {chw.shape} != {reference_shape}"
        )
    return np.ascontiguousarray(np.moveaxis(chw, 0, -1))


def _to_hwc_depth(value, key, reference_shape):
    depth = np.asarray(value)
    if depth.ndim != 3 or depth.shape[0] != 3:
        raise ValueError(f"{key}: expected 3-channel CHW Depth")
    if tuple(depth.shape) != tuple(reference_shape):
        raise ValueError(f"{key}: incompatible Depth shape")
    if np.issubdtype(depth.dtype, np.floating):
        if not np.isfinite(depth).all() or depth.min() < -1e-6 or depth.max() > 1 + 1e-6:
            raise ValueError(f"{key}: Depth float must be in [0,1]")
        # Match staging Depth's lossless per-frame uint8 quantization.
        depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    elif depth.dtype != np.uint8:
        raise ValueError(f"{key}: unsupported Depth dtype {depth.dtype}")
    return np.ascontiguousarray(np.moveaxis(depth, 0, -1))


def merge_lerobot_datasets(
    source_roots, output_root, repo_id, *,
    use_depth=False, batch_size=25, fallback_task="ARBiM",
):
    if not isinstance(source_roots, (list, tuple)) or len(source_roots) < 2:
        raise ValueError("merge_lerobot requires at least two input datasets")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    paths = [Path(root).expanduser().resolve() for root in source_roots]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate LeRobot input datasets")
    target = Path(output_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"LeRobot output already exists: {target}")
    if any(target == path or path in target.parents for path in paths):
        raise ValueError("output LeRobot cannot overwrite or nest inside an input")
    sources = []
    for path in paths:
        if not (path / "meta").is_dir():
            raise FileNotFoundError(f"LeRobot metadata missing: {path}")
        dataset = dp.load_lerobot_dataset(path)
        episodes = dp._group_episode_indices(dataset)
        if not episodes:
            raise ValueError(f"{path}: no episodes")
        sources.append({
            "path": str(path),
            "dataset": dataset,
            "episodes": episodes,
            "contract": _schema(dataset, use_depth),
        })
    _check_compatible(sources, use_depth)
    contract = sources[0]["contract"]
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(target),
        robot_type=contract["robot_type"],
        fps=contract["fps"],
        features=contract["features"],
        use_videos=contract["use_videos"],
    )
    output_episode_id = 0
    total_frames = 0
    records = []
    for source in sources:
        dataset = source["dataset"]
        use_batch = dp._can_batch_decode(dataset)
        input_metadata = Path(source["path"]) / "meta" / "arbim_staging_provenance.json"
        recorded = None
        if input_metadata.is_file():
            recorded = json.loads(input_metadata.read_text(encoding="utf-8"))
            if len(recorded.get("episodes", [])) != len(source["episodes"]):
                raise ValueError(f"{source['path']}: Staging metadata episode count mismatch")
        for episode_position, (episode_id, indices) in enumerate(source["episodes"].items()):
            task = _episode_task(dataset, indices[0], fallback_task)
            for start in range(0, len(indices), batch_size):
                frame_indices = indices[start:start + batch_size]
                if use_batch:
                    items = dp._load_lerobot_frame_batch(
                        dataset, episode_id, frame_indices, use_depth
                    )
                else:
                    items = [
                        dp._frame_arrays(dataset[index], index, use_depth)
                        for index in frame_indices
                    ]
                for item in items:
                    frame = {
                        "observation.state": torch.tensor(
                            np.asarray(item["agent_pos"], dtype=np.float32).copy()
                        ),
                        "action": torch.tensor(
                            np.asarray(item["action"], dtype=np.float32).copy()
                        ),
                    }
                    for key in RGB_KEYS:
                        frame[key] = _to_hwc_rgb(
                            item["rgb"][key], key,
                            contract["features"][key]["shape"],
                        )
                    if use_depth:
                        for key in DEPTH_KEYS:
                            frame[key] = _to_hwc_depth(
                                item["depth"][key], key,
                                contract["features"][key]["shape"],
                            )
                    destination.add_frame(frame, task=task)
                del items
            destination.save_episode()
            total_frames += len(indices)
            record = {
                "output_episode_index": output_episode_id,
                "source_root": source["path"],
                "source_episode_index": int(episode_id),
                "frames": len(indices),
                "task": task,
            }
            if recorded is not None:
                record["staging_reward"] = recorded["episodes"][episode_position]
            records.append(record)
            output_episode_id += 1
            # Keep parity with the existing Rosbag -> LeRobot writer.
            destination.hf_dataset = destination.create_hf_dataset()

    metadata = {
        "format": "arbim_lerobot_merge_provenance_v1",
        "use_depth": bool(use_depth),
        "sources": [source["path"] for source in sources],
        "episodes": records,
        "total_frames": total_frames,
        "total_episodes": output_episode_id,
    }
    provenance = target / "meta" / MERGE_METADATA
    provenance.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "lerobot_root": str(target),
        "num_frames": total_frames,
        "num_episodes": output_episode_id,
        "provenance": str(provenance),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Two or more input LeRobot directories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fallback-task", default="ARBiM")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--use-depth", action="store_true")
    args = parser.parse_args()
    print(merge_lerobot_datasets(
        args.sources, args.output_root, args.repo_id,
        use_depth=args.use_depth,
        batch_size=args.batch_size,
        fallback_task=args.fallback_task,
    ))


if __name__ == "__main__":
    main()
