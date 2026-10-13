"""Fast lossless merge of full-schema LeRobot datasets for ACT / IL.

Uses LeRobot's official merge_datasets() (Parquet metadata rewrite + video
copy/remux, never framewise decode/re-encode) and preserves ARBiM Reward V2
provenance for later Offline RL conversion. Physical merging is intentionally
strict: incompatible full schemas are rejected, not silently projected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for folder in (REPO_ROOT, REPO_ROOT / "post_training" / "src", REPO_ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import lerobot_patches.custom_patches  # noqa: F401
from post_rl.data.lerobot_v042_fast_merge import merge_v042_datasets as merge_datasets

from post_rl.data import data_prepare as dp


RGB_KEYS = tuple(dp.RGB_FEATURE_TO_BUFFER)
DEPTH_KEYS = tuple(dp.DEPTH_FEATURE_TO_BUFFER)
MERGE_METADATA = "arbim_merge_provenance.json"
STAGING_FORMAT = "arbim_lerobot_staging_provenance_v1"
MERGED_FORMAT = "arbim_lerobot_merge_provenance_v1"


def _validate_dataset(dataset, path, use_depth):
    features = dataset.meta.features
    for key in ("observation.state", "action", *RGB_KEYS, *(DEPTH_KEYS if use_depth else ())):
        if key not in features:
            raise ValueError(f"{path}: missing required LeRobot feature {key}")
    if tuple(features["observation.state"]["shape"]) != (16,):
        raise ValueError(f"{path}: observation.state must be 16D")
    if tuple(features["action"]["shape"]) != (16,):
        raise ValueError(f"{path}: action must be 16D")
    camera_keys = (*RGB_KEYS, *(DEPTH_KEYS if use_depth else ()))
    modes = {features[key]["dtype"] for key in camera_keys}
    if modes not in ({"video"}, {"image"}):
        raise ValueError(f"{path}: camera storage modes must match")
    for key in camera_keys:
        shape = tuple(features[key]["shape"])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"{path}: {key} must have 3-channel CHW shape")
    fps = float(dataset.meta.info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid FPS {fps}")
    return {
        "features": features,
        "fps": fps,
        "robot_type": dataset.meta.info.get("robot_type"),
    }


def _read_reward_rows(path, episode_items):
    """Read direct Staging or recursively merged Reward V2 annotations."""
    meta = path / "meta"
    staging_path = meta / "arbim_staging_provenance.json"
    merged_path = meta / MERGE_METADATA
    rows = None
    if staging_path.is_file():
        saved = json.loads(staging_path.read_text(encoding="utf-8"))
        if saved.get("format") != STAGING_FORMAT:
            raise ValueError(f"{path}: unrecognized Staging provenance")
        rows = saved.get("episodes")
        if not isinstance(rows, list):
            raise ValueError(f"{path}: invalid Staging episode records")
        direct = True
    elif merged_path.is_file():
        saved = json.loads(merged_path.read_text(encoding="utf-8"))
        if saved.get("format") != MERGED_FORMAT:
            raise ValueError(f"{path}: unrecognized merged provenance")
        rows = saved.get("episodes")
        if not isinstance(rows, list):
            raise ValueError(f"{path}: invalid merged episode records")
        direct = False
    if rows is None:
        return None
    if len(rows) != len(episode_items):
        raise ValueError(f"{path}: Reward provenance episode count mismatch")
    result = []
    for ordinal, ((episode_id, indices), record) in enumerate(zip(episode_items, rows, strict=True)):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: malformed provenance entry {ordinal}")
        if direct:
            if (record.get("lerobot_episode_index") != int(episode_id)
                or record.get("num_frames") != len(indices)):
                raise ValueError(f"{path}: Staging provenance Episode {episode_id} mismatch")
            reward = record
        else:
            if (record.get("output_episode_index") != int(episode_id)
                or record.get("frames") != len(indices)):
                raise ValueError(f"{path}: merged provenance Episode {episode_id} mismatch")
            reward = record.get("staging_reward")
        if reward is not None:
            if not isinstance(reward, dict) or reward.get("reward_status") not in ("valid", "invalid"):
                raise ValueError(f"{path}: invalid Reward provenance status")
            if reward.get("reward_status") == "valid":
                dp._validate_recorded_reward(episode_id, indices, reward)
        result.append(reward)
    return result


def _episode_task(dataset, first_index, fallback):
    """Resolve real LeRobot DataFrame tasks; retain support for dict fixtures."""
    hf = getattr(dataset, "hf_dataset", None)
    tasks = getattr(dataset.meta, "tasks", None)
    if hf is not None and tasks is not None:
        try:
            raw_index = hf[int(first_index)]["task_index"]
            task_index = int(raw_index.item() if hasattr(raw_index, "item") else raw_index)
            if isinstance(tasks, dict):
                task = tasks.get(task_index, tasks.get(str(task_index)))
                if isinstance(task, str) and task:
                    return task
            elif "task_index" in tasks.columns:
                matches = tasks.index[tasks["task_index"] == task_index]
                if len(matches) == 1 and isinstance(matches[0], str) and matches[0]:
                    return matches[0]
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            pass
    if not fallback:
        raise ValueError("Cannot determine source LeRobot task; set --fallback-task")
    return fallback


def merge_lerobot_datasets(
    source_roots, output_root, repo_id, *,
    use_depth=True, batch_size=25, fallback_task="ARBiM",
):
    """Merge complete LeRobot schemas; Depth is mandatory for archival by default.

    batch_size retained only for backward call compatibility; official
    merge_datasets() operates on Parquet/video files rather than frame batches.
    """
    if not isinstance(source_roots, (list, tuple)) or len(source_roots) < 2:
        raise ValueError("merge_lerobot requires at least two input datasets")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    paths = [Path(p).expanduser().resolve() for p in source_roots]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate LeRobot input datasets")
    target = Path(output_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"LeRobot output already exists: {target}")
    if any(target == p or p in target.parents for p in paths):
        raise ValueError("output LeRobot cannot overwrite or nest inside an input")
    sources = []
    for path in paths:
        if not (path / "meta").is_dir():
            raise FileNotFoundError(f"LeRobot metadata missing: {path}")
        dataset = dp.load_lerobot_dataset(path)
        episodes = dp._group_episode_indices(dataset)
        if not episodes:
            raise ValueError(f"{path}: no episodes")
        contract = _validate_dataset(dataset, path, use_depth)
        sources.append({
            "path": path, "dataset": dataset, "episodes": episodes,
            "contract": contract,
            "reward_rows": _read_reward_rows(path, list(episodes.items())),
        })

    reference = sources[0]["contract"]
    for source in sources[1:]:
        current = source["contract"]
        for key in ("fps", "robot_type"):
            if current[key] != reference[key]:
                raise ValueError(f"{source['path']}: incompatible {key}")
        # Official aggregate_datasets requires *complete* feature equality.
        if current["features"] != reference["features"]:
            missing = sorted(set(reference["features"]) - set(current["features"]))
            extra = sorted(set(current["features"]) - set(reference["features"]))
            raise ValueError(
                f"{source['path']}: incompatible full LeRobot schema "
                f"(missing={missing}, extra={extra}); archive Depth, do not drop it"
            )

    records = []
    ordinal = 0
    total_frames = 0
    for source in sources:
        for position, (episode_id, indices) in enumerate(source["episodes"].items()):
            record = {
                "output_episode_index": ordinal,
                "source_root": str(source["path"]),
                "source_episode_index": int(episode_id),
                "frames": len(indices),
                "task": _episode_task(source["dataset"], indices[0], fallback_task),
            }
            if source["reward_rows"] is not None:
                event = source["reward_rows"][position]
                if event is not None:
                    record["staging_reward"] = event
            records.append(record)
            total_frames += len(indices)
            ordinal += 1

    # This is LeRobot's official metadata/video-copy/remux implementation.
    target.parent.mkdir(parents=True, exist_ok=True)
    combined = merge_datasets(
        datasets=[source["dataset"] for source in sources],
        output_repo_id=repo_id, output_dir=target,
    )
    if len(combined) != total_frames or int(combined.meta.total_episodes) != ordinal:
        raise RuntimeError("Official LeRobot merge produced mismatched Episode/frame counts")
    if combined.meta.features != reference["features"]:
        raise RuntimeError("Official LeRobot merge changed the feature schema")
    manifest = {
        "format": MERGED_FORMAT,
        "use_depth": bool(set(DEPTH_KEYS).issubset(reference["features"])),
        "sources": [str(s["path"]) for s in sources],
        "episodes": records,
        "total_frames": total_frames,
        "total_episodes": ordinal,
        "merge_backend": "lerobot.dataset_tools.merge_datasets+v0.4.2_shard_fix",
    }
    provenance = target / "meta" / MERGE_METADATA
    provenance.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "lerobot_root": str(target), "num_frames": total_frames,
        "num_episodes": ordinal, "provenance": str(provenance),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Two or more LeRobot directories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fallback-task", default="ARBiM")
    parser.add_argument("--batch-size", type=int, default=25, help="Legacy compatibility only")
    parser.add_argument("--allow-rgb-only", action="store_true",
                        help="Legacy archive only; default requires all 3 Depth cameras")
    args = parser.parse_args()
    print(merge_lerobot_datasets(
        args.sources, args.output_root, args.repo_id,
        use_depth=not args.allow_rgb_only, batch_size=args.batch_size,
        fallback_task=args.fallback_task,
    ))


if __name__ == "__main__":
    main()
