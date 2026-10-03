from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import zarr


REWARD_MODE_LAST_GRIPPER_RELEASE_SPARSE = "last_gripper_release_sparse"


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    episode_ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if episode_ends.size == 0:
        raise ValueError("episode_ends must be non-empty")
    if np.any(np.diff(episode_ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing")
    starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends, strict=True)]


def _infer_thresholds(
    values: np.ndarray,
    episode_ranges: list[tuple[int, int]],
    *,
    open_side: str,
    start_frames: int,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("gripper action contains no finite values")

    q10, q90 = np.quantile(finite, [0.10, 0.90]).tolist()
    span = q90 - q10
    if span <= 1e-8:
        raise ValueError(
            f"gripper action has almost no dynamic range: q10={q10}, q90={q90}"
        )

    low = float(low_threshold) if low_threshold is not None else q10 + 0.35 * span
    high = float(high_threshold) if high_threshold is not None else q10 + 0.65 * span
    if not low < high:
        raise ValueError(f"expected low_threshold < high_threshold, got {low} >= {high}")

    starts = []
    for start, end in episode_ranges:
        take_end = min(start + int(start_frames), end)
        starts.extend(values[start:take_end].tolist())
    start_median = float(np.median(np.asarray(starts, dtype=np.float64)))

    if open_side == "auto":
        open_side = (
            "low"
            if abs(start_median - q10) <= abs(start_median - q90)
            else "high"
        )
    if open_side not in {"low", "high"}:
        raise ValueError(f"open_side must be auto/low/high, got {open_side!r}")

    return {
        "low": float(low),
        "high": float(high),
        "open_side": open_side,
        "q10": float(q10),
        "q90": float(q90),
        "start_median": start_median,
    }


def _classify(value: float, thresholds: dict) -> str | None:
    if value <= thresholds["low"]:
        side = "low"
    elif value >= thresholds["high"]:
        side = "high"
    else:
        return None
    return "open" if side == thresholds["open_side"] else "closed"


def _debounced_transitions(
    values: np.ndarray,
    thresholds: dict,
    min_dwell: int,
) -> list[dict]:
    state = None
    candidate = None
    candidate_start = None
    candidate_count = 0
    transitions = []

    for index, raw in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
        classified = _classify(float(raw), thresholds)
        if classified is None:
            continue

        if state is None:
            if candidate == classified:
                candidate_count += 1
            else:
                candidate = classified
                candidate_start = index
                candidate_count = 1
            if candidate_count >= min_dwell:
                state = classified
                candidate = None
                candidate_start = None
                candidate_count = 0
            continue

        if classified == state:
            candidate = None
            candidate_start = None
            candidate_count = 0
            continue

        if candidate == classified:
            candidate_count += 1
        else:
            candidate = classified
            candidate_start = index
            candidate_count = 1

        if candidate_count >= min_dwell:
            transitions.append(
                {
                    "frame_offset": int(candidate_start),
                    "from_state": state,
                    "to_state": classified,
                }
            )
            state = classified
            candidate = None
            candidate_start = None
            candidate_count = 0

    return transitions


def detect_last_release_indices(
    action: np.ndarray,
    episode_ends: np.ndarray,
    *,
    left_gripper_index: int = 7,
    right_gripper_index: int = 15,
    left_open_side: str = "auto",
    right_open_side: str = "auto",
    open_infer_frames: int = 5,
    min_dwell: int = 3,
    expected_cycles: int = 1,
    left_low_threshold: float | None = None,
    left_high_threshold: float | None = None,
    right_low_threshold: float | None = None,
    right_high_threshold: float | None = None,
) -> tuple[np.ndarray, list[dict], dict]:
    action = np.asarray(action)
    if action.ndim != 2:
        raise ValueError(f"expected action [N,D], got {action.shape}")
    max_index = max(int(left_gripper_index), int(right_gripper_index))
    if action.shape[1] <= max_index:
        raise ValueError(
            f"action dim={action.shape[1]} cannot access gripper index {max_index}"
        )
    if open_infer_frames < 1 or min_dwell < 1 or expected_cycles < 1:
        raise ValueError(
            "open_infer_frames, min_dwell, and expected_cycles must all be >= 1"
        )

    ranges = _episode_ranges(episode_ends)
    left_values = action[:, int(left_gripper_index)]
    right_values = action[:, int(right_gripper_index)]
    left_thresholds = _infer_thresholds(
        left_values,
        ranges,
        open_side=left_open_side,
        start_frames=open_infer_frames,
        low_threshold=left_low_threshold,
        high_threshold=left_high_threshold,
    )
    right_thresholds = _infer_thresholds(
        right_values,
        ranges,
        open_side=right_open_side,
        start_frames=open_infer_frames,
        low_threshold=right_low_threshold,
        high_threshold=right_high_threshold,
    )

    reward_indices = []
    rows = []
    for episode, (start, end) in enumerate(ranges):
        left_transitions = _debounced_transitions(
            left_values[start:end],
            left_thresholds,
            int(min_dwell),
        )
        right_transitions = _debounced_transitions(
            right_values[start:end],
            right_thresholds,
            int(min_dwell),
        )
        left_release = [
            event["frame_offset"]
            for event in left_transitions
            if event["from_state"] == "closed" and event["to_state"] == "open"
        ]
        right_release = [
            event["frame_offset"]
            for event in right_transitions
            if event["from_state"] == "closed" and event["to_state"] == "open"
        ]
        if len(left_release) != expected_cycles or len(right_release) != expected_cycles:
            raise ValueError(
                f"episode {episode} release detection failed: "
                f"left_release={left_release}, right_release={right_release}, "
                f"expected_cycles={expected_cycles}"
            )

        left_offset = int(left_release[-1])
        right_offset = int(right_release[-1])
        reward_offset = max(left_offset, right_offset)
        global_index = start + reward_offset
        if global_index >= end - 1:
            raise ValueError(
                f"episode {episode} final release index={global_index} must occur before "
                f"recording end={end - 1}"
            )

        reward_indices.append(global_index)
        rows.append(
            {
                "episode": episode,
                "start": start,
                "end_exclusive": end,
                "length": end - start,
                "left_release_offset": left_offset,
                "right_release_offset": right_offset,
                "reward_offset": reward_offset,
                "reward_global_index": global_index,
                "tail_after_reward_frames": (end - 1) - global_index,
            }
        )

    metadata = {
        "left_gripper_index": int(left_gripper_index),
        "right_gripper_index": int(right_gripper_index),
        "left_thresholds": left_thresholds,
        "right_thresholds": right_thresholds,
        "open_infer_frames": int(open_infer_frames),
        "min_dwell": int(min_dwell),
        "expected_cycles": int(expected_cycles),
    }
    return np.asarray(reward_indices, dtype=np.int64), rows, metadata


def compute_discounted_return(
    reward: np.ndarray,
    done: np.ndarray,
    gamma: float,
) -> np.ndarray:
    reward = np.asarray(reward, dtype=np.float32).reshape(-1)
    done = np.asarray(done, dtype=bool).reshape(-1)
    if reward.shape != done.shape:
        raise ValueError(f"reward/done shape mismatch: {reward.shape} vs {done.shape}")
    returns = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    for index in range(len(reward) - 1, -1, -1):
        running = float(reward[index]) + float(gamma) * running * float(not done[index])
        returns[index] = running
    return returns


def _validate_terminal_contract(done: np.ndarray, timeout: np.ndarray, episode_ends: np.ndarray) -> None:
    done = np.asarray(done, dtype=bool).reshape(-1)
    timeout = np.asarray(timeout, dtype=bool).reshape(-1)
    if not np.array_equal(done, timeout):
        raise ValueError("source Zarr requires done == timeout")
    expected = np.zeros_like(done, dtype=bool)
    expected[np.asarray(episode_ends, dtype=np.int64) - 1] = True
    if not np.array_equal(done, expected):
        raise ValueError(
            "source Zarr terminal contract mismatch: done must be true exactly at episode ends"
        )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def clone_zarr_tree(
    source: str | Path,
    target: str | Path,
    *,
    overwrite: bool = False,
    copy_mode: str = "auto",
) -> str:
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source Zarr not found: {source}")
    if source == target:
        raise ValueError("source and target Zarr paths must differ")
    if str(target).startswith(str(source) + os.sep):
        raise ValueError("target Zarr must not be nested inside source Zarr")

    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"target already exists: {target}; pass --overwrite to replace it"
            )
        _remove_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if copy_mode not in {"auto", "copy", "reflink"}:
        raise ValueError("copy_mode must be auto/copy/reflink")

    cp = shutil.which("cp")
    if copy_mode in {"auto", "reflink"} and cp is not None:
        reflink_mode = "auto" if copy_mode == "auto" else "always"
        result = subprocess.run(
            [cp, "-a", f"--reflink={reflink_mode}", str(source), str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return "reflink_or_copy"
        _remove_path(target)
        if copy_mode == "reflink":
            raise RuntimeError(
                f"cp --reflink=always failed: {result.stderr.strip()}"
            )

    shutil.copytree(source, target, copy_function=shutil.copy2)
    return "copy"


def _episode_ends_sha256(episode_ends: np.ndarray) -> str:
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    return hashlib.sha256(episode_ends.tobytes()).hexdigest()


def migrate_reward_zarr(
    source_zarr: str | Path,
    target_zarr: str | Path,
    *,
    gamma: float = 0.99,
    horizon: int = 32,
    left_gripper_index: int = 7,
    right_gripper_index: int = 15,
    left_open_side: str = "auto",
    right_open_side: str = "auto",
    open_infer_frames: int = 5,
    min_dwell: int = 3,
    expected_cycles: int = 1,
    overwrite: bool = False,
    copy_mode: str = "auto",
) -> dict:
    source_zarr = Path(source_zarr).expanduser().resolve()
    target_zarr = Path(target_zarr).expanduser().resolve()

    source_root = zarr.open_group(str(source_zarr), mode="r")
    if "data" not in source_root or "meta" not in source_root:
        raise KeyError("source Zarr must contain data and meta groups")
    source_data = source_root["data"]
    source_meta = source_root["meta"]
    required = ("action", "reward", "return", "done", "timeout")
    missing = [key for key in required if key not in source_data]
    if missing:
        raise KeyError(f"source Zarr is missing fields: {missing}")
    if "episode_ends" not in source_meta:
        raise KeyError("source Zarr is missing meta/episode_ends")

    action = np.asarray(source_data["action"][:], dtype=np.float32)
    episode_ends = np.asarray(source_meta["episode_ends"][:], dtype=np.int64)
    done = np.asarray(source_data["done"][:], dtype=bool).reshape(-1)
    timeout = np.asarray(source_data["timeout"][:], dtype=bool).reshape(-1)
    if len(action) != len(done) or int(episode_ends[-1]) != len(action):
        raise ValueError("source Zarr frame count/episode_ends mismatch")
    _validate_terminal_contract(done, timeout, episode_ends)

    reward_indices, event_rows, event_metadata = detect_last_release_indices(
        action,
        episode_ends,
        left_gripper_index=left_gripper_index,
        right_gripper_index=right_gripper_index,
        left_open_side=left_open_side,
        right_open_side=right_open_side,
        open_infer_frames=open_infer_frames,
        min_dwell=min_dwell,
        expected_cycles=expected_cycles,
    )

    reward = np.zeros(len(action), dtype=np.float32)
    reward[reward_indices] = 1.0
    returns = compute_discounted_return(reward, done, gamma)

    copy_backend = clone_zarr_tree(
        source_zarr,
        target_zarr,
        overwrite=overwrite,
        copy_mode=copy_mode,
    )
    target_root = zarr.open_group(str(target_zarr), mode="a")
    target_data = target_root["data"]
    target_meta = target_root["meta"]

    target_data["reward"][:] = reward.reshape(target_data["reward"].shape)
    target_data["return"][:] = returns.reshape(target_data["return"].shape)

    target_episode_ends = np.asarray(target_meta["episode_ends"][:], dtype=np.int64)
    target_done = np.asarray(target_data["done"][:], dtype=bool).reshape(-1)
    target_timeout = np.asarray(target_data["timeout"][:], dtype=bool).reshape(-1)
    if not np.array_equal(target_episode_ends, episode_ends):
        raise RuntimeError("target episode_ends changed during migration")
    if not np.array_equal(target_done, done) or not np.array_equal(target_timeout, timeout):
        raise RuntimeError("target done/timeout changed during migration")

    reward_check = np.asarray(target_data["reward"][:], dtype=np.float32).reshape(-1)
    return_check = np.asarray(target_data["return"][:], dtype=np.float32).reshape(-1)
    if int(np.count_nonzero(reward_check == 1.0)) != len(episode_ends):
        raise RuntimeError("target must contain exactly one +1 reward per episode")
    if not np.allclose(reward_check, reward, rtol=0.0, atol=0.0):
        raise RuntimeError("target reward verification failed")
    if not np.allclose(return_check, returns, rtol=1e-6, atol=1e-6):
        raise RuntimeError("target return verification failed")

    final_contains = []
    final_reward_offsets = []
    for row in event_rows:
        final_start = max(row["start"], row["end_exclusive"] - int(horizon))
        contains = final_start <= row["reward_global_index"] < row["end_exclusive"]
        final_contains.append(bool(contains))
        if contains:
            final_reward_offsets.append(row["reward_global_index"] - final_start)

    target_root.attrs["reward_mode"] = REWARD_MODE_LAST_GRIPPER_RELEASE_SPARSE
    target_root.attrs["reward_migration"] = {
        "source_zarr": str(source_zarr),
        "gamma": float(gamma),
        "horizon_diagnostic": int(horizon),
        **event_metadata,
    }

    tail = np.asarray(
        [row["tail_after_reward_frames"] for row in event_rows],
        dtype=np.float64,
    )
    summary = {
        "source_zarr": str(source_zarr),
        "target_zarr": str(target_zarr),
        "copy_backend": copy_backend,
        "frames": int(len(action)),
        "episodes": int(len(episode_ends)),
        "reward_mode": REWARD_MODE_LAST_GRIPPER_RELEASE_SPARSE,
        "reward_count": int(np.count_nonzero(reward == 1.0)),
        "reward_sum": float(reward.sum()),
        "gamma": float(gamma),
        "episode_ends_sha256": _episode_ends_sha256(episode_ends),
        "tail_after_reward_frames": {
            "min": float(tail.min()),
            "mean": float(tail.mean()),
            "median": float(np.median(tail)),
            "max": float(tail.max()),
        },
        "final_h_chunk_contains_reward_fraction": float(np.mean(final_contains)),
        "final_h_chunk_reward_offset_mean": (
            float(np.mean(final_reward_offsets)) if final_reward_offsets else None
        ),
        "event_metadata": event_metadata,
        "event_rows": event_rows,
    }
    return summary


def clone_latent_cache_for_reward_clone(
    source_cache: str | Path,
    source_zarr: str | Path,
    target_zarr: str | Path,
    *,
    target_cache: str | Path | None = None,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> dict:
    source_cache = Path(source_cache).expanduser().resolve()
    source_zarr = Path(source_zarr).expanduser().resolve()
    target_zarr = Path(target_zarr).expanduser().resolve()
    if not source_cache.is_dir():
        raise FileNotFoundError(f"source latent cache not found: {source_cache}")
    if not target_zarr.is_dir():
        raise FileNotFoundError(f"target Zarr not found: {target_zarr}")
    if link_mode not in {"hardlink", "symlink", "copy"}:
        raise ValueError("link_mode must be hardlink/symlink/copy")

    required = ("metadata.json", "obs_latent.npy", "next_indices.npy")
    missing = [name for name in required if not (source_cache / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source latent cache is incomplete: {missing}")

    with (source_cache / "metadata.json").open("r") as file:
        metadata = json.load(file)

    target_root = zarr.open_group(str(target_zarr), mode="r")
    episode_ends = np.asarray(target_root["meta"]["episode_ends"][:], dtype=np.int64)
    size = int(episode_ends[-1])
    episode_hash = _episode_ends_sha256(episode_ends)

    if int(metadata.get("size", -1)) != size:
        raise RuntimeError(
            f"latent cache size={metadata.get('size')} does not match target dataset size={size}"
        )
    if metadata.get("episode_ends_sha256") != episode_hash:
        raise RuntimeError("latent cache episode_ends_sha256 does not match target dataset")
    source_dataset = metadata.get("source_dataset")
    if source_dataset is not None:
        if os.path.realpath(source_dataset) != os.path.realpath(source_zarr):
            raise RuntimeError(
                "source latent cache metadata points to a different source dataset: "
                f"{source_dataset}"
            )

    obs = np.load(source_cache / "obs_latent.npy", mmap_mode="r")
    next_indices = np.load(source_cache / "next_indices.npy", mmap_mode="r")
    if len(obs) != size or len(next_indices) != size:
        raise RuntimeError("latent cache array length does not match target dataset")

    if target_cache is None:
        target_cache = (
            Path(str(target_zarr) + ".act_latent_cache")
            / source_cache.name
        )
    target_cache = Path(target_cache).expanduser().resolve()
    if target_cache.exists():
        if not overwrite:
            raise FileExistsError(
                f"target latent cache already exists: {target_cache}; "
                "pass --overwrite to replace it"
            )
        _remove_path(target_cache)

    target_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(target_cache) + f".tmp.{os.getpid()}")
    if tmp.exists():
        _remove_path(tmp)
    tmp.mkdir(parents=True)

    try:
        for name in ("obs_latent.npy", "next_indices.npy"):
            src = source_cache / name
            dst = tmp / name
            if link_mode == "hardlink":
                try:
                    os.link(src, dst)
                except OSError as exc:
                    raise RuntimeError(
                        f"hardlink failed for {src} -> {dst}: {exc}. "
                        "Use --latent-link-mode symlink or copy if the paths are on "
                        "different filesystems."
                    ) from exc
            elif link_mode == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)

        new_metadata = dict(metadata)
        new_metadata["source_dataset"] = os.path.realpath(target_zarr)
        new_metadata["derived_from_cache"] = str(source_cache)
        new_metadata["reward_only_dataset_clone"] = True
        with (tmp / "metadata.json").open("w") as file:
            json.dump(new_metadata, file, indent=2, sort_keys=True)

        os.replace(tmp, target_cache)
    except Exception:
        if tmp.exists():
            _remove_path(tmp)
        raise

    with (target_cache / "metadata.json").open("r") as file:
        check_metadata = json.load(file)
    if check_metadata.get("source_dataset") != os.path.realpath(target_zarr):
        raise RuntimeError("target latent cache metadata source_dataset verification failed")
    target_obs = np.load(target_cache / "obs_latent.npy", mmap_mode="r")
    target_next = np.load(target_cache / "next_indices.npy", mmap_mode="r")
    if target_obs.shape != obs.shape or not np.array_equal(target_next, next_indices):
        raise RuntimeError("target latent cache verification failed")

    return {
        "source_cache": str(source_cache),
        "target_cache": str(target_cache),
        "link_mode": link_mode,
        "latent_shape": list(target_obs.shape),
        "encoder_sha256": check_metadata.get("encoder_sha256"),
        "normalizer_sha256": check_metadata.get("normalizer_sha256"),
        "source_dataset": check_metadata.get("source_dataset"),
    }
