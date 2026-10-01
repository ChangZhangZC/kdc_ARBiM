from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from post_rl.data.offline_buffer import OfflineBuffer  # noqa: E402
from post_rl.data.sampler import SequenceSampler  # noqa: E402


OPEN = "open"
CLOSED = "closed"


def _scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _load_lerobot(root: pathlib.Path) -> LeRobotDataset:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return LeRobotDataset(repo_id=root.name, root=root)


def _column(hf_dataset, key: str) -> list:
    if key not in hf_dataset.column_names:
        raise KeyError(f"LeRobot dataset is missing required column {key!r}")
    return hf_dataset[key]


def _stack_numeric_column(hf_dataset, key: str, dtype=np.float32) -> np.ndarray:
    values = _column(hf_dataset, key)
    arrays = [_as_numpy(value) for value in values]
    return np.stack(arrays, axis=0).astype(dtype, copy=False)


def _scalar_column(hf_dataset, key: str, dtype) -> np.ndarray | None:
    if key not in hf_dataset.column_names:
        return None
    return np.asarray([_scalar(value) for value in hf_dataset[key]], dtype=dtype)


def _appearance_order(values: np.ndarray) -> list[int]:
    seen = set()
    order = []
    for value in values.tolist():
        value = int(value)
        if value not in seen:
            seen.add(value)
            order.append(value)
    return order


def _positions_by_episode(
    episode_index: np.ndarray,
    episode_order: list[int],
) -> dict[int, np.ndarray]:
    return {
        ep_id: np.flatnonzero(episode_index == ep_id).astype(np.int64)
        for ep_id in episode_order
    }


def _count_runs(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(indices) != 1))


@dataclass
class GripperThresholds:
    low: float
    high: float
    open_side: str
    q10: float
    q90: float
    start_median: float

    def classify(self, value: float) -> str | None:
        if value <= self.low:
            side = "low"
        elif value >= self.high:
            side = "high"
        else:
            return None
        if self.open_side == "low":
            return OPEN if side == "low" else CLOSED
        return OPEN if side == "high" else CLOSED


def _infer_thresholds(
    values: np.ndarray,
    episode_positions: Iterable[np.ndarray],
    *,
    open_side: str,
    start_frames: int,
    low_threshold: float | None,
    high_threshold: float | None,
) -> GripperThresholds:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("Gripper data contains no finite values.")

    q10, q90 = np.quantile(finite, [0.10, 0.90]).tolist()
    span = q90 - q10
    if span <= 1e-8:
        raise ValueError(
            f"Gripper action has almost no dynamic range: q10={q10}, q90={q90}"
        )

    low = float(low_threshold) if low_threshold is not None else q10 + 0.35 * span
    high = float(high_threshold) if high_threshold is not None else q10 + 0.65 * span
    if not low < high:
        raise ValueError(f"Expected low_threshold < high_threshold, got {low} >= {high}")

    starts = []
    for positions in episode_positions:
        if len(positions) == 0:
            continue
        take = positions[: min(start_frames, len(positions))]
        starts.extend(values[take].astype(np.float64).tolist())
    start_median = float(np.median(np.asarray(starts, dtype=np.float64)))

    if open_side == "auto":
        low_distance = abs(start_median - q10)
        high_distance = abs(start_median - q90)
        inferred = "low" if low_distance <= high_distance else "high"
    else:
        inferred = open_side

    return GripperThresholds(
        low=low,
        high=high,
        open_side=inferred,
        q10=float(q10),
        q90=float(q90),
        start_median=start_median,
    )


def _debounced_gripper_states(
    values: np.ndarray,
    thresholds: GripperThresholds,
    min_dwell: int,
) -> tuple[str | None, int | None, str | None, list[dict]]:
    state = None
    state_start = None
    candidate = None
    candidate_start = None
    candidate_count = 0
    transitions = []

    for index, raw in enumerate(np.asarray(values, dtype=np.float64)):
        classified = thresholds.classify(float(raw))
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
                state_start = int(candidate_start)
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
            old_state = state
            state = classified
            event_index = int(candidate_start)
            transitions.append(
                {
                    "frame_offset": event_index,
                    "from_state": old_state,
                    "to_state": state,
                    "value": float(values[event_index]),
                }
            )
            candidate = None
            candidate_start = None
            candidate_count = 0

    return (
        None if state_start is None else thresholds.classify(float(values[state_start])),
        state_start,
        state,
        transitions,
    )


def _gripper_semantics(
    values: np.ndarray,
    thresholds: GripperThresholds,
    min_dwell: int,
    expected_cycles: int,
) -> dict:
    initial_state, initial_offset, final_state, transitions = _debounced_gripper_states(
        values,
        thresholds,
        min_dwell,
    )
    close_events = [
        event for event in transitions
        if event["from_state"] == OPEN and event["to_state"] == CLOSED
    ]
    reopen_events = [
        event for event in transitions
        if event["from_state"] == CLOSED and event["to_state"] == OPEN
    ]

    sequence = []
    if initial_state is not None:
        sequence.append(initial_state)
        sequence.extend(event["to_state"] for event in transitions)

    expected_transition_count = expected_cycles * 2
    valid = (
        initial_state == OPEN
        and final_state == OPEN
        and len(close_events) == expected_cycles
        and len(reopen_events) == expected_cycles
        and len(transitions) == expected_transition_count
    )

    reasons = []
    if initial_state != OPEN:
        reasons.append(f"initial_state={initial_state}")
    if len(close_events) != expected_cycles:
        reasons.append(f"close_events={len(close_events)}")
    if len(reopen_events) != expected_cycles:
        reasons.append(f"reopen_events={len(reopen_events)}")
    if len(transitions) != expected_transition_count:
        reasons.append(f"transitions={len(transitions)}")
    if final_state != OPEN:
        reasons.append(f"final_state={final_state}")

    return {
        "initial_state": initial_state,
        "initial_state_offset": initial_offset,
        "final_state": final_state,
        "close_events": close_events,
        "reopen_events": reopen_events,
        "transitions": transitions,
        "sequence": "->".join(sequence),
        "valid": bool(valid),
        "reasons": reasons,
    }


def _paired_event_lag(
    action_result: dict,
    state_result: dict,
    event_key: str,
) -> int | None:
    action_events = action_result[event_key]
    state_events = state_result[event_key]
    if len(action_events) != 1 or len(state_events) != 1:
        return None
    return int(state_events[0]["frame_offset"] - action_events[0]["frame_offset"])


def _episode_structural_row(
    *,
    ep_id: int,
    positions: np.ndarray,
    frame_index: np.ndarray | None,
    timestamp: np.ndarray | None,
    fps: float | None,
    timestamp_gap_factor: float,
) -> dict:
    runs = _count_runs(positions)
    length = int(len(positions))

    frame_valid = None
    frame_first = None
    frame_last = None
    if frame_index is not None:
        frames = frame_index[positions]
        frame_first = int(frames[0])
        frame_last = int(frames[-1])
        frame_valid = bool(
            np.array_equal(frames, np.arange(length, dtype=frames.dtype))
        )

    timestamp_monotonic = None
    timestamp_max_gap = None
    timestamp_gap_valid = None
    timestamp_start = None
    timestamp_end = None
    if timestamp is not None:
        ts = timestamp[positions].astype(np.float64)
        timestamp_start = float(ts[0])
        timestamp_end = float(ts[-1])
        diffs = np.diff(ts)
        timestamp_monotonic = bool(np.all(diffs > 0)) if len(diffs) else True
        timestamp_max_gap = float(diffs.max()) if len(diffs) else 0.0
        if fps is not None and fps > 0:
            timestamp_gap_valid = bool(
                timestamp_max_gap <= timestamp_gap_factor / fps
            )

    structural_valid = runs == 1
    if frame_valid is not None:
        structural_valid = structural_valid and frame_valid
    if timestamp_monotonic is not None:
        structural_valid = structural_valid and timestamp_monotonic
    if timestamp_gap_valid is not None:
        structural_valid = structural_valid and timestamp_gap_valid

    return {
        "episode_id": int(ep_id),
        "global_start": int(positions[0]),
        "global_end_inclusive": int(positions[-1]),
        "length": length,
        "contiguous_runs": runs,
        "frame_index_first": frame_first,
        "frame_index_last": frame_last,
        "frame_index_valid": frame_valid,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "timestamp_monotonic": timestamp_monotonic,
        "timestamp_max_gap": timestamp_max_gap,
        "timestamp_gap_valid": timestamp_gap_valid,
        "structural_valid": bool(structural_valid),
    }


def _zarr_contract_rows(
    zarr_path: pathlib.Path,
    expected_lengths: list[int],
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    root = zarr.open_group(str(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError("Offline RL Zarr must contain data/ and meta/ groups.")
    data = root["data"]
    meta = root["meta"]

    required = ("action", "state", "next_state", "next_action", "reward", "done", "timeout", "next_index", "return")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Offline RL Zarr data/ is missing keys: {missing}")
    if "episode_ends" not in meta:
        raise KeyError("Offline RL Zarr meta/ is missing episode_ends.")

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate(([0], episode_ends[:-1]))
    zarr_lengths = episode_ends - starts

    done = np.asarray(data["done"][:], dtype=bool).reshape(-1)
    timeout = np.asarray(data["timeout"][:], dtype=bool).reshape(-1)
    reward = np.asarray(data["reward"][:], dtype=np.float64).reshape(-1)
    next_index = np.asarray(data["next_index"][:], dtype=np.int64).reshape(-1)
    returns = np.asarray(data["return"][:], dtype=np.float64).reshape(-1)

    rows = []
    for zarr_ep, (start, end) in enumerate(zip(starts, episode_ends, strict=True)):
        start = int(start)
        end = int(end)
        length = end - start
        local_done = done[start:end]
        local_timeout = timeout[start:end]
        local_reward = reward[start:end]
        local_next_index = next_index[start:end]

        expected_length = expected_lengths[zarr_ep] if zarr_ep < len(expected_lengths) else None
        expected_nonterminal_next = np.arange(start + 1, end, dtype=np.int64)
        nonterminal_next_valid = bool(
            np.array_equal(local_next_index[:-1], expected_nonterminal_next)
        ) if length > 1 else True
        terminal_self_loop = bool(local_next_index[-1] == end - 1)
        done_valid = bool(
            local_done.sum() == 1 and local_done[-1] and not local_done[:-1].any()
        )
        timeout_valid = bool(np.array_equal(local_timeout, local_done))
        nonterminal_positive_rewards = int(np.count_nonzero(local_reward[:-1] > 0))
        terminal_reward_positive = bool(local_reward[-1] > 0)

        length_match = expected_length is not None and int(expected_length) == length
        zarr_valid = (
            length_match
            and done_valid
            and timeout_valid
            and nonterminal_next_valid
            and terminal_self_loop
            and nonterminal_positive_rewards == 0
            and terminal_reward_positive
        )

        rows.append(
            {
                "zarr_episode": zarr_ep,
                "zarr_start": start,
                "zarr_end_exclusive": end,
                "zarr_length": length,
                "expected_lerobot_length": expected_length,
                "length_match": bool(length_match),
                "done_count": int(local_done.sum()),
                "done_last_only": done_valid,
                "timeout_matches_done": timeout_valid,
                "nonterminal_next_index_valid": nonterminal_next_valid,
                "terminal_next_index_self_loop": terminal_self_loop,
                "nonterminal_positive_reward_count": nonterminal_positive_rewards,
                "terminal_reward": float(local_reward[-1]),
                "terminal_reward_positive": terminal_reward_positive,
                "zarr_valid": bool(zarr_valid),
            }
        )

    arrays = {
        "episode_ends": episode_ends,
        "action": np.asarray(data["action"][:]),
        "state": np.asarray(data["state"][:]),
        "next_state": np.asarray(data["next_state"][:]),
        "next_action": np.asarray(data["next_action"][:]),
        "reward": reward,
        "done": done,
        "timeout": timeout,
        "next_index": next_index,
        "return": returns,
    }
    summary = {
        "frames": int(len(done)),
        "episodes": int(len(episode_ends)),
        "episode_lengths": zarr_lengths.astype(int).tolist(),
    }
    return rows, summary, arrays


def _transition_contract_summary(arrays: dict[str, np.ndarray], gamma: float) -> dict:
    done = arrays["done"]
    state = arrays["state"]
    action = arrays["action"]
    next_state = arrays["next_state"]
    next_action = arrays["next_action"]
    reward = arrays["reward"]
    returns = arrays["return"]

    nonterminal = ~done
    terminal = done

    # Validate next_index before using it for vectorized transition checks.
    ni = arrays["next_index"]
    next_index_in_range = bool(
        np.all((ni >= 0) & (ni < len(state)))
    )
    if next_index_in_range:
        state_by_index_error = np.abs(next_state - state[ni])
        action_by_index_error = np.abs(next_action - action[ni])
        state_by_index_max = float(state_by_index_error.max(initial=0.0))
        action_by_index_max = float(action_by_index_error.max(initial=0.0))
    else:
        state_by_index_max = float("inf")
        action_by_index_max = float("inf")

    target_return = reward.copy()
    nonterminal_indices = np.flatnonzero(nonterminal)
    valid_nonterminal = nonterminal_indices[nonterminal_indices + 1 < len(returns)]
    target_return[valid_nonterminal] += gamma * returns[valid_nonterminal + 1]
    return_error = np.abs(returns - target_return)
    dangling_last_nonterminal = bool(len(done) > 0 and not done[-1])

    terminal_state_error = np.abs(next_state[terminal] - state[terminal])
    terminal_action_error = np.abs(next_action[terminal] - action[terminal])

    return {
        "next_index_in_range": next_index_in_range,
        "next_state_vs_next_index_max_abs": state_by_index_max,
        "next_action_vs_next_index_max_abs": action_by_index_max,
        "terminal_self_loop_state_max_abs": float(terminal_state_error.max(initial=0.0)),
        "terminal_self_loop_action_max_abs": float(terminal_action_error.max(initial=0.0)),
        "return_recurrence_max_abs": float(return_error.max(initial=0.0)),
        "return_recurrence_mean_abs": float(return_error.mean()) if len(return_error) else 0.0,
        "dangling_last_nonterminal": dangling_last_nonterminal,
    }


def _sampler_contract(
    zarr_path: pathlib.Path,
    episode_ends: np.ndarray,
    *,
    horizon: int,
    strides: list[int],
    gamma: float,
) -> list[dict]:
    buffer = OfflineBuffer(device=torch.device("cpu"), gamma=gamma, use_depth=False)
    buffer.load_zarr(str(zarr_path))
    rows = []

    for stride in strides:
        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=horizon,
            pad_before=0,
            pad_after=0,
            keys=["action"],
            episode_mask=np.ones(len(episode_ends), dtype=bool),
            sequence_stride=stride,
        )
        crossing = 0
        invalid_length = 0
        for buffer_start, buffer_end, sample_start, sample_end in sampler.indices:
            if sample_start != 0 or sample_end != horizon:
                invalid_length += 1
            start_ep = int(np.searchsorted(episode_ends, buffer_start, side="right"))
            end_ep = int(np.searchsorted(episode_ends, buffer_end - 1, side="right"))
            if start_ep != end_ep:
                crossing += 1

        rows.append(
            {
                "horizon": int(horizon),
                "stride": int(stride),
                "windows": int(len(sampler.indices)),
                "cross_episode_windows": int(crossing),
                "padded_or_short_windows": int(invalid_length),
                "valid": bool(crossing == 0 and invalid_length == 0),
            }
        )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[int] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()) if len(array) else None,
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def _fmt_optional(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _print_summary(summary: dict) -> None:
    print("\n=== Dataset Episode Integrity Diagnostic ===")
    print(
        f"LeRobot: frames={summary['lerobot']['frames']} "
        f"episodes={summary['lerobot']['episodes']}"
    )
    print(
        f"Zarr:    frames={summary['zarr']['frames']} "
        f"episodes={summary['zarr']['episodes']}"
    )
    print(
        "Episode length min/mean/median/max: "
        f"{summary['lerobot']['episode_length_stats']['min']:.0f} / "
        f"{summary['lerobot']['episode_length_stats']['mean']:.2f} / "
        f"{summary['lerobot']['episode_length_stats']['median']:.0f} / "
        f"{summary['lerobot']['episode_length_stats']['max']:.0f}"
    )
    print(
        "Structural valid: "
        f"{summary['counts']['structural_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Zarr contract valid: "
        f"{summary['counts']['zarr_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Action command - left grasp+release valid: "
        f"{summary['counts']['left_gripper_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Action command - right grasp+release valid: "
        f"{summary['counts']['right_gripper_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Observed state - left grasp+release valid: "
        f"{summary['counts']['left_state_gripper_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Observed state - right grasp+release valid: "
        f"{summary['counts']['right_state_gripper_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Both arms command-semantic valid: "
        f"{summary['counts']['command_semantic_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Both arms observed-state semantic valid: "
        f"{summary['counts']['execution_semantic_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Both command+state task-semantic valid: "
        f"{summary['counts']['task_semantic_valid']} / {summary['lerobot']['episodes']}"
    )
    print(
        "Fully valid episodes: "
        f"{summary['counts']['fully_valid']} / {summary['lerobot']['episodes']}"
    )

    for source_name, key in (("Action", "gripper_thresholds"), ("State", "state_gripper_thresholds")):
        for side in ("left", "right"):
            thresholds = summary[key][side]
            print(
                f"{source_name} {side} gripper: open_side={thresholds['open_side']} "
                f"q10/q90={thresholds['q10']:.5f}/{thresholds['q90']:.5f} "
                f"low/high thresholds={thresholds['low']:.5f}/{thresholds['high']:.5f} "
                f"start_median={thresholds['start_median']:.5f}"
            )

    lag = summary["action_state_event_lag_frames"]
    print(
        "Action->state gripper event lag frames mean/median: "
        f"L-close={_fmt_optional(lag['left_close']['mean'])}/"
        f"{_fmt_optional(lag['left_close']['median'])}, "
        f"L-open={_fmt_optional(lag['left_reopen']['mean'])}/"
        f"{_fmt_optional(lag['left_reopen']['median'])}, "
        f"R-close={_fmt_optional(lag['right_close']['mean'])}/"
        f"{_fmt_optional(lag['right_close']['median'])}, "
        f"R-open={_fmt_optional(lag['right_reopen']['mean'])}/"
        f"{_fmt_optional(lag['right_reopen']['median'])}"
    )

    print("\nSampler checks:")
    for row in summary["sampler_checks"]:
        print(
            f"  H={row['horizon']} stride={row['stride']}: "
            f"windows={row['windows']} cross_episode={row['cross_episode_windows']} "
            f"short/padded={row['padded_or_short_windows']} valid={row['valid']}"
        )

    print(
        "\nSuspicious episodes: "
        f"{summary['counts']['suspicious']} "
        f"(see suspicious_episodes.csv)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit LeRobot -> Offline RL Zarr episode boundaries, transition contracts, "
            "sampler windows, and bimanual grasp/release semantics."
        )
    )
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/dataset_episode_integrity",
    )
    parser.add_argument("--left-gripper-index", type=int, default=7)
    parser.add_argument("--right-gripper-index", type=int, default=15)
    parser.add_argument("--left-state-gripper-index", type=int, default=7)
    parser.add_argument("--right-state-gripper-index", type=int, default=15)
    parser.add_argument(
        "--left-open-side",
        choices=("auto", "low", "high"),
        default="auto",
    )
    parser.add_argument(
        "--right-open-side",
        choices=("auto", "low", "high"),
        default="auto",
    )
    parser.add_argument("--left-low-threshold", type=float, default=None)
    parser.add_argument("--left-high-threshold", type=float, default=None)
    parser.add_argument("--right-low-threshold", type=float, default=None)
    parser.add_argument("--right-high-threshold", type=float, default=None)
    parser.add_argument(
        "--left-state-open-side",
        choices=("auto", "low", "high"),
        default="auto",
    )
    parser.add_argument(
        "--right-state-open-side",
        choices=("auto", "low", "high"),
        default="auto",
    )
    parser.add_argument("--left-state-low-threshold", type=float, default=None)
    parser.add_argument("--left-state-high-threshold", type=float, default=None)
    parser.add_argument("--right-state-low-threshold", type=float, default=None)
    parser.add_argument("--right-state-high-threshold", type=float, default=None)
    parser.add_argument("--open-infer-frames", type=int, default=5)
    parser.add_argument("--min-dwell", type=int, default=3)
    parser.add_argument("--expected-cycles", type=int, default=1)
    parser.add_argument("--timestamp-gap-factor", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--sampler-strides",
        type=int,
        nargs="+",
        default=[1, 32],
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--compare-atol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.min_dwell < 1:
        raise ValueError("--min-dwell must be >= 1")
    if args.expected_cycles < 1:
        raise ValueError("--expected-cycles must be >= 1")
    if args.open_infer_frames < 1:
        raise ValueError("--open-infer-frames must be >= 1")

    lerobot_root = pathlib.Path(args.lerobot_root).expanduser().resolve()
    zarr_path = pathlib.Path(args.zarr).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_lerobot(lerobot_root)
    hf_dataset = dataset.hf_dataset
    episode_index = _scalar_column(hf_dataset, "episode_index", np.int64)
    if episode_index is None:
        raise KeyError("LeRobot dataset has no episode_index.")
    frame_index = _scalar_column(hf_dataset, "frame_index", np.int64)
    timestamp = _scalar_column(hf_dataset, "timestamp", np.float64)
    action = _stack_numeric_column(hf_dataset, "action", np.float32)
    state = _stack_numeric_column(hf_dataset, "observation.state", np.float32)

    if action.ndim != 2:
        raise ValueError(f"Expected LeRobot action [N,D], got {action.shape}")
    max_index = max(args.left_gripper_index, args.right_gripper_index)
    if action.shape[1] <= max_index:
        raise ValueError(
            f"Action dim={action.shape[1]} cannot access gripper index {max_index}"
        )
    max_state_index = max(
        args.left_state_gripper_index,
        args.right_state_gripper_index,
    )
    if state.ndim != 2 or state.shape[1] <= max_state_index:
        raise ValueError(
            f"State shape={state.shape} cannot access gripper index {max_state_index}"
        )

    episode_order = _appearance_order(episode_index)
    positions_by_episode = _positions_by_episode(episode_index, episode_order)
    episode_positions = [positions_by_episode[ep] for ep in episode_order]
    expected_order = np.concatenate(episode_positions, axis=0)
    expected_lengths = [len(positions) for positions in episode_positions]

    fps = float(dataset.meta.fps) if getattr(dataset.meta, "fps", None) else None
    structural_rows = [
        _episode_structural_row(
            ep_id=ep_id,
            positions=positions_by_episode[ep_id],
            frame_index=frame_index,
            timestamp=timestamp,
            fps=fps,
            timestamp_gap_factor=float(args.timestamp_gap_factor),
        )
        for ep_id in episode_order
    ]

    left_values = action[:, args.left_gripper_index]
    right_values = action[:, args.right_gripper_index]
    left_state_values = state[:, args.left_state_gripper_index]
    right_state_values = state[:, args.right_state_gripper_index]
    left_thresholds = _infer_thresholds(
        left_values,
        episode_positions,
        open_side=args.left_open_side,
        start_frames=args.open_infer_frames,
        low_threshold=args.left_low_threshold,
        high_threshold=args.left_high_threshold,
    )
    right_thresholds = _infer_thresholds(
        right_values,
        episode_positions,
        open_side=args.right_open_side,
        start_frames=args.open_infer_frames,
        low_threshold=args.right_low_threshold,
        high_threshold=args.right_high_threshold,
    )
    left_state_thresholds = _infer_thresholds(
        left_state_values,
        episode_positions,
        open_side=args.left_state_open_side,
        start_frames=args.open_infer_frames,
        low_threshold=args.left_state_low_threshold,
        high_threshold=args.left_state_high_threshold,
    )
    right_state_thresholds = _infer_thresholds(
        right_state_values,
        episode_positions,
        open_side=args.right_state_open_side,
        start_frames=args.open_infer_frames,
        low_threshold=args.right_state_low_threshold,
        high_threshold=args.right_state_high_threshold,
    )

    gripper_rows = []
    gripper_event_rows = []
    for ep_id, positions in zip(episode_order, episode_positions, strict=True):
        left = _gripper_semantics(
            left_values[positions],
            left_thresholds,
            args.min_dwell,
            args.expected_cycles,
        )
        right = _gripper_semantics(
            right_values[positions],
            right_thresholds,
            args.min_dwell,
            args.expected_cycles,
        )
        left_state_result = _gripper_semantics(
            left_state_values[positions],
            left_state_thresholds,
            args.min_dwell,
            args.expected_cycles,
        )
        right_state_result = _gripper_semantics(
            right_state_values[positions],
            right_state_thresholds,
            args.min_dwell,
            args.expected_cycles,
        )
        left_close_lag = _paired_event_lag(left, left_state_result, "close_events")
        left_reopen_lag = _paired_event_lag(left, left_state_result, "reopen_events")
        right_close_lag = _paired_event_lag(right, right_state_result, "close_events")
        right_reopen_lag = _paired_event_lag(right, right_state_result, "reopen_events")
        command_semantic_valid = bool(left["valid"] and right["valid"])
        execution_semantic_valid = bool(
            left_state_result["valid"] and right_state_result["valid"]
        )
        gripper_rows.append(
            {
                "episode_id": int(ep_id),
                "left_initial_state": left["initial_state"],
                "left_final_state": left["final_state"],
                "left_sequence": left["sequence"],
                "left_close_events": len(left["close_events"]),
                "left_reopen_events": len(left["reopen_events"]),
                "left_valid": left["valid"],
                "left_reasons": ";".join(left["reasons"]),
                "right_initial_state": right["initial_state"],
                "right_final_state": right["final_state"],
                "right_sequence": right["sequence"],
                "right_close_events": len(right["close_events"]),
                "right_reopen_events": len(right["reopen_events"]),
                "right_valid": right["valid"],
                "right_reasons": ";".join(right["reasons"]),
                "left_state_initial": left_state_result["initial_state"],
                "left_state_final": left_state_result["final_state"],
                "left_state_sequence": left_state_result["sequence"],
                "left_state_close_events": len(left_state_result["close_events"]),
                "left_state_reopen_events": len(left_state_result["reopen_events"]),
                "left_state_valid": left_state_result["valid"],
                "left_state_reasons": ";".join(left_state_result["reasons"]),
                "right_state_initial": right_state_result["initial_state"],
                "right_state_final": right_state_result["final_state"],
                "right_state_sequence": right_state_result["sequence"],
                "right_state_close_events": len(right_state_result["close_events"]),
                "right_state_reopen_events": len(right_state_result["reopen_events"]),
                "right_state_valid": right_state_result["valid"],
                "right_state_reasons": ";".join(right_state_result["reasons"]),
                "left_close_action_to_state_lag_frames": left_close_lag,
                "left_reopen_action_to_state_lag_frames": left_reopen_lag,
                "right_close_action_to_state_lag_frames": right_close_lag,
                "right_reopen_action_to_state_lag_frames": right_reopen_lag,
                "command_semantic_valid": command_semantic_valid,
                "execution_semantic_valid": execution_semantic_valid,
                "task_semantic_valid": bool(
                    command_semantic_valid and execution_semantic_valid
                ),
            }
        )
        for source, side, result in (
            ("action", "left", left),
            ("action", "right", right),
            ("state", "left", left_state_result),
            ("state", "right", right_state_result),
        ):
            for event_index, event in enumerate(result["transitions"]):
                gripper_event_rows.append(
                    {
                        "episode_id": int(ep_id),
                        "source": source,
                        "side": side,
                        "event_index": event_index,
                        "frame_offset": event["frame_offset"],
                        "global_frame_index": int(positions[event["frame_offset"]]),
                        "from_state": event["from_state"],
                        "to_state": event["to_state"],
                        "value": event["value"],
                    }
                )

    zarr_rows, zarr_summary, zarr_arrays = _zarr_contract_rows(
        zarr_path,
        expected_lengths,
    )
    if len(zarr_rows) != len(episode_order):
        raise RuntimeError(
            f"Episode count mismatch: LeRobot={len(episode_order)}, Zarr={len(zarr_rows)}"
        )

    expected_action = action[expected_order]
    expected_state = state[expected_order]
    if expected_action.shape != zarr_arrays["action"].shape:
        action_match = False
        action_max_abs = float("inf")
    else:
        action_diff = np.abs(
            expected_action.astype(np.float64)
            - zarr_arrays["action"].astype(np.float64)
        )
        action_max_abs = float(action_diff.max(initial=0.0))
        action_match = bool(action_max_abs <= args.compare_atol)

    if expected_state.shape != zarr_arrays["state"].shape:
        state_match = False
        state_max_abs = float("inf")
    else:
        state_diff = np.abs(
            expected_state.astype(np.float64)
            - zarr_arrays["state"].astype(np.float64)
        )
        state_max_abs = float(state_diff.max(initial=0.0))
        state_match = bool(state_max_abs <= args.compare_atol)

    transition_summary = _transition_contract_summary(
        zarr_arrays,
        gamma=float(args.gamma),
    )
    sampler_checks = _sampler_contract(
        zarr_path,
        zarr_arrays["episode_ends"],
        horizon=int(args.horizon),
        strides=[int(value) for value in args.sampler_strides],
        gamma=float(args.gamma),
    )

    combined_rows = []
    suspicious_rows = []
    for index, ep_id in enumerate(episode_order):
        structural = structural_rows[index]
        gripper = gripper_rows[index]
        zrow = zarr_rows[index]
        start = int(zrow["zarr_start"])
        end = int(zrow["zarr_end_exclusive"])

        if index + 1 < len(zarr_rows):
            next_start = int(zarr_rows[index + 1]["zarr_start"])
            action_reset_jump = float(
                np.linalg.norm(
                    zarr_arrays["action"][end - 1].astype(np.float64)
                    - zarr_arrays["action"][next_start].astype(np.float64)
                )
            )
            state_reset_jump = float(
                np.linalg.norm(
                    zarr_arrays["state"][end - 1].astype(np.float64)
                    - zarr_arrays["state"][next_start].astype(np.float64)
                )
            )
        else:
            action_reset_jump = None
            state_reset_jump = None

        row = {
            **structural,
            **{key: value for key, value in zrow.items() if key != "zarr_episode"},
            **{key: value for key, value in gripper.items() if key != "episode_id"},
            "action_reset_jump_l2_to_next_episode": action_reset_jump,
            "state_reset_jump_l2_to_next_episode": state_reset_jump,
        }
        row["fully_valid"] = bool(
            row["structural_valid"]
            and row["zarr_valid"]
            and row["task_semantic_valid"]
        )

        reasons = []
        if not row["structural_valid"]:
            reasons.append("lerobot_structure")
        if not row["zarr_valid"]:
            reasons.append("zarr_contract")
        if not row["left_valid"]:
            reasons.append("left_action_gripper")
        if not row["right_valid"]:
            reasons.append("right_action_gripper")
        if not row["left_state_valid"]:
            reasons.append("left_state_gripper")
        if not row["right_state_valid"]:
            reasons.append("right_state_gripper")
        row["suspicious_reasons"] = ";".join(reasons)

        combined_rows.append(row)
        if reasons:
            suspicious_rows.append(row)

    counts = {
        "structural_valid": int(sum(row["structural_valid"] for row in combined_rows)),
        "zarr_valid": int(sum(row["zarr_valid"] for row in combined_rows)),
        "left_gripper_valid": int(sum(row["left_valid"] for row in combined_rows)),
        "right_gripper_valid": int(sum(row["right_valid"] for row in combined_rows)),
        "left_state_gripper_valid": int(sum(row["left_state_valid"] for row in combined_rows)),
        "right_state_gripper_valid": int(sum(row["right_state_valid"] for row in combined_rows)),
        "command_semantic_valid": int(sum(row["command_semantic_valid"] for row in combined_rows)),
        "execution_semantic_valid": int(sum(row["execution_semantic_valid"] for row in combined_rows)),
        "task_semantic_valid": int(sum(row["task_semantic_valid"] for row in combined_rows)),
        "fully_valid": int(sum(row["fully_valid"] for row in combined_rows)),
        "suspicious": int(len(suspicious_rows)),
    }

    summary = {
        "lerobot": {
            "root": str(lerobot_root),
            "frames": int(len(hf_dataset)),
            "episodes": int(len(episode_order)),
            "episode_ids_in_appearance_order": episode_order,
            "episode_id_consecutive_0_based": bool(
                episode_order == list(range(len(episode_order)))
            ),
            "episode_length_stats": _stats(expected_lengths),
            "fps": fps,
            "has_frame_index": frame_index is not None,
            "has_timestamp": timestamp is not None,
        },
        "zarr": {
            "path": str(zarr_path),
            **zarr_summary,
            "action_matches_lerobot_reordered": action_match,
            "action_max_abs_diff": action_max_abs,
            "state_matches_lerobot_reordered": state_match,
            "state_max_abs_diff": state_max_abs,
        },
        "gripper_thresholds": {
            "left": vars(left_thresholds),
            "right": vars(right_thresholds),
            "min_dwell": int(args.min_dwell),
            "expected_cycles": int(args.expected_cycles),
            "left_index": int(args.left_gripper_index),
            "right_index": int(args.right_gripper_index),
        },
        "state_gripper_thresholds": {
            "left": vars(left_state_thresholds),
            "right": vars(right_state_thresholds),
            "min_dwell": int(args.min_dwell),
            "expected_cycles": int(args.expected_cycles),
            "left_index": int(args.left_state_gripper_index),
            "right_index": int(args.right_state_gripper_index),
        },
        "action_state_event_lag_frames": {
            "left_close": _stats([
                row["left_close_action_to_state_lag_frames"]
                for row in combined_rows
                if row["left_close_action_to_state_lag_frames"] is not None
            ]),
            "left_reopen": _stats([
                row["left_reopen_action_to_state_lag_frames"]
                for row in combined_rows
                if row["left_reopen_action_to_state_lag_frames"] is not None
            ]),
            "right_close": _stats([
                row["right_close_action_to_state_lag_frames"]
                for row in combined_rows
                if row["right_close_action_to_state_lag_frames"] is not None
            ]),
            "right_reopen": _stats([
                row["right_reopen_action_to_state_lag_frames"]
                for row in combined_rows
                if row["right_reopen_action_to_state_lag_frames"] is not None
            ]),
        },
        "counts": counts,
        "transition_contract": transition_summary,
        "sampler_checks": sampler_checks,
        "converter_contract_note": (
            "Current kuavo_data/CvtRosbag2Lerobot.py processes each selected rosbag file "
            "inside one loop iteration and calls dataset.save_episode() exactly once after "
            "that bag is processed. Therefore, under this converter, one rosbag file maps "
            "to one LeRobot episode unless a bag fails and is skipped."
        ),
        "semantic_contract": (
            "Expected sim_task1 episode: both left and right gripper commands and observed "
            "gripper states each complete exactly one debounced open->closed->open cycle, "
            "corresponding to one commanded and executed grasp/release sequence per hand."
        ),
    }

    _write_csv(output_dir / "episode_integrity.csv", combined_rows)
    _write_csv(output_dir / "suspicious_episodes.csv", suspicious_rows)
    _write_csv(output_dir / "gripper_events.csv", gripper_event_rows)
    _write_csv(output_dir / "sampler_checks.csv", sampler_checks)
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    _print_summary(summary)
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
