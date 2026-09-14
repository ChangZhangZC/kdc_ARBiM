from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np
import zarr


def _resolve_processor_dir(checkpoint: str, processor_dir: str | None) -> pathlib.Path:
    if processor_dir is not None:
        candidates = [pathlib.Path(processor_dir).expanduser().resolve()]
    else:
        checkpoint_path = pathlib.Path(checkpoint).expanduser().resolve()
        candidates = [checkpoint_path, checkpoint_path.parent]

    checked = []
    for candidate in candidates:
        config_path = candidate / "policy_preprocessor.json"
        checked.append(str(config_path))
        if config_path.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find policy_preprocessor.json. Checked:\n- " + "\n- ".join(checked)
    )


def _find_action_stats(node):
    if isinstance(node, dict):
        stats = node.get("stats")
        if isinstance(stats, dict):
            action_stats = stats.get("action")
            if (
                isinstance(action_stats, dict)
                and "mean" in action_stats
                and "std" in action_stats
            ):
                return action_stats
        for value in node.values():
            found = _find_action_stats(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_action_stats(value)
            if found is not None:
                return found
    return None


def _as_numeric_array(value, name: str) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("data", "values", "value"):
            if key in value:
                value = value[key]
                break
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not decode action normalization {name} from policy_preprocessor.json"
        ) from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"Action normalization {name} must be finite and non-empty.")
    return array


def _load_action_normalization_stats(
    checkpoint: str,
    processor_dir: str | None,
) -> tuple[np.ndarray, np.ndarray, pathlib.Path]:
    resolved_processor_dir = _resolve_processor_dir(checkpoint, processor_dir)
    config_path = resolved_processor_dir / "policy_preprocessor.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    action_stats = _find_action_stats(config)
    if action_stats is None:
        raise KeyError(
            "Could not find stats.action.mean/std in policy_preprocessor.json."
        )

    mean = _as_numeric_array(action_stats["mean"], "mean")
    std = _as_numeric_array(action_stats["std"], "std")
    if np.any(std <= 0):
        raise ValueError("Checkpoint action std must be strictly positive.")
    return mean, std, resolved_processor_dir


def _load_zarr_arrays(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(dataset_path, mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError("Offline RL Zarr must contain top-level 'data' and 'meta' groups.")

    data = root["data"]
    meta = root["meta"]
    if "action" not in data:
        raise KeyError("Offline RL Zarr does not contain data/action.")
    if "episode_ends" not in meta:
        raise KeyError("Offline RL Zarr does not contain meta/episode_ends.")

    actions = np.asarray(data["action"][:], dtype=np.float64)
    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    return actions, episode_ends


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray) -> np.ndarray:
    return np.asarray(numerator, dtype=np.float64) / np.maximum(
        np.asarray(denominator, dtype=np.float64),
        1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze action variation in ARBiM Offline RL data and compare it with "
            "stochastic ACT sigma in checkpoint-normalized action space."
        )
    )
    parser.add_argument("--dataset", required=True, help="Offline RL Zarr path")
    parser.add_argument("--checkpoint", required=True, help="ACT checkpoint directory")
    parser.add_argument(
        "--processor-dir",
        default=None,
        help=(
            "Directory containing policy_preprocessor.json. If omitted, the script "
            "checks the checkpoint directory first and then its parent run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/action_sigma_analysis",
    )
    parser.add_argument("--init-log-std", type=float, default=-3.5)
    parser.add_argument("--min-log-std", type=float, default=-5.0)
    parser.add_argument("--max-log-std", type=float, default=-2.3)
    args = parser.parse_args()

    if not args.min_log_std < args.init_log_std < args.max_log_std:
        raise ValueError(
            "Expected min_log_std < init_log_std < max_log_std, got "
            f"{args.min_log_std} < {args.init_log_std} < {args.max_log_std}"
        )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actions, episode_ends = _load_zarr_arrays(args.dataset)
    if actions.ndim != 2:
        raise ValueError(f"Expected action [N,D], got {actions.shape}")
    if len(actions) == 0:
        raise ValueError("Action dataset is empty.")
    if len(episode_ends) == 0 or int(episode_ends[-1]) != len(actions):
        raise ValueError(
            "episode_ends must be non-empty and terminate at the dataset length."
        )

    num_frames, action_dim = actions.shape
    checkpoint_mean, checkpoint_std, resolved_processor_dir = (
        _load_action_normalization_stats(args.checkpoint, args.processor_dir)
    )
    if checkpoint_mean.shape != (action_dim,) or checkpoint_std.shape != (action_dim,):
        raise ValueError(
            "Checkpoint action normalization shape mismatch: "
            f"mean={checkpoint_mean.shape}, std={checkpoint_std.shape}, "
            f"dataset action_dim={action_dim}"
        )

    normalized_actions = (actions - checkpoint_mean[None]) / checkpoint_std[None]
    raw_mean = actions.mean(axis=0)
    raw_std = actions.std(axis=0)
    raw_min = actions.min(axis=0)
    raw_max = actions.max(axis=0)
    raw_p01 = np.percentile(actions, 1, axis=0)
    raw_p99 = np.percentile(actions, 99, axis=0)
    normalized_mean = normalized_actions.mean(axis=0)
    normalized_std = normalized_actions.std(axis=0)

    starts = np.concatenate(([0], episode_ends[:-1]))
    episode_means = []
    episode_stds = []
    episode_rows = []
    raw_deltas = []
    normalized_deltas = []

    for episode_idx, (start, end) in enumerate(zip(starts, episode_ends)):
        episode = actions[int(start):int(end)]
        if len(episode) == 0:
            continue

        ep_mean = episode.mean(axis=0)
        ep_std = episode.std(axis=0)
        episode_means.append(ep_mean)
        episode_stds.append(ep_std)
        for joint in range(action_dim):
            episode_rows.append(
                {
                    "episode": episode_idx,
                    "joint": joint,
                    "frames": len(episode),
                    "mean": ep_mean[joint],
                    "std": ep_std[joint],
                    "min": episode[:, joint].min(),
                    "max": episode[:, joint].max(),
                }
            )

        if len(episode) > 1:
            delta = np.diff(episode, axis=0)
            raw_deltas.append(delta)
            normalized_deltas.append(delta / checkpoint_std[None])

    if not raw_deltas:
        raise ValueError("No episode contains at least two frames; cannot compute deltas.")

    episode_means = np.asarray(episode_means, dtype=np.float64)
    episode_stds = np.asarray(episode_stds, dtype=np.float64)
    raw_deltas = np.concatenate(raw_deltas, axis=0)
    normalized_deltas = np.concatenate(normalized_deltas, axis=0)

    raw_delta_std = raw_deltas.std(axis=0)
    raw_delta_abs_median = np.median(np.abs(raw_deltas), axis=0)
    raw_delta_abs_p95 = np.percentile(np.abs(raw_deltas), 95, axis=0)
    normalized_delta_std = normalized_deltas.std(axis=0)
    normalized_delta_abs_median = np.median(np.abs(normalized_deltas), axis=0)
    normalized_delta_abs_p95 = np.percentile(np.abs(normalized_deltas), 95, axis=0)

    sigma_min = math.exp(args.min_log_std)
    sigma_init = math.exp(args.init_log_std)
    sigma_max = math.exp(args.max_log_std)
    raw_sigma_min = sigma_min * checkpoint_std
    raw_sigma_init = sigma_init * checkpoint_std
    raw_sigma_max = sigma_max * checkpoint_std

    init_noise_delta_std = math.sqrt(2.0) * sigma_init
    max_noise_delta_std = math.sqrt(2.0) * sigma_max
    init_noise_vs_demo_delta = _safe_ratio(init_noise_delta_std, normalized_delta_std)
    max_noise_vs_demo_delta = _safe_ratio(max_noise_delta_std, normalized_delta_std)
    sigma_equal_demo_delta = normalized_delta_std / math.sqrt(2.0)
    log_std_equal_demo_delta = np.log(np.maximum(sigma_equal_demo_delta, 1e-12))

    summary_path = output_dir / "joint_summary.csv"
    summary_fields = [
        "joint", "raw_mean", "raw_std", "raw_min", "raw_max", "raw_p01", "raw_p99",
        "checkpoint_mean", "checkpoint_std", "normalized_mean", "normalized_std",
        "episode_mean_std", "episode_std_median", "delta_std_raw",
        "delta_abs_median_raw", "delta_abs_p95_raw", "delta_std_normalized",
        "delta_abs_median_normalized", "delta_abs_p95_normalized",
        "sigma_min_normalized", "sigma_init_normalized", "sigma_max_normalized",
        "sigma_min_raw", "sigma_init_raw", "sigma_max_raw",
        "init_noise_delta_vs_demo_delta", "max_noise_delta_vs_demo_delta",
        "sigma_equal_demo_delta", "log_std_equal_demo_delta",
    ]

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for joint in range(action_dim):
            writer.writerow(
                {
                    "joint": joint,
                    "raw_mean": raw_mean[joint],
                    "raw_std": raw_std[joint],
                    "raw_min": raw_min[joint],
                    "raw_max": raw_max[joint],
                    "raw_p01": raw_p01[joint],
                    "raw_p99": raw_p99[joint],
                    "checkpoint_mean": checkpoint_mean[joint],
                    "checkpoint_std": checkpoint_std[joint],
                    "normalized_mean": normalized_mean[joint],
                    "normalized_std": normalized_std[joint],
                    "episode_mean_std": episode_means[:, joint].std(),
                    "episode_std_median": np.median(episode_stds[:, joint]),
                    "delta_std_raw": raw_delta_std[joint],
                    "delta_abs_median_raw": raw_delta_abs_median[joint],
                    "delta_abs_p95_raw": raw_delta_abs_p95[joint],
                    "delta_std_normalized": normalized_delta_std[joint],
                    "delta_abs_median_normalized": normalized_delta_abs_median[joint],
                    "delta_abs_p95_normalized": normalized_delta_abs_p95[joint],
                    "sigma_min_normalized": sigma_min,
                    "sigma_init_normalized": sigma_init,
                    "sigma_max_normalized": sigma_max,
                    "sigma_min_raw": raw_sigma_min[joint],
                    "sigma_init_raw": raw_sigma_init[joint],
                    "sigma_max_raw": raw_sigma_max[joint],
                    "init_noise_delta_vs_demo_delta": init_noise_vs_demo_delta[joint],
                    "max_noise_delta_vs_demo_delta": max_noise_vs_demo_delta[joint],
                    "sigma_equal_demo_delta": sigma_equal_demo_delta[joint],
                    "log_std_equal_demo_delta": log_std_equal_demo_delta[joint],
                }
            )

    episode_path = output_dir / "episode_joint_stats.csv"
    episode_fields = ["episode", "joint", "frames", "mean", "std", "min", "max"]
    with episode_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(episode_rows)

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {pathlib.Path(args.checkpoint).expanduser().resolve()}")
    print(f"Processor dir: {resolved_processor_dir}")
    print(f"Frames: {num_frames}")
    print(f"Episodes: {len(episode_ends)}")
    print(f"Action dim: {action_dim}")
    print()
    print("Stochastic ACT normalized sigma:")
    print(f"  min : exp({args.min_log_std:.4f}) = {sigma_min:.6f}")
    print(f"  init: exp({args.init_log_std:.4f}) = {sigma_init:.6f}")
    print(f"  max : exp({args.max_log_std:.4f}) = {sigma_max:.6f}")
    print()
    header = (
        "joint | ckpt_std | delta_std_norm | init/demo | max/demo | "
        "sigma_equal_delta | log_std_equal"
    )
    print(header)
    print("-" * len(header))
    for joint in range(action_dim):
        print(
            f"{joint:5d} | {checkpoint_std[joint]:8.5f} | "
            f"{normalized_delta_std[joint]:14.5f} | "
            f"{init_noise_vs_demo_delta[joint]:9.3f} | "
            f"{max_noise_vs_demo_delta[joint]:8.3f} | "
            f"{sigma_equal_demo_delta[joint]:17.5f} | "
            f"{log_std_equal_demo_delta[joint]:13.5f}"
        )
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {episode_path}")


if __name__ == "__main__":
    main()
