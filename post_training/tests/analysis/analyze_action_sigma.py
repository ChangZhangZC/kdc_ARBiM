from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

import numpy as np
import torch
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

for path in (REPO_ROOT, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_action_normalization_stats(checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
    )
    normalizers = [
        step
        for step in preprocessor.steps
        if isinstance(step, NormalizerProcessorStep)
    ]
    if len(normalizers) != 1:
        raise RuntimeError(
            f"Expected exactly one NormalizerProcessorStep, got {len(normalizers)}"
        )

    stats = normalizers[0].stats
    if "action" not in stats:
        raise KeyError("ACT checkpoint normalization stats do not contain 'action'.")
    action_stats = stats["action"]
    if "mean" not in action_stats or "std" not in action_stats:
        raise KeyError("ACT action normalization stats require both mean and std.")

    mean = _to_numpy(action_stats["mean"]).astype(np.float64).reshape(-1)
    std = _to_numpy(action_stats["std"]).astype(np.float64).reshape(-1)
    if np.any(std <= 0):
        raise ValueError("Checkpoint action std must be strictly positive.")
    return mean, std


def _load_episode_ends(root) -> np.ndarray:
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise KeyError("Offline RL Zarr does not contain meta/episode_ends.")
    return np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray) -> np.ndarray:
    return np.asarray(numerator, dtype=np.float64) / np.maximum(
        np.asarray(denominator, dtype=np.float64),
        1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ARBiM action variation and compare it with stochastic ACT sigma "
            "in the checkpoint-normalized action space."
        )
    )
    parser.add_argument("--dataset", required=True, help="Offline RL Zarr path")
    parser.add_argument("--checkpoint", required=True, help="ACT checkpoint directory")
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

    root = zarr.open_group(args.dataset, mode="r")
    if "action" not in root:
        raise KeyError("Offline RL Zarr does not contain action.")
    actions = np.asarray(root["action"][:], dtype=np.float64)
    episode_ends = _load_episode_ends(root)

    if actions.ndim != 2:
        raise ValueError(f"Expected action [N,D], got {actions.shape}")
    if len(actions) == 0:
        raise ValueError("Action dataset is empty.")
    if len(episode_ends) == 0 or int(episode_ends[-1]) != len(actions):
        raise ValueError(
            "episode_ends must be non-empty and terminate at the dataset length."
        )

    num_frames, action_dim = actions.shape
    checkpoint_mean, checkpoint_std = _load_action_normalization_stats(args.checkpoint)
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
        start = int(start)
        end = int(end)
        episode = actions[start:end]
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
    normalized_delta_abs_p95 = np.percentile(
        np.abs(normalized_deltas), 95, axis=0
    )

    sigma_min = math.exp(args.min_log_std)
    sigma_init = math.exp(args.init_log_std)
    sigma_max = math.exp(args.max_log_std)

    raw_sigma_min = sigma_min * checkpoint_std
    raw_sigma_init = sigma_init * checkpoint_std
    raw_sigma_max = sigma_max * checkpoint_std

    # Independent Gaussian noise at adjacent timesteps contributes
    # sqrt(2) * sigma standard deviation to the action difference.
    init_noise_delta_std = math.sqrt(2.0) * sigma_init
    max_noise_delta_std = math.sqrt(2.0) * sigma_max
    init_noise_vs_demo_delta = _safe_ratio(
        init_noise_delta_std,
        normalized_delta_std,
    )
    max_noise_vs_demo_delta = _safe_ratio(
        max_noise_delta_std,
        normalized_delta_std,
    )

    # Reference only: sigma for which independent Gaussian noise would contribute
    # the same adjacent-step delta std as the demonstrations.
    sigma_equal_demo_delta = normalized_delta_std / math.sqrt(2.0)
    log_std_equal_demo_delta = np.log(np.maximum(sigma_equal_demo_delta, 1e-12))

    summary_path = output_dir / "joint_summary.csv"
    summary_fields = [
        "joint",
        "raw_mean",
        "raw_std",
        "raw_min",
        "raw_max",
        "raw_p01",
        "raw_p99",
        "checkpoint_mean",
        "checkpoint_std",
        "normalized_mean",
        "normalized_std",
        "episode_mean_std",
        "episode_std_median",
        "delta_std_raw",
        "delta_abs_median_raw",
        "delta_abs_p95_raw",
        "delta_std_normalized",
        "delta_abs_median_normalized",
        "delta_abs_p95_normalized",
        "sigma_min_normalized",
        "sigma_init_normalized",
        "sigma_max_normalized",
        "sigma_min_raw",
        "sigma_init_raw",
        "sigma_max_raw",
        "init_noise_delta_vs_demo_delta",
        "max_noise_delta_vs_demo_delta",
        "sigma_equal_demo_delta",
        "log_std_equal_demo_delta",
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
            f"{joint:5d} | "
            f"{checkpoint_std[joint]:8.5f} | "
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
