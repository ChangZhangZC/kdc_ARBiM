from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np
import zarr
from safetensors.torch import load_file


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


def _load_action_normalization_stats(
    checkpoint: str,
    processor_dir: str | None,
) -> tuple[np.ndarray, np.ndarray, pathlib.Path, pathlib.Path]:
    resolved_processor_dir = _resolve_processor_dir(checkpoint, processor_dir)
    config_path = resolved_processor_dir / "policy_preprocessor.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    normalizer_step = None
    for step in config.get("steps", []):
        registry_name = step.get("registry_name")
        class_name = str(step.get("class", ""))
        if registry_name == "normalizer_processor" or class_name.endswith("NormalizerProcessorStep"):
            normalizer_step = step
            break

    if normalizer_step is None:
        raise KeyError("policy_preprocessor.json does not contain a normalizer_processor step.")

    state_file = normalizer_step.get("state_file")
    if not state_file:
        raise KeyError("normalizer_processor does not reference a state_file.")

    state_path = resolved_processor_dir / state_file
    if not state_path.is_file():
        raise FileNotFoundError(f"Normalizer state file not found: {state_path}")

    state = load_file(str(state_path), device="cpu")
    required = ("action.mean", "action.std")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(
            f"Normalizer state is missing {missing}. Available keys: {sorted(state.keys())}"
        )

    mean = state["action.mean"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    std = state["action.std"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    if mean.size == 0 or std.size == 0:
        raise ValueError("Checkpoint action normalization stats must be non-empty.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Checkpoint action normalization stats must be finite.")
    if np.any(std <= 0):
        raise ValueError("Checkpoint action std must be strictly positive.")
    return mean, std, resolved_processor_dir, state_path


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


def _split_episodes(actions: np.ndarray, episode_ends: np.ndarray) -> list[np.ndarray]:
    starts = np.concatenate(([0], episode_ends[:-1]))
    episodes = [actions[int(start):int(end)] for start, end in zip(starts, episode_ends)]
    if any(len(episode) == 0 for episode in episodes):
        raise ValueError("Found an empty episode in episode_ends.")
    return episodes


def _normalize(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (actions - mean) / std


def _cross_trajectory_stats(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """aligned: [E, T, D]. Return mean[T,D], std[T,D], pooled_sigma[D]."""
    cross_mean = aligned.mean(axis=0)
    cross_std = aligned.std(axis=0)
    residual = aligned - cross_mean[None]
    pooled_sigma = np.sqrt(np.mean(np.square(residual), axis=(0, 1)))
    return cross_mean, cross_std, pooled_sigma


def _build_frame_aligned(episodes: list[np.ndarray]) -> np.ndarray:
    common_frames = min(len(episode) for episode in episodes)
    return np.stack([episode[:common_frames] for episode in episodes], axis=0)


def _build_phase_aligned(episodes: list[np.ndarray], phase_points: int) -> np.ndarray:
    if phase_points < 2:
        raise ValueError("phase_points must be >= 2.")

    target_phase = np.linspace(0.0, 1.0, phase_points, dtype=np.float64)
    action_dim = episodes[0].shape[1]
    aligned = np.empty((len(episodes), phase_points, action_dim), dtype=np.float64)

    for episode_idx, episode in enumerate(episodes):
        if len(episode) == 1:
            aligned[episode_idx] = episode[0]
            continue
        source_phase = np.linspace(0.0, 1.0, len(episode), dtype=np.float64)
        for joint in range(action_dim):
            aligned[episode_idx, :, joint] = np.interp(
                target_phase,
                source_phase,
                episode[:, joint],
            )
    return aligned


def _write_alignment_stats(
    path: pathlib.Path,
    raw_aligned: np.ndarray,
    norm_aligned: np.ndarray,
    axis_name: str,
) -> None:
    raw_mean, raw_std, _ = _cross_trajectory_stats(raw_aligned)
    norm_mean, norm_std, _ = _cross_trajectory_stats(norm_aligned)
    steps = raw_aligned.shape[1]
    action_dim = raw_aligned.shape[2]

    fields = [
        axis_name,
        "joint",
        "mean_raw",
        "std_raw",
        "mean_normalized",
        "std_normalized",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for step in range(steps):
            axis_value = step if axis_name == "frame" else step / (steps - 1)
            for joint in range(action_dim):
                writer.writerow(
                    {
                        axis_name: axis_value,
                        "joint": joint,
                        "mean_raw": raw_mean[step, joint],
                        "std_raw": raw_std[step, joint],
                        "mean_normalized": norm_mean[step, joint],
                        "std_normalized": norm_std[step, joint],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate stochastic ACT action dispersion by comparing different episodes "
            "at matched frame/phase positions. No temporal a[t+1]-a[t] differences are used."
        )
    )
    parser.add_argument("--dataset", required=True, help="Offline RL Zarr path")
    parser.add_argument("--checkpoint", required=True, help="ACT checkpoint directory")
    parser.add_argument(
        "--processor-dir",
        default=None,
        help=(
            "Directory containing policy_preprocessor.json. If omitted, check the "
            "checkpoint directory first and then its parent run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/action_sigma_analysis",
    )
    parser.add_argument(
        "--phase-points",
        type=int,
        default=101,
        help="Number of normalized trajectory phase positions, including 0%% and 100%%.",
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
        raise ValueError("episode_ends must terminate exactly at the dataset length.")

    num_frames, action_dim = actions.shape
    checkpoint_mean, checkpoint_std, resolved_processor_dir, normalizer_state_path = (
        _load_action_normalization_stats(args.checkpoint, args.processor_dir)
    )
    if checkpoint_mean.shape != (action_dim,) or checkpoint_std.shape != (action_dim,):
        raise ValueError(
            "Checkpoint action normalization shape mismatch: "
            f"mean={checkpoint_mean.shape}, std={checkpoint_std.shape}, "
            f"dataset action_dim={action_dim}"
        )

    episodes = _split_episodes(actions, episode_ends)
    episode_lengths = np.asarray([len(episode) for episode in episodes], dtype=np.int64)

    frame_raw = _build_frame_aligned(episodes)
    frame_norm = _normalize(frame_raw, checkpoint_mean, checkpoint_std)
    frame_mean_raw, frame_std_raw, frame_sigma_raw = _cross_trajectory_stats(frame_raw)
    _, frame_std_norm, frame_sigma_norm = _cross_trajectory_stats(frame_norm)

    phase_raw = _build_phase_aligned(episodes, args.phase_points)
    phase_norm = _normalize(phase_raw, checkpoint_mean, checkpoint_std)
    _, phase_std_raw, phase_sigma_raw = _cross_trajectory_stats(phase_raw)
    _, phase_std_norm, phase_sigma_norm = _cross_trajectory_stats(phase_norm)

    sigma_min = math.exp(args.min_log_std)
    sigma_init = math.exp(args.init_log_std)
    sigma_max = math.exp(args.max_log_std)

    frame_stats_path = output_dir / "frame_joint_stats.csv"
    phase_stats_path = output_dir / "phase_joint_stats.csv"
    _write_alignment_stats(frame_stats_path, frame_raw, frame_norm, "frame")
    _write_alignment_stats(phase_stats_path, phase_raw, phase_norm, "phase")

    summary_path = output_dir / "joint_summary.csv"
    summary_fields = [
        "joint",
        "checkpoint_mean",
        "checkpoint_std",
        "frame_pooled_sigma_raw",
        "frame_pooled_sigma_normalized",
        "frame_log_std",
        "frame_std_median_normalized",
        "frame_std_p90_normalized",
        "frame_std_max_normalized",
        "phase_pooled_sigma_raw",
        "phase_pooled_sigma_normalized",
        "phase_log_std",
        "phase_std_median_normalized",
        "phase_std_p90_normalized",
        "phase_std_max_normalized",
        "current_sigma_min",
        "current_sigma_init",
        "current_sigma_max",
        "init_over_frame_sigma",
        "max_over_frame_sigma",
        "init_over_phase_sigma",
        "max_over_phase_sigma",
    ]

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for joint in range(action_dim):
            writer.writerow(
                {
                    "joint": joint,
                    "checkpoint_mean": checkpoint_mean[joint],
                    "checkpoint_std": checkpoint_std[joint],
                    "frame_pooled_sigma_raw": frame_sigma_raw[joint],
                    "frame_pooled_sigma_normalized": frame_sigma_norm[joint],
                    "frame_log_std": math.log(max(frame_sigma_norm[joint], 1e-12)),
                    "frame_std_median_normalized": np.median(frame_std_norm[:, joint]),
                    "frame_std_p90_normalized": np.percentile(frame_std_norm[:, joint], 90),
                    "frame_std_max_normalized": frame_std_norm[:, joint].max(),
                    "phase_pooled_sigma_raw": phase_sigma_raw[joint],
                    "phase_pooled_sigma_normalized": phase_sigma_norm[joint],
                    "phase_log_std": math.log(max(phase_sigma_norm[joint], 1e-12)),
                    "phase_std_median_normalized": np.median(phase_std_norm[:, joint]),
                    "phase_std_p90_normalized": np.percentile(phase_std_norm[:, joint], 90),
                    "phase_std_max_normalized": phase_std_norm[:, joint].max(),
                    "current_sigma_min": sigma_min,
                    "current_sigma_init": sigma_init,
                    "current_sigma_max": sigma_max,
                    "init_over_frame_sigma": sigma_init / max(frame_sigma_norm[joint], 1e-12),
                    "max_over_frame_sigma": sigma_max / max(frame_sigma_norm[joint], 1e-12),
                    "init_over_phase_sigma": sigma_init / max(phase_sigma_norm[joint], 1e-12),
                    "max_over_phase_sigma": sigma_max / max(phase_sigma_norm[joint], 1e-12),
                }
            )

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {pathlib.Path(args.checkpoint).expanduser().resolve()}")
    print(f"Processor dir: {resolved_processor_dir}")
    print(f"Normalizer state: {normalizer_state_path}")
    print(f"Frames: {num_frames}")
    print(f"Episodes: {len(episodes)}")
    print(f"Episode length: min={episode_lengths.min()}, median={np.median(episode_lengths):.1f}, max={episode_lengths.max()}")
    print(f"Action dim: {action_dim}")
    print(f"Exact-frame comparison uses common prefix: 0..{frame_raw.shape[1] - 1}")
    print(f"Phase comparison uses {args.phase_points} normalized positions from 0%% to 100%%")
    print()
    print("Current stochastic ACT normalized sigma:")
    print(f"  min : exp({args.min_log_std:.4f}) = {sigma_min:.6f}")
    print(f"  init: exp({args.init_log_std:.4f}) = {sigma_init:.6f}")
    print(f"  max : exp({args.max_log_std:.4f}) = {sigma_max:.6f}")
    print()
    header = (
        "joint | frame_sigma | frame_logstd | phase_sigma | phase_logstd | "
        "init/frame | max/frame"
    )
    print(header)
    print("-" * len(header))
    for joint in range(action_dim):
        print(
            f"{joint:5d} | "
            f"{frame_sigma_norm[joint]:11.5f} | "
            f"{math.log(max(frame_sigma_norm[joint], 1e-12)):12.5f} | "
            f"{phase_sigma_norm[joint]:11.5f} | "
            f"{math.log(max(phase_sigma_norm[joint], 1e-12)):12.5f} | "
            f"{sigma_init / max(frame_sigma_norm[joint], 1e-12):10.3f} | "
            f"{sigma_max / max(frame_sigma_norm[joint], 1e-12):9.3f}"
        )

    print()
    print("Interpretation:")
    print("  frame_sigma: std across episodes at the same absolute frame, pooled over common frames.")
    print("  phase_sigma: std across episodes at the same normalized trajectory phase, pooled over phase.")
    print("  Neither metric uses a[t+1] - a[t].")
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {frame_stats_path}")
    print(f"Saved: {phase_stats_path}")


if __name__ == "__main__":
    main()
