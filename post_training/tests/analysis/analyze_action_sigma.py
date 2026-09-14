from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import (  # noqa: E402
    CustomLeRobotDataset,
    filter_depth_policy_features,
)
from kuavo_train.wrapper.policy.act.ACTConfigWrapper import (  # noqa: E402
    CustomACTConfigWrapper,
)
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import (  # noqa: E402
    CustomACTPolicyWrapper,
)


def _resolve_processor_dir(
    checkpoint: str,
    processor_dir: str | None,
) -> pathlib.Path:
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


def _load_action_normalization_std(
    processor_dir: pathlib.Path,
) -> tuple[np.ndarray, pathlib.Path]:
    config_path = processor_dir / "policy_preprocessor.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    normalizer_step = None
    for step in config.get("steps", []):
        registry_name = step.get("registry_name")
        class_name = str(step.get("class", ""))
        if (
            registry_name == "normalizer_processor"
            or class_name.endswith("NormalizerProcessorStep")
        ):
            normalizer_step = step
            break
    if normalizer_step is None:
        raise KeyError(
            "policy_preprocessor.json does not contain a normalizer_processor step."
        )

    state_file = normalizer_step.get("state_file")
    if not state_file:
        raise KeyError("normalizer_processor does not reference a state_file.")
    state_path = processor_dir / state_file
    if not state_path.is_file():
        raise FileNotFoundError(f"Normalizer state file not found: {state_path}")

    state = load_file(str(state_path), device="cpu")
    key = "action.std"
    if key not in state:
        raise KeyError(
            f"Normalizer state is missing {key!r}. Available keys: {sorted(state.keys())}"
        )
    action_std = state[key].detach().cpu().numpy().astype(np.float64).reshape(-1)
    if (
        action_std.size == 0
        or not np.all(np.isfinite(action_std))
        or np.any(action_std <= 0)
    ):
        raise ValueError("Checkpoint action std must be finite and strictly positive.")
    return action_std, state_path


def _build_delta_timestamps(
    dataset_metadata,
    policy_cfg,
) -> dict[str, list[float]] | None:
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    active_input_keys = set(policy_cfg.input_features)
    active_output_keys = set(policy_cfg.output_features)
    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if key in active_input_keys and "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif key in active_output_keys and "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]
    return delta_timestamps if delta_timestamps else None


def _build_dataset(
    lerobot_root: pathlib.Path,
    repo_id: str,
    policy_cfg,
) -> tuple[CustomLeRobotDataset, LeRobotDatasetMetadata, set[str]]:
    metadata = LeRobotDatasetMetadata(repo_id, root=lerobot_root)
    features = dataset_to_policy_features(metadata.features)
    dataset_inputs = {
        key: feature
        for key, feature in features.items()
        if feature.type is not FeatureType.ACTION
    }
    use_depth = bool(getattr(policy_cfg, "use_depth", False))
    _, excluded_depth_keys = filter_depth_policy_features(
        dataset_inputs,
        use_depth=use_depth,
    )
    delta_timestamps = _build_delta_timestamps(metadata, policy_cfg)
    dataset = CustomLeRobotDataset(
        repo_id,
        root=lerobot_root,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        excluded_keys=excluded_depth_keys,
    )
    return dataset, metadata, excluded_depth_keys


def _select_indices(
    dataset_length: int,
    stride: int,
    max_samples: int | None,
) -> np.ndarray:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    indices = np.arange(0, dataset_length, stride, dtype=np.int64)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}")
        if len(indices) > max_samples:
            positions = np.linspace(
                0,
                len(indices) - 1,
                num=max_samples,
                dtype=np.int64,
            )
            indices = indices[positions]
    return np.unique(indices)


def _move_tensor_values(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _finalize_stats(
    count: int,
    sum_residual: np.ndarray,
    sum_square: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("No valid residual samples were accumulated.")
    mean = sum_residual / count
    second_moment = sum_square / count
    std = np.sqrt(np.maximum(second_moment - np.square(mean), 0.0))
    sigma_mle = np.sqrt(np.maximum(second_moment, 0.0))
    return mean, std, sigma_mle


def _correlation_from_moments(
    count: int,
    sum_residual: np.ndarray,
    sum_outer: np.ndarray,
) -> np.ndarray:
    mean = sum_residual / count
    covariance = sum_outer / count - np.outer(mean, mean)
    diagonal = np.maximum(np.diag(covariance), 0.0)
    denom = np.sqrt(np.outer(diagonal, diagonal))
    correlation = np.zeros_like(covariance)
    np.divide(covariance, denom, out=correlation, where=denom > 1e-12)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _write_joint_summary(
    path: pathlib.Path,
    action_std: np.ndarray,
    pooled_mean: np.ndarray,
    pooled_std: np.ndarray,
    pooled_sigma_mle: np.ndarray,
    h0_mean: np.ndarray,
    h0_std: np.ndarray,
    h0_sigma_mle: np.ndarray,
    sigma_min: float,
    sigma_init: float,
    sigma_max: float,
) -> None:
    fields = [
        "joint",
        "checkpoint_action_std",
        "pooled_residual_mean_normalized",
        "pooled_residual_std_normalized",
        "pooled_sigma_mle_normalized",
        "pooled_log_sigma_mle",
        "pooled_sigma_mle_raw",
        "h0_residual_mean_normalized",
        "h0_residual_std_normalized",
        "h0_sigma_mle_normalized",
        "h0_log_sigma_mle",
        "h0_sigma_mle_raw",
        "current_sigma_min",
        "current_sigma_init",
        "current_sigma_max",
        "init_over_pooled_sigma_mle",
        "max_over_pooled_sigma_mle",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for joint in range(len(action_std)):
            pooled_sigma = max(float(pooled_sigma_mle[joint]), 1e-12)
            h0_sigma = max(float(h0_sigma_mle[joint]), 1e-12)
            writer.writerow(
                {
                    "joint": joint,
                    "checkpoint_action_std": action_std[joint],
                    "pooled_residual_mean_normalized": pooled_mean[joint],
                    "pooled_residual_std_normalized": pooled_std[joint],
                    "pooled_sigma_mle_normalized": pooled_sigma_mle[joint],
                    "pooled_log_sigma_mle": math.log(pooled_sigma),
                    "pooled_sigma_mle_raw": pooled_sigma_mle[joint] * action_std[joint],
                    "h0_residual_mean_normalized": h0_mean[joint],
                    "h0_residual_std_normalized": h0_std[joint],
                    "h0_sigma_mle_normalized": h0_sigma_mle[joint],
                    "h0_log_sigma_mle": math.log(h0_sigma),
                    "h0_sigma_mle_raw": h0_sigma_mle[joint] * action_std[joint],
                    "current_sigma_min": sigma_min,
                    "current_sigma_init": sigma_init,
                    "current_sigma_max": sigma_max,
                    "init_over_pooled_sigma_mle": sigma_init / pooled_sigma,
                    "max_over_pooled_sigma_mle": sigma_max / pooled_sigma,
                }
            )


def _write_horizon_summary(
    path: pathlib.Path,
    horizon_count: np.ndarray,
    horizon_sum: np.ndarray,
    horizon_square: np.ndarray,
) -> None:
    fields = [
        "chunk_position",
        "joint",
        "count",
        "residual_mean_normalized",
        "residual_std_normalized",
        "sigma_mle_normalized",
        "log_sigma_mle",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for h in range(len(horizon_count)):
            if int(horizon_count[h]) == 0:
                continue
            mean, std, sigma_mle = _finalize_stats(
                int(horizon_count[h]),
                horizon_sum[h],
                horizon_square[h],
            )
            for joint in range(horizon_sum.shape[1]):
                sigma = max(float(sigma_mle[joint]), 1e-12)
                writer.writerow(
                    {
                        "chunk_position": h,
                        "joint": joint,
                        "count": int(horizon_count[h]),
                        "residual_mean_normalized": mean[joint],
                        "residual_std_normalized": std[joint],
                        "sigma_mle_normalized": sigma_mle[joint],
                        "log_sigma_mle": math.log(sigma),
                    }
                )


def _write_correlation(path: pathlib.Path, correlation: np.ndarray) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["joint", *[f"joint_{i}" for i in range(correlation.shape[1])]])
        for joint, row in enumerate(correlation):
            writer.writerow([joint, *row.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate stochastic ACT sigma from true ACT residuals on the original "
            "LeRobot dataset: residual = normalized demo action - deterministic ACT mean. "
            "No Zarr/latent cache and no a[t+1]-a[t] proxy are used."
        )
    )
    parser.add_argument(
        "--lerobot-root",
        required=True,
        help="Original LeRobot dataset root used to train/evaluate ACT.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo_id. Defaults to the dataset root directory name.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Deterministic ACT checkpoint directory, e.g. epoch120.",
    )
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Analyze every Nth LeRobot anchor frame. Default 1 uses all frames.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional cap on anchor observations. Samples are spread across the full "
            "dataset; useful for a quick smoke run before the full analysis."
        ),
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
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device!r}, but CUDA is not available.")

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    lerobot_root = pathlib.Path(args.lerobot_root).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"ACT checkpoint not found: {checkpoint}")
    if not lerobot_root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root not found: {lerobot_root}")

    processor_dir = _resolve_processor_dir(str(checkpoint), args.processor_dir)
    action_std, normalizer_state_path = _load_action_normalization_std(processor_dir)

    policy_cfg = CustomACTConfigWrapper.from_pretrained(str(checkpoint))
    policy_cfg.device = str(device)
    action_dim = int(policy_cfg.action_feature.shape[0])
    chunk_size = int(policy_cfg.chunk_size)
    if action_std.shape != (action_dim,):
        raise ValueError(
            f"Normalizer action std shape {action_std.shape} != action_dim={action_dim}"
        )

    policy = CustomACTPolicyWrapper.from_pretrained(
        str(checkpoint),
        config=policy_cfg,
        strict=True,
    ).to(device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(processor_dir),
        config_filename="policy_preprocessor.json",
    )

    repo_id = args.repo_id or lerobot_root.name
    dataset, metadata, excluded_depth_keys = _build_dataset(
        lerobot_root,
        repo_id,
        policy_cfg,
    )
    indices = _select_indices(len(dataset), args.stride, args.max_samples)
    subset = Subset(dataset, indices.tolist())
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_count = 0
    pooled_sum = np.zeros(action_dim, dtype=np.float64)
    pooled_square = np.zeros(action_dim, dtype=np.float64)
    pooled_outer = np.zeros((action_dim, action_dim), dtype=np.float64)
    horizon_count = np.zeros(chunk_size, dtype=np.int64)
    horizon_sum = np.zeros((chunk_size, action_dim), dtype=np.float64)
    horizon_square = np.zeros((chunk_size, action_dim), dtype=np.float64)

    processed_anchors = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="ACT residual inference"):
            processed = preprocessor(batch)
            processed = _move_tensor_values(processed, device)

            if ACTION not in processed:
                raise KeyError("Preprocessed LeRobot batch is missing 'action'.")
            target = processed[ACTION]
            if target.ndim != 3:
                raise ValueError(
                    f"Expected normalized action target [B,H,D], got {tuple(target.shape)}"
                )
            if target.shape[1] != chunk_size or target.shape[2] != action_dim:
                raise ValueError(
                    "Action target shape does not match ACT contract: "
                    f"target={tuple(target.shape)}, expected H={chunk_size}, D={action_dim}"
                )

            action_is_pad = processed.get("action_is_pad")
            if action_is_pad is None:
                valid = torch.ones(
                    target.shape[:2],
                    dtype=torch.bool,
                    device=target.device,
                )
            else:
                valid = ~action_is_pad.to(device=target.device, dtype=torch.bool)
                if tuple(valid.shape) != tuple(target.shape[:2]):
                    raise ValueError(
                        f"action_is_pad shape {tuple(valid.shape)} != {tuple(target.shape[:2])}"
                    )

            policy_input = {
                key: value
                for key, value in processed.items()
                if key not in (ACTION, "action_is_pad")
            }
            mu = policy.predict_action_chunk(policy_input)
            if tuple(mu.shape) != tuple(target.shape):
                raise ValueError(
                    f"ACT mean shape {tuple(mu.shape)} != target shape {tuple(target.shape)}"
                )

            residual = target - mu
            residual_np = residual.detach().float().cpu().numpy().astype(np.float64)
            valid_np = valid.detach().cpu().numpy()

            flat = residual_np[valid_np]
            if flat.size:
                pooled_count += int(flat.shape[0])
                pooled_sum += flat.sum(axis=0)
                pooled_square += np.square(flat).sum(axis=0)
                pooled_outer += flat.T @ flat

            for h in range(chunk_size):
                mask_h = valid_np[:, h]
                if not np.any(mask_h):
                    continue
                residual_h = residual_np[mask_h, h, :]
                horizon_count[h] += residual_h.shape[0]
                horizon_sum[h] += residual_h.sum(axis=0)
                horizon_square[h] += np.square(residual_h).sum(axis=0)

            processed_anchors += int(target.shape[0])

    pooled_mean, pooled_std, pooled_sigma_mle = _finalize_stats(
        pooled_count,
        pooled_sum,
        pooled_square,
    )
    h0_mean, h0_std, h0_sigma_mle = _finalize_stats(
        int(horizon_count[0]),
        horizon_sum[0],
        horizon_square[0],
    )
    correlation = _correlation_from_moments(
        pooled_count,
        pooled_sum,
        pooled_outer,
    )

    sigma_min = math.exp(args.min_log_std)
    sigma_init = math.exp(args.init_log_std)
    sigma_max = math.exp(args.max_log_std)

    joint_summary_path = output_dir / "residual_joint_summary.csv"
    horizon_summary_path = output_dir / "residual_horizon_summary.csv"
    correlation_path = output_dir / "residual_correlation.csv"
    _write_joint_summary(
        joint_summary_path,
        action_std,
        pooled_mean,
        pooled_std,
        pooled_sigma_mle,
        h0_mean,
        h0_std,
        h0_sigma_mle,
        sigma_min,
        sigma_init,
        sigma_max,
    )
    _write_horizon_summary(
        horizon_summary_path,
        horizon_count,
        horizon_sum,
        horizon_square,
    )
    _write_correlation(correlation_path, correlation)

    print()
    print(f"LeRobot dataset: {lerobot_root}")
    print(f"repo_id: {repo_id}")
    print(f"Frames in dataset: {len(dataset)}")
    print(f"Anchor observations analyzed: {processed_anchors}")
    print(f"Valid chunk residual vectors: {pooled_count}")
    print(f"ACT checkpoint: {checkpoint}")
    print(f"Processor dir: {processor_dir}")
    print(f"Normalizer state: {normalizer_state_path}")
    print(f"Device: {device}")
    print(f"Chunk size: {chunk_size}")
    print(f"Action dim: {action_dim}")
    print(f"Excluded depth keys: {sorted(excluded_depth_keys)}")
    print()
    print("Residual definition:")
    print("  r[t,h,d] = normalized_demo_action[t+h,d] - ACT_mean(s_t)[h,d]")
    print("  The ACT forward pass receives observations only; demo actions are removed.")
    print("  pooled_sigma_mle = sqrt(mean(r^2)) across valid t and all chunk positions h.")
    print("  h0_sigma_mle uses only h=0 and is the closest analogue to the current action.")
    print()
    print("Current stochastic ACT normalized sigma:")
    print(f"  min : exp({args.min_log_std:.4f}) = {sigma_min:.6f}")
    print(f"  init: exp({args.init_log_std:.4f}) = {sigma_init:.6f}")
    print(f"  max : exp({args.max_log_std:.4f}) = {sigma_max:.6f}")
    print()
    header = (
        "joint | pooled_sigma | pooled_logstd | h0_sigma | h0_logstd | "
        "residual_mean | init/pooled | max/pooled"
    )
    print(header)
    print("-" * len(header))
    for joint in range(action_dim):
        pooled_sigma = max(float(pooled_sigma_mle[joint]), 1e-12)
        h0_sigma = max(float(h0_sigma_mle[joint]), 1e-12)
        print(
            f"{joint:5d} | "
            f"{pooled_sigma_mle[joint]:12.5f} | "
            f"{math.log(pooled_sigma):13.5f} | "
            f"{h0_sigma_mle[joint]:8.5f} | "
            f"{math.log(h0_sigma):9.5f} | "
            f"{pooled_mean[joint]:13.5f} | "
            f"{sigma_init / pooled_sigma:11.3f} | "
            f"{sigma_max / pooled_sigma:10.3f}"
        )

    pairs = []
    for i in range(action_dim):
        for j in range(i + 1, action_dim):
            pairs.append((abs(float(correlation[i, j])), i, j, float(correlation[i, j])))
    pairs.sort(reverse=True)
    print()
    print("Strongest pooled residual correlations (diagnostic for independent-Gaussian assumption):")
    for _, i, j, value in pairs[: min(8, len(pairs))]:
        print(f"  joint {i:2d} <-> joint {j:2d}: corr={value:+.4f}")

    print()
    print(f"Saved: {joint_summary_path}")
    print(f"Saved: {horizon_summary_path}")
    print(f"Saved: {correlation_path}")


if __name__ == "__main__":
    main()
