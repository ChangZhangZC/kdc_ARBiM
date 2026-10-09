from __future__ import annotations

import argparse
import csv
import hashlib
import math
import pathlib
import sys
from collections import defaultdict

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


def _resolve_processor_dir(checkpoint: pathlib.Path, explicit: str | None) -> pathlib.Path:
    candidates = (
        [pathlib.Path(explicit).expanduser().resolve()]
        if explicit is not None
        else [checkpoint, checkpoint.parent]
    )
    checked = []
    for candidate in candidates:
        required = (
            candidate / "policy_preprocessor.json",
            candidate / "policy_postprocessor.json",
        )
        checked.extend(str(path) for path in required)
        if all(path.is_file() for path in required):
            return candidate
    raise FileNotFoundError(
        "Could not resolve ACT processor directory. Checked:\n- " + "\n- ".join(checked)
    )


def _processor_fingerprint(processor_dir: pathlib.Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        [
            *processor_dir.glob("policy_preprocessor*"),
            *processor_dir.glob("policy_postprocessor*"),
        ],
        key=lambda path: path.name,
    )
    for path in files:
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_action_norm(processor_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    import json

    config_path = processor_dir / "policy_preprocessor.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    normalizer = None
    for step in config.get("steps", []):
        if step.get("registry_name") == "normalizer_processor" or str(
            step.get("class", "")
        ).endswith("NormalizerProcessorStep"):
            normalizer = step
            break
    if normalizer is None or not normalizer.get("state_file"):
        raise KeyError("policy_preprocessor.json does not define a normalizer state_file")
    state_path = processor_dir / normalizer["state_file"]
    state = load_file(str(state_path), device="cpu")
    for key in ("action.mean", "action.std"):
        if key not in state:
            raise KeyError(f"Normalizer state is missing {key!r}")
    mean = state["action.mean"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    std = state["action.std"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise ValueError("action.std must be finite and strictly positive")
    return mean, std


def _build_delta_timestamps(metadata, policy_cfg) -> dict[str, list[float]] | None:
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None
    active_input_keys = set(policy_cfg.input_features)
    active_output_keys = set(policy_cfg.output_features)
    result = {}
    for key in metadata.info["features"]:
        if key in active_input_keys and "observation" in key and obs_indices is not None:
            result[key] = [index / metadata.fps for index in obs_indices]
        elif key in active_output_keys and "action" in key and act_indices is not None:
            result[key] = [index / metadata.fps for index in act_indices]
    return result or None


def _build_dataset(root: pathlib.Path, repo_id: str, policy_cfg):
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    features = dataset_to_policy_features(metadata.features)
    inputs = {
        key: feature
        for key, feature in features.items()
        if feature.type is not FeatureType.ACTION
    }
    _, excluded_depth = filter_depth_policy_features(
        inputs,
        use_depth=bool(getattr(policy_cfg, "use_depth", False)),
    )
    dataset = CustomLeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=_build_delta_timestamps(metadata, policy_cfg),
        image_transforms=None,
        excluded_keys=excluded_depth,
    )
    return dataset


def _select_indices(length: int, stride: int, max_samples: int | None) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    indices = np.arange(0, length, stride, dtype=np.int64)
    if max_samples is not None and len(indices) > max_samples:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        positions = np.linspace(0, len(indices) - 1, max_samples, dtype=np.int64)
        indices = indices[positions]
    return np.unique(indices)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _parameter_group(name: str) -> str:
    if "action_head" in name:
        return "action_head"
    if "decoder_pos_embed" in name:
        return "decoder_pos_embed"
    if "decoder" in name:
        return "decoder"
    if "backbone" in name or "encoder" in name:
        return "observation_encoder"
    return "other"


def _safe_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.reshape(-1).double()
    second = second.reshape(-1).double()
    denom = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denom) <= 1e-30:
        return float("nan")
    return float(torch.dot(first, second) / denom)


def _parameter_drift(il_policy, rl_policy, output_dir: pathlib.Path) -> dict[str, dict[str, float]]:
    il_state = il_policy.state_dict()
    rl_state = rl_policy.state_dict()
    if set(il_state) != set(rl_state):
        missing = sorted(set(il_state) - set(rl_state))
        unexpected = sorted(set(rl_state) - set(il_state))
        raise RuntimeError(
            f"IL/Post-RL parameter key mismatch: missing={missing}, unexpected={unexpected}"
        )

    detail_path = output_dir / "parameter_drift.csv"
    group_acc = defaultdict(
        lambda: {
            "numel": 0,
            "base_sq": 0.0,
            "diff_sq": 0.0,
            "abs_sum": 0.0,
            "max_abs": 0.0,
            "dot": 0.0,
            "il_sq": 0.0,
            "rl_sq": 0.0,
        }
    )

    with detail_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "parameter",
                "group",
                "numel",
                "l2_il",
                "l2_diff",
                "relative_l2",
                "mean_abs_diff",
                "max_abs_diff",
                "cosine_similarity",
            ]
        )
        for name in sorted(il_state):
            il = il_state[name].detach().cpu().double()
            rl = rl_state[name].detach().cpu().double()
            diff = rl - il
            numel = int(il.numel())
            l2_il = float(torch.linalg.vector_norm(il))
            l2_diff = float(torch.linalg.vector_norm(diff))
            relative_l2 = l2_diff / max(l2_il, 1e-30)
            mean_abs = float(diff.abs().mean()) if numel else 0.0
            max_abs = float(diff.abs().max()) if numel else 0.0
            cosine = _safe_cosine(il, rl)
            group = _parameter_group(name)
            writer.writerow(
                [
                    name,
                    group,
                    numel,
                    f"{l2_il:.12g}",
                    f"{l2_diff:.12g}",
                    f"{relative_l2:.12g}",
                    f"{mean_abs:.12g}",
                    f"{max_abs:.12g}",
                    "" if math.isnan(cosine) else f"{cosine:.12g}",
                ]
            )
            acc = group_acc[group]
            acc["numel"] += numel
            acc["base_sq"] += float((il * il).sum())
            acc["diff_sq"] += float((diff * diff).sum())
            acc["abs_sum"] += float(diff.abs().sum())
            acc["max_abs"] = max(acc["max_abs"], max_abs)
            acc["dot"] += float((il * rl).sum())
            acc["il_sq"] += float((il * il).sum())
            acc["rl_sq"] += float((rl * rl).sum())

    summary = {}
    summary_path = output_dir / "parameter_group_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "group",
                "numel",
                "relative_l2",
                "mean_abs_diff",
                "max_abs_diff",
                "cosine_similarity",
            ]
        )
        for group in sorted(group_acc):
            acc = group_acc[group]
            l2_base = math.sqrt(acc["base_sq"])
            l2_diff = math.sqrt(acc["diff_sq"])
            relative_l2 = l2_diff / max(l2_base, 1e-30)
            mean_abs = acc["abs_sum"] / max(acc["numel"], 1)
            denom = math.sqrt(acc["il_sq"] * acc["rl_sq"])
            cosine = acc["dot"] / denom if denom > 1e-30 else float("nan")
            summary[group] = {
                "numel": int(acc["numel"]),
                "relative_l2": relative_l2,
                "mean_abs_diff": mean_abs,
                "max_abs_diff": acc["max_abs"],
                "cosine_similarity": cosine,
            }
            writer.writerow(
                [
                    group,
                    acc["numel"],
                    f"{relative_l2:.12g}",
                    f"{mean_abs:.12g}",
                    f"{acc['max_abs']:.12g}",
                    "" if math.isnan(cosine) else f"{cosine:.12g}",
                ]
            )
    return summary


def _action_groups(action_dim: int) -> dict[str, list[int]]:
    if action_dim == 16:
        return {
            "left_arm": list(range(0, 7)),
            "left_gripper": [7],
            "right_arm": list(range(8, 15)),
            "right_gripper": [15],
        }
    return {"all": list(range(action_dim))}


def _action_drift(
    il_policy,
    rl_policy,
    policy_cfg,
    preprocessor,
    dataset,
    indices: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: pathlib.Path,
) -> None:
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    action_dim = int(policy_cfg.action_feature.shape[0])
    chunk_size = int(policy_cfg.chunk_size)
    joint_abs = np.zeros(action_dim, dtype=np.float64)
    joint_sq = np.zeros(action_dim, dtype=np.float64)
    joint_max = np.zeros(action_dim, dtype=np.float64)
    horizon_abs = np.zeros(chunk_size, dtype=np.float64)
    horizon_sq = np.zeros(chunk_size, dtype=np.float64)
    horizon_max = np.zeros(chunk_size, dtype=np.float64)
    joint_count = 0
    horizon_count = 0
    sample_rows = []
    offset = 0

    with torch.inference_mode():
        for raw_batch in tqdm(loader, desc="IL vs Post-RL action drift"):
            processed = _move_batch(preprocessor(raw_batch), device)
            policy_input = {
                key: value
                for key, value in processed.items()
                if key not in (ACTION, "action_is_pad")
            }
            il_mu = il_policy.predict_action_chunk(policy_input)
            rl_mu = rl_policy.predict_action_chunk(policy_input)
            if tuple(il_mu.shape) != tuple(rl_mu.shape):
                raise RuntimeError(
                    f"Action shape mismatch: IL={tuple(il_mu.shape)}, RL={tuple(rl_mu.shape)}"
                )
            if il_mu.shape[1:] != (chunk_size, action_dim):
                raise RuntimeError(
                    f"Unexpected ACT action shape {tuple(il_mu.shape)}; expected [B,{chunk_size},{action_dim}]"
                )

            delta_norm = (rl_mu - il_mu).detach().float().cpu().numpy().astype(np.float64)
            delta_phys = delta_norm * action_std.reshape(1, 1, -1)
            abs_phys = np.abs(delta_phys)
            sq_phys = np.square(delta_phys)
            batch_n = delta_phys.shape[0]

            joint_abs += abs_phys.sum(axis=(0, 1))
            joint_sq += sq_phys.sum(axis=(0, 1))
            joint_max = np.maximum(joint_max, abs_phys.max(axis=(0, 1)))
            joint_count += batch_n * chunk_size

            horizon_abs += abs_phys.sum(axis=(0, 2))
            horizon_sq += sq_phys.sum(axis=(0, 2))
            horizon_max = np.maximum(horizon_max, abs_phys.max(axis=(0, 2)))
            horizon_count += batch_n * action_dim

            rms = np.sqrt(sq_phys.mean(axis=(1, 2)))
            max_abs = abs_phys.max(axis=(1, 2))
            mean_abs = abs_phys.mean(axis=(1, 2))
            for local in range(batch_n):
                sample_rows.append(
                    (
                        int(indices[offset + local]),
                        float(mean_abs[local]),
                        float(rms[local]),
                        float(max_abs[local]),
                    )
                )
            offset += batch_n

    joint_mae = joint_abs / max(joint_count, 1)
    joint_rmse = np.sqrt(joint_sq / max(joint_count, 1))
    horizon_mae = horizon_abs / max(horizon_count, 1)
    horizon_rmse = np.sqrt(horizon_sq / max(horizon_count, 1))

    with (output_dir / "action_joint_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action_dim", "physical_mae", "physical_rmse", "physical_max_abs"])
        for dim in range(action_dim):
            writer.writerow(
                [
                    dim,
                    f"{joint_mae[dim]:.12g}",
                    f"{joint_rmse[dim]:.12g}",
                    f"{joint_max[dim]:.12g}",
                ]
            )

    with (output_dir / "action_horizon_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["horizon_step", "physical_mae", "physical_rmse", "physical_max_abs"])
        for step in range(chunk_size):
            writer.writerow(
                [
                    step,
                    f"{horizon_mae[step]:.12g}",
                    f"{horizon_rmse[step]:.12g}",
                    f"{horizon_max[step]:.12g}",
                ]
            )

    sample_rows.sort(key=lambda row: row[2], reverse=True)
    with (output_dir / "top_action_drift_samples.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_index", "physical_mae", "physical_rmse", "physical_max_abs"])
        writer.writerows(sample_rows[: min(100, len(sample_rows))])

    print("\nAction drift on identical observations (physical action units):")
    for name, dims in _action_groups(action_dim).items():
        print(
            f"  {name:14s} MAE={float(joint_mae[dims].mean()):.6g} "
            f"RMSE={float(np.sqrt(np.mean(np.square(joint_rmse[dims])))):.6g} "
            f"max={float(joint_max[dims].max()):.6g}"
        )
    print(
        f"  all            MAE={float(joint_mae.mean()):.6g} "
        f"RMSE={float(np.sqrt(np.mean(np.square(joint_rmse)))):.6g} "
        f"max={float(joint_max.max()):.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic IL ACT and exported deterministic Post-RL ACT. "
            "Always reports parameter drift; optionally runs both policies on the same "
            "LeRobot observations and reports action drift."
        )
    )
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--il-processor-dir", default=None)
    parser.add_argument("--postrl-processor-dir", default=None)
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/policy_drift_analysis",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument(
        "--allow-processor-mismatch",
        action="store_true",
        help=(
            "Allow different IL/Post-RL processor bundles. Action comparison still uses "
            "the IL preprocessor so the reported drift isolates policy weights."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    il_checkpoint = pathlib.Path(args.il_checkpoint).expanduser().resolve()
    rl_checkpoint = pathlib.Path(args.postrl_checkpoint).expanduser().resolve()
    il_cfg = CustomACTConfigWrapper.from_pretrained(str(il_checkpoint))
    rl_cfg = CustomACTConfigWrapper.from_pretrained(str(rl_checkpoint))
    il_cfg.device = str(device)
    rl_cfg.device = str(device)

    if int(il_cfg.chunk_size) != int(rl_cfg.chunk_size):
        raise RuntimeError(
            f"Chunk-size mismatch: IL={il_cfg.chunk_size}, Post-RL={rl_cfg.chunk_size}"
        )
    il_action_dim = int(il_cfg.action_feature.shape[0])
    rl_action_dim = int(rl_cfg.action_feature.shape[0])
    if il_action_dim != rl_action_dim:
        raise RuntimeError(
            f"Action-dim mismatch: IL={il_action_dim}, Post-RL={rl_action_dim}"
        )

    il_policy = CustomACTPolicyWrapper.from_pretrained(
        str(il_checkpoint), config=il_cfg, strict=True
    ).to(device)
    rl_policy = CustomACTPolicyWrapper.from_pretrained(
        str(rl_checkpoint), config=rl_cfg, strict=True
    ).to(device)
    il_policy.eval()
    rl_policy.eval()

    print("=== Parameter drift ===")
    group_summary = _parameter_drift(il_policy, rl_policy, output_dir)
    for group, values in sorted(group_summary.items()):
        print(
            f"  {group:20s} rel_L2={values['relative_l2']:.6g} "
            f"mean_abs={values['mean_abs_diff']:.6g} "
            f"max_abs={values['max_abs_diff']:.6g}"
        )

    if args.lerobot_root is None:
        print("\nNo --lerobot-root supplied; parameter-only analysis complete.")
        print(f"Saved reports to: {output_dir}")
        return

    print("\n=== Same-observation action drift ===")
    il_processor_dir = _resolve_processor_dir(il_checkpoint, args.il_processor_dir)
    rl_processor_dir = _resolve_processor_dir(rl_checkpoint, args.postrl_processor_dir)
    il_fp = _processor_fingerprint(il_processor_dir)
    rl_fp = _processor_fingerprint(rl_processor_dir)
    if il_fp != rl_fp and not args.allow_processor_mismatch:
        raise RuntimeError(
            "IL and Post-RL processor bundles differ. This test expects export to preserve "
            "the original processor contract. Pass --allow-processor-mismatch only if intentional."
        )
    if il_fp == rl_fp:
        print(f"  processor bundle fingerprint: identical ({il_fp[:12]}...)")
    else:
        print("  WARNING: processor bundles differ; using IL preprocessing for both models")

    action_mean, action_std = _load_action_norm(il_processor_dir)
    if action_mean.shape != (il_action_dim,) or action_std.shape != (il_action_dim,):
        raise RuntimeError("Action normalizer shape does not match policy action dimension")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(il_processor_dir),
        config_filename="policy_preprocessor.json",
    )
    lerobot_root = pathlib.Path(args.lerobot_root).expanduser().resolve()
    repo_id = args.repo_id or lerobot_root.name
    dataset = _build_dataset(lerobot_root, repo_id, il_cfg)
    indices = _select_indices(len(dataset), args.stride, args.max_samples)
    print(f"  dataset frames={len(dataset)}, selected anchors={len(indices)}")

    _action_drift(
        il_policy,
        rl_policy,
        il_cfg,
        preprocessor,
        dataset,
        indices,
        action_std,
        device,
        args.batch_size,
        args.num_workers,
        output_dir,
    )
    print(f"\nSaved reports to: {output_dir}")


if __name__ == "__main__":
    main()
