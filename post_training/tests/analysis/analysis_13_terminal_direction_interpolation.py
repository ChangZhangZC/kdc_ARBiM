from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from analysis_05_terminal_advantage import _episode_bounds, _q_components  # noqa: E402
from analysis_12_terminal_tail_aliasing import (  # noqa: E402
    _build_workspace,
    _finite_mean,
    _parse_int,
    _read_integrity_rows,
    _write_csv,
)


DEFAULT_ALPHAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)


def _finite_fraction(values) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _build_targets(
    base: torch.Tensor,
    terminal: torch.Tensor,
    first_side_left: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if base.shape != terminal.shape:
        raise ValueError(
            f"Base/terminal shapes must match, got {tuple(base.shape)} and {tuple(terminal.shape)}"
        )
    if base.ndim != 3 or base.shape[-1] != 16:
        raise ValueError(
            "Analysis 13 expects sim_task1 action layout "
            "[left7,left_gripper,right7,right_gripper] with D=16."
        )

    batch = base.shape[0]
    side_left = first_side_left.reshape(batch, 1, 1)

    def replace(mask_left: slice, mask_right: slice) -> torch.Tensor:
        left = base.clone()
        left[:, :, mask_left] = terminal[:, :, mask_left]
        right = base.clone()
        right[:, :, mask_right] = terminal[:, :, mask_right]
        return torch.where(side_left, left, right)

    completed_arm = replace(slice(0, 8), slice(8, 16))
    remaining_arm = torch.where(
        side_left,
        torch.cat([base[:, :, 0:8], terminal[:, :, 8:16]], dim=-1),
        torch.cat([terminal[:, :, 0:8], base[:, :, 8:16]], dim=-1),
    )

    completed_joints = replace(slice(0, 7), slice(8, 15))
    completed_gripper = replace(slice(7, 8), slice(15, 16))
    remaining_joints = torch.where(
        side_left,
        torch.cat(
            [
                base[:, :, 0:8],
                terminal[:, :, 8:15],
                base[:, :, 15:16],
            ],
            dim=-1,
        ),
        torch.cat(
            [
                terminal[:, :, 0:7],
                base[:, :, 7:16],
            ],
            dim=-1,
        ),
    )
    remaining_gripper = torch.where(
        side_left,
        torch.cat(
            [
                base[:, :, 0:15],
                terminal[:, :, 15:16],
            ],
            dim=-1,
        ),
        torch.cat(
            [
                base[:, :, 0:7],
                terminal[:, :, 7:8],
                base[:, :, 8:16],
            ],
            dim=-1,
        ),
    )

    return {
        "full_terminal": terminal,
        "completed_arm_terminal": completed_arm,
        "remaining_arm_terminal": remaining_arm,
        "completed_arm_joints": completed_joints,
        "completed_arm_gripper": completed_gripper,
        "remaining_arm_joints": remaining_joints,
        "remaining_arm_gripper": remaining_gripper,
    }


@torch.no_grad()
def _evaluate_batch(
    workspace,
    specs: list[dict],
    alphas: list[float],
    chunk_size: int,
) -> tuple[list[dict], list[dict]]:
    device = workspace.device
    anchors = np.asarray([row["anchor"] for row in specs], dtype=np.int64)
    episode_ends = np.asarray([row["episode_end"] for row in specs], dtype=np.int64)
    first_left = torch.as_tensor(
        [row["first_side"] == "left" for row in specs],
        device=device,
        dtype=torch.bool,
    )

    latent_np = np.asarray(workspace.latent_cache.obs[anchors], dtype=np.float32)
    latent = torch.from_numpy(latent_np).to(device)
    state_latent = latent.mean(dim=1)

    base = workspace.model.get_action_mean({"latent": latent})
    terminal_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][int(end) - chunk_size:int(end)],
                dtype=np.float32,
            )
            for end in episode_ends
        ],
        axis=0,
    )
    terminal = workspace.obs_adapter.normalize_action(
        torch.from_numpy(terminal_raw).to(device)
    )

    demo_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][int(anchor):int(anchor) + chunk_size],
                dtype=np.float32,
            )
            for anchor in anchors
        ],
        axis=0,
    )
    demo = workspace.obs_adapter.normalize_action(
        torch.from_numpy(demo_raw).to(device)
    )

    value = workspace.critic._value(state_latent).reshape(-1)
    q_base, q1_base, q2_base, gap_base = _q_components(
        workspace.critic, latent, base
    )
    q_demo, _, _, gap_demo = _q_components(workspace.critic, latent, demo)

    dist = workspace.model.get_distribution({"latent": latent})
    logprob_base = dist.log_prob(base).sum(dim=(1, 2))
    std = workspace.model._get_std().detach().reshape(1, 1, -1)
    log_std = workspace.model._get_log_std().detach()

    targets = _build_targets(base, terminal, first_left)
    rows = []
    endpoint_rows = []

    for direction, target in targets.items():
        delta = target - base
        target_rmse = torch.sqrt(delta.square().mean(dim=(1, 2)))
        for alpha in alphas:
            action = base + float(alpha) * delta
            q, q1, q2, q_gap = _q_components(workspace.critic, latent, action)
            logprob = dist.log_prob(action).sum(dim=(1, 2))
            z = (action - base) / std
            sigma_rms = torch.sqrt(z.square().mean(dim=(1, 2)))
            sigma_l2 = torch.sqrt(z.square().sum(dim=(1, 2)))
            sigma_max = z.abs().amax(dim=(1, 2))

            for i, spec in enumerate(specs):
                row = {
                    "episode": int(spec["episode"]),
                    "anchor": int(spec["anchor"]),
                    "first_side": spec["first_side"],
                    "first_release": int(spec["first_release"]),
                    "last_release": int(spec["last_release"]),
                    "partial_gap": int(spec["last_release"] - spec["first_release"]),
                    "direction": direction,
                    "alpha": float(alpha),
                    "v": float(value[i].item()),
                    "q_base": float(q_base[i].item()),
                    "q_demo": float(q_demo[i].item()),
                    "q": float(q[i].item()),
                    "adv": float((q[i] - value[i]).item()),
                    "q_minus_base": float((q[i] - q_base[i]).item()),
                    "q_minus_demo": float((q[i] - q_demo[i]).item()),
                    "q1": float(q1[i].item()),
                    "q2": float(q2[i].item()),
                    "q_gap": float(q_gap[i].item()),
                    "q_base_gap": float(gap_base[i].item()),
                    "q_demo_gap": float(gap_demo[i].item()),
                    "target_rmse_from_base": float(target_rmse[i].item()),
                    "sigma_rms_from_base": float(sigma_rms[i].item()),
                    "sigma_l2_from_base": float(sigma_l2[i].item()),
                    "sigma_max_abs_from_base": float(sigma_max[i].item()),
                    "logprob_sum": float(logprob[i].item()),
                    "logprob_drop_from_base_mean": float(
                        (logprob[i] - logprob_base[i]).item()
                    ),
                    "q_gt_base": int(q[i] > q_base[i]),
                    "q_gt_demo": int(q[i] > q_demo[i]),
                }
                rows.append(row)
                if float(alpha) == 1.0:
                    endpoint_rows.append(row)

    policy_stats = [
        {
            "log_std_min": float(log_std.min().item()),
            "log_std_mean": float(log_std.mean().item()),
            "log_std_max": float(log_std.max().item()),
            "std_min": float(log_std.exp().min().item()),
            "std_mean": float(log_std.exp().mean().item()),
            "std_max": float(log_std.exp().max().item()),
        }
    ]
    return rows, policy_stats


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["direction"], float(row["alpha"]))].append(row)

    output = []
    for (direction, alpha), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        output.append(
            {
                "direction": direction,
                "alpha": float(alpha),
                "count": int(len(group)),
                "q_mean": _finite_mean([row["q"] for row in group]),
                "q_base_mean": _finite_mean([row["q_base"] for row in group]),
                "q_demo_mean": _finite_mean([row["q_demo"] for row in group]),
                "q_minus_base_mean": _finite_mean(
                    [row["q_minus_base"] for row in group]
                ),
                "q_minus_demo_mean": _finite_mean(
                    [row["q_minus_demo"] for row in group]
                ),
                "adv_mean": _finite_mean([row["adv"] for row in group]),
                "q_gap_mean": _finite_mean([row["q_gap"] for row in group]),
                "q_gt_base_fraction": _finite_fraction(
                    [row["q_gt_base"] for row in group]
                ),
                "q_gt_demo_fraction": _finite_fraction(
                    [row["q_gt_demo"] for row in group]
                ),
                "target_rmse_from_base_mean": _finite_mean(
                    [row["target_rmse_from_base"] for row in group]
                ),
                "sigma_rms_from_base_mean": _finite_mean(
                    [row["sigma_rms_from_base"] for row in group]
                ),
                "sigma_l2_from_base_mean": _finite_mean(
                    [row["sigma_l2_from_base"] for row in group]
                ),
                "sigma_max_abs_from_base_mean": _finite_mean(
                    [row["sigma_max_abs_from_base"] for row in group]
                ),
                "logprob_drop_from_base_mean": _finite_mean(
                    [row["logprob_drop_from_base_mean"] for row in group]
                ),
            }
        )
    return output


def _side_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if float(row["alpha"]) != 1.0:
            continue
        grouped[(row["first_side"], row["direction"])].append(row)

    output = []
    for (side, direction), group in sorted(grouped.items()):
        output.append(
            {
                "first_side": side,
                "direction": direction,
                "count": int(len(group)),
                "q_minus_base_mean": _finite_mean(
                    [row["q_minus_base"] for row in group]
                ),
                "q_gt_base_fraction": _finite_fraction(
                    [row["q_gt_base"] for row in group]
                ),
                "q_minus_demo_mean": _finite_mean(
                    [row["q_minus_demo"] for row in group]
                ),
                "q_gt_demo_fraction": _finite_fraction(
                    [row["q_gt_demo"] for row in group]
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the local Critic landscape from Base ACT toward the same-episode "
            "terminal action chunk at the first-release partial-completion state. "
            "Also measures Base Gaussian policy support and bimanual arm/joint/gripper hybrids."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--integrity-dir", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/analysis_13_terminal_direction_interpolation",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    alphas = sorted(set(float(value) for value in args.alphas))
    if not alphas:
        raise ValueError("--alphas must not be empty")
    if any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("--alphas must be within [0, 1]")
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("--alphas must include both 0 and 1")

    integrity_path = (
        pathlib.Path(args.integrity_dir).expanduser().resolve()
        / "episode_integrity.csv"
    )
    if not integrity_path.is_file():
        raise FileNotFoundError(integrity_path)
    integrity_rows = _read_integrity_rows(integrity_path)

    (
        workspace,
        cfg,
        stage1_dir,
        checkpoint,
        dataset,
        latent_cache_dir,
        config_path,
    ) = _build_workspace(args)

    chunk_size = int(cfg.n_action_steps)
    episode_starts, episode_ends = _episode_bounds(workspace.buffer.episode_ends)
    if len(integrity_rows) != len(episode_ends):
        raise RuntimeError(
            f"Integrity episodes={len(integrity_rows)} != buffer episodes={len(episode_ends)}"
        )

    specs = []
    skipped_tied_release = 0
    skipped_no_demo_chunk = 0
    for buffer_ep, (integrity, start, end) in enumerate(
        zip(integrity_rows, episode_starts, episode_ends, strict=True)
    ):
        left_release = _parse_int(integrity, "left_reopen_frame_offset")
        right_release = _parse_int(integrity, "right_reopen_frame_offset")
        if left_release == right_release:
            skipped_tied_release += 1
            continue

        first_release = min(left_release, right_release)
        last_release = max(left_release, right_release)
        anchor = int(start + first_release)
        if anchor + chunk_size > int(end):
            skipped_no_demo_chunk += 1
            continue

        specs.append(
            {
                "episode": int(buffer_ep),
                "anchor": int(anchor),
                "episode_start": int(start),
                "episode_end": int(end),
                "first_side": "left" if left_release < right_release else "right",
                "first_release": int(first_release),
                "last_release": int(last_release),
            }
        )

    if not specs:
        raise RuntimeError("No valid partial-completion probes were found.")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    policy_stats = None
    for offset in range(0, len(specs), args.batch_size):
        batch_rows, batch_policy_stats = _evaluate_batch(
            workspace,
            specs[offset:offset + args.batch_size],
            alphas,
            chunk_size,
        )
        rows.extend(batch_rows)
        if policy_stats is None:
            policy_stats = batch_policy_stats[0]

    aggregate_rows = _aggregate(rows)
    side_rows = _side_summary(rows)
    _write_csv(output_dir / "interpolation_per_episode.csv", rows)
    _write_csv(output_dir / "interpolation_summary.csv", aggregate_rows)
    _write_csv(output_dir / "terminal_hybrid_side_summary.csv", side_rows)

    lookup = {
        (row["direction"], float(row["alpha"])): row
        for row in aggregate_rows
    }
    local_alphas = [value for value in alphas if 0.0 < value <= 0.10]
    local_full = {
        str(alpha): lookup.get(("full_terminal", alpha))
        for alpha in local_alphas
    }

    summary = {
        "analysis": "13_terminal_direction_interpolation",
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "latent_cache_dir": str(latent_cache_dir),
        "integrity_csv": str(integrity_path),
        "episodes_analyzed": int(len(specs)),
        "skipped_tied_release": int(skipped_tied_release),
        "skipped_no_complete_demo_chunk": int(skipped_no_demo_chunk),
        "chunk_size": int(chunk_size),
        "alphas": alphas,
        "base_policy_support": policy_stats,
        "full_terminal_local": local_full,
        "full_terminal_endpoint": lookup.get(("full_terminal", 1.0)),
        "completed_arm_endpoint": lookup.get(("completed_arm_terminal", 1.0)),
        "remaining_arm_endpoint": lookup.get(("remaining_arm_terminal", 1.0)),
        "interpretation": {
            "stronger_critic_to_ppo_link_if": (
                "Q rises above Base at small alpha (for example 0.02-0.10) for a large "
                "fraction of episodes while the corresponding Base-policy sigma/log-prob "
                "distance is still relatively small. This means the high-Q terminal direction "
                "is locally reachable by samples from the Base stochastic policy."
            ),
            "weaker_link_if": (
                "Q stays flat or falls near alpha=0 and rises only near alpha=1, especially "
                "when those actions are many sigma away and have a very large summed log-prob drop."
            ),
            "hybrid_use": (
                "Compare completed_arm_terminal versus remaining_arm_terminal and the "
                "joint/gripper-only directions to identify which side/component produces "
                "the terminal Q preference."
            ),
            "support_caveat": (
                "Sigma/log-prob values use the Base stochastic ACT initialized from the IL "
                "checkpoint. They characterize the initial PPO support, not later moving "
                "old-policy distributions after PPO refreshes."
            ),
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print("\n=== Analysis 13: Base -> terminal interpolation ===")
    print(f"Episodes analyzed: {len(specs)}")
    print(
        "Base log_std min/mean/max: "
        f"{policy_stats['log_std_min']:.4f} / "
        f"{policy_stats['log_std_mean']:.4f} / "
        f"{policy_stats['log_std_max']:.4f}"
    )
    print("\nFull-terminal direction:")
    for alpha in alphas:
        row = lookup[("full_terminal", alpha)]
        print(
            f"  alpha={alpha:>4.2f} "
            f"Q-Qbase={row['q_minus_base_mean']:+.6f} "
            f"Q>Base={100.0 * row['q_gt_base_fraction']:.1f}% "
            f"sigma_rms={row['sigma_rms_from_base_mean']:.3f} "
            f"sigma_l2={row['sigma_l2_from_base_mean']:.3f} "
            f"dlogp={row['logprob_drop_from_base_mean']:.2f}"
        )

    print("\nTerminal hybrid endpoints (alpha=1):")
    for direction in (
        "completed_arm_terminal",
        "remaining_arm_terminal",
        "completed_arm_joints",
        "completed_arm_gripper",
        "remaining_arm_joints",
        "remaining_arm_gripper",
    ):
        row = lookup[(direction, 1.0)]
        print(
            f"  {direction:28s} "
            f"Q-Qbase={row['q_minus_base_mean']:+.6f} "
            f"Q>Base={100.0 * row['q_gt_base_fraction']:.1f}%"
        )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
