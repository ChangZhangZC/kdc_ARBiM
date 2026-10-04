from __future__ import annotations

import argparse
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


def _fraction(values) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


@torch.no_grad()
def _evaluate_batch(
    workspace,
    specs: list[dict],
    alphas: list[float],
    chunk_size: int,
) -> tuple[list[dict], dict]:
    device = workspace.device
    anchors = np.asarray([row["anchor"] for row in specs], dtype=np.int64)

    latent_np = np.asarray(workspace.latent_cache.obs[anchors], dtype=np.float32)
    latent = torch.from_numpy(latent_np).to(device)
    state_latent = latent.mean(dim=1)

    base = workspace.model.get_action_mean({"latent": latent})

    raw_state_np = np.asarray(workspace.buffer["state"][anchors], dtype=np.float32)
    raw_state = torch.from_numpy(raw_state_np).to(device)
    if raw_state.shape[-1] != base.shape[-1]:
        raise RuntimeError(
            "Hold counterfactual requires state/action dims to match, got "
            f"{raw_state.shape[-1]} and {base.shape[-1]}"
        )
    hold_raw = raw_state.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()
    hold = workspace.obs_adapter.normalize_action(hold_raw)

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
    q_base, _, _, _ = _q_components(workspace.critic, latent, base)
    q_demo, _, _, _ = _q_components(workspace.critic, latent, demo)

    dist = workspace.model.get_distribution({"latent": latent})
    logprob_base = dist.log_prob(base).sum(dim=(1, 2))
    std = workspace.model._get_std().detach().reshape(1, 1, -1)
    log_std = workspace.model._get_log_std().detach()

    delta = hold - base
    hold_rmse = torch.sqrt(delta.square().mean(dim=(1, 2)))

    rows = []
    for alpha in alphas:
        action = base + float(alpha) * delta
        q, _, _, _ = _q_components(workspace.critic, latent, action)
        logprob = dist.log_prob(action).sum(dim=(1, 2))
        z = (action - base) / std
        sigma_rms = torch.sqrt(z.square().mean(dim=(1, 2)))
        sigma_l2 = torch.sqrt(z.square().sum(dim=(1, 2)))
        sigma_max = z.abs().amax(dim=(1, 2))

        for i, spec in enumerate(specs):
            rows.append(
                {
                    "episode": int(spec["episode"]),
                    "anchor_offset_from_first_release": int(spec["probe_offset"]),
                    "first_release": int(spec["first_release"]),
                    "last_release": int(spec["last_release"]),
                    "frames_to_last_release": int(
                        spec["last_release"] - spec["probe_local"]
                    ),
                    "alpha": float(alpha),
                    "v": float(value[i].item()),
                    "q_base": float(q_base[i].item()),
                    "q_demo": float(q_demo[i].item()),
                    "q": float(q[i].item()),
                    "adv": float((q[i] - value[i]).item()),
                    "q_minus_base": float((q[i] - q_base[i]).item()),
                    "q_minus_demo": float((q[i] - q_demo[i]).item()),
                    "q_gt_base": int(q[i] > q_base[i]),
                    "q_gt_demo": int(q[i] > q_demo[i]),
                    "hold_rmse_from_base": float(hold_rmse[i].item()),
                    "sigma_rms_from_base": float(sigma_rms[i].item()),
                    "sigma_l2_from_base": float(sigma_l2[i].item()),
                    "sigma_max_abs_from_base": float(sigma_max[i].item()),
                    "logprob_sum": float(logprob[i].item()),
                    "logprob_drop_from_base": float(
                        (logprob[i] - logprob_base[i]).item()
                    ),
                }
            )

    policy_stats = {
        "log_std_min": float(log_std.min().item()),
        "log_std_mean": float(log_std.mean().item()),
        "log_std_max": float(log_std.max().item()),
        "std_min": float(log_std.exp().min().item()),
        "std_mean": float(log_std.exp().mean().item()),
        "std_max": float(log_std.exp().max().item()),
    }
    return rows, policy_stats


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["alpha"])].append(row)

    output = []
    for alpha in sorted(grouped):
        group = grouped[alpha]
        output.append(
            {
                "alpha": float(alpha),
                "count": int(len(group)),
                "frames_to_last_release_mean": _finite_mean(
                    [row["frames_to_last_release"] for row in group]
                ),
                "v_mean": _finite_mean([row["v"] for row in group]),
                "q_base_mean": _finite_mean([row["q_base"] for row in group]),
                "q_demo_mean": _finite_mean([row["q_demo"] for row in group]),
                "q_mean": _finite_mean([row["q"] for row in group]),
                "q_minus_base_mean": _finite_mean(
                    [row["q_minus_base"] for row in group]
                ),
                "q_minus_demo_mean": _finite_mean(
                    [row["q_minus_demo"] for row in group]
                ),
                "q_gt_base_fraction": _fraction(
                    [row["q_gt_base"] for row in group]
                ),
                "q_gt_demo_fraction": _fraction(
                    [row["q_gt_demo"] for row in group]
                ),
                "hold_rmse_from_base_mean": _finite_mean(
                    [row["hold_rmse_from_base"] for row in group]
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
                    [row["logprob_drop_from_base"] for row in group]
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the local Critic landscape from Base ACT toward the hold "
            "counterfactual at first_release + offset. The default offset=5 targets "
            "the residual Q-hold spike found by Analysis 12."
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
        default="post_training/outputs/analysis_14_first_release_hold_interpolation",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--offset",
        type=int,
        default=5,
        help="Probe frame relative to first release; default reproduces Analysis 12 +5.",
    )
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
    skipped_no_complete_chunk = 0
    for episode, (integrity, start, end) in enumerate(
        zip(integrity_rows, episode_starts, episode_ends, strict=True)
    ):
        left_release = _parse_int(integrity, "left_reopen_frame_offset")
        right_release = _parse_int(integrity, "right_reopen_frame_offset")
        first_release = min(left_release, right_release)
        last_release = max(left_release, right_release)
        probe_local = int(first_release + args.offset)
        anchor = int(start + probe_local)
        if anchor < int(start) or anchor + chunk_size > int(end):
            skipped_no_complete_chunk += 1
            continue

        specs.append(
            {
                "episode": int(episode),
                "anchor": int(anchor),
                "episode_start": int(start),
                "episode_end": int(end),
                "first_release": int(first_release),
                "last_release": int(last_release),
                "probe_local": int(probe_local),
                "probe_offset": int(args.offset),
            }
        )

    if not specs:
        raise RuntimeError("No valid first-release hold-interpolation probes were found.")

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
            policy_stats = batch_policy_stats

    aggregate_rows = _aggregate(rows)
    _write_csv(output_dir / "hold_interpolation_per_episode.csv", rows)
    _write_csv(output_dir / "hold_interpolation_summary.csv", aggregate_rows)

    lookup = {float(row["alpha"]): row for row in aggregate_rows}
    summary = {
        "analysis": "14_first_release_hold_interpolation",
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "latent_cache_dir": str(latent_cache_dir),
        "integrity_csv": str(integrity_path),
        "episodes_analyzed": int(len(specs)),
        "skipped_no_complete_chunk": int(skipped_no_complete_chunk),
        "chunk_size": int(chunk_size),
        "probe_offset_from_first_release": int(args.offset),
        "alphas": alphas,
        "base_policy_support": policy_stats,
        "local_alpha_summary": {
            str(alpha): lookup.get(alpha)
            for alpha in alphas
            if 0.0 < alpha <= 0.10
        },
        "hold_endpoint": lookup.get(1.0),
        "interpretation": {
            "risk_if": (
                "Q rises above Base already at small alpha (0.02-0.10) for many "
                "episodes. That means the residual high-Q hold direction is locally "
                "reachable from the Base stochastic policy."
            ),
            "lower_risk_if": (
                "Q stays flat or falls at small alpha and rises only far from Base. "
                "Then the alpha=1 hold spike is more consistent with Critic OOD extrapolation."
            ),
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print("\n=== Analysis 14: first_release + offset, Base -> hold interpolation ===")
    print(f"Probe offset from first release: +{args.offset}")
    print(f"Episodes analyzed: {len(specs)}")
    print(f"Skipped without complete H-step chunk: {skipped_no_complete_chunk}")
    print(
        "Mean frames from probe to last release: "
        f"{aggregate_rows[0]['frames_to_last_release_mean']:.2f}"
    )
    print(
        "Base log_std min/mean/max: "
        f"{policy_stats['log_std_min']:.4f} / "
        f"{policy_stats['log_std_mean']:.4f} / "
        f"{policy_stats['log_std_max']:.4f}"
    )
    print("\nBase -> hold direction:")
    for row in aggregate_rows:
        print(
            f"  alpha={row['alpha']:.2f} "
            f"Q-Qbase={row['q_minus_base_mean']:+.6f} "
            f"Q>Base={100.0 * row['q_gt_base_fraction']:.1f}% "
            f"Q-Qdemo={row['q_minus_demo_mean']:+.6f} "
            f"sigma_rms={row['sigma_rms_from_base_mean']:.3f} "
            f"sigma_l2={row['sigma_l2_from_base_mean']:.3f} "
            f"dlogp={row['logprob_drop_from_base_mean']:.2f}"
        )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
