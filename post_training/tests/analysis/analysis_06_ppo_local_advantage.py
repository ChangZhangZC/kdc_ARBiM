from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from analysis_05_terminal_advantage import (  # noqa: E402
    _episode_bounds,
    _load_postrl_policy,
    _processor_fingerprint,
    _resolve_config,
    _resolve_processor_dir,
    _resolve_required_path,
)
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


ACTION_GROUPS = {
    "all": slice(0, 16),
    "left": slice(0, 8),
    "left_joints": slice(0, 7),
    "left_gripper": slice(7, 8),
    "right": slice(8, 16),
    "right_joints": slice(8, 15),
    "right_gripper": slice(15, 16),
}


def _phase(progress: float) -> str:
    if progress < 0.25:
        return "early_0_25"
    if progress < 0.50:
        return "middle_25_50"
    if progress < 0.75:
        return "late_50_75"
    return "tail_75_100"


def _window_starts(length: int, chunk_size: int, stride: int) -> np.ndarray:
    if length < chunk_size:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, length - chunk_size + 1, stride, dtype=np.int64)


def _select_anchors(
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    chunk_size: int,
    stride: int,
    max_anchors: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchors = []
    episodes = []
    starts = []
    ends = []
    for episode_id, (start, end) in enumerate(
        zip(episode_starts, episode_ends, strict=True)
    ):
        length = int(end - start)
        local = _window_starts(length, chunk_size, stride)
        if len(local) == 0:
            continue
        anchors.append(int(start) + local)
        episodes.append(np.full(len(local), episode_id, dtype=np.int64))
        starts.append(np.full(len(local), int(start), dtype=np.int64))
        ends.append(np.full(len(local), int(end), dtype=np.int64))

    if not anchors:
        raise RuntimeError("No valid PPO anchors found.")

    anchors = np.concatenate(anchors)
    episodes = np.concatenate(episodes)
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)

    if max_anchors is not None and len(anchors) > max_anchors:
        if max_anchors < 1:
            raise ValueError("--max-anchors must be >= 1")
        positions = np.linspace(0, len(anchors) - 1, max_anchors, dtype=np.int64)
        anchors = anchors[positions]
        episodes = episodes[positions]
        starts = starts[positions]
        ends = ends[positions]

    return anchors, episodes, starts, ends


def _q_value(critic, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    prepared = critic._prepare_action(action)
    if critic._is_double_q:
        q1, q2 = critic._Q(state, prepared)
        q = torch.minimum(q1, q2)
    else:
        q = critic._Q(state, prepared)
    return q.reshape(-1)


def _row_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(
            f"Expected matching [B,K] tensors, got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt(
        x_centered.square().sum(dim=1) * y_centered.square().sum(dim=1)
    )
    corr = numerator / denominator.clamp_min(1e-12)
    return torch.where(
        denominator > 1e-12,
        corr,
        torch.full_like(corr, float("nan")),
    )


def _terminal_projection(
    perturbation: torch.Tensor,
    terminal_direction: torch.Tensor,
    group_slice: slice,
) -> tuple[torch.Tensor, torch.Tensor]:
    # perturbation: [B,K,H,D], terminal_direction: [B,H,D]
    delta = perturbation[..., group_slice]
    direction = terminal_direction[..., group_slice]
    direction_flat = direction.reshape(direction.shape[0], -1)
    delta_flat = delta.reshape(delta.shape[0], delta.shape[1], -1)
    direction_norm = torch.linalg.vector_norm(direction_flat, dim=1)
    projection = (
        delta_flat * direction_flat.unsqueeze(1)
    ).sum(dim=-1) / direction_norm.unsqueeze(1).clamp_min(1e-12)
    projection = torch.where(
        direction_norm.unsqueeze(1) > 1e-12,
        projection,
        torch.full_like(projection, float("nan")),
    )
    return projection, direction_norm


def _ppo_score(
    advantage: torch.Tensor,
    temperature: float | None,
) -> torch.Tensor:
    if temperature is None:
        return advantage
    return torch.minimum(
        torch.exp(advantage * float(temperature)),
        torch.full_like(advantage, 100.0),
    )


def _rank_projection_stats(
    score: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = score.shape[1]
    q = max(1, k // 4)
    order = torch.argsort(score, dim=1)
    bottom_idx = order[:, :q]
    top_idx = order[:, -q:]
    top_projection = torch.gather(projection, 1, top_idx).mean(dim=1)
    bottom_projection = torch.gather(projection, 1, bottom_idx).mean(dim=1)
    best_idx = order[:, -1:]
    best_projection = torch.gather(projection, 1, best_idx).squeeze(1)
    return top_projection, bottom_projection, best_projection


def _mean_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _median_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _fraction(values: list[float], predicate) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(predicate(finite)))


def _aggregate_rows(
    rows: list[dict],
    group_key: str,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row[group_key])].append(row)

    result = []
    for (policy, group), items in sorted(grouped.items()):
        out = {
            "policy": policy,
            group_key: group,
            "count": len(items),
            "mean_action_adv": _mean_or_nan(
                [row["mean_action_adv"] for row in items]
            ),
            "sample_adv_mean": _mean_or_nan(
                [row["sample_adv_mean"] for row in items]
            ),
            "sample_adv_std": _mean_or_nan(
                [row["sample_adv_std"] for row in items]
            ),
            "sample_adv_positive_fraction": _mean_or_nan(
                [row["sample_adv_positive_fraction"] for row in items]
            ),
            "terminal_adv": _mean_or_nan(
                [row["terminal_adv"] for row in items]
            ),
            "terminal_distance_rmse": _mean_or_nan(
                [row["terminal_distance_rmse"] for row in items]
            ),
        }
        for name in ACTION_GROUPS:
            corr_key = f"corr_score_projection_{name}"
            adv_corr_key = f"corr_adv_projection_{name}"
            top_key = f"top_minus_bottom_projection_{name}"
            best_key = f"best_score_projection_{name}"
            out[f"mean_{corr_key}"] = _mean_or_nan(
                [row[corr_key] for row in items]
            )
            out[f"median_{corr_key}"] = _median_or_nan(
                [row[corr_key] for row in items]
            )
            out[f"fraction_{corr_key}_positive"] = _fraction(
                [row[corr_key] for row in items],
                lambda x: x > 0,
            )
            out[f"mean_{adv_corr_key}"] = _mean_or_nan(
                [row[adv_corr_key] for row in items]
            )
            out[f"mean_{top_key}"] = _mean_or_nan(
                [row[top_key] for row in items]
            )
            out[f"fraction_{top_key}_positive"] = _fraction(
                [row[top_key] for row in items],
                lambda x: x > 0,
            )
            out[f"mean_{best_key}"] = _mean_or_nan(
                [row[best_key] for row in items]
            )
            out[f"fraction_{best_key}_positive"] = _fraction(
                [row[best_key] for row in items],
                lambda x: x > 0,
            )
        result.append(out)
    return result


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summaries(
    output_dir: pathlib.Path,
    progress_rows: list[dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    policies = sorted({row["policy"] for row in progress_rows})
    for policy in policies:
        current = [row for row in progress_rows if row["policy"] == policy]
        current.sort(key=lambda row: int(row["progress_bin"]))
        if not current:
            continue
        bin_count = max(
            int(row["progress_bin"]) for row in current
        ) + 1
        x = np.asarray(
            [
                (int(row["progress_bin"]) + 0.5) / max(bin_count, 1)
                for row in current
            ],
            dtype=np.float64,
        )

        plt.figure(figsize=(10, 5))
        plt.plot(
            x,
            [row["mean_corr_score_projection_all"] for row in current],
            label="all",
        )
        plt.plot(
            x,
            [row["mean_corr_score_projection_left_gripper"] for row in current],
            label="left_gripper",
        )
        plt.plot(
            x,
            [row["mean_corr_score_projection_right_gripper"] for row in current],
            label="right_gripper",
        )
        plt.plot(
            x,
            [row["mean_corr_score_projection_left_joints"] for row in current],
            label="left_joints",
        )
        plt.plot(
            x,
            [row["mean_corr_score_projection_right_joints"] for row in current],
            label="right_joints",
        )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("episode progress")
        plt.ylabel("mean local corr(PPO score, terminal projection)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{policy}_local_corr_vs_progress.png",
            dpi=160,
        )
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(
            x,
            [row["mean_top_minus_bottom_projection_all"] for row in current],
            label="top-score minus bottom-score terminal projection",
        )
        plt.plot(
            x,
            [row["mean_best_score_projection_all"] for row in current],
            label="best-score sample terminal projection",
        )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("episode progress")
        plt.ylabel("normalized-action terminal projection")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{policy}_rank_projection_vs_progress.png",
            dpi=160,
        )
        plt.close()


@torch.no_grad()
def _evaluate_policy(
    *,
    workspace,
    policy,
    policy_name: str,
    anchors: np.ndarray,
    episode_ids: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    terminal_bank_raw: np.ndarray,
    samples_per_anchor: int,
    anchor_batch_size: int,
    progress_bins: int,
    temperature: float | None,
) -> list[dict]:
    if int(workspace.action_dim) != 16:
        raise RuntimeError(
            "analysis_06 currently expects the sim_task1 16D bimanual action contract."
        )

    policy.eval()
    policy.set_frozen_encoder_pos_embed(workspace._frozen_encoder_pos_embed)
    log_std = policy._get_log_std().detach().float()
    std = log_std.exp()
    chunk_size = int(workspace.cfg.n_action_steps)
    rows = []

    iterator = range(0, len(anchors), anchor_batch_size)
    for offset in tqdm(
        iterator,
        desc=f"PPO local advantage [{policy_name}]",
    ):
        batch_anchors = anchors[offset:offset + anchor_batch_size]
        batch_episode_ids = episode_ids[offset:offset + anchor_batch_size]
        batch_starts = episode_starts[offset:offset + anchor_batch_size]
        batch_ends = episode_ends[offset:offset + anchor_batch_size]
        batch_n = len(batch_anchors)

        latent_np = np.asarray(
            workspace.latent_cache.obs[batch_anchors],
            dtype=np.float32,
        )
        latent = torch.from_numpy(latent_np).to(workspace.device)
        state = latent.mean(dim=1)
        value = workspace.critic._value(state).reshape(-1)
        mean_action = policy.get_action_mean({"latent": latent})

        terminal_raw_np = terminal_bank_raw[batch_episode_ids]
        terminal_raw = torch.from_numpy(terminal_raw_np).to(workspace.device)
        terminal_action = workspace.obs_adapter.normalize_action(terminal_raw)
        terminal_direction = terminal_action - mean_action

        epsilon = torch.randn(
            (
                batch_n,
                samples_per_anchor,
                chunk_size,
                int(workspace.action_dim),
            ),
            device=workspace.device,
            dtype=mean_action.dtype,
        )
        samples = (
            mean_action.unsqueeze(1)
            + epsilon * std.view(1, 1, 1, -1)
        )
        perturbation = samples - mean_action.unsqueeze(1)

        flat_state = state.repeat_interleave(samples_per_anchor, dim=0)
        flat_samples = samples.reshape(
            batch_n * samples_per_anchor,
            chunk_size,
            int(workspace.action_dim),
        )
        q_samples = _q_value(
            workspace.critic,
            flat_state,
            flat_samples,
        ).reshape(batch_n, samples_per_anchor)
        advantage = q_samples - value.unsqueeze(1)
        score = _ppo_score(advantage, temperature)

        q_mean = _q_value(workspace.critic, state, mean_action)
        q_terminal = _q_value(workspace.critic, state, terminal_action)
        mean_action_adv = q_mean - value
        terminal_adv = q_terminal - value
        terminal_distance_rmse = torch.sqrt(
            terminal_direction.square().mean(dim=(1, 2))
        )

        group_stats = {}
        for group_name, group_slice in ACTION_GROUPS.items():
            projection, direction_norm = _terminal_projection(
                perturbation,
                terminal_direction,
                group_slice,
            )
            corr_adv = _row_corr(advantage, projection)
            corr_score = _row_corr(score, projection)
            top_projection, bottom_projection, best_projection = (
                _rank_projection_stats(score, projection)
            )
            group_stats[group_name] = {
                "projection": projection,
                "direction_norm": direction_norm,
                "corr_adv": corr_adv,
                "corr_score": corr_score,
                "top_minus_bottom": top_projection - bottom_projection,
                "best_projection": best_projection,
            }

        local_anchor_np = batch_anchors - batch_starts
        length_np = batch_ends - batch_starts
        frame_progress_np = local_anchor_np / np.maximum(length_np - 1, 1)

        for local in range(batch_n):
            progress = float(frame_progress_np[local])
            phase = _phase(progress)
            progress_bin = min(
                int(progress * progress_bins),
                progress_bins - 1,
            )
            row = {
                "policy": policy_name,
                "episode": int(batch_episode_ids[local]),
                "anchor": int(batch_anchors[local]),
                "local_anchor": int(local_anchor_np[local]),
                "episode_length": int(length_np[local]),
                "frame_progress": progress,
                "progress_bin": f"{progress_bin:02d}",
                "phase": phase,
                "samples_per_anchor": int(samples_per_anchor),
                "mean_action_adv": float(mean_action_adv[local].item()),
                "terminal_adv": float(terminal_adv[local].item()),
                "sample_adv_mean": float(advantage[local].mean().item()),
                "sample_adv_std": float(
                    advantage[local].std(unbiased=False).item()
                ),
                "sample_adv_positive_fraction": float(
                    (advantage[local] > 0).float().mean().item()
                ),
                "sample_score_mean": float(score[local].mean().item()),
                "sample_score_std": float(
                    score[local].std(unbiased=False).item()
                ),
                "terminal_distance_rmse": float(
                    terminal_distance_rmse[local].item()
                ),
            }
            for group_name, stats in group_stats.items():
                row[f"terminal_direction_norm_{group_name}"] = float(
                    stats["direction_norm"][local].item()
                )
                row[f"corr_adv_projection_{group_name}"] = float(
                    stats["corr_adv"][local].item()
                )
                row[f"corr_score_projection_{group_name}"] = float(
                    stats["corr_score"][local].item()
                )
                row[f"top_minus_bottom_projection_{group_name}"] = float(
                    stats["top_minus_bottom"][local].item()
                )
                row[f"best_score_projection_{group_name}"] = float(
                    stats["best_projection"][local].item()
                )
            rows.append(row)

    return rows


def _print_report(
    phase_summary: list[dict],
    policy_stats: dict[str, dict],
    *,
    stride: int,
    samples_per_anchor: int,
    temperature: float | None,
) -> None:
    print("\n=== PPO Local Sampling Advantage Diagnostic ===")
    print(f"PPO anchor stride: {stride}")
    print(f"samples per anchor: {samples_per_anchor}")
    print(f"PPO temperature: {temperature}")
    print("\nSampling policy log_std/std:")
    for name, stats in policy_stats.items():
        print(
            f"  {name:18s} "
            f"log_std(mean/min/max)="
            f"{stats['log_std_mean']:+.4f}/"
            f"{stats['log_std_min']:+.4f}/"
            f"{stats['log_std_max']:+.4f} "
            f"std_mean={stats['std_mean']:.5f}"
        )

    print(
        "\nInterpretation: positive local correlation means that, inside the Gaussian "
        "neighborhood the PPO policy can actually sample, actions moving toward the "
        "same-episode terminal template tend to receive larger Critic advantage/PPO score."
    )

    phase_order = [
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
    ]
    by_policy_phase = {
        (row["policy"], row["phase"]): row
        for row in phase_summary
    }
    for policy_name in policy_stats:
        print(f"\n[{policy_name}]")
        print(
            "  phase             corr(all)  corr(Lgrip) corr(Rgrip) "
            "top-bottom(all)  corr>0(all)"
        )
        for phase in phase_order:
            row = by_policy_phase.get((policy_name, phase))
            if row is None:
                continue
            print(
                f"  {phase:17s} "
                f"{row['mean_corr_score_projection_all']:+.4f}      "
                f"{row['mean_corr_score_projection_left_gripper']:+.4f}      "
                f"{row['mean_corr_score_projection_right_gripper']:+.4f}      "
                f"{row['mean_top_minus_bottom_projection_all']:+.5f}         "
                f"{100.0 * row['fraction_corr_score_projection_all_positive']:.1f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the trained IQL critic rewards terminal-directed perturbations "
            "inside the local Gaussian neighborhoods actually sampled by Offline PPO. "
            "This is analysis-only and does not update policy or critic weights."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override Base/IL ACT checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override Offline RL Zarr path.",
    )
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--samples-per-anchor",
        type=int,
        default=64,
        help="Gaussian action chunks sampled around each policy mean.",
    )
    parser.add_argument(
        "--anchor-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="Optional evenly spaced cap over the actual PPO stride anchors.",
    )
    parser.add_argument("--progress-bins", type=int, default=16)
    parser.add_argument(
        "--policies",
        choices=("both", "il", "postrl"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.samples_per_anchor < 8:
        raise ValueError("--samples-per-anchor must be >= 8")
    if args.anchor_batch_size < 1:
        raise ValueError("--anchor-batch-size must be >= 1")
    if args.progress_bins < 4:
        raise ValueError("--progress-bins must be >= 4")

    stage1_dir = pathlib.Path(args.stage1_dir).expanduser().resolve()
    critic_final = stage1_dir / "critic" / "checkpoints" / "final"
    for name in ("Q.pt", "value.pt", "contract.json"):
        path = critic_final / name
        if not path.is_file():
            raise FileNotFoundError(path)

    config_path = _resolve_config(stage1_dir, args.config)
    cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_required_path(
        args.checkpoint,
        cfg.input.get("policy_checkpoint"),
        "Base ACT checkpoint",
    )
    dataset = _resolve_required_path(
        args.dataset,
        cfg.input.get("dataset_path"),
        "Offline dataset",
    )
    latent_cache_dir = _resolve_required_path(
        args.latent_cache_dir,
        None,
        "Frozen latent cache",
    )
    postrl_checkpoint = _resolve_required_path(
        args.postrl_checkpoint,
        None,
        "Post-RL checkpoint",
    )

    cfg.input.policy_checkpoint = str(checkpoint)
    cfg.input.policy_checkpoint_type = "il"
    cfg.input.dataset_path = str(dataset)
    cfg.training.device = str(args.device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    cfg.dataset.use_latent_cache = True
    cfg.dataset.endpoint_obs_only = True
    cfg.dataset.latent_cache_dir = str(latent_cache_dir)
    cfg.critic.load_pretrain = True
    cfg.critic.artifact_dir = str(stage1_dir / "critic")

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "post_training"
        / "outputs"
        / "ppo_local_advantage_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.training.seed) if args.seed is None else int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError(
            "Expected pretrained Q/V; analysis_06 must not train the Critic."
        )
    workspace.critic.eval()
    workspace.model.eval()
    workspace.model.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )

    postrl_policy, postrl_kind = _load_postrl_policy(
        postrl_checkpoint,
        workspace.device,
        cfg,
    )
    postrl_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )

    if int(postrl_policy.config.chunk_size) != int(workspace.model.config.chunk_size):
        raise RuntimeError(
            "Post-RL/Base ACT chunk mismatch: "
            f"{postrl_policy.config.chunk_size} vs {workspace.model.config.chunk_size}"
        )

    il_processor = _resolve_processor_dir(checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_processor_fp = _processor_fingerprint(il_processor)
    postrl_processor_fp = _processor_fingerprint(postrl_processor)
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ. Refusing local PPO comparison."
        )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    postrl_encoder_sha = workspace._fingerprint_module(postrl_encoder)
    cache_encoder_sha = str(
        workspace.latent_cache.metadata.get("encoder_sha256", "")
    )
    if postrl_encoder_sha != cache_encoder_sha:
        raise RuntimeError(
            "Post-RL encoder does not match the frozen latent cache: "
            f"postrl={postrl_encoder_sha[:12]}..., "
            f"cache={cache_encoder_sha[:12]}..."
        )

    chunk_size = int(cfg.n_action_steps)
    stride = int(cfg.dataset.finetune_sequence_stride)
    episode_starts, episode_ends = _episode_bounds(
        workspace.buffer.episode_ends
    )
    episode_lengths = episode_ends - episode_starts
    if np.any(episode_lengths < chunk_size):
        short_ids = np.flatnonzero(episode_lengths < chunk_size).tolist()
        raise RuntimeError(
            "Terminal template requires every episode to contain a full chunk; "
            f"short episode ids={short_ids[:10]}"
        )

    anchors, anchor_episode_ids, anchor_starts, anchor_ends = _select_anchors(
        episode_starts,
        episode_ends,
        chunk_size,
        stride,
        args.max_anchors,
    )
    terminal_bank_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][int(end) - chunk_size:int(end)],
                dtype=np.float32,
            )
            for end in episode_ends
        ],
        axis=0,
    )

    temperature = cfg.unio4.get("temperature", None)
    temperature = None if temperature is None else float(temperature)

    policies = {}
    if args.policies in ("both", "il"):
        policies["il_init"] = workspace.model
    if args.policies in ("both", "postrl"):
        policies["postrl_best_ope"] = postrl_policy

    all_rows = []
    policy_stats = {}
    for policy_name, policy in policies.items():
        log_std = policy._get_log_std().detach().float().cpu().numpy()
        std = np.exp(log_std)
        policy_stats[policy_name] = {
            "log_std_mean": float(log_std.mean()),
            "log_std_min": float(log_std.min()),
            "log_std_max": float(log_std.max()),
            "std_mean": float(std.mean()),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        }
        # Reuse the same Gaussian epsilon stream for each policy so IL/Post-RL
        # comparisons are paired rather than dominated by Monte-Carlo seed noise.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rows = _evaluate_policy(
            workspace=workspace,
            policy=policy,
            policy_name=policy_name,
            anchors=anchors,
            episode_ids=anchor_episode_ids,
            episode_starts=anchor_starts,
            episode_ends=anchor_ends,
            terminal_bank_raw=terminal_bank_raw,
            samples_per_anchor=int(args.samples_per_anchor),
            anchor_batch_size=int(args.anchor_batch_size),
            progress_bins=int(args.progress_bins),
            temperature=temperature,
        )
        all_rows.extend(rows)

    phase_summary = _aggregate_rows(all_rows, "phase")
    progress_summary = _aggregate_rows(all_rows, "progress_bin")
    _write_csv(output_dir / "local_advantage_per_anchor.csv", all_rows)
    _write_csv(output_dir / "local_advantage_phase_summary.csv", phase_summary)
    _write_csv(
        output_dir / "local_advantage_progress_summary.csv",
        progress_summary,
    )
    _plot_summaries(output_dir, progress_summary)

    summary = {
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "postrl_checkpoint": str(postrl_checkpoint),
        "postrl_checkpoint_kind": postrl_kind,
        "dataset": str(dataset),
        "latent_cache_dir": str(latent_cache_dir),
        "processor_fingerprint": il_processor_fp,
        "encoder_sha256": cache_encoder_sha,
        "seed": seed,
        "chunk_size": chunk_size,
        "ppo_anchor_stride": stride,
        "anchors": int(len(anchors)),
        "samples_per_anchor": int(args.samples_per_anchor),
        "sampled_chunks_per_policy": int(
            len(anchors) * int(args.samples_per_anchor)
        ),
        "ppo_temperature": temperature,
        "policy_stats": policy_stats,
        "phase_summary": phase_summary,
        "progress_summary": progress_summary,
        "metric_contract": {
            "terminal_direction": (
                "same-episode final H-action chunk minus the sampling policy mean "
                "in normalized ACT action space"
            ),
            "projection": (
                "signed Euclidean projection of a sampled Gaussian perturbation onto "
                "the unit terminal direction; positive means locally terminal-ward"
            ),
            "raw_advantage": "IQL Q(s,a_sample)-V(s)",
            "ppo_score": (
                "the exact pre-normalization PPO advantage score: raw advantage when "
                "temperature is null, otherwise min(exp(temperature*advantage),100)"
            ),
            "local_correlation": (
                "Pearson correlation across samples from one state, then averaged over "
                "anchors; this avoids state-to-state value differences confounding the result"
            ),
            "top_minus_bottom_projection": (
                "mean terminal projection of top-quartile PPO-score samples minus "
                "bottom-quartile samples at the same state"
            ),
        },
        "scope_note": (
            "il_init reproduces the initial Gaussian PPO neighborhood from the Base ACT "
            "checkpoint and configured init log_std. postrl_best_ope probes the final best-OPE "
            "policy neighborhood. Intermediate old-policy reference snapshots are not reconstructed."
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    _print_report(
        phase_summary,
        policy_stats,
        stride=stride,
        samples_per_anchor=int(args.samples_per_anchor),
        temperature=temperature,
    )
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
