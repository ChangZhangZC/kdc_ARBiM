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
from analysis_06_ppo_local_advantage import (  # noqa: E402
    ACTION_GROUPS,
    _phase,
    _ppo_score,
    _q_value,
    _select_anchors,
)
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


def _cosine(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            f"Vector shapes must match, got {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    numerator = (left * right).sum(dim=-1)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    denom = left_norm * right_norm
    cosine = numerator / denom.clamp_min(1e-12)
    return torch.where(
        denom > 1e-12,
        cosine,
        torch.full_like(cosine, float("nan")),
    )


def _flatten_group(
    tensor: torch.Tensor,
    group_slice: slice,
) -> torch.Tensor:
    return tensor[..., group_slice].reshape(tensor.shape[0], -1)


def _direction_stats(
    gradient: torch.Tensor,
    terminal_direction: torch.Tensor,
    postrl_drift: torch.Tensor,
    group_slice: slice,
) -> dict[str, torch.Tensor]:
    grad = _flatten_group(gradient, group_slice)
    terminal = _flatten_group(terminal_direction, group_slice)
    drift = _flatten_group(postrl_drift, group_slice)

    grad_norm = torch.linalg.vector_norm(grad, dim=-1)
    terminal_norm = torch.linalg.vector_norm(terminal, dim=-1)
    drift_norm = torch.linalg.vector_norm(drift, dim=-1)

    terminal_unit = terminal / terminal_norm.unsqueeze(-1).clamp_min(1e-12)
    drift_unit = drift / drift_norm.unsqueeze(-1).clamp_min(1e-12)

    grad_terminal_projection = (grad * terminal_unit).sum(dim=-1)
    grad_drift_projection = (grad * drift_unit).sum(dim=-1)
    grad_terminal_projection = torch.where(
        terminal_norm > 1e-12,
        grad_terminal_projection,
        torch.full_like(grad_terminal_projection, float("nan")),
    )
    grad_drift_projection = torch.where(
        drift_norm > 1e-12,
        grad_drift_projection,
        torch.full_like(grad_drift_projection, float("nan")),
    )

    return {
        "gradient_norm": grad_norm,
        "terminal_direction_norm": terminal_norm,
        "postrl_drift_norm": drift_norm,
        "cos_gradient_terminal": _cosine(grad, terminal),
        "cos_gradient_postrl_drift": _cosine(grad, drift),
        "cos_terminal_postrl_drift": _cosine(terminal, drift),
        "gradient_terminal_projection": grad_terminal_projection,
        "gradient_postrl_drift_projection": grad_drift_projection,
    }


def _score_function_mean_gradient(
    *,
    score: torch.Tensor,
    perturbation: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score.ndim != 2:
        raise ValueError(f"Expected score [B,K], got {tuple(score.shape)}")
    if perturbation.ndim != 4:
        raise ValueError(
            f"Expected perturbation [B,K,H,D], got {tuple(perturbation.shape)}"
        )
    if score.shape[:2] != perturbation.shape[:2]:
        raise ValueError(
            f"Score/perturbation batch mismatch: {tuple(score.shape)} vs "
            f"{tuple(perturbation.shape)}"
        )

    inv_variance = std.square().reciprocal().view(1, 1, 1, -1)
    score_function = perturbation * inv_variance

    raw_gradient = (
        score[:, :, None, None] * score_function
    ).mean(dim=1)

    # Per-state centering is a control variate. For a Gaussian policy,
    # E[grad log pi(a|s)] = 0, so subtracting an action-independent baseline
    # leaves the expected policy-gradient direction unchanged while reducing
    # Monte-Carlo variance. PPO's later positive std normalization also does
    # not change this expected direction.
    centered_score = score - score.mean(dim=1, keepdim=True)
    centered_gradient = (
        centered_score[:, :, None, None] * score_function
    ).mean(dim=1)
    return centered_gradient, raw_gradient


def _mean_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _median_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _fraction_positive(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite > 0))


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
            "sample_score_mean": _mean_or_nan(
                [row["sample_score_mean"] for row in items]
            ),
            "sample_score_std": _mean_or_nan(
                [row["sample_score_std"] for row in items]
            ),
            "sample_adv_mean": _mean_or_nan(
                [row["sample_adv_mean"] for row in items]
            ),
            "sample_adv_std": _mean_or_nan(
                [row["sample_adv_std"] for row in items]
            ),
        }
        for name in ACTION_GROUPS:
            for metric in (
                "cos_gradient_terminal",
                "cos_gradient_postrl_drift",
                "cos_terminal_postrl_drift",
                "gradient_terminal_projection",
                "gradient_postrl_drift_projection",
                "gradient_norm",
                "terminal_direction_norm",
                "postrl_drift_norm",
                "raw_cos_gradient_terminal",
                "raw_cos_gradient_postrl_drift",
            ):
                key = f"{metric}_{name}"
                values = [row[key] for row in items]
                out[f"mean_{key}"] = _mean_or_nan(values)
                out[f"median_{key}"] = _median_or_nan(values)
                if metric.startswith("cos_") or metric.startswith("gradient_") or metric.startswith("raw_cos_"):
                    out[f"fraction_{key}_positive"] = _fraction_positive(values)
        result.append(out)
    return result


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_progress(
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

        bin_count = max(int(row["progress_bin"]) for row in current) + 1
        x = np.asarray(
            [
                (int(row["progress_bin"]) + 0.5) / max(bin_count, 1)
                for row in current
            ],
            dtype=np.float64,
        )

        plt.figure(figsize=(10, 5))
        for name in (
            "all",
            "left_gripper",
            "right_gripper",
            "left_joints",
            "right_joints",
        ):
            plt.plot(
                x,
                [row[f"mean_cos_gradient_terminal_{name}"] for row in current],
                label=name,
            )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("episode progress")
        plt.ylabel("cos(mean-space PPO gradient, terminal direction)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{policy}_gradient_terminal_alignment.png",
            dpi=160,
        )
        plt.close()

        plt.figure(figsize=(10, 5))
        for name in (
            "all",
            "left_gripper",
            "right_gripper",
            "left_joints",
            "right_joints",
        ):
            plt.plot(
                x,
                [
                    row[f"mean_cos_gradient_postrl_drift_{name}"]
                    for row in current
                ],
                label=name,
            )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("episode progress")
        plt.ylabel("cos(mean-space PPO gradient, observed IL->Post-RL drift)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{policy}_gradient_postrl_drift_alignment.png",
            dpi=160,
        )
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(
            x,
            [row["mean_cos_terminal_postrl_drift_all"] for row in current],
            label="terminal vs observed Post-RL drift",
        )
        plt.plot(
            x,
            [row["mean_cos_gradient_terminal_all"] for row in current],
            label="gradient vs terminal",
        )
        plt.plot(
            x,
            [row["mean_cos_gradient_postrl_drift_all"] for row in current],
            label="gradient vs observed Post-RL drift",
        )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("episode progress")
        plt.ylabel("cosine alignment")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{policy}_causal_chain_alignment.png",
            dpi=160,
        )
        plt.close()


@torch.no_grad()
def _evaluate_policy(
    *,
    workspace,
    sampling_policy,
    sampling_policy_name: str,
    il_policy,
    postrl_policy,
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
            "analysis_07 currently expects the sim_task1 16D bimanual action contract."
        )

    for policy in (sampling_policy, il_policy, postrl_policy):
        policy.eval()
        policy.set_frozen_encoder_pos_embed(
            workspace._frozen_encoder_pos_embed
        )

    log_std = sampling_policy._get_log_std().detach().float()
    std = log_std.exp()
    chunk_size = int(workspace.cfg.n_action_steps)
    rows = []

    for offset in tqdm(
        range(0, len(anchors), anchor_batch_size),
        desc=f"PPO gradient alignment [{sampling_policy_name}]",
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

        sampling_mean = sampling_policy.get_action_mean({"latent": latent})
        il_mean = il_policy.get_action_mean({"latent": latent})
        postrl_mean = postrl_policy.get_action_mean({"latent": latent})
        observed_postrl_drift = postrl_mean - il_mean

        terminal_raw_np = terminal_bank_raw[batch_episode_ids]
        terminal_raw = torch.from_numpy(terminal_raw_np).to(workspace.device)
        terminal_action = workspace.obs_adapter.normalize_action(
            terminal_raw
        )
        terminal_direction = terminal_action - sampling_mean

        epsilon = torch.randn(
            (
                batch_n,
                samples_per_anchor,
                chunk_size,
                int(workspace.action_dim),
            ),
            device=workspace.device,
            dtype=sampling_mean.dtype,
        )
        samples = (
            sampling_mean.unsqueeze(1)
            + epsilon * std.view(1, 1, 1, -1)
        )
        perturbation = samples - sampling_mean.unsqueeze(1)

        flat_state = state.repeat_interleave(
            samples_per_anchor,
            dim=0,
        )
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

        gradient, raw_gradient = _score_function_mean_gradient(
            score=score,
            perturbation=perturbation,
            std=std,
        )

        local_anchor_np = batch_anchors - batch_starts
        length_np = batch_ends - batch_starts
        progress_np = (
            local_anchor_np / np.maximum(length_np - 1, 1)
        )

        group_stats = {}
        for name, group_slice in ACTION_GROUPS.items():
            stats = _direction_stats(
                gradient,
                terminal_direction,
                observed_postrl_drift,
                group_slice,
            )
            raw_stats = _direction_stats(
                raw_gradient,
                terminal_direction,
                observed_postrl_drift,
                group_slice,
            )
            stats["raw_cos_gradient_terminal"] = raw_stats[
                "cos_gradient_terminal"
            ]
            stats["raw_cos_gradient_postrl_drift"] = raw_stats[
                "cos_gradient_postrl_drift"
            ]
            group_stats[name] = stats

        for local in range(batch_n):
            progress = float(progress_np[local])
            phase = _phase(progress)
            progress_bin = min(
                int(progress * progress_bins),
                progress_bins - 1,
            )
            row = {
                "policy": sampling_policy_name,
                "episode": int(batch_episode_ids[local]),
                "anchor": int(batch_anchors[local]),
                "local_anchor": int(local_anchor_np[local]),
                "episode_length": int(length_np[local]),
                "frame_progress": progress,
                "progress_bin": f"{progress_bin:02d}",
                "phase": phase,
                "samples_per_anchor": int(samples_per_anchor),
                "sample_adv_mean": float(
                    advantage[local].mean().item()
                ),
                "sample_adv_std": float(
                    advantage[local].std(unbiased=False).item()
                ),
                "sample_score_mean": float(
                    score[local].mean().item()
                ),
                "sample_score_std": float(
                    score[local].std(unbiased=False).item()
                ),
            }
            for name, stats in group_stats.items():
                for metric, tensor in stats.items():
                    row[f"{metric}_{name}"] = float(
                        tensor[local].item()
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
    print("\n=== PPO Gradient Alignment Diagnostic ===")
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
        "\nPrimary interpretation: positive cos(g,terminal) means the expected "
        "first-order PPO mean update points toward the terminal action template. "
        "Positive cos(g,PostRL-drift) means that estimated mean-space update also "
        "aligns with the observed deterministic IL->Post-RL action drift."
    )

    phase_order = [
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
    ]
    lookup = {
        (row["policy"], row["phase"]): row
        for row in phase_summary
    }
    for policy_name in policy_stats:
        print(f"\n[{policy_name}]")
        print(
            "  phase             g->terminal  g->PostRL  "
            "Lgrip g->term  Rgrip g->term  terminal->PostRL"
        )
        for phase in phase_order:
            row = lookup.get((policy_name, phase))
            if row is None:
                continue
            print(
                f"  {phase:17s} "
                f"{row['mean_cos_gradient_terminal_all']:+.4f}       "
                f"{row['mean_cos_gradient_postrl_drift_all']:+.4f}      "
                f"{row['mean_cos_gradient_terminal_left_gripper']:+.4f}          "
                f"{row['mean_cos_gradient_terminal_right_gripper']:+.4f}          "
                f"{row['mean_cos_terminal_postrl_drift_all']:+.4f}"
            )

        print("  fraction of anchors with positive alignment:")
        for phase in phase_order:
            row = lookup.get((policy_name, phase))
            if row is None:
                continue
            print(
                f"    {phase:17s} "
                f"g->terminal="
                f"{100.0 * row['fraction_cos_gradient_terminal_all_positive']:.1f}%  "
                f"g->PostRL="
                f"{100.0 * row['fraction_cos_gradient_postrl_drift_all_positive']:.1f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the score-function PPO gradient with respect to the ACT action "
            "mean and test whether it points toward terminal-like actions and toward "
            "the observed IL->Post-RL deterministic action drift. Analysis-only."
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
        default=128,
        help="Monte-Carlo samples used to estimate the local score-function gradient.",
    )
    parser.add_argument(
        "--anchor-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="Optional evenly spaced cap over real PPO stride anchors.",
    )
    parser.add_argument("--progress-bins", type=int, default=16)
    parser.add_argument(
        "--policies",
        choices=("both", "il", "postrl"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.samples_per_anchor < 16:
        raise ValueError("--samples-per-anchor must be >= 16")
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
        / "ppo_gradient_alignment_diagnostic"
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
            "Expected pretrained Q/V; analysis_07 must not train the Critic."
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

    if int(postrl_policy.config.chunk_size) != int(
        workspace.model.config.chunk_size
    ):
        raise RuntimeError(
            "Post-RL/Base ACT chunk mismatch: "
            f"{postrl_policy.config.chunk_size} vs "
            f"{workspace.model.config.chunk_size}"
        )

    il_processor = _resolve_processor_dir(checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_processor_fp = _processor_fingerprint(il_processor)
    postrl_processor_fp = _processor_fingerprint(postrl_processor)
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ. "
            "Refusing gradient alignment comparison."
        )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    postrl_encoder_sha = workspace._fingerprint_module(
        postrl_encoder
    )
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
        short_ids = np.flatnonzero(
            episode_lengths < chunk_size
        ).tolist()
        raise RuntimeError(
            "Terminal template requires every episode to contain a full "
            f"chunk; short episode ids={short_ids[:10]}"
        )

    anchors, anchor_episode_ids, anchor_starts, anchor_ends = (
        _select_anchors(
            episode_starts,
            episode_ends,
            chunk_size,
            stride,
            args.max_anchors,
        )
    )
    terminal_bank_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][
                    int(end) - chunk_size:int(end)
                ],
                dtype=np.float32,
            )
            for end in episode_ends
        ],
        axis=0,
    )

    temperature = cfg.unio4.get("temperature", None)
    temperature = (
        None if temperature is None else float(temperature)
    )

    policies = {}
    if args.policies in ("both", "il"):
        policies["il_init"] = workspace.model
    if args.policies in ("both", "postrl"):
        policies["postrl_best_ope"] = postrl_policy

    all_rows = []
    policy_stats = {}
    for policy_name, sampling_policy in policies.items():
        log_std = (
            sampling_policy._get_log_std()
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        std = np.exp(log_std)
        policy_stats[policy_name] = {
            "log_std_mean": float(log_std.mean()),
            "log_std_min": float(log_std.min()),
            "log_std_max": float(log_std.max()),
            "std_mean": float(std.mean()),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        }

        # Pair Monte-Carlo epsilon streams across IL and Post-RL.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rows = _evaluate_policy(
            workspace=workspace,
            sampling_policy=sampling_policy,
            sampling_policy_name=policy_name,
            il_policy=workspace.model,
            postrl_policy=postrl_policy,
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
    progress_summary = _aggregate_rows(
        all_rows,
        "progress_bin",
    )
    _write_csv(
        output_dir / "gradient_alignment_per_anchor.csv",
        all_rows,
    )
    _write_csv(
        output_dir / "gradient_alignment_phase_summary.csv",
        phase_summary,
    )
    _write_csv(
        output_dir / "gradient_alignment_progress_summary.csv",
        progress_summary,
    )
    _plot_progress(output_dir, progress_summary)

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
            "mean_space_policy_gradient": (
                "Monte-Carlo estimate of E[score * (a-mu)/sigma^2] "
                "for the Gaussian ACT policy at fixed state. The primary "
                "estimate subtracts the within-state sample-score mean as "
                "a control variate; raw uncentered estimates are also stored."
            ),
            "ppo_score": (
                "same pre-normalization score as Offline PPO: Q-V when "
                "temperature is null, otherwise "
                "min(exp(temperature*(Q-V)),100)"
            ),
            "terminal_direction": (
                "same-episode final H-action chunk minus the current "
                "sampling-policy mean in normalized action space"
            ),
            "observed_postrl_drift": (
                "deterministic Post-RL mean minus deterministic Base IL mean "
                "on the exact same cached observation"
            ),
            "cos_gradient_terminal": (
                "cosine between estimated mean-space PPO gradient and "
                "terminal direction"
            ),
            "cos_gradient_postrl_drift": (
                "cosine between estimated mean-space PPO gradient and "
                "observed IL->Post-RL deterministic action drift"
            ),
            "cos_terminal_postrl_drift": (
                "cosine between terminal direction and observed IL->Post-RL "
                "drift; measures whether the final policy drift itself is "
                "terminal-ward"
            ),
        },
        "scope_note": (
            "At the moment actions are sampled from the old/reference policy, "
            "the PPO ratio is 1 and clipping is inactive. This diagnostic "
            "therefore estimates the first-order score-function direction in "
            "action-mean space. It does not reconstruct the full Transformer "
            "parameter Jacobian, repeated clipped optimization on the same "
            "samples, or intermediate old-policy/OPE reference snapshots."
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
