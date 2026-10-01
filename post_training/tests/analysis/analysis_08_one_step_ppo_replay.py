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
    _select_anchors,
)
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            f"Vector shapes must match, got {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    numerator = (left * right).sum(dim=-1)
    denom = (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)
    )
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


def _alignment_stats(
    one_step_drift: torch.Tensor,
    terminal_direction: torch.Tensor,
    final_postrl_drift: torch.Tensor,
    group_slice: slice,
) -> dict[str, torch.Tensor]:
    step = _flatten_group(one_step_drift, group_slice)
    terminal = _flatten_group(terminal_direction, group_slice)
    final = _flatten_group(final_postrl_drift, group_slice)

    step_norm = torch.linalg.vector_norm(step, dim=-1)
    terminal_norm = torch.linalg.vector_norm(terminal, dim=-1)
    final_norm = torch.linalg.vector_norm(final, dim=-1)

    step_rmse = torch.sqrt(step.square().mean(dim=-1))
    final_rmse = torch.sqrt(final.square().mean(dim=-1))

    terminal_unit = terminal / terminal_norm.unsqueeze(-1).clamp_min(1e-12)
    final_unit = final / final_norm.unsqueeze(-1).clamp_min(1e-12)

    step_terminal_projection = (step * terminal_unit).sum(dim=-1)
    step_final_projection = (step * final_unit).sum(dim=-1)
    step_terminal_projection = torch.where(
        terminal_norm > 1e-12,
        step_terminal_projection,
        torch.full_like(step_terminal_projection, float("nan")),
    )
    step_final_projection = torch.where(
        final_norm > 1e-12,
        step_final_projection,
        torch.full_like(step_final_projection, float("nan")),
    )

    return {
        "one_step_norm": step_norm,
        "one_step_rmse": step_rmse,
        "final_postrl_norm": final_norm,
        "final_postrl_rmse": final_rmse,
        "terminal_direction_norm": terminal_norm,
        "cos_one_step_terminal": _cosine(step, terminal),
        "cos_one_step_final_postrl": _cosine(step, final),
        "cos_terminal_final_postrl": _cosine(terminal, final),
        "one_step_terminal_projection": step_terminal_projection,
        "one_step_final_postrl_projection": step_final_projection,
        "one_step_over_final_norm": (
            step_norm / final_norm.clamp_min(1e-12)
        ),
    }


@torch.no_grad()
def _probe_policy(
    workspace,
    policy,
    anchors: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    policy.eval()
    policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    chunks = []
    for offset in tqdm(
        range(0, len(anchors), batch_size),
        desc="Probing deterministic policy",
        leave=False,
    ):
        batch_anchors = anchors[offset:offset + batch_size]
        latent_np = np.asarray(
            workspace.latent_cache.obs[batch_anchors],
            dtype=np.float32,
        )
        latent = torch.from_numpy(latent_np).to(workspace.device)
        action = policy.get_action_mean({"latent": latent})
        chunks.append(action.detach().cpu())
    return torch.cat(chunks, dim=0)


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    return cpu_state, cuda_state


def _restore_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def _preview_exact_update_batch(
    workspace,
    batch: dict,
) -> dict[str, float]:
    unio4 = workspace.unio4
    normalized_obs = workspace.obs_adapter.normalize_obs(batch["obs"])
    policy_obs = {
        key: value[:, 0]
        for key, value in normalized_obs.items()
    }
    action_chunk, _, _ = unio4._old_policy.sample_action_chunk(
        policy_obs
    )
    raw_advantage = workspace.critic.get_advantage(
        batch["obs"],
        action_chunk,
    ).reshape(-1)

    score = raw_advantage
    if unio4.temperature is not None:
        score = torch.minimum(
            torch.exp(raw_advantage * float(unio4.temperature)),
            torch.full_like(raw_advantage, 100.0),
        )
    normalized_advantage = unio4._normalize_advantage(score)
    clip_value = workspace.cfg.get("chunk_adv_clip", None)
    if clip_value is not None:
        normalized_advantage = torch.clamp(
            normalized_advantage,
            -float(clip_value),
            float(clip_value),
        )

    return {
        "batch_size": int(raw_advantage.shape[0]),
        "raw_adv_mean": float(raw_advantage.mean().item()),
        "raw_adv_std": float(
            raw_advantage.std(unbiased=False).item()
        ),
        "raw_adv_positive_fraction": float(
            (raw_advantage > 0).float().mean().item()
        ),
        "score_mean": float(score.mean().item()),
        "score_std": float(score.std(unbiased=False).item()),
        "normalized_adv_mean": float(
            normalized_advantage.mean().item()
        ),
        "normalized_adv_std": float(
            normalized_advantage.std(unbiased=False).item()
        ),
        "normalized_adv_positive_fraction": float(
            (normalized_advantage > 0).float().mean().item()
        ),
        "sample_action_mean": float(action_chunk.mean().item()),
        "sample_action_std": float(
            action_chunk.std(unbiased=False).item()
        ),
    }


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
        grouped[row[group_key]].append(row)

    result = []
    for group, items in sorted(grouped.items()):
        out = {
            group_key: group,
            "count": len(items),
        }
        for name in ACTION_GROUPS:
            for metric in (
                "one_step_norm",
                "one_step_rmse",
                "final_postrl_norm",
                "final_postrl_rmse",
                "terminal_direction_norm",
                "cos_one_step_terminal",
                "cos_one_step_final_postrl",
                "cos_terminal_final_postrl",
                "one_step_terminal_projection",
                "one_step_final_postrl_projection",
                "one_step_over_final_norm",
            ):
                key = f"{metric}_{name}"
                values = [row[key] for row in items]
                out[f"mean_{key}"] = _mean_or_nan(values)
                out[f"median_{key}"] = _median_or_nan(values)
                if (
                    metric.startswith("cos_")
                    or metric.endswith("_projection")
                ):
                    out[f"fraction_{key}_positive"] = (
                        _fraction_positive(values)
                    )
        result.append(out)
    return result


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_probe_rows(
    *,
    base_action: torch.Tensor,
    one_step_action: torch.Tensor,
    final_postrl_action: torch.Tensor,
    terminal_action: torch.Tensor,
    anchors: np.ndarray,
    episode_ids: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    progress_bins: int,
) -> list[dict]:
    one_step_drift = one_step_action - base_action
    final_postrl_drift = final_postrl_action - base_action
    terminal_direction = terminal_action - base_action

    group_stats = {
        name: _alignment_stats(
            one_step_drift,
            terminal_direction,
            final_postrl_drift,
            group_slice,
        )
        for name, group_slice in ACTION_GROUPS.items()
    }

    local_anchor = anchors - episode_starts
    episode_length = episode_ends - episode_starts
    frame_progress = (
        local_anchor
        / np.maximum(episode_length - 1, 1)
    )

    rows = []
    for index in range(len(anchors)):
        progress = float(frame_progress[index])
        progress_bin = min(
            int(progress * progress_bins),
            progress_bins - 1,
        )
        row = {
            "episode": int(episode_ids[index]),
            "anchor": int(anchors[index]),
            "local_anchor": int(local_anchor[index]),
            "episode_length": int(episode_length[index]),
            "frame_progress": progress,
            "progress_bin": f"{progress_bin:02d}",
            "phase": _phase(progress),
        }
        for name, stats in group_stats.items():
            for metric, tensor in stats.items():
                row[f"{metric}_{name}"] = float(
                    tensor[index].item()
                )
        rows.append(row)
    return rows


def _plot_phase_summary(
    output_dir: pathlib.Path,
    phase_rows: list[dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    phase_order = [
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
    ]
    lookup = {
        row["phase"]: row
        for row in phase_rows
    }
    present = [
        phase
        for phase in phase_order
        if phase in lookup
    ]
    if not present:
        return

    x = np.arange(len(present), dtype=np.float64)

    plt.figure(figsize=(10, 5))
    for name in (
        "left_joints",
        "left_gripper",
        "right_joints",
        "right_gripper",
    ):
        plt.plot(
            x,
            [
                lookup[phase][f"mean_one_step_rmse_{name}"]
                for phase in present
            ],
            marker="o",
            label=name,
        )
    plt.xticks(x, present, rotation=20)
    plt.ylabel("one-step deterministic action RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "one_step_drift_by_component.png",
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
                lookup[phase][
                    f"mean_cos_one_step_terminal_{name}"
                ]
                for phase in present
            ],
            marker="o",
            label=name,
        )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, present, rotation=20)
    plt.ylabel("cos(one-step drift, terminal direction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "one_step_terminal_alignment.png",
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
                lookup[phase][
                    f"mean_cos_one_step_final_postrl_{name}"
                ]
                for phase in present
            ],
            marker="o",
            label=name,
        )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, present, rotation=20)
    plt.ylabel("cos(one-step drift, final Post-RL drift)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "one_step_final_postrl_alignment.png",
        dpi=160,
    )
    plt.close()


def _print_report(
    *,
    update_stats: dict,
    phase_rows: list[dict],
    cfg,
) -> None:
    print("\n=== One-Step Real PPO Replay Diagnostic ===")
    print(
        f"finetune stride: {int(cfg.dataset.finetune_sequence_stride)}"
    )
    print(
        f"finetune batch size: {int(cfg.unio4.finetune_batch_size)}"
    )
    print(f"loss after one real update: {update_stats['loss']:+.8f}")
    print(
        "raw advantage mean/std/positive: "
        f"{update_stats['batch_preview']['raw_adv_mean']:+.6f} / "
        f"{update_stats['batch_preview']['raw_adv_std']:.6f} / "
        f"{100.0 * update_stats['batch_preview']['raw_adv_positive_fraction']:.1f}%"
    )
    print(
        "normalized advantage mean/std/positive: "
        f"{update_stats['batch_preview']['normalized_adv_mean']:+.6f} / "
        f"{update_stats['batch_preview']['normalized_adv_std']:.6f} / "
        f"{100.0 * update_stats['batch_preview']['normalized_adv_positive_fraction']:.1f}%"
    )
    print(
        "log_std mean before/after/delta: "
        f"{update_stats['log_std_mean_before']:+.6f} / "
        f"{update_stats['log_std_mean_after']:+.6f} / "
        f"{update_stats['log_std_mean_delta']:+.8f}"
    )

    lookup = {
        row["phase"]: row
        for row in phase_rows
    }
    print(
        "\nPer-phase one-step deterministic output drift "
        "(normalized action space):"
    )
    print(
        "  phase             all_rmse   Ljoint     Lgrip      "
        "Rjoint     Rgrip      step->terminal  step->PostRL"
    )
    for phase in (
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
    ):
        row = lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:17s} "
            f"{row['mean_one_step_rmse_all']:.7f}  "
            f"{row['mean_one_step_rmse_left_joints']:.7f}  "
            f"{row['mean_one_step_rmse_left_gripper']:.7f}  "
            f"{row['mean_one_step_rmse_right_joints']:.7f}  "
            f"{row['mean_one_step_rmse_right_gripper']:.7f}  "
            f"{row['mean_cos_one_step_terminal_all']:+.4f}          "
            f"{row['mean_cos_one_step_final_postrl_all']:+.4f}"
        )

    print("\nGripper vs arm alignment:")
    print(
        "  phase             Lgrip->terminal  Rgrip->terminal  "
        "Ljoints->terminal  Rjoints->terminal"
    )
    for phase in (
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
    ):
        row = lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:17s} "
            f"{row['mean_cos_one_step_terminal_left_gripper']:+.4f}            "
            f"{row['mean_cos_one_step_terminal_right_gripper']:+.4f}            "
            f"{row['mean_cos_one_step_terminal_left_joints']:+.4f}             "
            f"{row['mean_cos_one_step_terminal_right_joints']:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay exactly one real Offline PPO optimizer update from the Base ACT "
            "initialization, then measure how the deterministic ACT output moves on "
            "fixed probe states. Analysis-only; no training artifact is overwritten."
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
        "--probe-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
        help="Optional evenly spaced cap over real PPO stride probe anchors.",
    )
    parser.add_argument("--progress-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.probe_batch_size < 1:
        raise ValueError("--probe-batch-size must be >= 1")
    if args.progress_bins < 4:
        raise ValueError("--progress-bins must be >= 4")

    stage1_dir = pathlib.Path(
        args.stage1_dir
    ).expanduser().resolve()
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
    if cfg.get("ppo") is not None:
        cfg.ppo.enable_ratio_logging = False
        cfg.ppo.enable_monitoring_csv = False

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "post_training"
        / "outputs"
        / "one_step_ppo_replay_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = (
        int(cfg.training.seed)
        if args.seed is None
        else int(args.seed)
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    workspace = TrainACTWorkspace(
        cfg,
        output_dir=str(output_dir),
    )
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError(
            "Expected pretrained Q/V; analysis_08 must not train the Critic."
        )
    workspace.critic.eval()

    if int(workspace.action_dim) != 16:
        raise RuntimeError(
            "analysis_08 currently expects the sim_task1 16D bimanual action contract."
        )

    episode_starts, episode_ends = _episode_bounds(
        workspace.buffer.episode_ends
    )
    chunk_size = int(cfg.n_action_steps)
    stride = int(cfg.dataset.finetune_sequence_stride)
    anchors, anchor_episode_ids, anchor_starts, anchor_ends = (
        _select_anchors(
            episode_starts,
            episode_ends,
            chunk_size,
            stride,
            args.max_probes,
        )
    )

    episode_lengths = episode_ends - episode_starts
    if np.any(episode_lengths < chunk_size):
        short_ids = np.flatnonzero(
            episode_lengths < chunk_size
        ).tolist()
        raise RuntimeError(
            "Terminal template requires every episode to contain one full "
            f"chunk; short episode ids={short_ids[:10]}"
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
    terminal_raw = torch.from_numpy(
        terminal_bank_raw[anchor_episode_ids]
    ).to(workspace.device)
    terminal_action = (
        workspace.obs_adapter.normalize_action(terminal_raw)
        .detach()
        .cpu()
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
            "Refusing one-step replay comparison."
        )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    postrl_encoder_sha = workspace._fingerprint_module(
        postrl_encoder
    )
    cache_encoder_sha = str(
        workspace.latent_cache.metadata.get(
            "encoder_sha256",
            "",
        )
    )
    if postrl_encoder_sha != cache_encoder_sha:
        raise RuntimeError(
            "Post-RL encoder does not match the frozen latent cache: "
            f"postrl={postrl_encoder_sha[:12]}..., "
            f"cache={cache_encoder_sha[:12]}..."
        )

    print("Probing Base ACT before update...")
    base_action = _probe_policy(
        workspace,
        workspace.model,
        anchors,
        int(args.probe_batch_size),
    )
    print("Probing final Post-RL reference...")
    final_postrl_action = _probe_policy(
        workspace,
        postrl_policy,
        anchors,
        int(args.probe_batch_size),
    )

    # Free the final policy before constructing the two PPO policy copies.
    postrl_policy.to("cpu")
    del postrl_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    workspace._build_ppo()
    workspace.unio4._policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    workspace.unio4._old_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )

    # Validate that the trainable PPO clone starts from the same deterministic
    # Base ACT action function before mutating it.
    check_count = min(len(anchors), 64)
    initial_clone_action = _probe_policy(
        workspace,
        workspace.unio4._policy,
        anchors[:check_count],
        int(args.probe_batch_size),
    )
    max_init_diff = float(
        (initial_clone_action - base_action[:check_count])
        .abs()
        .max()
        .item()
    )
    if max_init_diff > 1e-6:
        raise RuntimeError(
            "PPO policy clone does not match Base ACT before update: "
            f"max_abs={max_init_diff:.6g}"
        )

    # The immutable Base policy is no longer needed on GPU.
    workspace.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    workspace._build_finetune_dataloader()

    # Reseed immediately before the first shuffled PPO batch so this replay is
    # reproducible. The batch and policy action sample then follow the same
    # production code path used by Stage-2.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch = workspace.sample_finetune_batch()

    # Preview the exact sampled batch action/advantage, then restore RNG so
    # update_distribution samples the identical stochastic action chunk.
    replay_rng = _rng_state()
    batch_preview = _preview_exact_update_batch(
        workspace,
        batch,
    )
    _restore_rng_state(replay_rng)

    log_std_before = (
        workspace.unio4._policy._get_log_std()
        .detach()
        .float()
        .clone()
    )
    lr_before = float(
        workspace.unio4._optimizer.param_groups[0]["lr"]
    )
    clip_before = float(workspace.unio4._clip_ratio)

    step = 0
    decay_stop_step = int(cfg.unio4.decay_stop_step)
    decay_active = (
        decay_stop_step < 0
        or step <= decay_stop_step
    )
    linear_active = (
        bool(cfg.unio4.is_linear_decay)
        and decay_active
    )
    if linear_active:
        progress = step / max(
            int(cfg.unio4.bppo_steps),
            1,
        )
        bppo_lr_now = float(cfg.unio4.bppo_lr) * (
            1.0 - progress
        )
        clip_ratio_now = float(
            cfg.unio4.clip_ratio
        ) * (1.0 - progress)
    else:
        bppo_lr_now = None
        clip_ratio_now = None

    loss = workspace.unio4.update_distribution(
        batch=batch,
        critic=workspace.critic,
        is_clip_decay=(
            bool(cfg.unio4.is_clip_decay)
            and decay_active
        ),
        is_lr_decay=(
            bool(cfg.unio4.is_bppo_lr_decay)
            and decay_active
        ),
        is_linear_decay=linear_active,
        bppo_lr_now=bppo_lr_now,
        clip_ratio_now=clip_ratio_now,
    )

    log_std_after = (
        workspace.unio4._policy._get_log_std()
        .detach()
        .float()
        .clone()
    )
    lr_after = float(
        workspace.unio4._optimizer.param_groups[0]["lr"]
    )
    clip_after = float(workspace.unio4._clip_ratio)

    print("Probing ACT after one real PPO optimizer step...")
    one_step_action = _probe_policy(
        workspace,
        workspace.unio4._policy,
        anchors,
        int(args.probe_batch_size),
    )

    probe_rows = _build_probe_rows(
        base_action=base_action,
        one_step_action=one_step_action,
        final_postrl_action=final_postrl_action,
        terminal_action=terminal_action,
        anchors=anchors,
        episode_ids=anchor_episode_ids,
        episode_starts=anchor_starts,
        episode_ends=anchor_ends,
        progress_bins=int(args.progress_bins),
    )
    phase_summary = _aggregate_rows(
        probe_rows,
        "phase",
    )
    progress_summary = _aggregate_rows(
        probe_rows,
        "progress_bin",
    )

    _write_csv(
        output_dir / "one_step_probe_per_anchor.csv",
        probe_rows,
    )
    _write_csv(
        output_dir / "one_step_probe_phase_summary.csv",
        phase_summary,
    )
    _write_csv(
        output_dir / "one_step_probe_progress_summary.csv",
        progress_summary,
    )
    _plot_phase_summary(
        output_dir,
        phase_summary,
    )

    update_stats = {
        "loss": float(loss),
        "lr_before": lr_before,
        "lr_after": lr_after,
        "clip_ratio_before": clip_before,
        "clip_ratio_after": clip_after,
        "iteration_after": int(workspace.unio4.iteration),
        "log_std_mean_before": float(
            log_std_before.mean().item()
        ),
        "log_std_mean_after": float(
            log_std_after.mean().item()
        ),
        "log_std_mean_delta": float(
            (log_std_after - log_std_before)
            .mean()
            .item()
        ),
        "log_std_max_abs_delta": float(
            (log_std_after - log_std_before)
            .abs()
            .max()
            .item()
        ),
        "initial_clone_max_abs_action_diff": max_init_diff,
        "batch_preview": batch_preview,
    }

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
        "ppo_finetune_batch_size": int(
            cfg.unio4.finetune_batch_size
        ),
        "probe_anchors": int(len(anchors)),
        "update_stats": update_stats,
        "phase_summary": phase_summary,
        "progress_summary": progress_summary,
        "metric_contract": {
            "real_update": (
                "one call to the production BehaviorProximalPolicyOptimization."
                "update_distribution using the first reproducible shuffled "
                "finetune batch, the trained Stage-1 critic, configured LR/clip "
                "decay flags, scalar IQL advantage, and the actual optimizer"
            ),
            "one_step_drift": (
                "deterministic ACT mean after one real optimizer.step minus "
                "the Base ACT deterministic mean on the exact same cached "
                "observation latent"
            ),
            "terminal_direction": (
                "same-episode final H-action chunk minus Base ACT mean in "
                "normalized action space"
            ),
            "final_postrl_drift": (
                "final best-OPE Post-RL deterministic mean minus Base ACT mean"
            ),
            "cos_one_step_terminal": (
                "whether the true one-step network output drift points toward "
                "the terminal action template"
            ),
            "cos_one_step_final_postrl": (
                "whether the true one-step network output drift points toward "
                "the final observed Post-RL policy drift"
            ),
        },
        "scope_note": (
            "This replays one real optimizer update from the initial Base ACT "
            "policy and includes the actual ACT decoder/action-head parameter "
            "coupling. It does not reproduce 8000 accumulated updates, OPE-based "
            "old-policy refreshes, EMA selection, or intermediate optimizer state."
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=True,
        )

    _print_report(
        update_stats=update_stats,
        phase_rows=phase_summary,
        cfg=cfg,
    )
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
