from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

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
from analysis_08_one_step_ppo_replay import _probe_policy  # noqa: E402
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


PHASE_ORDER = (
    "early_0_25",
    "middle_25_50",
    "late_50_75",
    "tail_75_100",
)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cosine_rows(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denom = (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)
    )
    return torch.where(
        denom > 1e-12,
        numerator / denom.clamp_min(1e-12),
        torch.full_like(numerator, float("nan")),
    )


def _nanmean(tensor: torch.Tensor) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())


def _parameter_drift(
    current,
    reference,
    *,
    trainable_only: bool = True,
) -> dict[str, float]:
    reference_params = dict(reference.named_parameters())
    sum_sq = 0.0
    count = 0
    max_abs = 0.0
    group_sq = {
        "decoder": 0.0,
        "decoder_pos_embed": 0.0,
        "action_head": 0.0,
        "raw_log_std": 0.0,
    }
    group_count = {key: 0 for key in group_sq}

    with torch.no_grad():
        for name, param in current.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            ref = reference_params.get(name)
            if ref is None:
                continue
            delta = param.detach().float() - ref.detach().float()
            sq = float(delta.square().sum().item())
            n = delta.numel()
            sum_sq += sq
            count += n
            max_abs = max(
                max_abs,
                float(delta.abs().max().item()),
            )
            for group in group_sq:
                if group == "raw_log_std":
                    match = name.endswith("raw_log_std")
                else:
                    match = (
                        f".{group}." in f".{name}."
                        or name.startswith(group + ".")
                    )
                if match:
                    group_sq[group] += sq
                    group_count[group] += n
                    break

    result = {
        "param_drift_l2": float(np.sqrt(sum_sq)),
        "param_drift_rms": float(
            np.sqrt(sum_sq / max(count, 1))
        ),
        "param_drift_max_abs": max_abs,
    }
    for group in group_sq:
        result[f"param_{group}_rms"] = float(
            np.sqrt(
                group_sq[group]
                / max(group_count[group], 1)
            )
        )
    return result


def _decay_args(cfg, completed_updates: int) -> dict:
    # Match production Stage-2 scheduling. The denominator remains the
    # configured full BPPO run length even when this diagnostic stops early.
    production_steps = max(int(cfg.unio4.bppo_steps), 1)
    step_index = max(int(completed_updates), 0)
    decay_stop_step = int(cfg.unio4.decay_stop_step)
    decay_active = (
        decay_stop_step < 0
        or step_index <= decay_stop_step
    )
    linear_active = (
        bool(cfg.unio4.is_linear_decay)
        and decay_active
    )
    if linear_active:
        progress = step_index / production_steps
        bppo_lr_now = (
            float(cfg.unio4.bppo_lr)
            * (1.0 - progress)
        )
        clip_ratio_now = (
            float(cfg.unio4.clip_ratio)
            * (1.0 - progress)
        )
    else:
        bppo_lr_now = None
        clip_ratio_now = None

    return {
        "is_clip_decay": (
            bool(cfg.unio4.is_clip_decay)
            and decay_active
        ),
        "is_lr_decay": (
            bool(cfg.unio4.is_bppo_lr_decay)
            and decay_active
        ),
        "is_linear_decay": linear_active,
        "bppo_lr_now": bppo_lr_now,
        "clip_ratio_now": clip_ratio_now,
    }


def _build_probe_context(
    workspace,
    *,
    cfg,
    postrl_checkpoint: pathlib.Path,
    max_probes: int | None,
    probe_batch_size: int,
):
    episode_starts, episode_ends = _episode_bounds(
        workspace.buffer.episode_ends
    )
    chunk_size = int(cfg.n_action_steps)
    stride = int(cfg.dataset.finetune_sequence_stride)
    (
        anchors,
        episode_ids,
        anchor_starts,
        anchor_ends,
    ) = _select_anchors(
        episode_starts,
        episode_ends,
        chunk_size,
        stride,
        max_probes,
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
        terminal_bank_raw[episode_ids]
    ).to(workspace.device)
    terminal_action = (
        workspace.obs_adapter.normalize_action(terminal_raw)
        .detach()
        .cpu()
    )

    state_raw = torch.from_numpy(
        np.asarray(
            workspace.buffer["state"][anchors],
            dtype=np.float32,
        )
    ).to(workspace.device)
    if (
        state_raw.ndim != 2
        or state_raw.shape[-1] != int(workspace.action_dim)
    ):
        raise RuntimeError(
            "Physical hold counterfactual requires state/action "
            "dimensions to match; "
            f"state={tuple(state_raw.shape)}, "
            f"action_dim={workspace.action_dim}."
        )
    hold_step = workspace.obs_adapter.normalize_action(
        state_raw
    )
    hold_action = (
        hold_step[:, None, :]
        .expand(-1, chunk_size, -1)
        .contiguous()
        .cpu()
    )

    latent_np = np.asarray(
        workspace.latent_cache.obs[anchors],
        dtype=np.float32,
    )
    latent = torch.from_numpy(latent_np).to(
        workspace.device
    )

    base_action = _probe_policy(
        workspace,
        workspace.model,
        anchors,
        probe_batch_size,
    )
    postrl_policy, postrl_kind = _load_postrl_policy(
        postrl_checkpoint,
        workspace.device,
        cfg,
    )
    postrl_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    final_postrl_action = _probe_policy(
        workspace,
        postrl_policy,
        anchors,
        probe_batch_size,
    )
    postrl_policy.to("cpu")
    del postrl_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_span = np.maximum(
        anchor_ends - anchor_starts - chunk_size,
        1,
    )
    progress = (
        (anchors - anchor_starts)
        / valid_span
    )
    phases = np.asarray(
        [_phase(float(x)) for x in progress],
        dtype=object,
    )
    return {
        "anchors": anchors,
        "episode_ids": episode_ids,
        "anchor_starts": anchor_starts,
        "anchor_ends": anchor_ends,
        "progress": progress,
        "phases": phases,
        "latent": latent,
        "base_action": base_action,
        "final_postrl_action": final_postrl_action,
        "terminal_action": terminal_action,
        "hold_action": hold_action,
        "postrl_kind": postrl_kind,
    }


@torch.no_grad()
def _qva(
    workspace,
    latent: torch.Tensor,
    action: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    state = latent.mean(dim=1)
    action = action.to(workspace.device)
    value = workspace.critic.value(state).reshape(-1)
    q = workspace.critic.minQ(
        state,
        action,
    ).reshape(-1)
    return (
        q.cpu(),
        value.cpu(),
        (q - value).cpu(),
    )


def _probe_checkpoint(
    workspace,
    context: dict,
    *,
    completed_updates: int,
    probe_batch_size: int,
) -> tuple[list[dict], dict]:
    current_action = _probe_policy(
        workspace,
        workspace.unio4._policy,
        context["anchors"],
        probe_batch_size,
    )
    base = context["base_action"]
    final = context["final_postrl_action"]
    terminal = context["terminal_action"]
    hold = context["hold_action"]
    latent = context["latent"]

    q_base, value, a_base = _qva(
        workspace,
        latent,
        base,
    )
    q_current, _, a_current = _qva(
        workspace,
        latent,
        current_action,
    )
    q_hold, _, a_hold = _qva(
        workspace,
        latent,
        hold,
    )
    q_final, _, a_final = _qva(
        workspace,
        latent,
        final,
    )

    drift = current_action - base
    final_direction = final - base
    terminal_direction = terminal - base
    rows = []

    for phase in ("all", *PHASE_ORDER):
        mask_np = (
            np.ones(len(base), dtype=bool)
            if phase == "all"
            else context["phases"] == phase
        )
        indices = np.flatnonzero(mask_np)
        if len(indices) == 0:
            continue
        idx = torch.as_tensor(
            indices,
            dtype=torch.long,
        )
        q_slice = {
            "value_mean": float(
                value[idx].mean().item()
            ),
            "q_base_mean": float(
                q_base[idx].mean().item()
            ),
            "q_current_mean": float(
                q_current[idx].mean().item()
            ),
            "q_hold_mean": float(
                q_hold[idx].mean().item()
            ),
            "q_final_postrl_mean": float(
                q_final[idx].mean().item()
            ),
            "adv_base_mean": float(
                a_base[idx].mean().item()
            ),
            "adv_current_mean": float(
                a_current[idx].mean().item()
            ),
            "adv_hold_mean": float(
                a_hold[idx].mean().item()
            ),
            "adv_final_postrl_mean": float(
                a_final[idx].mean().item()
            ),
            "q_current_gt_base_fraction": float(
                (
                    q_current[idx]
                    > q_base[idx]
                ).float().mean().item()
            ),
            "q_hold_gt_base_fraction": float(
                (
                    q_hold[idx]
                    > q_base[idx]
                ).float().mean().item()
            ),
        }

        for group_name, group_slice in ACTION_GROUPS.items():
            d = drift[
                idx,
                :,
                group_slice,
            ].reshape(len(indices), -1).float()
            f = final_direction[
                idx,
                :,
                group_slice,
            ].reshape(len(indices), -1).float()
            t = terminal_direction[
                idx,
                :,
                group_slice,
            ].reshape(len(indices), -1).float()
            current = current_action[
                idx,
                :,
                group_slice,
            ].reshape(len(indices), -1).float()
            hold_group = hold[
                idx,
                :,
                group_slice,
            ].reshape(len(indices), -1).float()

            row = {
                "updates": int(completed_updates),
                "phase": phase,
                "group": group_name,
                "n": int(len(indices)),
                "drift_rmse": float(
                    torch.sqrt(
                        d.square().mean()
                    ).item()
                ),
                "drift_l2_mean": float(
                    torch.linalg.vector_norm(
                        d,
                        dim=-1,
                    ).mean().item()
                ),
                "cos_drift_final": _nanmean(
                    _cosine_rows(d, f)
                ),
                "cos_drift_terminal": _nanmean(
                    _cosine_rows(d, t)
                ),
                "current_hold_rmse": float(
                    torch.sqrt(
                        (
                            current
                            - hold_group
                        ).square().mean()
                    ).item()
                ),
                **q_slice,
            }
            rows.append(row)

    param_stats = _parameter_drift(
        workspace.unio4._policy,
        workspace.model,
    )
    old_param_stats = _parameter_drift(
        workspace.unio4._old_policy,
        workspace.model,
        trainable_only=False,
    )
    checkpoint_summary = {
        "updates": int(completed_updates),
        **param_stats,
        "old_policy_param_drift_rms": (
            old_param_stats["param_drift_rms"]
        ),
        "old_policy_param_drift_max_abs": (
            old_param_stats["param_drift_max_abs"]
        ),
        "log_std_mean": float(
            workspace.unio4._policy._get_log_std()
            .detach()
            .float()
            .mean()
            .item()
        ),
        "lr": float(
            workspace.unio4._optimizer.param_groups[0]["lr"]
        ),
        "clip_ratio": float(
            workspace.unio4._clip_ratio
        ),
    }
    return rows, checkpoint_summary


def _prepare_workspace(
    *,
    stage1_dir: pathlib.Path,
    config_path: pathlib.Path,
    checkpoint: pathlib.Path,
    dataset: pathlib.Path,
    latent_cache_dir: pathlib.Path,
    device: str,
    output_dir: pathlib.Path,
):
    cfg = OmegaConf.load(config_path)
    cfg.input.policy_checkpoint = str(checkpoint)
    cfg.input.policy_checkpoint_type = "il"
    cfg.input.dataset_path = str(dataset)
    cfg.training.device = str(device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    cfg.dataset.use_latent_cache = True
    cfg.dataset.endpoint_obs_only = True
    cfg.dataset.latent_cache_dir = str(
        latent_cache_dir
    )
    cfg.critic.load_pretrain = True
    cfg.critic.artifact_dir = str(
        stage1_dir / "critic"
    )
    # Never write diagnostic outputs into historical production paths
    # embedded in a resolved Stage-2 config.
    cfg.unio4.artifact_dir = None
    cfg.unio4.global_best_dir = None
    cfg.unio4.global_best_ema_dir = None

    if cfg.get("ppo") is not None:
        cfg.ppo.enable_ratio_logging = True
        cfg.ppo.ratio_log_every_updates = 1
        cfg.ppo.enable_monitoring_csv = True
        cfg.ppo.monitor_every_updates = 1

    if not bool(cfg.chunk_as_single_action):
        raise RuntimeError(
            "Analysis 10 currently targets the sim_task1 "
            "whole-chunk PPO contract and requires "
            "chunk_as_single_action=true."
        )

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
            "Expected pretrained Stage-1 Q/V; "
            "Analysis 10 must not train the Critic."
        )
    workspace.critic.eval()
    return workspace, cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analysis 10: start from Base ACT, freeze the "
            "Stage-1 Critic and the initial PPO old/reference "
            "policy, then run many real production "
            "update_distribution calls. Periodically probe "
            "fixed states to test whether PPO accumulation "
            "alone creates the observed Post-RL/terminal/"
            "hold-like drift."
        )
    )
    parser.add_argument(
        "--stage1-dir",
        required=True,
    )
    parser.add_argument(
        "--postrl-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        default=None,
    )
    parser.add_argument(
        "--latent-cache-dir",
        required=True,
    )
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--probe-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        nargs="*",
        default=[
            0,
            1,
            10,
            50,
            100,
            250,
            500,
            1000,
        ],
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.probe_every < 1:
        raise ValueError("--probe-every must be >= 1")

    stage1_dir = pathlib.Path(
        args.stage1_dir
    ).expanduser().resolve()
    config_path = _resolve_config(
        stage1_dir,
        args.config,
    )
    raw_cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_required_path(
        args.checkpoint,
        raw_cfg.input.get("policy_checkpoint"),
        "Base ACT checkpoint",
    )
    dataset = _resolve_required_path(
        args.dataset,
        raw_cfg.input.get("dataset_path"),
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
    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            REPO_ROOT
            / "post_training"
            / "outputs"
            / "analysis_10_multi_step_fixed_old"
        )
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = (
        int(raw_cfg.training.seed)
        if args.seed is None
        else int(args.seed)
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    workspace, cfg = _prepare_workspace(
        stage1_dir=stage1_dir,
        config_path=config_path,
        checkpoint=checkpoint,
        dataset=dataset,
        latent_cache_dir=latent_cache_dir,
        device=args.device,
        output_dir=output_dir,
    )

    il_processor_fp = _processor_fingerprint(
        _resolve_processor_dir(checkpoint)
    )
    postrl_processor_fp = _processor_fingerprint(
        _resolve_processor_dir(postrl_checkpoint)
    )
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ; "
            "refusing drift comparison."
        )

    # Verify final Post-RL uses the frozen encoder/latent contract
    # before using it as a directional reference.
    postrl_policy, _ = _load_postrl_policy(
        postrl_checkpoint,
        workspace.device,
        cfg,
    )
    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    cache_encoder_sha = str(
        workspace.latent_cache.metadata.get(
            "encoder_sha256",
            "",
        )
    )
    if (
        workspace._fingerprint_module(postrl_encoder)
        != cache_encoder_sha
    ):
        raise RuntimeError(
            "Post-RL encoder does not match the frozen "
            "latent cache."
        )
    postrl_policy.to("cpu")
    del postrl_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    context = _build_probe_context(
        workspace,
        cfg=cfg,
        postrl_checkpoint=postrl_checkpoint,
        max_probes=args.max_probes,
        probe_batch_size=args.probe_batch_size,
    )

    workspace._build_ppo()
    workspace.unio4._policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    workspace.unio4._old_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    # This copies Base ACT into old_policy exactly once. Analysis 10
    # deliberately never calls set_old_policy() again.
    workspace._build_finetune_dataloader()

    probe_steps = {
        int(x)
        for x in args.probe_steps
        if 0 <= int(x) <= args.steps
    }
    probe_steps.update(
        range(
            0,
            args.steps + 1,
            args.probe_every,
        )
    )
    probe_steps.add(args.steps)

    probe_rows = []
    checkpoint_rows = []
    update_rows = []

    rows, checkpoint_row = _probe_checkpoint(
        workspace,
        context,
        completed_updates=0,
        probe_batch_size=args.probe_batch_size,
    )
    probe_rows.extend(rows)
    checkpoint_rows.append(checkpoint_row)

    for completed in tqdm(
        range(1, args.steps + 1),
        desc="Analysis 10 fixed-old PPO",
    ):
        batch = workspace.sample_finetune_batch()
        loss = workspace.unio4.update_distribution(
            batch=batch,
            critic=workspace.critic,
            **_decay_args(
                cfg,
                completed - 1,
            ),
        )
        update_rows.append(
            {
                "updates": completed,
                "loss": float(loss),
                "lr": float(
                    workspace.unio4._optimizer
                    .param_groups[0]["lr"]
                ),
                "clip_ratio": float(
                    workspace.unio4._clip_ratio
                ),
                "log_std_mean": float(
                    workspace.unio4._policy
                    ._get_log_std()
                    .detach()
                    .float()
                    .mean()
                    .item()
                ),
            }
        )

        if completed in probe_steps:
            rows, checkpoint_row = _probe_checkpoint(
                workspace,
                context,
                completed_updates=completed,
                probe_batch_size=args.probe_batch_size,
            )
            probe_rows.extend(rows)
            checkpoint_rows.append(
                checkpoint_row
            )

    workspace.unio4.flush_ratio_logs(
        force=True
    )
    workspace.unio4.flush_monitor_logs(
        force=True
    )
    _write_csv(
        output_dir / "updates.csv",
        update_rows,
    )
    _write_csv(
        output_dir / "probe_checkpoints.csv",
        checkpoint_rows,
    )
    _write_csv(
        output_dir / "probe_phase_group.csv",
        probe_rows,
    )

    summary = {
        "analysis": "10_multi_step_fixed_old",
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "postrl_checkpoint": str(
            postrl_checkpoint
        ),
        "dataset": str(dataset),
        "latent_cache_dir": str(
            latent_cache_dir
        ),
        "seed": seed,
        "updates": int(args.steps),
        "probe_steps": sorted(probe_steps),
        "probe_anchors": int(
            len(context["anchors"])
        ),
        "chunk_size": int(
            cfg.n_action_steps
        ),
        "fixed_old_policy": True,
        "processor_fingerprint": (
            il_processor_fp
        ),
        "production_reference_bppo_steps": int(
            cfg.unio4.bppo_steps
        ),
        "scope": (
            "Real Offline PPO update_distribution calls with the "
            "trained Stage-1 Critic, real finetune DataLoader, "
            "optimizer, clipping, LR/clip schedule, and shared "
            "ACT decoder. old_policy is intentionally frozen at "
            "the initial Base ACT; Dynamics OPE and old-policy "
            "refresh are intentionally excluded. Analysis 11 "
            "restores those production mechanisms."
        ),
        "outputs": {
            "updates": "updates.csv",
            "ppo_monitoring": (
                "ppo/monitoring/ppo_metrics.csv "
                "(path under this analysis output tree)"
            ),
            "probe_checkpoints": (
                "probe_checkpoints.csv"
            ),
            "probe_phase_group": (
                "probe_phase_group.csv"
            ),
        },
    }
    with (
        output_dir / "summary.json"
    ).open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=True,
        )

    final_checkpoint = checkpoint_rows[-1]
    print(
        "\n=== Analysis 10: "
        "Multi-step fixed-old PPO ==="
    )
    print(f"Updates: {args.steps}")
    print(
        "Probe anchors: "
        f"{len(context['anchors'])}"
    )
    print(
        "Final policy param RMS drift from Base: "
        f"{final_checkpoint['param_drift_rms']:.6g}"
    )
    print(
        "Fixed old-policy max abs param drift from Base: "
        f"{final_checkpoint['old_policy_param_drift_max_abs']:.6g}"
    )
    print(
        f"Saved diagnostics to: {output_dir}"
    )


if __name__ == "__main__":
    main()
