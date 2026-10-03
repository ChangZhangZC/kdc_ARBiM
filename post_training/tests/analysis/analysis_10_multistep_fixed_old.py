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
from analysis_06_ppo_local_advantage import ACTION_GROUPS, _phase, _select_anchors  # noqa: E402
from analysis_08_one_step_ppo_replay import _probe_policy  # noqa: E402
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    denom = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    out = (a * b).sum(dim=-1) / denom.clamp_min(1e-12)
    return torch.where(denom > 1e-12, out, torch.full_like(out, float("nan")))


def _mean_finite(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _parameter_group(name: str) -> str:
    if name == "raw_log_std":
        return "raw_log_std"
    if "action_head" in name:
        return "action_head"
    if "decoder_pos_embed" in name:
        return "decoder_pos_embed"
    if "decoder" in name:
        return "decoder"
    return "other"


def _capture_trainable(policy) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().double().clone()
        for name, param in policy.named_parameters()
        if param.requires_grad
    }


def _parameter_drift_rows(
    policy,
    base_state: dict[str, torch.Tensor],
    step: int,
) -> list[dict]:
    accum = defaultdict(lambda: {"base_sq": 0.0, "diff_sq": 0.0, "abs_sum": 0.0, "max": 0.0, "n": 0})
    current = dict(policy.named_parameters())
    for name, base in base_state.items():
        now = current[name].detach().cpu().double()
        diff = now - base
        group = _parameter_group(name)
        a = accum[group]
        a["base_sq"] += float((base * base).sum())
        a["diff_sq"] += float((diff * diff).sum())
        a["abs_sum"] += float(diff.abs().sum())
        a["max"] = max(a["max"], float(diff.abs().max()) if diff.numel() else 0.0)
        a["n"] += int(diff.numel())
    rows = []
    for group, a in sorted(accum.items()):
        rows.append({
            "step": int(step),
            "group": group,
            "relative_l2": (a["diff_sq"] ** 0.5) / max(a["base_sq"] ** 0.5, 1e-30),
            "mean_abs_diff": a["abs_sum"] / max(a["n"], 1),
            "max_abs_diff": a["max"],
        })
    return rows


def _decay_args(cfg, step: int) -> dict:
    total_steps = int(cfg.unio4.bppo_steps)
    decay_stop_step = int(cfg.unio4.decay_stop_step)
    decay_active = decay_stop_step < 0 or step <= decay_stop_step
    linear_active = bool(cfg.unio4.is_linear_decay) and decay_active
    if linear_active:
        progress = step / max(total_steps, 1)
        bppo_lr_now = float(cfg.unio4.bppo_lr) * (1.0 - progress)
        clip_ratio_now = float(cfg.unio4.clip_ratio) * (1.0 - progress)
    else:
        bppo_lr_now = None
        clip_ratio_now = None
    return {
        "is_clip_decay": bool(cfg.unio4.is_clip_decay) and decay_active,
        "is_lr_decay": bool(cfg.unio4.is_bppo_lr_decay) and decay_active,
        "is_linear_decay": linear_active,
        "bppo_lr_now": bppo_lr_now,
        "clip_ratio_now": clip_ratio_now,
    }


def _probe_states_from_cache(workspace, anchors: np.ndarray, batch_size: int) -> torch.Tensor:
    states = []
    for offset in range(0, len(anchors), batch_size):
        idx = anchors[offset:offset + batch_size]
        latent = torch.from_numpy(
            np.asarray(workspace.latent_cache.obs[idx], dtype=np.float32)
        ).to(workspace.device)
        if latent.ndim != 3:
            raise RuntimeError(f"Expected cached latent [B,S,D], got {tuple(latent.shape)}")
        states.append(latent.mean(dim=1).cpu())
    return torch.cat(states, dim=0)


@torch.no_grad()
def _qva(
    workspace,
    states_cpu: torch.Tensor,
    actions_cpu: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, np.ndarray]:
    result = {"v": []}
    for name in actions_cpu:
        result[f"q_{name}"] = []
        result[f"a_{name}"] = []
    for offset in range(0, len(states_cpu), batch_size):
        states = states_cpu[offset:offset + batch_size].to(workspace.device)
        v = workspace.critic.value(states).reshape(-1)
        result["v"].append(v.cpu())
        for name, action_all in actions_cpu.items():
            action = action_all[offset:offset + batch_size].to(workspace.device)
            q = workspace.critic.minQ(states, action).reshape(-1)
            result[f"q_{name}"].append(q.cpu())
            result[f"a_{name}"].append((q - v).cpu())
    return {
        key: torch.cat(parts).numpy()
        for key, parts in result.items()
    }


def _snapshot_rows(
    *,
    step: int,
    base_action: torch.Tensor,
    current_action: torch.Tensor,
    final_action: torch.Tensor,
    terminal_action: torch.Tensor,
    qva: dict[str, np.ndarray],
    anchors: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
) -> list[dict]:
    progress = (anchors - episode_starts) / np.maximum(episode_ends - episode_starts - 1, 1)
    phases = np.asarray([_phase(float(x)) for x in progress], dtype=object)
    base_to_current = current_action - base_action
    base_to_final = final_action - base_action
    base_to_terminal = terminal_action - base_action
    rows = []
    for phase in ("early_0_25", "middle_25_50", "late_50_75", "tail_75_100", "all"):
        mask_np = np.ones(len(anchors), dtype=bool) if phase == "all" else phases == phase
        idx = np.flatnonzero(mask_np)
        if len(idx) == 0:
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        for group, slc in ACTION_GROUPS.items():
            drift = base_to_current.index_select(0, idx_t)[..., slc]
            final_drift = base_to_final.index_select(0, idx_t)[..., slc]
            terminal_drift = base_to_terminal.index_select(0, idx_t)[..., slc]
            cur_to_final = (current_action - final_action).index_select(0, idx_t)[..., slc]
            cur_to_terminal = (current_action - terminal_action).index_select(0, idx_t)[..., slc]
            terminal_norm = torch.linalg.vector_norm(
                terminal_drift.reshape(terminal_drift.shape[0], -1), dim=-1
            )
            terminal_unit = terminal_drift.reshape(terminal_drift.shape[0], -1) / (
                terminal_norm.unsqueeze(-1).clamp_min(1e-12)
            )
            current_projection_terminal = (
                drift.reshape(drift.shape[0], -1) * terminal_unit
            ).sum(dim=-1)
            current_projection_terminal = torch.where(
                terminal_norm > 1e-12,
                current_projection_terminal,
                torch.full_like(current_projection_terminal, float("nan")),
            )
            rows.append({
                "step": int(step),
                "phase": phase,
                "group": group,
                "count": int(len(idx)),
                "base_to_current_rmse": float(torch.sqrt(drift.square().mean()).item()),
                "current_to_final_rmse": float(torch.sqrt(cur_to_final.square().mean()).item()),
                "current_to_terminal_rmse": float(
                    torch.sqrt(cur_to_terminal.square().mean()).item()
                ),
                "cos_base_current_to_base_final": _mean_finite(
                    _cosine(drift, final_drift).numpy()
                ),
                "cos_base_current_to_terminal": _mean_finite(
                    _cosine(drift, terminal_drift).numpy()
                ),
                "cos_base_final_to_terminal": _mean_finite(
                    _cosine(final_drift, terminal_drift).numpy()
                ),
                "projection_base_current_onto_terminal": _mean_finite(
                    current_projection_terminal.numpy()
                ),
                "v_mean": _mean_finite(qva["v"][idx]),
                "q_base_mean": _mean_finite(qva["q_base"][idx]),
                "q_current_mean": _mean_finite(qva["q_current"][idx]),
                "q_final_mean": _mean_finite(qva["q_final"][idx]),
                "q_terminal_mean": _mean_finite(qva["q_terminal"][idx]),
                "q_current_minus_base_mean": _mean_finite(
                    qva["q_current"][idx] - qva["q_base"][idx]
                ),
                "q_terminal_minus_base_mean": _mean_finite(
                    qva["q_terminal"][idx] - qva["q_base"][idx]
                ),
                "a_base_mean": _mean_finite(qva["a_base"][idx]),
                "a_current_mean": _mean_finite(qva["a_current"][idx]),
                "a_final_mean": _mean_finite(qva["a_final"][idx]),
                "a_terminal_mean": _mean_finite(qva["a_terminal"][idx]),
            })
    return rows


def _prepare_experiment(args, *, need_dynamics: bool = False):
    stage1_dir = pathlib.Path(args.stage1_dir).expanduser().resolve()
    config_path = _resolve_config(stage1_dir, args.config)
    cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_required_path(
        args.checkpoint, cfg.input.get("policy_checkpoint"), "Base ACT checkpoint"
    )
    dataset = _resolve_required_path(
        args.dataset, cfg.input.get("dataset_path"), "Offline dataset"
    )
    latent_cache_dir = _resolve_required_path(
        args.latent_cache_dir, None, "Frozen latent cache"
    )
    postrl_checkpoint = _resolve_required_path(
        args.postrl_checkpoint, None, "Post-RL checkpoint"
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
    if need_dynamics:
        cfg.dynamics.load_pretrain = True
        cfg.dynamics.artifact_dir = str(stage1_dir / "dynamics")
    cfg.ppo.enable_ratio_logging = True
    cfg.ppo.ratio_log_every_updates = 1
    cfg.ppo.enable_monitoring_csv = True
    cfg.ppo.monitor_every_updates = 1

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError("Analysis requires pretrained Stage-1 Q/V; it must not retrain Critic.")
    workspace.critic.eval()

    if need_dynamics:
        workspace._build_dynamics()
        if not workspace._load_dynamics_if_needed():
            raise RuntimeError("Analysis 11 requires pretrained Stage-1 dynamics.")
        workspace.dynamics.model.eval()

    if int(workspace.action_dim) != 16:
        raise RuntimeError("These sim_task1 diagnostics expect the 16D bimanual action contract.")

    episode_starts, episode_ends = _episode_bounds(workspace.buffer.episode_ends)
    anchors, anchor_episode_ids, anchor_starts, anchor_ends = _select_anchors(
        episode_starts,
        episode_ends,
        int(cfg.n_action_steps),
        int(cfg.dataset.finetune_sequence_stride),
        args.max_probes,
    )

    chunk_size = int(cfg.n_action_steps)
    terminal_raw = np.stack([
        np.asarray(workspace.buffer["action"][int(end) - chunk_size:int(end)], dtype=np.float32)
        for end in episode_ends
    ])
    terminal_action = (
        workspace.obs_adapter.normalize_action(
            torch.from_numpy(terminal_raw[anchor_episode_ids]).to(workspace.device)
        ).cpu()
    )

    # Loading the diagnostic final policy constructs a model and can consume RNG.
    # Preserve the training RNG so reference probing cannot change PPO samples.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    postrl_policy, postrl_kind = _load_postrl_policy(
        postrl_checkpoint, workspace.device, cfg
    )
    postrl_policy.set_frozen_encoder_pos_embed(workspace._frozen_encoder_pos_embed)
    if int(postrl_policy.config.chunk_size) != int(workspace.model.config.chunk_size):
        raise RuntimeError("Base/Post-RL chunk-size mismatch.")

    base_proc = _resolve_processor_dir(checkpoint)
    post_proc = _resolve_processor_dir(postrl_checkpoint)
    base_fp = _processor_fingerprint(base_proc)
    post_fp = _processor_fingerprint(post_proc)
    if base_fp != post_fp:
        raise RuntimeError("Base ACT and Post-RL processor bundles differ.")

    postrl_encoder = ACTCriticEncoder(postrl_policy.model, copy_model=False).to(workspace.device).eval()
    cache_encoder_sha = str(workspace.latent_cache.metadata.get("encoder_sha256", ""))
    if workspace._fingerprint_module(postrl_encoder) != cache_encoder_sha:
        raise RuntimeError("Post-RL encoder does not match the frozen latent cache.")

    base_action = _probe_policy(
        workspace, workspace.model, anchors, int(args.probe_batch_size)
    )
    final_action = _probe_policy(
        workspace, postrl_policy, anchors, int(args.probe_batch_size)
    )
    states = _probe_states_from_cache(
        workspace, anchors, int(args.probe_batch_size)
    )

    postrl_policy.to("cpu")
    del postrl_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    workspace._build_ppo()
    workspace.unio4._policy.set_frozen_encoder_pos_embed(workspace._frozen_encoder_pos_embed)
    workspace.unio4._old_policy.set_frozen_encoder_pos_embed(workspace._frozen_encoder_pos_embed)
    workspace.unio4.set_ratio_log_dir(None)
    workspace._build_finetune_dataloader()

    return {
        "workspace": workspace,
        "cfg": cfg,
        "config_path": config_path,
        "stage1_dir": stage1_dir,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "latent_cache_dir": latent_cache_dir,
        "postrl_checkpoint": postrl_checkpoint,
        "postrl_kind": postrl_kind,
        "processor_fingerprint": base_fp,
        "output_dir": output_dir,
        "anchors": anchors,
        "anchor_episode_ids": anchor_episode_ids,
        "anchor_starts": anchor_starts,
        "anchor_ends": anchor_ends,
        "base_action": base_action,
        "final_action": final_action,
        "terminal_action": terminal_action,
        "states": states,
    }


def _take_snapshot(exp: dict, step: int) -> tuple[list[dict], list[dict]]:
    workspace = exp["workspace"]
    current_action = _probe_policy(
        workspace,
        workspace.unio4._policy,
        exp["anchors"],
        int(exp["args"].probe_batch_size),
    )
    actions = {
        "base": exp["base_action"],
        "current": current_action,
        "final": exp["final_action"],
        "terminal": exp["terminal_action"],
    }
    qva = _qva(
        workspace,
        exp["states"],
        actions,
        int(exp["args"].probe_batch_size),
    )
    rows = _snapshot_rows(
        step=step,
        base_action=exp["base_action"],
        current_action=current_action,
        final_action=exp["final_action"],
        terminal_action=exp["terminal_action"],
        qva=qva,
        anchors=exp["anchors"],
        episode_starts=exp["anchor_starts"],
        episode_ends=exp["anchor_ends"],
    )
    param_rows = _parameter_drift_rows(
        workspace.unio4._policy, exp["base_trainable_state"], step
    )
    return rows, param_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analysis 10: sequential production Offline-PPO updates with the Critic "
            "and PPO old/reference policy held fixed at the initial Base ACT."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/analysis_10_multistep_fixed_old",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--max-probes", type=int, default=256)
    args = parser.parse_args()
    if args.steps < 1 or args.probe_every < 1:
        raise ValueError("--steps and --probe-every must be >= 1")

    exp = _prepare_experiment(args, need_dynamics=False)
    exp["args"] = args
    workspace = exp["workspace"]
    cfg = exp["cfg"]

    old_version = int(workspace.unio4._old_policy_version)
    exp["base_trainable_state"] = _capture_trainable(workspace.unio4._policy)

    update_rows = []
    snapshot_rows = []
    parameter_rows = []
    rows, params = _take_snapshot(exp, 0)
    snapshot_rows.extend(rows)
    parameter_rows.extend(params)

    for step in range(int(args.steps)):
        if int(workspace.unio4._old_policy_version) != old_version:
            raise RuntimeError("Analysis 10 contract violated: old_policy changed.")
        batch = workspace.sample_finetune_batch()
        loss = workspace.unio4.update_distribution(
            batch=batch,
            critic=workspace.critic,
            **_decay_args(cfg, step),
        )
        workspace.global_step += 1
        monitor = workspace.unio4._monitor_records[-1]
        ratio = workspace.unio4._ratio_records[-1]
        update_rows.append({
            "step": int(workspace.global_step),
            "loss_returned": float(loss),
            "old_policy_version": int(workspace.unio4._old_policy_version),
            "policy_loss": monitor["policy_loss"],
            "ratio_mean": monitor["ratio_mean"],
            "approx_kl": monitor["approx_kl"],
            "clip_fraction": monitor["clip_fraction"],
            "adv_pre_norm_mean": monitor["adv_pre_norm_mean"],
            "adv_pre_norm_std": monitor["adv_pre_norm_std"],
            "grad_norm": monitor["grad_norm"],
            "lr": monitor["lr"],
            "clip_ratio": monitor["clip_ratio"],
            "log_std_mean": monitor["log_std_mean"],
            "ratio_q05": ratio["ratio_q05"],
            "ratio_q50": ratio["ratio_q50"],
            "ratio_q95": ratio["ratio_q95"],
        })
        if workspace.global_step % int(args.probe_every) == 0 or step + 1 == int(args.steps):
            rows, params = _take_snapshot(exp, int(workspace.global_step))
            snapshot_rows.extend(rows)
            parameter_rows.extend(params)
            print(
                f"[analysis10] step={workspace.global_step}/{args.steps} "
                f"loss={loss:+.6f} old_version={workspace.unio4._old_policy_version}"
            )

    output_dir = exp["output_dir"]
    _write_csv(output_dir / "update_metrics.csv", update_rows)
    _write_csv(output_dir / "snapshot_phase_group.csv", snapshot_rows)
    _write_csv(output_dir / "parameter_drift.csv", parameter_rows)

    summary = {
        "analysis": "10_multistep_fixed_old",
        "config": str(exp["config_path"]),
        "stage1_dir": str(exp["stage1_dir"]),
        "base_checkpoint": str(exp["checkpoint"]),
        "postrl_checkpoint": str(exp["postrl_checkpoint"]),
        "postrl_checkpoint_kind": exp["postrl_kind"],
        "dataset": str(exp["dataset"]),
        "latent_cache_dir": str(exp["latent_cache_dir"]),
        "processor_fingerprint": exp["processor_fingerprint"],
        "steps": int(args.steps),
        "probe_every": int(args.probe_every),
        "probe_anchors": int(len(exp["anchors"])),
        "fixed_old_policy_version": old_version,
        "final_old_policy_version": int(workspace.unio4._old_policy_version),
        "metric_contract": {
            "base_to_current_rmse": (
                "deterministic current-policy drift from the immutable Base ACT on fixed "
                "offline probe observations"
            ),
            "cos_base_current_to_terminal": (
                "cosine between Base->current drift and Base->same-episode terminal-chunk "
                "direction; positive means terminal-ward output drift"
            ),
            "projection_base_current_onto_terminal": (
                "signed Base->current displacement along the unit terminal direction in "
                "normalized action-chunk space"
            ),
            "q_current_minus_base_mean": (
                "same-state Critic preference for current policy chunk over Base chunk"
            ),
        },
        "contract": (
            "Critic fixed; old/reference policy copied from Base ACT once before step 1 "
            "and never refreshed. Batches, PPO loss, clipping, optimizer, and decay use "
            "the production Offline-PPO implementation."
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    print("\n=== Analysis 10 complete ===")
    print(f"steps={args.steps}, fixed old-policy version={old_version}")
    print(f"probe anchors={len(exp['anchors'])}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
