from __future__ import annotations

import argparse
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
    _processor_fingerprint,
    _resolve_config,
    _resolve_processor_dir,
    _resolve_required_path,
)
from analysis_10_multi_step_fixed_old import (  # noqa: E402
    _build_probe_context,
    _decay_args,
    _prepare_workspace,
    _probe_checkpoint,
    _write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analysis 11: replay the production Offline PPO accumulation "
            "from Base ACT, including the trained transition model, "
            "Dynamics OPE gate, and moving old-policy refresh. Compare "
            "the same fixed probe states against Analysis 10 to isolate "
            "the effect of OPE-accepted moving references."
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
        default=None,
        help=(
            "Completed PPO updates. Default: cfg.unio4.bppo_steps "
            "for a full production replay."
        ),
    )
    parser.add_argument(
        "--ope-every",
        type=int,
        default=None,
        help=(
            "Override cfg.unio4.eval_step. Leave unset for "
            "production-faithful OPE cadence."
        ),
    )
    parser.add_argument(
        "--probe-every",
        type=int,
        default=50,
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

    if args.probe_every < 1:
        raise ValueError(
            "--probe-every must be >= 1"
        )

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
        pathlib.Path(args.output_dir)
        .expanduser()
        .resolve()
        if args.output_dir
        else (
            REPO_ROOT
            / "post_training"
            / "outputs"
            / "analysis_11_production_ppo_replay"
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

    cfg.dynamics.load_pretrain = True
    cfg.dynamics.artifact_dir = str(
        stage1_dir / "dynamics"
    )
    cfg.unio4.artifact_dir = None
    cfg.unio4.global_best_dir = None
    cfg.unio4.global_best_ema_dir = None
    workspace.cfg = cfg

    if not bool(
        cfg.unio4.is_update_old_policy
    ):
        print(
            "WARNING: cfg.unio4.is_update_old_policy=false. "
            "Analysis 11 will still replay Dynamics OPE, but "
            "old_policy will remain fixed and should converge "
            "toward the Analysis-10 mechanism."
        )

    il_processor_fp = _processor_fingerprint(
        _resolve_processor_dir(checkpoint)
    )
    postrl_processor_fp = _processor_fingerprint(
        _resolve_processor_dir(
            postrl_checkpoint
        )
    )
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ; "
            "refusing drift comparison."
        )

    context = _build_probe_context(
        workspace,
        cfg=cfg,
        postrl_checkpoint=postrl_checkpoint,
        max_probes=args.max_probes,
        probe_batch_size=args.probe_batch_size,
    )

    workspace._build_dynamics()
    if not workspace._load_dynamics_if_needed():
        raise RuntimeError(
            "Expected pretrained Stage-1 transition model; "
            "Analysis 11 must not train Dynamics."
        )
    workspace.dynamics.model.eval()

    workspace._build_ppo()
    workspace.unio4._policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    workspace.unio4._old_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    workspace._build_ema()
    workspace._build_finetune_dataloader()

    total_steps = (
        int(cfg.unio4.bppo_steps)
        if args.steps is None
        else int(args.steps)
    )
    if total_steps < 1:
        raise ValueError(
            "--steps must be >= 1"
        )

    ope_every = (
        int(cfg.unio4.eval_step)
        if args.ope_every is None
        else int(args.ope_every)
    )
    if ope_every < 1:
        raise ValueError(
            "--ope-every must be >= 1"
        )

    # Production Stage 2 performs one initial OPE before the first
    # PPO update. This also consumes one finetune batch and therefore
    # must happen before the update loop to keep the same
    # DataLoader/RNG contract.
    best_mean_q, initial_reward = (
        workspace._evaluate_dynamics_ope()
    )
    ope_rows = [
        {
            "updates": 0,
            "mean_q": float(best_mean_q),
            "mean_reward": float(
                initial_reward
            ),
            "best_mean_q": float(
                best_mean_q
            ),
            "accepted_old_policy_refresh": False,
            "old_policy_version": int(
                workspace.unio4
                ._old_policy_version
            ),
        }
    ]

    probe_rows = []
    checkpoint_rows = []
    update_rows = []

    rows, checkpoint_row = _probe_checkpoint(
        workspace,
        context,
        completed_updates=0,
        probe_batch_size=args.probe_batch_size,
    )
    checkpoint_row["old_policy_version"] = int(
        workspace.unio4._old_policy_version
    )
    checkpoint_row["best_ope_mean_q"] = float(
        best_mean_q
    )
    probe_rows.extend(rows)
    checkpoint_rows.append(
        checkpoint_row
    )

    refresh_count = 0
    for completed in tqdm(
        range(1, total_steps + 1),
        desc="Analysis 11 production PPO replay",
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
        if workspace.ema is not None:
            workspace.ema.step(
                workspace.unio4._policy
            )
        workspace.global_step += 1

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
                "old_policy_version": int(
                    workspace.unio4
                    ._old_policy_version
                ),
            }
        )

        ope_happened = (
            completed % ope_every == 0
        )
        accepted = False

        if ope_happened:
            (
                current_mean_q,
                mean_reward,
            ) = workspace._evaluate_dynamics_ope()

            if (
                current_mean_q > best_mean_q
                and bool(
                    cfg.unio4
                    .is_update_old_policy
                )
            ):
                best_mean_q = current_mean_q
                workspace.unio4.set_old_policy()
                workspace.unio4._old_policy.set_frozen_encoder_pos_embed(
                    workspace._frozen_encoder_pos_embed
                )
                accepted = True
                refresh_count += 1

            ope_rows.append(
                {
                    "updates": completed,
                    "mean_q": float(
                        current_mean_q
                    ),
                    "mean_reward": float(
                        mean_reward
                    ),
                    "best_mean_q": float(
                        best_mean_q
                    ),
                    "accepted_old_policy_refresh": bool(
                        accepted
                    ),
                    "old_policy_version": int(
                        workspace.unio4
                        ._old_policy_version
                    ),
                }
            )

        # Probe at a regular cadence and always immediately after every
        # OPE/refresh event.
        if (
            completed
            % int(args.probe_every)
            == 0
            or ope_happened
            or completed == total_steps
        ):
            rows, checkpoint_row = _probe_checkpoint(
                workspace,
                context,
                completed_updates=completed,
                probe_batch_size=args.probe_batch_size,
            )
            checkpoint_row[
                "old_policy_version"
            ] = int(
                workspace.unio4
                ._old_policy_version
            )
            checkpoint_row[
                "best_ope_mean_q"
            ] = float(best_mean_q)
            checkpoint_row[
                "ope_happened"
            ] = bool(ope_happened)
            checkpoint_row[
                "old_policy_refreshed_here"
            ] = bool(accepted)
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
        output_dir / "ope_gate.csv",
        ope_rows,
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
        "analysis": "11_production_ppo_replay",
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
        "updates": total_steps,
        "configured_bppo_steps": int(
            cfg.unio4.bppo_steps
        ),
        "ope_every": ope_every,
        "configured_eval_step": int(
            cfg.unio4.eval_step
        ),
        "old_policy_refresh_enabled": bool(
            cfg.unio4.is_update_old_policy
        ),
        "old_policy_refresh_count": (
            refresh_count
        ),
        "final_old_policy_version": int(
            workspace.unio4
            ._old_policy_version
        ),
        "initial_ope_mean_q": float(
            ope_rows[0]["mean_q"]
        ),
        "final_best_ope_mean_q": float(
            best_mean_q
        ),
        "probe_anchors": int(
            len(context["anchors"])
        ),
        "chunk_size": int(
            cfg.n_action_steps
        ),
        "processor_fingerprint": (
            il_processor_fp
        ),
        "scope": (
            "Replays the production mechanisms that can change the "
            "Actor trajectory: real Offline PPO updates, optimizer/"
            "scheduler, EMA updates, transition-model Dynamics OPE "
            "at the configured cadence, and OPE-gated old_policy "
            "refresh. External env_runner evaluation/global-best "
            "checkpointing is intentionally omitted because it does "
            "not feed gradients or update old_policy in Stage 2."
        ),
        "comparison_to_analysis_10": (
            "If Analysis 10 remains near Base but Analysis 11 drifts "
            "strongly after accepted OPE refreshes, the moving PPO "
            "reference/OPE gate is implicated. If both drift "
            "similarly, the core fixed-Critic PPO accumulation is "
            "sufficient to explain the effect."
        ),
        "outputs": {
            "updates": "updates.csv",
            "ope_gate": "ope_gate.csv",
            "probe_checkpoints": (
                "probe_checkpoints.csv"
            ),
            "probe_phase_group": (
                "probe_phase_group.csv"
            ),
            "ppo_monitoring": (
                "ppo/monitoring/ppo_metrics.csv"
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
        "\n=== Analysis 11: Production PPO "
        "+ OPE old-policy replay ==="
    )
    print(f"Updates: {total_steps}")
    print(
        "OPE evaluations: "
        f"{len(ope_rows)}"
    )
    print(
        "Accepted old-policy refreshes: "
        f"{refresh_count}"
    )
    print(
        "Final best OPE mean Q: "
        f"{best_mean_q:.6f}"
    )
    print(
        "Final policy param RMS drift from Base: "
        f"{final_checkpoint['param_drift_rms']:.6g}"
    )
    print(
        "Final old-policy param RMS drift from Base: "
        f"{final_checkpoint['old_policy_param_drift_rms']:.6g}"
    )
    print(
        f"Saved diagnostics to: {output_dir}"
    )


if __name__ == "__main__":
    main()
