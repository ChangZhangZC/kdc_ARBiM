from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from analysis_06_ppo_local_advantage import ACTION_GROUPS, _phase  # noqa: E402
from analysis_08_one_step_ppo_replay import _probe_policy  # noqa: E402
from analysis_10_multistep_fixed_old import (  # noqa: E402
    _capture_trainable,
    _decay_args,
    _mean_finite,
    _prepare_experiment,
    _qva,
    _take_snapshot,
    _write_csv,
)


def _old_reference_rows(exp: dict, step: int) -> list[dict]:
    workspace = exp["workspace"]
    batch_size = int(exp["args"].probe_batch_size)
    old_action = _probe_policy(
        workspace,
        workspace.unio4._old_policy,
        exp["anchors"],
        batch_size,
    )
    current_action = _probe_policy(
        workspace,
        workspace.unio4._policy,
        exp["anchors"],
        batch_size,
    )
    qva = _qva(
        workspace,
        exp["states"],
        {
            "old": old_action,
            "current": current_action,
            "base": exp["base_action"],
        },
        batch_size,
    )
    progress = (
        (exp["anchors"] - exp["anchor_starts"])
        / np.maximum(exp["anchor_ends"] - exp["anchor_starts"] - 1, 1)
    )
    phases = np.asarray([_phase(float(x)) for x in progress], dtype=object)
    rows = []
    for phase in ("early_0_25", "middle_25_50", "late_50_75", "tail_75_100", "all"):
        mask = np.ones(len(exp["anchors"]), dtype=bool) if phase == "all" else phases == phase
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        idx = torch.as_tensor(indices, dtype=torch.long)
        for group, slc in ACTION_GROUPS.items():
            base_to_old = (old_action - exp["base_action"]).index_select(0, idx)[..., slc]
            old_to_current = (current_action - old_action).index_select(0, idx)[..., slc]
            rows.append({
                "step": int(step),
                "old_policy_version": int(workspace.unio4._old_policy_version),
                "phase": phase,
                "group": group,
                "count": int(len(indices)),
                "base_to_old_rmse": float(torch.sqrt(base_to_old.square().mean()).item()),
                "old_to_current_rmse": float(torch.sqrt(old_to_current.square().mean()).item()),
                "v_mean": _mean_finite(qva["v"][indices]),
                "q_base_mean": _mean_finite(qva["q_base"][indices]),
                "q_old_mean": _mean_finite(qva["q_old"][indices]),
                "q_current_mean": _mean_finite(qva["q_current"][indices]),
                "a_base_mean": _mean_finite(qva["a_base"][indices]),
                "a_old_mean": _mean_finite(qva["a_old"][indices]),
                "a_current_mean": _mean_finite(qva["a_current"][indices]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analysis 11: replay production Offline-PPO with pretrained dynamics OPE "
            "and the real current_mean_q > best_mean_q old-policy refresh rule."
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
        default="post_training/outputs/analysis_11_ope_refresh_replay",
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

    exp = _prepare_experiment(args, need_dynamics=True)
    exp["args"] = args
    workspace = exp["workspace"]
    cfg = exp["cfg"]
    exp["base_trainable_state"] = _capture_trainable(workspace.unio4._policy)

    workspace._build_ema()
    initial_old_version = int(workspace.unio4._old_policy_version)

    # Production ordering: _build_finetune_dataloader() has already copied Base ACT
    # into old_policy, then initial OPE consumes one finetune batch before step 1.
    best_mean_q, initial_mean_reward = workspace._evaluate_dynamics_ope()
    ope_rows = [{
        "step": 0,
        "current_mean_q": float(best_mean_q),
        "best_mean_q_before": float(best_mean_q),
        "best_mean_q_after": float(best_mean_q),
        "mean_reward": float(initial_mean_reward),
        "old_policy_refreshed": 0,
        "old_policy_version": int(workspace.unio4._old_policy_version),
    }]

    update_rows = []
    snapshot_rows = []
    parameter_rows = []
    old_rows = []

    rows, params = _take_snapshot(exp, 0)
    snapshot_rows.extend(rows)
    parameter_rows.extend(params)
    old_rows.extend(_old_reference_rows(exp, 0))

    for step in range(int(args.steps)):
        batch = workspace.sample_finetune_batch()
        loss = workspace.unio4.update_distribution(
            batch=batch,
            critic=workspace.critic,
            **_decay_args(cfg, step),
        )
        if workspace.ema is not None:
            workspace.ema.step(workspace.unio4._policy)
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

        refreshed = False
        if workspace.global_step % int(cfg.unio4.eval_step) == 0:
            current_mean_q, mean_reward = workspace._evaluate_dynamics_ope()
            best_before = best_mean_q
            if (
                current_mean_q > best_mean_q
                and bool(cfg.unio4.is_update_old_policy)
            ):
                best_mean_q = current_mean_q
                workspace.unio4.set_old_policy()
                refreshed = True
            ope_rows.append({
                "step": int(workspace.global_step),
                "current_mean_q": float(current_mean_q),
                "best_mean_q_before": float(best_before),
                "best_mean_q_after": float(best_mean_q),
                "mean_reward": float(mean_reward),
                "old_policy_refreshed": int(refreshed),
                "old_policy_version": int(workspace.unio4._old_policy_version),
            })
            print(
                f"[analysis11:ope] step={workspace.global_step} "
                f"q={current_mean_q:+.6f} best={best_mean_q:+.6f} "
                f"refresh={refreshed} old_version={workspace.unio4._old_policy_version}"
            )

        do_probe = (
            workspace.global_step % int(args.probe_every) == 0
            or refreshed
            or step + 1 == int(args.steps)
        )
        if do_probe:
            rows, params = _take_snapshot(exp, int(workspace.global_step))
            snapshot_rows.extend(rows)
            parameter_rows.extend(params)
            old_rows.extend(_old_reference_rows(exp, int(workspace.global_step)))

    output_dir = exp["output_dir"]
    _write_csv(output_dir / "update_metrics.csv", update_rows)
    _write_csv(output_dir / "ope_events.csv", ope_rows)
    _write_csv(output_dir / "snapshot_phase_group.csv", snapshot_rows)
    _write_csv(output_dir / "old_reference_phase_group.csv", old_rows)
    _write_csv(output_dir / "parameter_drift.csv", parameter_rows)

    refresh_count = int(sum(row["old_policy_refreshed"] for row in ope_rows))
    summary = {
        "analysis": "11_ope_refresh_replay",
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
        "eval_step": int(cfg.unio4.eval_step),
        "is_update_old_policy": bool(cfg.unio4.is_update_old_policy),
        "initial_old_policy_version": initial_old_version,
        "final_old_policy_version": int(workspace.unio4._old_policy_version),
        "old_policy_refresh_count": refresh_count,
        "initial_ope_mean_q": float(ope_rows[0]["current_mean_q"]),
        "final_best_ope_mean_q": float(best_mean_q),
        "contract": (
            "Uses production batch ordering including initial OPE, production PPO "
            "update_distribution, EMA stepping, dynamics OPE cadence, and the exact "
            "current_mean_q > best_mean_q && is_update_old_policy refresh condition. "
            "Environment evaluation/checkpoint selection is intentionally omitted."
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    print("\n=== Analysis 11 complete ===")
    print(
        f"steps={args.steps}, OPE refreshes={refresh_count}, "
        f"old version {initial_old_version}->{workspace.unio4._old_policy_version}"
    )
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
