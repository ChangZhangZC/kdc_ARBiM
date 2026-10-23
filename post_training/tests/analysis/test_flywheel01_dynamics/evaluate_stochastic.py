"""Flywheel01 single-step Dynamics: deterministic elite mean vs exact PPO stochastic step.

Offline and read-only. Stochastic states use EnsembleDynamics_batch.step() itself,
including random elite selection and Gaussian variance sampling.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import load_context, tensor_obs, zarr_windows, staging_windows
from evaluate import evaluate_window, write_csv

THRESHOLDS = (0.0, 0.00005, 0.0001, 0.0002, 0.0003)


@torch.no_grad()
def evaluate_stochastic_window(workspace, dynamics, window, repeats):
    """One call to Dynamics.step for all independent replicas at each time step.

    Advantage is evaluated PRE-action: A(z_t, a_t) vs A(z_hat_t, a_t).
    Next-state MSE is evaluated POST-action: z_(t+1) vs z_hat_(t+1).
    """
    device = workspace.device
    true_states = torch.cat([
        workspace.obs_adapter.encode(tensor_obs(obs, device), track_grad=False)[:, 0]
        for obs in window.observations
    ], dim=0)
    actions = workspace.obs_adapter.normalize_action(
        torch.as_tensor(window.actions, dtype=torch.float32, device=device)
    )
    if len(true_states) != len(actions) + 1:
        raise ValueError(f"Misaligned T+1 observations for {window.name}")
    if actions.shape[-1] != workspace.action_dim:
        raise ValueError(f"Action mismatch: {actions.shape[-1]} vs {workspace.action_dim}")

    rollout = true_states[0:1].expand(repeats, -1, -1).contiguous()
    rows = []
    for t, single_action in enumerate(actions):
        action = single_action[None].expand(repeats, -1).contiguous()
        truth_pre = true_states[t:t+1]
        truth_next = true_states[t+1:t+2]
        critic = workspace.critic

        real_feat = dynamics._critic_readout(truth_pre)
        predicted_feat = dynamics._critic_readout(rollout)
        real_q = float(critic.minQ(real_feat, action[:1]).item())
        real_v = float(critic.value(real_feat).item())
        real_adv = real_q - real_v
        pred_q = critic.minQ(predicted_feat, action).reshape(repeats).cpu().numpy()
        pred_v = critic.value(predicted_feat).reshape(repeats).cpu().numpy()
        pred_adv = pred_q - pred_v

        # This is the EXACT stochastic transition path used by PPO for the
        # per-step advantage rollout. Dynamics.step returns sampled next tokens.
        next_rollout_np, _, _, _ = dynamics.step(rollout, action)
        next_rollout = torch.as_tensor(
            next_rollout_np, dtype=rollout.dtype, device=device
        )
        # Teacher-forced stochastic one-step prediction for comparison.
        true_batch = truth_pre.expand(repeats, -1, -1).contiguous()
        next_teacher_np, _, _, _ = dynamics.step(true_batch, action)
        next_teacher = torch.as_tensor(
            next_teacher_np, dtype=rollout.dtype, device=device
        )
        if not torch.isfinite(next_rollout).all() or not torch.isfinite(next_teacher).all():
            raise FloatingPointError(
                f"Dynamics sampling gave nonfinite values in {window.name}, "
                f"anchor={window.start}, step={t+1}"
            )

        sq_rollout = (next_rollout - truth_next).square().mean(dim=(1, 2))
        sq_teacher = (next_teacher - truth_next).square().mean(dim=(1, 2))
        for r in range(repeats):
            av = float(pred_adv[r])
            rows.append({
                "trajectory":window.name,
                "anchor":window.start,
                "replica":r,
                "step":t+1,
                "adv_horizon":t,
                "stoch_rollout_mse":float(sq_rollout[r].item()),
                "stoch_teacher_mse":float(sq_teacher[r].item()),
                "q_real":real_q,
                "q_pred":float(pred_q[r]),
                "v_real":real_v,
                "v_pred":float(pred_v[r]),
                "adv_real":real_adv,
                "adv_pred":av,
                "adv_abs_error":abs(real_adv-av),
                "adv_sign_flip":int(real_adv*av<0),
            })
        rollout = next_rollout
    return rows


def _safe_mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def summarize(stochastic, deterministic, thresholds=THRESHOLDS):
    steps=sorted(set(r["step"] for r in deterministic))
    out=[]
    for step in steps:
        stochastic_rows=[r for r in stochastic if r["step"]==step]
        mean_rows=[r for r in deterministic if r["step"]==step]
        if not stochastic_rows or not mean_rows:
            raise ValueError(f"Missing records for step={step}")
        st_mse=np.array([r["stoch_rollout_mse"] for r in stochastic_rows])
        st_teacher=np.array([r["stoch_teacher_mse"] for r in stochastic_rows])
        entry={
            "step":step,
            "adv_horizon":step-1,
            "replica_samples":len(stochastic_rows),
            "windows":len(mean_rows),
            "deterministic_rollout_mse":_safe_mean([r["rollout_mse"] for r in mean_rows]),
            "deterministic_teacher_mse":_safe_mean([r["teacher_mse"] for r in mean_rows]),
            "zero_from_start_mse":_safe_mean([r["zero_from_start_mse"] for r in mean_rows]),
            "stochastic_rollout_mse":float(st_mse.mean()),
            "stochastic_rollout_p10":float(np.quantile(st_mse,0.1)),
            "stochastic_rollout_p50":float(np.quantile(st_mse,0.5)),
            "stochastic_rollout_p90":float(np.quantile(st_mse,0.9)),
            "stochastic_teacher_mse":float(st_teacher.mean()),
            "stochastic_adv_abs_error":_safe_mean([r["adv_abs_error"] for r in stochastic_rows]),
            "deterministic_adv_abs_error":_safe_mean([r["adv_abs_error"] for r in mean_rows]),
            "stochastic_adv_sign_flip":_safe_mean([r["adv_sign_flip"] for r in stochastic_rows]),
            "deterministic_adv_sign_flip":_safe_mean([r["adv_sign_flip"] for r in mean_rows]),
            "median_abs_real_advantage":float(np.median(np.abs([r["adv_real"] for r in mean_rows]))),
        }
        for i, threshold in enumerate(thresholds):
            d=[r for r in mean_rows if abs(r["adv_real"])>=threshold]
            s=[r for r in stochastic_rows if abs(r["adv_real"])>=threshold]
            entry[f"margin_{i}"]=float(threshold)
            entry[f"det_n_margin_{i}"]=len(d)
            entry[f"sto_n_margin_{i}"]=len(s)
            entry[f"det_flip_margin_{i}"]=_safe_mean([r["adv_sign_flip"] for r in d])
            entry[f"sto_flip_margin_{i}"]=_safe_mean([r["adv_sign_flip"] for r in s])
        out.append(entry)
    return out


def main():
    parser=argparse.ArgumentParser(
        description="Compare exact PPO stochastic Dynamics rollout to elite-mean prediction."
    )
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--source",choices=["zarr","staging"],default="zarr")
    parser.add_argument("--staging-dir")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--windows",type=int,default=64)
    parser.add_argument("--repeats",type=int,default=8)
    parser.add_argument("--horizon",type=int,default=32)
    parser.add_argument("--stride",type=int,default=32)
    parser.add_argument("--split",choices=["val","train"],default="val")
    parser.add_argument("--seed",type=int,default=20261023)
    args=parser.parse_args()
    if not 1<=args.horizon<=32 or args.repeats<2 or args.windows<1:
        parser.error("require 1<=horizon<=32, repeats>=2, windows>=1")
    if args.source=="staging" and not args.staging_dir:
        parser.error("--staging-dir is required for staging source")
    out=Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    ws,dynamics,cfg=load_context(args.run_dir,out,args.device,args.checkpoint,args.dataset)
    # Workspace initialization sets its own training seed. Apply diagnostic
    # seed AFTER loading so stochastic repeats are reproducible.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.source=="zarr":
        windows=zarr_windows(ws,args.horizon,args.stride,args.windows,args.split)
    else:
        windows=staging_windows(args.staging_dir,args.horizon,args.stride,args.windows)
    stochastic=[]
    deterministic=[]
    count=0
    for window in tqdm(windows,total=args.windows,desc="Stochastic Dynamics diagnostic"):
        deterministic.extend(evaluate_window(ws,dynamics,window,args.source))
        stochastic.extend(evaluate_stochastic_window(ws,dynamics,window,args.repeats))
        count+=1
    if count==0:
        raise RuntimeError("No usable 32-step windows")
    summary=summarize(stochastic,deterministic)
    write_csv(out/"stochastic_per_replica.csv",stochastic)
    write_csv(out/"deterministic_per_window.csv",deterministic)
    write_csv(out/"stochastic_by_step.csv",summary)
    meta={
        "run_dir":args.run_dir,"source":args.source,"split":args.split,
        "windows":count,"repeats":args.repeats,"seed":args.seed,
        "horizon":args.horizon,
        "action_source":"recorded executed actions",
        "stochastic_transition":"EnsembleDynamics_batch.step(): random elite per sample, Gaussian predicted variance",
        "deterministic_transition":"mean across ensemble elite predicted means",
        "advantage_alignment":"At row step=j+1, uses true/predicted pre-action state at j",
        "margin_thresholds":list(THRESHOLDS),
        "note":"Observed-vs-predicted Advantage only tests critic sensitivity, not correct RL advantage",
    }
    (out/"stochastic_metadata.json").write_text(json.dumps(meta,indent=2))
    from plot_stochastic import plot_stochastic
    plot_stochastic(out)
    for row in summary:
        if row["step"] in (1,2,4,8,16,24,32):
            print(f"step={row['step']:2d} "
                  f"elite_mean={row['deterministic_rollout_mse']:.5f} "
                  f"ppo_random={row['stochastic_rollout_mse']:.5f} "
                  f"ppo_random_P90={row['stochastic_rollout_p90']:.5f} "
                  f"adv_flip={row['stochastic_adv_sign_flip']:.1%} "
                  f"flip_abs_A>=1e-4={row['sto_flip_margin_2']:.1%} "
                  f"eligible={row['sto_n_margin_2']}")
    print(f"Saved diagnostics to: {out}")


if __name__=="__main__":
    main()
