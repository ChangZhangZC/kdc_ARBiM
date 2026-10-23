"""Evaluate Flywheel01 Single-Step latent Dynamics using Zarr or MuJoCo staging."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import numpy as np
import torch
from tqdm import tqdm

from common import load_context, tensor_obs, zarr_windows, staging_windows


@torch.no_grad()
def evaluate_window(workspace, dynamics, window, mode):
    device = workspace.device
    true_states = torch.cat([
        workspace.obs_adapter.encode(tensor_obs(o, device), track_grad=False)[:, 0]
        for o in window.observations
    ], dim=0)
    actions = workspace.obs_adapter.normalize_action(
        torch.as_tensor(window.actions, dtype=torch.float32, device=device)
    )
    if true_states.shape[0] != actions.shape[0] + 1:
        raise ValueError("Observed T+1 and action T misalignment")
    if actions.shape[-1] != workspace.action_dim:
        raise ValueError("Action dimension mismatch")

    elite_indices = dynamics._model().elites.long().to(device)
    predicted = true_states[0:1].clone()
    initial = predicted.clone()
    rows = []
    for i, action in enumerate(actions):
        action = action.view(1, -1)
        truth = true_states[i+1:i+2]
        # Predictive means, not stochastic samples. True teacher/rollout separation.
        means_rollout, _ = dynamics.model(predicted, action)
        means_teacher, _ = dynamics.model(true_states[i:i+1], action)
        if dynamics.predict_delta:
            means_rollout = means_rollout + predicted.unsqueeze(0)
            means_teacher = means_teacher + true_states[i:i+1].unsqueeze(0)
        next_pred = means_rollout.index_select(0, elite_indices).mean(dim=0)
        teacher = means_teacher.index_select(0, elite_indices).mean(dim=0)
        err = lambda a,b: float((a-b).square().mean().item())
        rollout_mse = err(next_pred, truth)
        teacher_mse = err(teacher, truth)
        zero_mse = err(initial, truth)
        step_zero_mse = err(true_states[i:i+1], truth)
        row = {
            "source":mode, "trajectory":window.name, "anchor":window.start,
            "step": i+1, "rollout_mse": rollout_mse,
            "teacher_mse": teacher_mse, "zero_from_start_mse": zero_mse,
            "zero_one_step_mse": step_zero_mse,
            "rollout_over_zero": rollout_mse/max(zero_mse,1e-12),
            "teacher_over_zero": teacher_mse/max(step_zero_mse,1e-12),
        }
        if workspace.critic is not None:
            critic = workspace.critic
            # The PPO trainer uses mean pooling of the ACT tokens.
            action_q = action
            real_z = dynamics._critic_readout(truth)
            pred_z = dynamics._critic_readout(next_pred)
            # Same action on both states: this probes the effect of state error,
            # not ground-truth validity of Q or Advantage.
            real_q = critic.minQ(real_z, action_q)
            pred_q = critic.minQ(pred_z, action_q)
            real_v = critic.value(real_z)
            pred_v = critic.value(pred_z)
            ar = float((real_q-real_v).item())
            ap = float((pred_q-pred_v).item())
            row.update({
                "q_real":float(real_q.item()), "q_pred":float(pred_q.item()),
                "v_real":float(real_v.item()), "v_pred":float(pred_v.item()),
                "adv_real":ar, "adv_pred":ap, "adv_abs_error":abs(ar-ap),
                "adv_sign_flip":int(ar*ap<0),
            })
        rows.append(row)
        predicted = next_pred
    return rows


def summarize(rows):
    fields = ["rollout_mse","teacher_mse","zero_from_start_mse",
              "zero_one_step_mse","rollout_over_zero","teacher_over_zero",
              "adv_abs_error","adv_sign_flip"]
    out = []
    for step in sorted(set(r["step"] for r in rows)):
        srows = [r for r in rows if r["step"] == step]
        item = {"step":step,"windows":len(srows)}
        for field in fields:
            data = np.array([r[field] for r in srows if field in r],dtype=np.float64)
            if len(data):
                item[field] = float(np.mean(data))
        out.append(item)
    return out


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No valid evaluation data")
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",required=True)
    p.add_argument("--source",choices=["zarr","staging"],default="zarr")
    p.add_argument("--staging-dir",default=None)
    p.add_argument("--dataset",default=None)
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--windows",type=int,default=64)
    p.add_argument("--horizon",type=int,default=32)
    p.add_argument("--stride",type=int,default=32)
    p.add_argument("--split",choices=["train","val"],default="val")
    args=p.parse_args()
    if not 1<=args.horizon<=32:
        p.error("This diagnostic expects a horizon from 1 to 32")
    output=pathlib.Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    ws,dynamics,cfg=load_context(args.run_dir,output,args.device,args.checkpoint,args.dataset)
    if args.source=="zarr":
        windows=zarr_windows(ws,args.horizon,args.stride,args.windows,args.split)
    else:
        if args.staging_dir is None:
            p.error("--staging-dir is required for --source staging")
        windows=staging_windows(args.staging_dir,args.horizon,args.stride,args.windows)
    rows=[]
    for window in tqdm(windows,total=args.windows,desc="Dynamics diagnostic"):
        rows.extend(evaluate_window(ws,dynamics,window,args.source))
    if not rows:
        raise RuntimeError("No usable sequences. Check episode length and data path.")
    by_step=summarize(rows)
    write_csv(output/"per_window.csv",rows)
    write_csv(output/"by_step.csv",by_step)
    metadata={"run_dir":args.run_dir,"source":args.source,"model_mode":"single_step",
              "horizon":args.horizon,"windows":len(rows)//args.horizon,
              "action_dim":ws.action_dim,"notes":"MSE is in ACT encoder latent space; Q(real) uses learned critic, not ground truth."}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2))
    for row in by_step:
        if row["step"] in [1,2,4,8,16,24,32]:
            print("step=%2d  rollout_mse=%.6g  teacher_mse=%.6g  zero_mse=%.6g  adv_flip=%.3f"%(
                row["step"],row["rollout_mse"],row["teacher_mse"],
                row["zero_from_start_mse"],row.get("adv_sign_flip",float("nan"))))
    print("Saved:",output)
    from plot import create_figures
    create_figures(output)


if __name__=="__main__":
    main()
