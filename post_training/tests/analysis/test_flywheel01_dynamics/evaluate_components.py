"""Decompose Flywheel01 Dynamics error: elite mean / random elite / Gaussian sample.

Read-only, single-step mode. PPO branch calls the unmodified dynamics.step()
function. For paired comparisons the Random-Elite-Mean branch reuses the exact
elite IDs that the PPO sampling branch chose at that environment step.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import load_context, tensor_obs, zarr_windows, staging_windows
from evaluate import evaluate_window, write_csv
from post_rl.utils.common import CONST_EPS


def _mse(pred, truth):
    return (pred - truth).square().mean(dim=tuple(range(1, pred.ndim)))


def _adv(workspace, dynamics, state, action):
    # Exactly the same IQL (min(Q1,Q2)-V) readout as per-step BPPO.
    state_vector = dynamics._critic_readout(state)
    return workspace.critic.get_advantage(state_vector, action).reshape(-1)


def _standardize(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return (values-values.mean())/(std+CONST_EPS)


@torch.no_grad()
def evaluate_components_window(workspace, dynamics, window, repeats):
    device=workspace.device
    states=torch.cat([
        workspace.obs_adapter.encode(tensor_obs(o,device),track_grad=False)[:,0]
        for o in window.observations
    ],dim=0)
    actions=workspace.obs_adapter.normalize_action(
        torch.as_tensor(window.actions,dtype=torch.float32,device=device)
    )
    if len(states)!=len(actions)+1:
        raise ValueError(f"T+1 observation/action mismatch in {window.name}")
    if actions.shape[-1]!=workspace.action_dim:
        raise ValueError("Wrong action dimension")

    # Independent trajectories with IDENTICAL actions and initial state.
    selected_mean=states[:1].expand(repeats,-1,-1).contiguous()
    ppo_sample=selected_mean.clone()
    ensemble=dynamics._model()
    elite_idx=ensemble.elites.to(device=device,dtype=torch.long)
    rows,calibration=[],[]

    for j,action_j in enumerate(actions):
        action=action_j[None].expand(repeats,-1).contiguous()
        real_pre=states[j:j+1]
        real_next=states[j+1:j+2]
        a_real=float(_adv(workspace,dynamics,real_pre,action[:1]).item())
        a_member=_adv(workspace,dynamics,selected_mean,action)
        a_ppo=_adv(workspace,dynamics,ppo_sample,action)

        # 1. Exact PPO sampled transition, including random torch noise and
        #    one np.random.choice per independent batch element.
        # 2. Replay only the np elite selection for the no-Gaussian branch,
        #    leaving RNG state as if the PPO step was called just once.
        before=np.random.get_state()
        sampled_np,_,_,_=dynamics.step(ppo_sample,action)
        after=np.random.get_state()
        np.random.set_state(before)
        sampled_indices=ensemble.random_elite_idxs(repeats)
        np.random.set_state(after)
        selected=torch.as_tensor(sampled_indices,dtype=torch.long,device=device)
        ii=torch.arange(repeats,device=device)
        ppo_next=torch.as_tensor(sampled_np,device=device,dtype=ppo_sample.dtype)

        member_means,_=dynamics.model(selected_mean,action)
        if dynamics.predict_delta:
            member_means=member_means+selected_mean.unsqueeze(0)
        member_next=member_means[selected,ii]

        if not torch.isfinite(member_next).all() or not torch.isfinite(ppo_next).all():
            raise FloatingPointError(f"Nonfinite Dynamics at {window.name}:{window.start}+{j}")

        # One-step uncertainty calibration using *true pre-action* latent.
        # This is a mixture of Gaussians over the retained Elite models.
        means,logvars=dynamics.model(real_pre,action[:1])
        if dynamics.predict_delta:
            means=means+real_pre.unsqueeze(0)
        elite_means=means.index_select(0,elite_idx)[:,0]
        elite_vars=logvars.index_select(0,elite_idx)[:,0].exp()
        mix_mean=elite_means.mean(dim=0)
        v_epistemic=elite_means.var(dim=0,unbiased=False)
        v_aleatoric=elite_vars.mean(dim=0)
        v_total=(v_epistemic+v_aleatoric).clamp_min(1e-12)
        error=real_next[0]-mix_mean
        # The coverage uses moment-matched Gaussian intervals, a calibration
        # diagnostic, NOT exact quantiles of the multi-Gaussian mixture.
        scaled=error.abs()/v_total.sqrt()
        cal_row={
            "trajectory":window.name,"anchor":window.start,"step":j+1,
            "teacher_elite_mean_mse":float(error.square().mean().item()),
            "predicted_epistemic_var":float(v_epistemic.mean().item()),
            "predicted_aleatoric_var":float(v_aleatoric.mean().item()),
            "predicted_total_var":float(v_total.mean().item()),
            "squared_error_over_predicted_var":float((error.square()/v_total).mean().item()),
            "coverage_1sigma":float((scaled<=1.0).float().mean().item()),
            "coverage_2sigma":float((scaled<=2.0).float().mean().item()),
        }
        calibration.append(cal_row)

        # Isolate mean prediction error at the same randomly selected Elite.
        selected_true_mean=means[selected,0]
        # The auxiliary teacher sample is independent of the PPO rollout, but
        # follows the same conditional distribution at the true state.
        true_selected_logvar=logvars[selected,0]
        sampled_teacher=selected_true_mean+torch.exp(.5*true_selected_logvar)*torch.randn_like(selected_true_mean)
        member_errors=_mse(member_next,real_next)
        sample_errors=_mse(ppo_next,real_next)
        teacher_member_errors=_mse(selected_true_mean,real_next)
        teacher_sample_errors=_mse(sampled_teacher,real_next)

        for k in range(repeats):
            rows.append({
                "trajectory":window.name,"anchor":window.start,
                "replica":k,"step":j+1,"adv_horizon":j,
                "elite_id":int(selected[k].item()),
                "random_elite_mse":float(member_errors[k].item()),
                "ppo_random_mse":float(sample_errors[k].item()),
                "teacher_random_elite_mse":float(teacher_member_errors[k].item()),
                "teacher_ppo_sample_mse":float(teacher_sample_errors[k].item()),
                "adv_real":a_real,
                "adv_random_elite":float(a_member[k].item()),
                "adv_ppo_random":float(a_ppo[k].item()),
            })
        selected_mean=member_next
        ppo_sample=ppo_next
    return rows,calibration


def _mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def summarize_components(replica_rows, calibration_rows, baseline_rows):
    samples=defaultdict(list)
    cals=defaultdict(list)
    det=defaultdict(list)
    for row in replica_rows: samples[row["step"]].append(row)
    for row in calibration_rows: cals[row["step"]].append(row)
    for row in baseline_rows: det[row["step"]].append(row)
    out=[]
    for step in sorted(samples):
        sr=samples[step]
        cr=cals[step]
        dr=det[step]
        pairs=sorted({(r["trajectory"],r["anchor"]) for r in dr})
        if len(pairs)!=len(dr):
            raise ValueError(f"Duplicate window IDs at step {step}")
        real_by_pair={(r["trajectory"],r["anchor"]):r["adv_real"] for r in dr}
        raw_real=np.asarray([real_by_pair[p] for p in pairs])
        norm_real=_standardize(raw_real)
        replicas=sorted({r["replica"] for r in sr})
        by_replica=defaultdict(dict)
        for row in sr:
            key=(row["trajectory"],row["anchor"])
            by_replica[row["replica"]][key]=row

        norm_member_flips=[]
        norm_ppo_flips=[]
        norm_member_large=[]
        norm_ppo_large=[]
        for k in replicas:
            if set(by_replica[k])!=set(pairs):
                raise ValueError(f"Inconsistent window keys for replica {k}, step {step}")
            ordered=[by_replica[k][p] for p in pairs]
            mm=_standardize([r["adv_random_elite"] for r in ordered])
            pp=_standardize([r["adv_ppo_random"] for r in ordered])
            for i,row in enumerate(ordered):
                flip_m=int(norm_real[i]*mm[i]<0)
                flip_p=int(norm_real[i]*pp[i]<0)
                row["norm_adv_real"]=float(norm_real[i])
                row["norm_adv_random_elite"]=float(mm[i])
                row["norm_adv_ppo_random"]=float(pp[i])
                row["norm_flip_random_elite"]=flip_m
                row["norm_flip_ppo_random"]=flip_p
                is_large=abs(norm_real[i])>=.5
                row["norm_adv_large"]=int(is_large)
                norm_member_flips.append(flip_m)
                norm_ppo_flips.append(flip_p)
                if is_large:
                    norm_member_large.append(flip_m)
                    norm_ppo_large.append(flip_p)

        entry={
            "step":step,"adv_horizon":step-1,
            "windows":len(dr),"replica_samples":len(sr),
            "elite_mean_mse":_mean([r["rollout_mse"] for r in dr]),
            "random_elite_mse":_mean([r["random_elite_mse"] for r in sr]),
            "ppo_random_mse":_mean([r["ppo_random_mse"] for r in sr]),
            "zero_from_start_mse":_mean([r["zero_from_start_mse"] for r in dr]),
            "teacher_elite_mean_mse":_mean([r["teacher_elite_mean_mse"] for r in cr]),
            "teacher_random_elite_mse":_mean([r["teacher_random_elite_mse"] for r in sr]),
            "teacher_ppo_sample_mse":_mean([r["teacher_ppo_sample_mse"] for r in sr]),
            "predicted_epistemic_var":_mean([r["predicted_epistemic_var"] for r in cr]),
            "predicted_aleatoric_var":_mean([r["predicted_aleatoric_var"] for r in cr]),
            "predicted_total_var":_mean([r["predicted_total_var"] for r in cr]),
            "coverage_1sigma":_mean([r["coverage_1sigma"] for r in cr]),
            "coverage_2sigma":_mean([r["coverage_2sigma"] for r in cr]),
            "squared_error_over_predicted_var":_mean([r["squared_error_over_predicted_var"] for r in cr]),
            "raw_flip_random_elite":_mean([r["adv_real"]*r["adv_random_elite"]<0 for r in sr]),
            "raw_flip_ppo_random":_mean([r["adv_real"]*r["adv_ppo_random"]<0 for r in sr]),
            "norm_flip_random_elite":_mean(norm_member_flips),
            "norm_flip_ppo_random":_mean(norm_ppo_flips),
            "norm_flip_random_elite_large":_mean(norm_member_large),
            "norm_flip_ppo_random_large":_mean(norm_ppo_large),
            "eligible_large_windows":sum(abs(v)>=.5 for v in norm_real),
        }
        out.append(entry)
    return out


def main():
    p=argparse.ArgumentParser(description="Isolate random-Elite vs Gaussian Dynamics prediction errors")
    p.add_argument("--run-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--source",choices=["zarr","staging"],default="zarr")
    p.add_argument("--staging-dir")
    p.add_argument("--dataset")
    p.add_argument("--checkpoint")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--windows",type=int,default=64)
    p.add_argument("--repeats",type=int,default=8)
    p.add_argument("--horizon",type=int,default=32)
    p.add_argument("--stride",type=int,default=32)
    p.add_argument("--split",choices=["val","train"],default="val")
    p.add_argument("--seed",type=int,default=20261028)
    args=p.parse_args()
    if not 1<=args.horizon<=32 or args.repeats<2 or args.windows<1:
        p.error("Expected 1<=horizon<=32, repeats>=2, windows>=1")
    if args.source=="staging" and not args.staging_dir:
        p.error("--staging-dir is required with source=staging")

    output=Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    ws,dyn,cfg=load_context(args.run_dir,output,args.device,args.checkpoint,args.dataset)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    seq=(zarr_windows(ws,args.horizon,args.stride,args.windows,args.split)
         if args.source=="zarr"
         else staging_windows(args.staging_dir,args.horizon,args.stride,args.windows))
    rows,cal_rows,baseline=[],[],[]
    count=0
    for window in tqdm(seq,total=args.windows,desc="Dynamics component diagnostics"):
        baseline.extend(evaluate_window(ws,dyn,window,args.source))
        out,cal=evaluate_components_window(ws,dyn,window,args.repeats)
        rows.extend(out)
        cal_rows.extend(cal)
        count+=1
    if count==0:
        raise RuntimeError("No complete observation/action windows")
    by_step=summarize_components(rows,cal_rows,baseline)
    write_csv(output/"component_per_replica.csv",rows)
    write_csv(output/"component_calibration.csv",cal_rows)
    write_csv(output/"component_by_step.csv",by_step)
    info={
        "run_dir":args.run_dir,"dataset":args.dataset or str(cfg.input.dataset_path),
        "source":args.source,"windows":count,"repeats":args.repeats,
        "horizon":args.horizon,"seed":args.seed,
        "note":"Same sampled Elite IDs for paired no-noise and PPO-noisy paths. Divergence after step 1 includes nonlinear state propagation.",
        "sampling":"Noisy path calls unmodified EnsembleDynamics_batch.step().",
        "calibration":"Moment-matched mixture variance: E_elite[exp(logvar)] + Var_elite(mu), coverage of 1 and 2 moment-matched sigmas.",
        "advantage":"Q-V from trained Critic; not ground truth. Per-step z-scoring uses selected offline windows, not PPO minibatches.",
    }
    (output/"component_metadata.json").write_text(json.dumps(info,indent=2))
    from plot_components import plot_components
    plot_components(output)
    for row in by_step:
        if row["step"] in (1,2,4,8,16,24,32):
            print(
                f"step={row['step']:2d} "
                f"elite_mean={row['elite_mean_mse']:.4f} "
                f"random_elite={row['random_elite_mse']:.4f} "
                f"ppo_sample={row['ppo_random_mse']:.4f} "
                f"teacher ale_var={row['predicted_aleatoric_var']:.4f} "
                f"epi_var={row['predicted_epistemic_var']:.4f} "
                f"norm_flip={row['norm_flip_ppo_random']:.1%}"
            )
    print("Saved:",output)


if __name__=="__main__":
    main()
