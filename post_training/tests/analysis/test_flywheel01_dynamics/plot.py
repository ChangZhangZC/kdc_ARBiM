"""Generate readable, reproducible plots from Dynamics CSV outputs."""
from __future__ import annotations
import argparse
import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path,newline="") as f:
        return [{k:float(v) for k,v in row.items() if k not in ("source","trajectory")}
                for row in csv.DictReader(f)]


def create_figures(directory):
    directory=pathlib.Path(directory)
    data=read_csv(directory/"by_step.csv")
    x=np.array([int(r["step"]) for r in data])
    fig,ax=plt.subplots(figsize=(9,5))
    for key,label in [("rollout_mse","Autoregressive prediction"),
                      ("teacher_mse","Teacher-forced prediction"),
                      ("zero_from_start_mse","No-change baseline (from start)")]:
        ax.plot(x,[r[key] for r in data],label=label,linewidth=2)
    ax.set(xlabel="Prediction horizon (environment steps)",ylabel="ACT latent-state MSE",
           title="Flywheel01 Dynamics | Real observation vs predicted latent")
    ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(directory/"latent_error_vs_horizon.png",dpi=180);plt.close(fig)

    if "adv_abs_error" in data[0]:
        x_adv=np.array([int(r.get("adv_horizon",r["step"]-1)) for r in data])
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        axes[0].plot(x_adv,[r["adv_abs_error"] for r in data])
        axes[0].set(xlabel="Prediction horizon",ylabel="Mean absolute advantage difference",
                    title="Critic sensitivity to predicted states")
        axes[1].plot(x_adv,[r["adv_sign_flip"]*100 for r in data])
        axes[1].set(xlabel="Prediction horizon",ylabel="Sign disagreement (%)",
                    title="Advantage sign disagreement")
        for ax in axes: ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(directory/"advantage_disagreement.png",dpi=180);plt.close(fig)

    raw=read_csv(directory/"per_window.csv")
    final=int(x.max())
    final_rows=[r for r in raw if int(r["step"])==final]
    if final_rows:
        err=np.asarray([r["rollout_mse"] for r in final_rows])
        fig,ax=plt.subplots(figsize=(8,4))
        ax.hist(err,bins=min(30,max(5,len(err)//3)))
        ax.set(xlabel=f"Rollout latent MSE at step {final}",ylabel="Number of windows",
               title="Distribution of terminal prediction errors")
        ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(directory/"terminal_error_distribution.png",dpi=180);plt.close(fig)
    return directory


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()
    create_figures(args.output_dir)
    print("Plots saved to:",args.output_dir)


if __name__=="__main__":
    main()
