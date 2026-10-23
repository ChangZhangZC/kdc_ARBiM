"""Visual comparison of stochastic PPO Dynamics and deterministic elite-mean rollout."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    with Path(path).open(newline="") as f:
        return [
            {name:float(value) for name,value in line.items()}
            for line in csv.DictReader(f)
        ]


def plot_stochastic(output):
    output=Path(output)
    data=load_rows(output/"stochastic_by_step.csv")
    x=np.array([r["step"] for r in data])

    # Two scales prevent the no-change baseline from hiding model differences.
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(x,[r["deterministic_rollout_mse"] for r in data],
                 label="Deterministic elite mean",linewidth=2)
    axes[0].plot(x,[r["stochastic_rollout_mse"] for r in data],
                 label="PPO random elite + noise",linewidth=2)
    axes[0].fill_between(x,[r["stochastic_rollout_p10"] for r in data],
                        [r["stochastic_rollout_p90"] for r in data],
                        alpha=.15,label="Random rollout 10th–90th percentile")
    axes[0].plot(x,[r["deterministic_teacher_mse"] for r in data],
                 label="Deterministic teacher-forced",linestyle="--")
    axes[0].plot(x,[r["stochastic_teacher_mse"] for r in data],
                 label="Stochastic teacher-forced",linestyle=":")
    axes[0].set(title="Dynamics: error from true ACT latent",
                xlabel="Predicted environment steps",ylabel="Latent MSE")
    axes[0].legend(fontsize=8)
    axes[1].plot(x,[r["deterministic_rollout_mse"] for r in data],
                 label="Elite mean rollout")
    axes[1].plot(x,[r["stochastic_rollout_mse"] for r in data],
                 label="PPO stochastic rollout")
    axes[1].plot(x,[r["zero_from_start_mse"] for r in data],
                 label="No-change baseline")
    axes[1].set(title="Does Dynamics beat no-change baseline?",
                xlabel="Predicted environment steps",ylabel="Latent MSE")
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.22)
    fig.suptitle("Flywheel01 Single-step Dynamics | 32-step trajectory comparison")
    fig.tight_layout()
    fig.savefig(output/"stochastic_vs_elite_mean.png",dpi=180)
    plt.close(fig)

    x_adv=np.array([r["adv_horizon"] for r in data])
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(x_adv,[100*r["deterministic_adv_sign_flip"] for r in data],
                 label="Elite mean")
    axes[0].plot(x_adv,[100*r["stochastic_adv_sign_flip"] for r in data],
                 label="PPO stochastic")
    axes[0].set(title="All actions: sign disagreement",xlabel="Predicted pre-action steps",
                ylabel="Sign disagreement (%)",ylim=(0,100))
    axes[0].legend()
    axes[1].plot(x_adv,[100*r["det_flip_margin_2"] for r in data],
                 label="Elite mean |A_true| >= 1e-4")
    axes[1].plot(x_adv,[100*r["sto_flip_margin_2"] for r in data],
                 label="PPO stochastic |A_true| >= 1e-4")
    axes[1].set(title="Larger advantage only: sign disagreement",
                xlabel="Predicted pre-action steps",
                ylabel="Sign disagreement (%)",ylim=(0,100))
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.22)
    fig.suptitle("Critic sensitivity: observed ACT latent vs predicted ACT latent")
    fig.tight_layout()
    fig.savefig(output/"stochastic_advantage_sign.png",dpi=180)
    plt.close(fig)

    # Illustrate how raw sign-disagreement can be caused by near-zero margins.
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for step in [min(8,int(x[-1])),min(16,int(x[-1])),int(x[-1])]:
        row=next(r for r in data if int(r["step"])==step)
        threshold=[row[f"margin_{i}"] for i in range(5)]
        flip=[100*row[f"sto_flip_margin_{i}"] for i in range(5)]
        count=[row[f"sto_n_margin_{i}"] for i in range(5)]
        axes[0].plot(threshold,flip,marker="o",label=f"step {step}, stochastic")
        axes[1].plot(threshold,count,marker="o",label=f"step {step}")
    axes[0].set(title="Sign disagreement after excluding small |Advantage|",
                xlabel="Minimum |A_real|",ylabel="Sign disagreement (%)",ylim=(0,100))
    axes[1].set(title="Number of eligible action samples",xlabel="Minimum |A_real|",
                ylabel="Number of samples")
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend()
        ax.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(output/"advantage_margin_sensitivity.png",dpi=180)
    plt.close(fig)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    plot_stochastic(args.output_dir)
    print("Plots written to:",args.output_dir)


if __name__=="__main__":
    main()
