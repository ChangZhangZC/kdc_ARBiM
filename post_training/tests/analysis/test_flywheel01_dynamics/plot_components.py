"""Plots for the Flywheel01 Dynamics error-decomposition experiment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return [
            {key:float(value) for key,value in record.items()}
            for record in csv.DictReader(handle)
        ]


def plot_components(output):
    output=Path(output)
    rows=read_rows(output/"component_by_step.csv")
    x=np.array([r["step"] for r in rows])
    xa=x-1

    fig,ax=plt.subplots(figsize=(10,5.5))
    for key,label in [
        ("elite_mean_mse","A: Elite mean (no noise)"),
        ("random_elite_mse","B: Random Elite mean (no noise)"),
        ("ppo_random_mse","C: PPO Random Elite + Gaussian"),
        ("zero_from_start_mse","No-change baseline"),
    ]:
        ax.plot(x,[r[key] for r in rows],linewidth=2,label=label)
    ax.set(
        xlabel="Environment transitions predicted from the initial observed state",
        ylabel="ACT latent state MSE",
        title="Flywheel01 | Three Dynamics rollout modes"
    )
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"three_mode_rollout_mse.png",dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.5))
    for key,label in [
        ("teacher_elite_mean_mse","Elite mean (no noise)"),
        ("teacher_random_elite_mse","Random Elite mean (no noise)"),
        ("teacher_ppo_sample_mse","Random Elite + Gaussian sample"),
        ("predicted_total_var","Predicted mixture variance (reference)"),
    ]:
        ax.plot(x,[r[key] for r in rows],linewidth=2,label=label)
    ax.set(
        xlabel="Teacher-forced transition position",
        ylabel="Per-coordinate latent MSE / variance",
        title="One-step error decomposition at observed latent states"
    )
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"teacher_one_step_components.png",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for key,label in [
        ("predicted_epistemic_var","Between-Elite variance"),
        ("predicted_aleatoric_var","Within-model Gaussian variance"),
        ("teacher_elite_mean_mse","Elite mean actual squared error"),
    ]:
        axes[0].plot(x,[r[key] for r in rows],label=label,linewidth=2)
    axes[0].set(xlabel="Teacher-forced transition position",
                ylabel="Latent variance / MSE",
                title="Predicted uncertainty vs observed mean error")
    axes[0].legend(fontsize=9)
    axes[1].plot(x,[100*r["coverage_1sigma"] for r in rows],label="Observed inside ±1σ")
    axes[1].plot(x,[100*r["coverage_2sigma"] for r in rows],label="Observed inside ±2σ")
    axes[1].axhline(68.27,linestyle="--",color="grey",alpha=.6,label="Gaussian ±1σ reference")
    axes[1].axhline(95.45,linestyle=":",color="grey",alpha=.6,label="Gaussian ±2σ reference")
    axes[1].set(xlabel="Teacher-forced transition position",
                ylabel="Coordinate coverage (%)",ylim=(0,100),
                title="Moment-matched uncertainty coverage (not exact mixture quantiles)")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(output/"variance_calibration.png",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(xa,[100*r["raw_flip_random_elite"] for r in rows],
                 label="Random Elite mean, raw Q-V")
    axes[0].plot(xa,[100*r["raw_flip_ppo_random"] for r in rows],
                 label="PPO random, raw Q-V")
    axes[0].set(title="Raw Advantage sign disagreement",xlabel="Pre-action predicted steps",
                ylabel="Sign disagreement (%)",ylim=(0,100))
    axes[0].legend(fontsize=9)
    axes[1].plot(xa,[100*r["norm_flip_random_elite"] for r in rows],
                 label="Random Elite mean, batch-normalized")
    axes[1].plot(xa,[100*r["norm_flip_ppo_random"] for r in rows],
                 label="PPO random, batch-normalized")
    axes[1].plot(xa,[100*r["norm_flip_ppo_random_large"] for r in rows],
                 label="PPO random, normalized |A| >= 0.5",linestyle="--")
    axes[1].set(title="Per-position normalized Advantage sign disagreement",
                xlabel="Pre-action predicted steps",ylabel="Sign disagreement (%)",
                ylim=(0,100))
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(output/"component_advantage_direction.png",dpi=180)
    plt.close(fig)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    print("Plots saved:",plot_components(args.output_dir))
