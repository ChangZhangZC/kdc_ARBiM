"""Print P0 PPO-normalized Advantage sign disagreement from the prior CSV.

No GPU, Dynamics model, or MuJoCo required.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",default="post_training/outputs/dynamics_diagnostics/flywheel01_stochastic/stochastic_by_step.csv")
    args=parser.parse_args()
    path=Path(args.csv).expanduser()
    if not path.is_file():
        parser.error(f"Missing CSV: {path}")
    with path.open(newline="") as file:
        data={int(float(row["step"])):row for row in csv.DictReader(file)}
    required=[
        "stochastic_norm_sign_flip","stochastic_norm_sign_flip_large",
        "normalized_large_margin_windows","deterministic_norm_sign_flip",
    ]
    if not data:
        parser.error("Empty CSV")
    missing=[key for key in required if key not in next(iter(data.values()))]
    if missing:
        parser.error(f"Old CSV schema missing {missing}; rerun run_stochastic_flywheel01.sh")
    print("P0: Advantage sign disagreement, real observation vs predicted latent")
    print("Rates use per-action-position batch normalization across recorded windows.")
    print("step | elite_mean_norm | PPO_sample_norm | PPO_large_A_norm | eligible_windows")
    for step in (1,2,4,8,16,24,32):
        if step not in data:
            continue
        row=data[step]
        print(
            f"{step:4d} | "
            f"{float(row['deterministic_norm_sign_flip']):15.1%} | "
            f"{float(row['stochastic_norm_sign_flip']):15.1%} | "
            f"{float(row['stochastic_norm_sign_flip_large']):16.1%} | "
            f"{int(float(row['normalized_large_margin_windows'])):16d}"
        )
    print("\nThese rates are diagnostic only: recorded actions and mini-batch composition")
    print("differ from the actual BPPO optimizer. Learned Q/V is not ground truth.")


if __name__=="__main__":
    main()
