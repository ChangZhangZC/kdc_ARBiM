"""CPU-only algorithmic smoke test; no Flywheel01 checkpoints required.

Run: python post_training/tests/analysis/test_flywheel01_dynamics/test_stochastic_smoke.py
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from common import Window
from test_smoke import FakeWorkspace, FakeDynamics
from evaluate import evaluate_window, write_csv
from evaluate_stochastic import evaluate_stochastic_window, summarize
from plot_stochastic import plot_stochastic


class SyntheticStochasticDynamics(FakeDynamics):
    def step(self, nobs_features, action, policy_features=None):
        # Toy model is perfect in expectation but injects random state noise.
        next_state=nobs_features + action[:, :1].reshape(-1,1,1)
        next_state=next_state + .05*torch.randn_like(next_state)
        n=len(action)
        return next_state.cpu().numpy(), np.zeros((n,1)), np.zeros((n,1),dtype=bool), {}


def main():
    n=5
    states=[]
    for j in range(n+1):
        states.append({
            "state":np.r_[np.float32(j),np.zeros(15,dtype=np.float32)],
            "head_rgb":np.zeros((3,2,2),dtype=np.uint8),
            "wrist_left_rgb":np.zeros((3,2,2),dtype=np.uint8),
            "wrist_right_rgb":np.zeros((3,2,2),dtype=np.uint8),
        })
    actions=np.zeros((n,16),dtype=np.float32)
    actions[:,0]=1.
    window=Window("synthetic",0,actions,states)
    ws=FakeWorkspace()
    dm=SyntheticStochasticDynamics()
    torch.manual_seed(23)
    deterministic=evaluate_window(ws,dm,window,"synthetic")
    stochastic=evaluate_stochastic_window(ws,dm,window,8)
    assert len(deterministic)==n
    assert len(stochastic)==n*8
    assert all(row["adv_horizon"]==row["step"]-1 for row in stochastic)
    assert all(row["adv_sign_flip"]==0 for row in stochastic)
    by_step=summarize(stochastic,deterministic)
    assert by_step[0]["stochastic_rollout_mse"]>0
    assert by_step[0]["deterministic_rollout_mse"]<1e-10
    assert by_step[-1]["stochastic_rollout_mse"]>0
    assert by_step[-1]["sto_n_margin_2"]==8
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)
        write_csv(out/"stochastic_by_step.csv",by_step)
        write_csv(out/"stochastic_per_replica.csv",stochastic)
        plot_stochastic(out)
        for name in ("stochastic_vs_elite_mean.png","stochastic_advantage_sign.png",
                     "advantage_margin_sensitivity.png"):
            assert (out/name).is_file(), name
    print("PASS: T+1 alignment, 8 stochastic replicas, critic consistency, plots")


if __name__=="__main__":
    main()
