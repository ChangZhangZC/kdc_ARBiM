"""A no-checkpoint smoke test for temporal alignment and chart generation.

Run in the ARBiM python environment:
    python post_training/tests/analysis/test_flywheel01_dynamics/test_smoke.py
"""
import tempfile
from pathlib import Path
import numpy as np
import torch

from evaluate import evaluate_window, summarize, write_csv
from common import Window
from plot import create_figures


class FakeAdapter:
    def encode(self, obs, track_grad=False):
        # [B=1, time=1, tokens=2, features=4]; state[0] is the synthetic time.
        x = obs["state"][..., :1]
        return x.reshape(1, 1, 1, 1).expand(1, 1, 2, 4)

    def normalize_action(self, actions):
        return actions


class FakeEnsembleModel:
    def __init__(self):
        self.elites = torch.tensor([0, 1])

    def __call__(self, state, action):
        # Perfect delta: action[:,0] is the synthetic state displacement.
        delta = action[:, :1].reshape(1, 1, 1, 1).expand(2, 1, 2, 4)
        return delta, torch.zeros_like(delta)


class FakeDynamics:
    def __init__(self):
        self.model = FakeEnsembleModel()
        self.predict_delta = True

    def _model(self):
        return self.model

    def _critic_readout(self, x):
        return x.mean(dim=1)


class FakeCritic:
    def minQ(self, s, a):
        return s[:, :1] + a[:, :1]

    def value(self, s):
        return s[:, :1]


class FakeWorkspace:
    device = torch.device("cpu")
    action_dim = 16
    obs_adapter = FakeAdapter()
    critic = FakeCritic()


def main():
    n=4
    obs=[]
    for i in range(n+1):
        obs.append({
            "state": np.r_[np.float32(i), np.zeros(15,dtype=np.float32)],
            "head_rgb": np.zeros((3,2,2),dtype=np.uint8),
            "wrist_left_rgb": np.zeros((3,2,2),dtype=np.uint8),
            "wrist_right_rgb": np.zeros((3,2,2),dtype=np.uint8),
        })
    actions=np.zeros((n,16),dtype=np.float32)
    actions[:,0]=1.
    window=Window("synthetic",0,actions,obs)
    rows=evaluate_window(FakeWorkspace(),FakeDynamics(),window,"synthetic")
    assert len(rows)==n
    assert max(row["rollout_mse"] for row in rows)<1e-10
    assert max(row["teacher_mse"] for row in rows)<1e-10
    assert rows[-1]["zero_from_start_mse"]>0
    assert max(abs(row["adv_real"]-row["adv_pred"]) for row in rows)<1e-10
    with tempfile.TemporaryDirectory() as d:
        directory=Path(d)
        write_csv(directory/"per_window.csv",rows)
        write_csv(directory/"by_step.csv",summarize(rows))
        create_figures(directory)
        assert (directory/"latent_error_vs_horizon.png").is_file()
        assert (directory/"advantage_disagreement.png").is_file()
        assert (directory/"terminal_error_distribution.png").is_file()
    print("PASS: single-step alignment, accumulated rollout, Q/V, CSV and plots")


if __name__=="__main__":
    main()
