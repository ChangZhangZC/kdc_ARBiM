"""CPU smoke for three-mode Dynamics decomposition; no model artifacts required."""
from pathlib import Path
import tempfile

import numpy as np
import torch

from common import Window
from test_smoke import FakeAdapter
from evaluate import write_csv
from evaluate_components import evaluate_components_window, summarize_components
from plot_components import plot_components


class ToyEnsemble:
    def __init__(self):
        self.elites=torch.tensor([0,1],dtype=torch.long)

    def random_elite_idxs(self,n):
        return np.random.choice(self.elites.numpy(),size=n)

    def __call__(self,states,actions):
        batch,tokens,features=states.shape
        delta=actions[:,0].reshape(1,batch,1,1).expand(2,batch,tokens,features)
        # Both models have different deterministic biases. This lets the
        # smoke detect the model-selection and Gaussian-noise components.
        biases=torch.tensor([-.1,.1],dtype=states.dtype,device=states.device).reshape(2,1,1,1)
        mean=delta+biases
        logvar=torch.full_like(mean,float(np.log(.04)))
        return mean,logvar


class ToyDynamics:
    def __init__(self):
        self.model=ToyEnsemble()
        self.predict_delta=True

    def _model(self):
        return self.model

    def _critic_readout(self,tokens):
        return tokens.mean(dim=1)

    def step(self,states,actions,policy_features=None):
        mean,logvar=self.model(states,actions)
        mean=mean+states.unsqueeze(0)
        # Exact order of RNG operations in the real Dynamics step:
        # torch.randn_like for all models, followed by NumPy elite selection.
        sampled=mean+torch.exp(.5*logvar)*torch.randn_like(mean)
        batch=len(states)
        ids=torch.as_tensor(self.model.random_elite_idxs(batch),dtype=torch.long)
        chosen=sampled[ids,torch.arange(batch)]
        return (chosen.cpu().numpy(),
                np.zeros((batch,1),dtype=np.float32),
                np.zeros((batch,1),dtype=bool),{})


class ToyCritic:
    def get_advantage(self,state,action):
        return state[:,0:1] + .1*action[:,0:1]


class ToyWorkspace:
    device=torch.device("cpu")
    action_dim=16
    obs_adapter=FakeAdapter()
    critic=ToyCritic()


def make_window(index,horizon=4):
    start=.25*index
    step=.5 + .05*index
    observations=[]
    for t in range(horizon+1):
        value=start+t*step
        observations.append({
            "state":np.r_[np.float32(value),np.zeros(15,dtype=np.float32)],
            "head_rgb":np.zeros((3,2,2),dtype=np.uint8),
            "wrist_left_rgb":np.zeros((3,2,2),dtype=np.uint8),
            "wrist_right_rgb":np.zeros((3,2,2),dtype=np.uint8),
        })
    actions=np.zeros((horizon,16),dtype=np.float32)
    actions[:,0]=step
    return Window(f"toy_episode_{index}",0,actions,observations)


def main():
    np.random.seed(1729)
    torch.manual_seed(1729)
    ws,dyn=ToyWorkspace(),ToyDynamics()
    samples,calibration,baseline=[],[],[]
    repeats=12
    for i in range(6):
        win=make_window(i)
        rows,cal=evaluate_components_window(ws,dyn,win,repeats)
        samples.extend(rows)
        calibration.extend(cal)
        for j in range(len(win.actions)):
            # Synthetic baseline used only to verify aggregation and charting.
            baseline.append({
                "trajectory":win.name,"anchor":win.start,"step":j+1,
                "adv_real":float(win.observations[j]["state"][0]+.1*win.actions[j,0]),
                "rollout_mse":0.0,
                "zero_from_start_mse":float(((j+1)*win.actions[j,0])**2),
            })
    assert len(samples)==6*4*repeats
    assert len(calibration)==6*4
    assert samples[0]["step"]==1 and samples[0]["adv_horizon"]==0
    assert all(row["adv_horizon"]==row["step"]-1 for row in samples)
    assert all(row["ppo_random_mse"]>=0 and row["random_elite_mse"]>=0 for row in samples)
    summary=summarize_components(samples,calibration,baseline)
    assert len(summary)==4
    assert summary[0]["ppo_random_mse"]>summary[0]["random_elite_mse"]
    assert summary[0]["predicted_epistemic_var"]>0
    assert summary[0]["predicted_aleatoric_var"]>0
    assert summary[0]["norm_flip_ppo_random"]==0
    for row in summary:
        assert np.isfinite(row["coverage_1sigma"])
        assert np.isfinite(row["norm_flip_random_elite"])
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        write_csv(folder/"component_per_replica.csv",samples)
        write_csv(folder/"component_calibration.csv",calibration)
        write_csv(folder/"component_by_step.csv",summary)
        plot_components(folder)
        for name in (
            "three_mode_rollout_mse.png","teacher_one_step_components.png",
            "variance_calibration.png","component_advantage_direction.png"
        ):
            assert (folder/name).is_file(),name
    print("PASS: paired Elite selection, Gaussian decomposition, Q-V alignment, normalized signs, plots")


if __name__=="__main__":
    main()
