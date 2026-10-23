"""Shared, read-only Flywheel01 Dynamics diagnostic loaders.

Uses the exact ACT encoder/normalizer and Stage-1 contracts from training.
No training, checkpoint mutation, or simulator side-effects.
"""
from __future__ import annotations

import copy
import pathlib
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[4]
for p in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import lerobot_patches.custom_patches  # noqa: F401
from post_rl.dynamics.models.dynamics_model import EnsembleDynamicsModel
from post_rl.dynamics.core.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from post_rl.dynamics.utils.termination_fns import get_termination_fn
from post_rl.training import TrainACTWorkspace
from post_rl.data.sampler import get_val_mask

CAMERAS = {
    "head_rgb": "observation.images.head_cam_h",
    "wrist_left_rgb": "observation.images.wrist_cam_l",
    "wrist_right_rgb": "observation.images.wrist_cam_r",
}


def load_context(run_dir, output_dir, device, checkpoint=None, dataset=None):
    run_dir = pathlib.Path(run_dir).expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Resolved training config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if checkpoint:
        cfg.input.policy_checkpoint = str(pathlib.Path(checkpoint).expanduser().resolve())
    if dataset:
        cfg.input.dataset_path = str(pathlib.Path(dataset).expanduser().resolve())
    if cfg.chunk_as_single_action or int(cfg.n_action_steps) != 32:
        raise ValueError("This experiment is for Flywheel01 single-step 32-action PPO only")
    if not pathlib.Path(cfg.input.policy_checkpoint).exists():
        raise FileNotFoundError(f"IL checkpoint not found: {cfg.input.policy_checkpoint}")
    cfg.training.device = device
    cfg.use_wandb = False
    cfg.eval = False
    cfg.dataset.use_latent_cache = False
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace._build_act_observation_frontends()
    stage1 = run_dir / "stage1"
    dynamic_dir = stage1 / "dynamics" / "checkpoints" / "final"
    critic_dir = stage1 / "critic" / "checkpoints" / "final"
    for needed in (dynamic_dir / "dynamics.pth", dynamic_dir / "contract.json",
                   critic_dir / "Q.pt", critic_dir / "value.pt", critic_dir / "contract.json"):
        if not needed.is_file():
            raise FileNotFoundError(f"Stage-1 weight/contract missing: {needed}")

    # The trained Single-step Dynamics has a 16-D input, NOT a 32*16-D chunk input.
    model_cfg = copy.deepcopy(cfg)
    model_cfg.n_action_steps = 1
    model = EnsembleDynamicsModel(
        obs_dim=int(workspace.obs_feature_dim),
        action_dim=int(workspace.action_dim),
        hidden_dims=list(cfg.dynamics.dynamics_hidden_dims),
        num_ensemble=int(cfg.dynamics.n_ensemble),
        num_elites=int(cfg.dynamics.n_elites),
        weight_decays=list(cfg.dynamics.dynamics_weight_decay),
        with_reward=False,
        device=workspace.device,
        cfg=model_cfg,
    )
    optim = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    dynamics = EnsembleDynamics_batch(
        model=model, optim=optim, terminal_fn=get_termination_fn(cfg.task_name),
        env=None, obs_adapter=workspace.obs_adapter, cfg=cfg,
        action_dim=int(workspace.action_dim),
        gamma=float(cfg.critic.gamma), device=workspace.device,
        chunk_as_single_action=False, n_action_steps=int(cfg.n_action_steps),
        prediction_mode="full",
    )
    workspace._validate_artifact_contract(str(dynamic_dir), "Dynamics")
    dynamics.load(str(dynamic_dir))
    dynamics.model.eval()
    workspace._build_critic()
    workspace._validate_artifact_contract(str(critic_dir), "Critic")
    workspace.critic.load(
        q_path=str(critic_dir / "Q.pt"), v_path=str(critic_dir / "value.pt"),
        encoder_path=str(critic_dir / "encoder.pt") if (critic_dir / "encoder.pt").is_file() else None,
    )
    workspace.critic.eval()
    return workspace, dynamics, cfg


def tensor_obs(snapshot, device):
    """Input: unnormalized RGB uint8 CHW + physical robot state."""
    obs = {"state": torch.as_tensor(np.asarray(snapshot["state"], dtype=np.float32)[None, None], device=device)}
    for key in CAMERAS:
        image = np.asarray(snapshot[key])
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{key}: expected [3,H,W], got {image.shape}")
        obs[key] = torch.as_tensor(image[None, None], device=device)
    return obs


def read_zarr_obs(buffer, idx, next_view=False):
    prefix = "next_" if next_view else ""
    obs = {"state": np.asarray(buffer[f"{prefix}state"][idx], dtype=np.float32)}
    for key in CAMERAS:
        obs[key] = np.asarray(buffer[f"{prefix}{key}"][idx])
    return obs


def read_staging_obs(item):
    rgb = {}
    for key, feature in CAMERAS.items():
        payload = item["rgb"][feature]
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot decode staging JPEG: {feature}")
        rgb[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
    return {"state": np.asarray(item["agent_pos"], dtype=np.float32), **rgb}


@dataclass
class Window:
    name: str
    start: int
    actions: np.ndarray
    observations: list  # T+1 raw unnormalized observations


def zarr_windows(workspace, horizon, stride, max_windows, split):
    cfg = workspace.cfg
    workspace.buffer = workspace._load_buffer()
    buffer = workspace.buffer
    episode_ends = np.asarray(buffer.episode_ends, dtype=np.int64)
    mask = get_val_mask(len(episode_ends), val_ratio=float(cfg.dataset.val_ratio),
                        seed=int(cfg.training.seed))
    starts = np.r_[0, episode_ends[:-1]]
    count = 0
    for ep, (first, end) in enumerate(zip(starts, episode_ends)):
        if split == "val" and not mask[ep] or split == "train" and mask[ep]:
            continue
        for anchor in range(int(first), int(end) - horizon + 1, stride):
            actions = np.asarray(buffer["action"][anchor:anchor + horizon], dtype=np.float32)
            observations = [read_zarr_obs(buffer, j) for j in range(anchor, anchor + horizon)]
            observations.append(read_zarr_obs(buffer, anchor + horizon - 1, next_view=True))
            yield Window(f"zarr_ep{ep:04d}", int(anchor-first), actions, observations)
            count += 1
            if count >= max_windows:
                return


def staging_windows(directory, horizon, stride, max_windows):
    directory = pathlib.Path(directory).expanduser().resolve()
    paths = [directory] if directory.is_file() else sorted(directory.rglob("*_staging.npy"))
    if not paths:
        raise FileNotFoundError(f"No *_staging.npy files in {directory}")
    count = 0
    for p in paths:
        payload = np.load(p, allow_pickle=True).item()  # trusted, self-recorded data only
        if payload.get("__format__") != "arbim_rollout_staging_v2":
            raise ValueError(f"Unsupported staging payload: {p}")
        actions = np.asarray(payload["actions"], dtype=np.float32)
        observations = payload["observations"]
        if len(observations) != len(actions) + 1:
            raise ValueError(f"Staging T+1 alignment violated: {p}")
        for start in range(0, len(actions)-horizon+1, stride):
            obs = [read_staging_obs(item) for item in observations[start:start+horizon+1]]
            yield Window(p.stem, start, actions[start:start+horizon], obs)
            count += 1
            if count >= max_windows:
                return
