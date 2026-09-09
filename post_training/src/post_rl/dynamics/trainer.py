import copy
import os
from typing import Dict

import hydra
import torch

from .core.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from .models.dynamics_model import EnsembleDynamicsModel
from .utils.termination_fns import get_termination_fn
from .utils.logger import Logger, make_log_dirs


def train_dynamics(
    env,
    obs_adapter,
    dynamics_save_path,
    cfg,
    feature_dim,
    action_dim,
    chunk_as_single_action=False,
    n_action_steps=1,
    n_obs_steps=1,
    device="cuda",
):
    if obs_adapter.fix_encoder != cfg.dynamics.fix_encoder:
        raise ValueError(
            "obs_adapter.fix_encoder must match cfg.dynamics.fix_encoder"
        )
    if not cfg.dynamics.fix_encoder:
        raise ValueError(
            "ACT transformer-latent dynamics requires dynamics.fix_encoder=true."
        )
    if cfg.dynamics.get("latent_mode", None) != "transformer_encoder":
        raise ValueError(
            "ACT Scheme C requires dynamics.latent_mode=transformer_encoder."
        )
    if n_obs_steps != 1 or obs_adapter.n_obs_steps != 1:
        raise ValueError(
            "ACT transformer-latent dynamics currently requires n_obs_steps=1."
        )
    if feature_dim != obs_adapter.feature_dim:
        raise ValueError(
            f"feature_dim={feature_dim} does not match "
            f"obs_adapter.feature_dim={obs_adapter.feature_dim}"
        )
    if cfg.predict_r:
        raise NotImplementedError(
            "Transformer-latent dynamics follows RL-100 OPE gating with predict_r=false."
        )
    if cfg.dynamics_type != "mlp":
        raise NotImplementedError(
            "ACT transformer-latent V1 supports token-structured MLP dynamics only."
        )

    effective_action_steps = n_action_steps if chunk_as_single_action else 1
    model_action_dim = action_dim * effective_action_steps
    model_cfg = copy.deepcopy(cfg)
    if not chunk_as_single_action:
        model_cfg.n_action_steps = 1

    dynamics_model = EnsembleDynamicsModel(
        obs_dim=feature_dim,
        action_dim=model_action_dim,
        hidden_dims=cfg.dynamics.dynamics_hidden_dims,
        num_ensemble=cfg.dynamics.n_ensemble,
        num_elites=cfg.dynamics.n_elites,
        weight_decays=cfg.dynamics.dynamics_weight_decay,
        device=device,
        cfg=model_cfg,
        with_reward=cfg.predict_r,
    )

    dynamics_optim = hydra.utils.instantiate(
        cfg.optimizer,
        params=dynamics_model.parameters(),
    )

    termination_fn = get_termination_fn(task=cfg.task_name)
    dynamics = EnsembleDynamics_batch(
        dynamics_model,
        dynamics_optim,
        termination_fn,
        env,
        obs_adapter,
        cfg=cfg,
        action_dim=model_action_dim,
        gamma=cfg.critic.gamma,
        device=device,
        chunk_as_single_action=chunk_as_single_action,
        n_action_steps=n_action_steps,
        prediction_mode=cfg.dynamics.prediction_mode,
    )

    os.makedirs(dynamics_save_path, exist_ok=True)
    log_dirs = make_log_dirs(
        cfg.task_name,
        cfg.name,
        cfg.training.seed,
        None,
        record_params=None,
    )
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboardX",
    }
    logger = Logger(log_dirs, output_config)
    dynamics.set_logger(logger)
    return dynamics


@torch.no_grad()
def rollout(
    policy,
    dynamics,
    Q,
    iql,
    batch: Dict,
    rollout_length: int,
    args,
):
    return dynamics.rollout(
        policy=policy,
        Q=Q,
        iql=iql,
        batch=batch,
        rollout_length=rollout_length,
        is_iql=args.is_iql,
        use_gae=getattr(args, "use_gae", False),
        first_action=getattr(args, "first_action", False),
    )


@torch.no_grad()
def dynamics_eval(
    args,
    policy,
    Q,
    iql,
    dynamics,
    batch: Dict,
):
    return rollout(
        policy,
        dynamics,
        Q,
        iql,
        batch,
        args.rollout_length,
        args,
    )


if __name__ == "__main__":
    raise RuntimeError(
        "ARBiM does not support the legacy Gym/D4RL standalone entry of "
        "dynamics_eval_batch.py. Build Dynamics from the post-training entry point."
    )
