import os
from typing import Dict, Tuple

import hydra
import torch

from transition_model.dynamics.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from transition_model.dynamics import EnsembleDynamics_batch
from transition_model.utils.termination_fns import get_termination_fn
from transition_model.utils.logger import Logger, make_log_dirs


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
    if n_obs_steps != cfg.n_obs_steps or n_obs_steps != obs_adapter.n_obs_steps:
        raise ValueError(
            "n_obs_steps must match cfg.n_obs_steps and obs_adapter.n_obs_steps"
        )
    if feature_dim != obs_adapter.feature_dim:
        raise ValueError(
            f"feature_dim={feature_dim} does not match "
            f"obs_adapter.feature_dim={obs_adapter.feature_dim}"
        )

    prediction_mode = getattr(cfg.dynamics, "prediction_mode", "last")

    if chunk_as_single_action:
        action_dim = action_dim * n_action_steps

    if prediction_mode == "full":
        output_obs_dim = feature_dim * n_obs_steps
        print(
            f"==========================Dynamics prediction mode: FULL "
            f"(output dim: {output_obs_dim})=========================="
        )
    else:
        output_obs_dim = feature_dim
        print(
            f"==========================Dynamics prediction mode: LAST "
            f"(output dim: {output_obs_dim})=========================="
        )

    if cfg.dynamics_type == "diffusion":
        raise NotImplementedError(
            "ARBiM ACT V1 does not migrate RL-100 Diffusion Dynamics."
        )

    dynamics_model = EnsembleDynamicsModel(
        obs_dim=output_obs_dim,
        action_dim=action_dim,
        hidden_dims=cfg.dynamics.dynamics_hidden_dims,
        num_ensemble=cfg.dynamics.n_ensemble,
        num_elites=cfg.dynamics.n_elites,
        weight_decays=cfg.dynamics.dynamics_weight_decay,
        device=device,
        cfg=cfg,
        with_reward=cfg.predict_r,
    )
    
    if not cfg.dynamics.fix_encoder:
        dynamics_optim = hydra.utils.instantiate(
            cfg.optimizer,
            params=list(dynamics_model.parameters())
            + list(obs_adapter.encoder.parameters()),
        )
    else:
        print("==========================fix encoder==========================")
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
        action_dim=action_dim,
        gamma=cfg.critic.gamma,
        device=device,
        chunk_as_single_action=chunk_as_single_action,
        n_action_steps=n_action_steps,
        prediction_mode=prediction_mode,
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
    
def get_args():
    from transition_model.configs import loaded_args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--env", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--is_state_norm", default=False, type=bool)
    parser.add_argument("--is_eval_state_norm", default=False, type=bool)

    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.env]
    for arg_key, default_value in default_args.items():
        parser.add_argument(
            f"--{arg_key}",
            default=default_value,
            type=type(default_value),
        )
    return parser.parse_args()


if __name__ == "__main__":
    raise RuntimeError(
        "ARBiM does not support the legacy Gym/D4RL standalone entry of "
        "dynamics_eval_batch.py. Build Dynamics from the post-training entry point."
    )