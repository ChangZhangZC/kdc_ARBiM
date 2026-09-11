from __future__ import annotations

import argparse
import copy
import os
import pathlib

import torch

from _common import (
    add_common_args,
    load_cfg,
    make_work_dir,
    make_workspace,
    max_param_diff,
    print_pass,
    print_section,
    snapshot_params,
)


def configure_tiny_run(cfg) -> None:
    stride = int(cfg.rl_chunk_size)
    cfg.use_wandb = False
    cfg.training.use_ema = False
    cfg.training.num_critic_epochs = 1
    cfg.dataset.sequence_stride = stride
    cfg.dataset.finetune_sequence_stride = stride
    cfg.dataset.val_ratio = 0.2
    cfg.dataset.max_train_episodes = 2
    cfg.critic.sequence_stride = stride
    cfg.critic.q_hidden_dim = 64
    cfg.critic.q_depth = 2
    cfg.critic.v_hidden_dim = 64
    cfg.critic.v_depth = 2
    cfg.dynamics.dynamics_hidden_dims = [64, 64]
    cfg.dynamics.dynamics_weight_decay = [2.5e-5, 5.0e-5, 1.0e-4]
    cfg.dynamics.n_ensemble = 2
    cfg.dynamics.n_elites = 1
    cfg.dynamics.dynamics_max_epochs = 1
    cfg.dynamics.max_epochs_since_update = 1
    cfg.unio4.bppo_steps = 2
    cfg.unio4.eval_step = 1
    cfg.unio4.eval_freq = 1000000
    cfg.unio4.checkpoint_every_steps = 1
    cfg.unio4.is_clip_decay = False
    cfg.unio4.is_bppo_lr_decay = False
    cfg.unio4.is_linear_decay = False
    cfg.ppo.enable_ratio_logging = False


def require_path(path: str) -> None:
    if not os.path.exists(path):
        raise AssertionError(f"Expected smoke-test artifact is missing: {path}")


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(
            description="Smoke 06: tiny Critic->Dynamics->Offline PPO E2E + exact resume"
        ),
    )
    args = parser.parse_args()
    root = pathlib.Path(make_work_dir(args, "smoke_06_e2e"))
    first_dir = root / "first_run"
    resume_dir = root / "resume_run"
    first_dir.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args, chunk_as_single_action=True, ratio_mode="scalar", adv_mode="scalar_iql")
    configure_tiny_run(cfg)

    print_section("tiny full post-RL run")
    workspace = make_workspace(cfg, str(first_dir))
    workspace.run()
    if workspace.global_step != 2:
        raise AssertionError(f"Expected Offline PPO global_step=2, got {workspace.global_step}")

    critic_final = first_dir / "critic" / "checkpoints" / "final"
    dynamics_dir = first_dir / "dynamics"
    dynamics_final = dynamics_dir / "checkpoints" / "final"
    dynamics_logs = dynamics_dir / "logs"
    ppo_step1 = first_dir / "offline_ppo" / "checkpoints" / "step_00000001"
    ppo_final = first_dir / "offline_ppo" / "checkpoints" / "final"
    for path in (
        critic_final / "Q.pt",
        critic_final / "value.pt",
        critic_final / "contract.json",
        dynamics_final / "contract.json",
        dynamics_logs,
        ppo_step1 / "training_state.pt",
        ppo_step1 / "resume_meta.json",
        ppo_step1 / "contract.json",
        ppo_final / "training_state.pt",
    ):
        require_path(str(path))
    if not any(path.is_dir() for path in dynamics_logs.iterdir()):
        raise AssertionError("Dynamics logs directory does not contain a run subdirectory")
    if workspace.unio4 is None:
        raise AssertionError("Offline PPO object was not built in the E2E run")
    first_trainable = snapshot_params(workspace.unio4._policy, trainable_only=True)
    print_pass("Critic, Dynamics, Offline PPO, local Dynamics logs and resume artifacts were produced")

    del workspace
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_section("resume from step 1 and reproduce step 2")
    resume_cfg = copy.deepcopy(cfg)
    resume_cfg.resume.checkpoint_dir = str(ppo_step1)
    resumed = make_workspace(resume_cfg, str(resume_dir))
    resumed.run()
    if resumed.global_step != 2:
        raise AssertionError(f"Resumed run did not continue to step 2: {resumed.global_step}")
    if resumed.unio4 is None:
        raise AssertionError("Offline PPO object was not restored in resumed run")
    max_diff = max_param_diff(first_trainable, resumed.unio4._policy)
    if max_diff > 1e-5:
        raise AssertionError(
            f"Exact-resume trainable policy mismatch after reproducing step 2: max diff={max_diff:.6g}"
        )
    print_pass(f"resume continued from step 1 to step 2; max trainable-param diff={max_diff:.3g}")

    print(f"\nSmoke artifacts: {root}")
    print("SMOKE 06 PASSED")


if __name__ == "__main__":
    main()
