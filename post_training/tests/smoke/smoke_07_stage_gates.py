from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

from omegaconf import OmegaConf

from _common import CONFIG_PATH, TrainACTWorkspace
from post_rl.training.checkpoint_compat_workspace import (
    TrainACTWorkspace as CheckpointCompatTrainACTWorkspace,
)


def make_workspace(cfg):
    workspace = TrainACTWorkspace.__new__(TrainACTWorkspace)
    workspace.cfg = cfg
    workspace.rank = 0
    workspace.ema = object()
    workspace.ema_model = object()
    return workspace


def expect_raises(fn, text: str):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        if text not in str(exc):
            raise AssertionError(f"Expected error containing {text!r}, got: {exc}") from exc
        return
    raise AssertionError(f"Expected an exception containing {text!r}")


def main():
    base = OmegaConf.load(CONFIG_PATH)

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = True
    cfg.stages.run_stage2 = False
    ws = make_workspace(cfg)
    ws._validate_stage_gates()
    assert ws._stage_flags() == (True, False)

    with patch.object(
        CheckpointCompatTrainACTWorkspace,
        "_build_ppo",
        side_effect=AssertionError("Stage 2 parent should not be called"),
    ):
        assert ws._build_ppo() is None
    print("[PASS] Stage-1-only gate skips PPO construction")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = True
    cfg.stages.run_stage2 = True
    ws = make_workspace(cfg)
    ws._validate_stage_gates()
    called = {"ppo": False}

    def mark_ppo(_self):
        called["ppo"] = True

    with patch.object(CheckpointCompatTrainACTWorkspace, "_build_ppo", mark_ppo):
        ws._build_ppo()
    assert called["ppo"]
    print("[PASS] Stage-1+Stage-2 gate allows PPO construction")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = False
    cfg.stages.run_stage2 = True
    cfg.unio4.stage1_resume_dir = "/tmp/stage1_artifacts"
    ws = make_workspace(cfg)
    ws._validate_stage_gates()
    assert ws._stage_flags() == (False, True)
    expect_raises(
        lambda: ws._train_critic(None),
        "Stage 1 is disabled",
    )
    expect_raises(
        lambda: ws._train_dynamics(),
        "Stage 1 is disabled",
    )
    print("[PASS] Stage-2-only gate requires loading Stage-1 artifacts, never retraining them")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = False
    cfg.stages.run_stage2 = True
    cfg.unio4.stage1_resume_dir = None
    cfg.resume.checkpoint_dir = None
    cfg.critic.artifact_dir = None
    cfg.dynamics.artifact_dir = None
    ws = make_workspace(cfg)
    expect_raises(
        ws._validate_stage_gates,
        "requires existing Stage 1 artifacts",
    )
    print("[PASS] Stage-2-only gate rejects missing Stage-1 artifacts")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = False
    cfg.stages.run_stage2 = False
    ws = make_workspace(cfg)
    expect_raises(
        ws._validate_stage_gates,
        "At least one stage must be enabled",
    )
    print("[PASS] disabling both stages is rejected")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = True
    cfg.stages.run_stage2 = False
    cfg.eval = True
    ws = make_workspace(cfg)
    expect_raises(
        ws._validate_stage_gates,
        "eval=true requires stages.run_stage2=true",
    )
    print("[PASS] evaluation cannot bypass the Stage-2 gate")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = True
    cfg.stages.run_stage2 = True
    cfg.eval = False
    cfg.unio4.bppo_steps = 0
    ws = make_workspace(cfg)
    expect_raises(
        ws._validate_stage_gates,
        "stages.run_stage2=true requires unio4.bppo_steps > 0",
    )
    print("[PASS] Stage-2 training cannot be enabled with zero PPO steps")

    cfg = deepcopy(base)
    cfg.stages.run_stage1 = True
    cfg.stages.run_stage2 = False
    cfg.resume.checkpoint_dir = "/tmp/full_ppo_resume"
    ws = make_workspace(cfg)
    expect_raises(
        ws._validate_stage_gates,
        "resume.checkpoint_dir is a full PPO resume",
    )
    print("[PASS] full PPO resume cannot bypass the Stage-2 gate")

    print("SMOKE 07 PASSED")


if __name__ == "__main__":
    main()
