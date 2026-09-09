from __future__ import annotations

import argparse
import math

import torch

from _common import (
    add_common_args,
    assert_finite,
    build_real_batch,
    changed_param_names,
    load_cfg,
    make_work_dir,
    make_workspace,
    print_pass,
    print_section,
    snapshot_params,
)


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Smoke 03: IQL Critic forward/backward/update"),
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    workspace = make_workspace(cfg, make_work_dir(args, "smoke_03_critic"))
    _, batch = build_real_batch(workspace, args.batch_size)
    workspace._build_critic()
    critic = workspace.critic

    print_section("Q / V / Advantage forward")
    action, reward, not_done = critic._select_transition(batch)
    with torch.no_grad():
        q = critic.minQ(batch["obs"], action)
        value = critic.value(batch["obs"])
        advantage = critic.get_advantage(batch["obs"], action)
    for name, tensor in {
        "Q": q,
        "V": value,
        "Advantage": advantage,
        "reward": reward,
        "not_done": not_done,
    }.items():
        assert_finite(tensor, name)
        if tensor.reshape(tensor.shape[0], -1).shape[1] != 1:
            raise AssertionError(f"{name} must be scalar per sample, got {tuple(tensor.shape)}")
    print_pass("Q, V and Q-V are finite scalar values per sample")

    print_section("one IQL update")
    q_before = snapshot_params(critic._Q)
    v_before = snapshot_params(critic._value)
    encoder_before = workspace._fingerprint_module(critic.obs_encoder)
    if any(param.requires_grad for param in critic._target_Q.parameters()):
        raise AssertionError("target_Q parameters must not require gradients")

    q_loss, v_loss = critic.update(batch)
    if not math.isfinite(q_loss) or not math.isfinite(v_loss):
        raise AssertionError(f"Non-finite critic loss: q={q_loss}, v={v_loss}")
    q_changed = changed_param_names(q_before, critic._Q)
    v_changed = changed_param_names(v_before, critic._value)
    if not q_changed:
        raise AssertionError("Q network parameters did not change after optimizer.step()")
    if not v_changed:
        raise AssertionError("Value network parameters did not change after optimizer.step()")
    encoder_after = workspace._fingerprint_module(critic.obs_encoder)
    if encoder_before != encoder_after:
        raise AssertionError("Frozen ACT critic encoder changed during IQL update")
    if any(param.grad is not None for param in critic._target_Q.parameters()):
        raise AssertionError("target_Q unexpectedly received gradients")

    print_pass(f"Q update is finite and changed {len(q_changed)} parameter tensors")
    print_pass(f"V update is finite and changed {len(v_changed)} parameter tensors")
    print_pass("ACT critic encoder and target_Q stayed frozen")
    print("\nSMOKE 03 PASSED")


if __name__ == "__main__":
    main()
