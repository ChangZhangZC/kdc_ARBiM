from __future__ import annotations

import argparse
import math

import torch

from _common import (
    add_common_args,
    assert_close_to_one,
    assert_finite,
    build_real_batch,
    changed_param_names,
    fingerprint_params,
    load_cfg,
    make_work_dir,
    make_workspace,
    policy_obs_from_batch,
    print_pass,
    print_section,
    snapshot_params,
)

CASES = {
    "chunk_scalar": (True, "scalar", "scalar_iql", False),
    "chunk_per_step": (True, "per_step", "scalar_iql", False),
    "single_step": (False, "per_step", "scalar_iql", False),
    "unsupported_vdelta": (True, "scalar", "chunk_vdelta_gae", True),
}


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Smoke 05: Offline PPO ratio/advantage/update"),
    )
    parser.add_argument("--case", choices=sorted(CASES), default="chunk_scalar")
    args = parser.parse_args()
    chunk_mode, ratio_mode, adv_mode, expect_not_implemented = CASES[args.case]
    cfg = load_cfg(
        args,
        chunk_as_single_action=chunk_mode,
        ratio_mode=ratio_mode,
        adv_mode=adv_mode,
    )
    cfg.unio4.is_clip_decay = False
    cfg.unio4.is_bppo_lr_decay = False
    cfg.unio4.is_linear_decay = False
    workspace = make_workspace(cfg, make_work_dir(args, f"smoke_05_{args.case}"))
    _, batch = build_real_batch(workspace, max(args.batch_size, 2))

    workspace._build_critic()
    q_loss, v_loss = workspace.critic.update(batch)
    if not math.isfinite(q_loss) or not math.isfinite(v_loss):
        raise AssertionError("Critic warm-up update produced non-finite loss")

    if not chunk_mode:
        workspace._build_dynamics()
        dynamics = workspace.dynamics
        batch_size = batch["action"].shape[0]
        state = dynamics.obs2latent(batch["obs"])
        next_state = dynamics.next_obs2latent(batch["next_obs"])
        dyn_loss = dynamics.learn(
            batch,
            state.reshape(batch_size, -1),
            next_state.reshape(batch_size, -1),
        )
        dynamics.optimize(dyn_loss)

    workspace._build_ppo()
    ppo = workspace.unio4
    policy_obs = policy_obs_from_batch(workspace, batch)

    print_section(f"PPO case: {args.case}")
    with torch.no_grad():
        action, old_raw, _ = ppo._old_policy.sample_action_chunk(policy_obs)
        new_raw, _ = ppo._policy.evaluate_action_chunk(policy_obs, action)
        if chunk_mode and ratio_mode == "scalar":
            old_lp = ppo._sum_chunk_event_dims(old_raw)
            new_lp = ppo._sum_chunk_event_dims(new_raw)
        else:
            old_lp = ppo._sum_step_event_dims(old_raw)
            new_lp = ppo._sum_step_event_dims(new_raw)
        initial_ratio = torch.exp(new_lp - old_lp)
    assert_close_to_one(initial_ratio)
    print_pass("old/new policies start identical, so PPO ratio is ~1")

    trainable_before = snapshot_params(ppo._policy, trainable_only=True)
    frozen_before = fingerprint_params(ppo._policy, trainable=False)

    if expect_not_implemented:
        try:
            ppo.update_distribution(
                batch=batch,
                critic=workspace.critic,
                is_clip_decay=False,
                is_lr_decay=False,
                is_linear_decay=False,
            )
        except NotImplementedError as exc:
            if "vdelta" not in str(exc):
                raise AssertionError(f"Unexpected NotImplementedError: {exc}") from exc
            print_pass("unsupported vdelta advantage mode fails explicitly instead of falling back")
            print("\nSMOKE 05 PASSED")
            return
        raise AssertionError("Expected unsupported vdelta mode to raise NotImplementedError")

    loss = ppo.update_distribution(
        batch=batch,
        critic=workspace.critic,
        is_clip_decay=False,
        is_lr_decay=False,
        is_linear_decay=False,
    )
    if not math.isfinite(loss):
        raise AssertionError(f"PPO loss is non-finite: {loss}")
    changed = changed_param_names(trainable_before, ppo._policy)
    if not changed:
        raise AssertionError("No trainable PPO policy parameter changed after optimizer.step()")
    frozen_after = fingerprint_params(ppo._policy, trainable=False)
    if frozen_before != frozen_after:
        raise AssertionError("A frozen PPO/ACT parameter changed during actor update")
    if not ppo._ratio_records:
        raise AssertionError("Ratio logging produced no smoke-test record")
    logged_ratio = torch.tensor(ppo._ratio_records[-1]["ratio_mean"])
    assert_finite(logged_ratio, "logged_ratio")
    if abs(float(logged_ratio) - 1.0) > 1e-5:
        raise AssertionError(f"Logged initial ratio mean is not ~1: {float(logged_ratio)}")

    print_pass(f"PPO loss is finite and {len(changed)} trainable parameter tensors changed")
    print_pass("frozen ACT parameters stayed unchanged")
    print("\nSMOKE 05 PASSED")


if __name__ == "__main__":
    main()
