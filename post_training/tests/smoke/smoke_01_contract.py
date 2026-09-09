from __future__ import annotations

import argparse
import copy

from _common import add_common_args, load_cfg, make_work_dir, make_workspace, print_pass, print_section


def expect_invalid(workspace, cfg, expected_text: str) -> None:
    try:
        workspace._validate_scheme_c_contract(cfg)
    except ValueError as exc:
        if expected_text not in str(exc):
            raise AssertionError(
                f"Expected error containing {expected_text!r}, got: {exc}"
            ) from exc
        return
    raise AssertionError(f"Expected invalid config containing {expected_text!r} to fail")


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Smoke 01: post-RL config/contract guards"),
        require_dataset=False,
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    workspace = make_workspace(cfg, make_work_dir(args, "smoke_01_contract"))

    print_section("valid contract")
    workspace._validate_scheme_c_contract(copy.deepcopy(cfg))
    print_pass("default resolved contract is valid")

    valid_horizon = copy.deepcopy(cfg)
    valid_horizon.horizon = int(cfg.n_action_steps) + 1
    workspace._validate_scheme_c_contract(valid_horizon)
    print_pass("horizon > n_action_steps is accepted")

    print_section("hard alignment failures")
    mismatch = copy.deepcopy(cfg)
    mismatch.rl_chunk_size = int(cfg.rl_chunk_size) + 1
    expect_invalid(workspace, mismatch, "rl_chunk_size")
    print_pass("rl_chunk_size mismatch is rejected")

    checkpoint_mismatch = copy.deepcopy(cfg)
    wrong = int(workspace.model.config.chunk_size) + 1
    checkpoint_mismatch.act_chunk_size = wrong
    checkpoint_mismatch.rl_chunk_size = wrong
    checkpoint_mismatch.n_action_steps = wrong
    checkpoint_mismatch.horizon = wrong
    expect_invalid(workspace, checkpoint_mismatch, "loaded ACT checkpoint")
    print_pass("ACT checkpoint chunk mismatch is rejected")

    short_horizon = copy.deepcopy(cfg)
    short_horizon.horizon = int(cfg.n_action_steps) - 1
    expect_invalid(workspace, short_horizon, "horizon")
    print_pass("horizon < n_action_steps is rejected")

    print_section("V1 fixed-interface failures")
    encoder_unfrozen = copy.deepcopy(cfg)
    encoder_unfrozen.critic.fix_encoder = False
    expect_invalid(workspace, encoder_unfrozen, "critic.fix_encoder")
    print_pass("critic encoder unfreeze is rejected")

    reward_dynamics = copy.deepcopy(cfg)
    reward_dynamics.predict_r = True
    expect_invalid(workspace, reward_dynamics, "predict_r")
    print_pass("predict_r=true is rejected")

    gae = copy.deepcopy(cfg)
    gae.unio4.use_gae = True
    expect_invalid(workspace, gae, "use_gae")
    print_pass("online-style GAE is rejected")

    bad_ratio = copy.deepcopy(cfg)
    bad_ratio.offline_chunk_ratio_mode = "invalid"
    expect_invalid(workspace, bad_ratio, "offline_chunk_ratio_mode")
    print_pass("unknown ratio mode is rejected")

    print("\nSMOKE 01 PASSED")


if __name__ == "__main__":
    main()
