from __future__ import annotations

import argparse
import gc

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


def run_mode(workspace, batch: dict, chunk_mode: bool) -> None:
    workspace.cfg.chunk_as_single_action = chunk_mode
    workspace._build_dynamics()
    dynamics = workspace.dynamics
    batch_size = batch["action"].shape[0]

    state = dynamics.obs2latent(batch["obs"])
    next_state = dynamics.next_obs2latent(batch["next_obs"])
    inputs, targets = dynamics.format_samples_for_training(
        batch,
        state.reshape(batch_size, -1),
        next_state.reshape(batch_size, -1),
    )
    state_tokens, action = inputs
    expected_action_shape = (
        (batch_size, int(workspace.cfg.n_action_steps), workspace.action_dim)
        if chunk_mode
        else (batch_size, workspace.action_dim)
    )
    if tuple(action.shape) != expected_action_shape:
        raise AssertionError(
            f"Dynamics action shape {tuple(action.shape)} != {expected_action_shape}"
        )

    expected_delta = dynamics._as_tokens(next_state) - dynamics._as_tokens(state)
    torch.testing.assert_close(targets, expected_delta, rtol=1e-5, atol=1e-6)
    assert_finite((state_tokens, action, targets), "dynamics_inputs")

    encoder_before = workspace._fingerprint_module(workspace.obs_adapter.encoder)
    model_before = snapshot_params(dynamics.model)
    loss = dynamics.learn(
        batch=batch,
        nobs_features=state.reshape(batch_size, -1),
        next_nobs_features=next_state.reshape(batch_size, -1),
    )
    assert_finite(loss, "dynamics_loss")
    dynamics.optimize(loss)
    changed = changed_param_names(model_before, dynamics.model)
    if not changed:
        raise AssertionError("Dynamics model parameters did not change after optimizer.step()")
    encoder_after = workspace._fingerprint_module(workspace.obs_adapter.encoder)
    if encoder_before != encoder_after:
        raise AssertionError("Frozen ACT dynamics encoder changed during dynamics update")

    with torch.no_grad():
        predicted_next, reward, terminal, _ = dynamics.step(state, action)
    predicted_next = torch.as_tensor(predicted_next)
    if tuple(predicted_next.shape) != tuple(state.shape):
        raise AssertionError(
            f"Predicted next latent shape {tuple(predicted_next.shape)} != state {tuple(state.shape)}"
        )
    assert_finite(predicted_next, "predicted_next")
    if reward.shape[0] != batch_size or terminal.shape[0] != batch_size:
        raise AssertionError("Dynamics step returned invalid batch dimensions")

    label = "chunk transition s_t,a[t:t+H]->s_t+H" if chunk_mode else "single-step transition s_t,a_t->s_t+1"
    print_pass(f"{label}: shape, delta target, optimizer update and frozen encoder are correct")


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Smoke 04: token dynamics in chunk and single-step modes"),
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    workspace = make_workspace(cfg, make_work_dir(args, "smoke_04_dynamics"))
    _, batch = build_real_batch(workspace, args.batch_size)

    for chunk_mode in (True, False):
        print_section(f"chunk_as_single_action={str(chunk_mode).lower()}")
        run_mode(workspace, batch, chunk_mode)
        workspace.dynamics = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSMOKE 04 PASSED")


if __name__ == "__main__":
    main()
