from __future__ import annotations

import argparse

import torch

from _common import (
    add_common_args,
    assert_finite,
    build_real_batch,
    load_cfg,
    make_work_dir,
    make_workspace,
    policy_obs_from_batch,
    print_pass,
    print_section,
)
from post_rl.data.offline_buffer import LazyZarrArray


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Smoke 02: real Zarr + stochastic ACT + Scheme-C latent"),
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    workspace = make_workspace(cfg, make_work_dir(args, "smoke_02_policy_data"))
    dataset, batch = build_real_batch(workspace, args.batch_size)

    print_section("offline data")
    modal_keys = [
        *workspace.buffer.RGB_KEYS,
        *(f"next_{key}" for key in workspace.buffer.RGB_KEYS),
    ]
    if bool(cfg.dataset.use_depth):
        modal_keys.extend(workspace.buffer.DEPTH_KEYS)
        modal_keys.extend(f"next_{key}" for key in workspace.buffer.DEPTH_KEYS)
    non_lazy = [
        key for key in modal_keys
        if not isinstance(workspace.buffer[key], LazyZarrArray)
    ]
    if non_lazy:
        raise AssertionError(
            f"Zarr image/depth modalities must remain lazy in OfflineBuffer: {non_lazy}"
        )
    print_pass("Zarr RGB/depth modalities remain lazy and are loaded only per sampled sequence")

    required = {"obs", "next_obs", "action", "next_action", "reward", "not_done", "return"}
    missing = required.difference(batch)
    if missing:
        raise AssertionError(f"Dataset batch is missing keys: {sorted(missing)}")
    assert_finite(batch, "batch")
    batch_size = batch["action"].shape[0]
    horizon = int(cfg.horizon)
    action_dim = int(workspace.action_dim)
    if batch["action"].shape[1] != horizon:
        raise AssertionError(
            f"Action horizon mismatch: {batch['action'].shape[1]} != {horizon}"
        )
    if batch["action"].shape[-1] != action_dim:
        raise AssertionError(
            f"Action dim mismatch: {batch['action'].shape[-1]} != {action_dim}"
        )
    print_pass(f"real dataset batch is valid: B={batch_size}, H={horizon}, D={action_dim}")

    print_section("stochastic ACT")
    policy_obs = policy_obs_from_batch(workspace, batch)
    with torch.no_grad():
        mu = workspace.model.get_action_mean(policy_obs)
        dist = workspace.model.get_distribution(policy_obs)
        sample, log_prob, entropy = workspace.model.sample_action_chunk(policy_obs)
        log_std = workspace.model._get_log_std()
        std = workspace.model._get_std()

    expected = (batch_size, int(workspace.model.config.chunk_size), action_dim)
    for name, tensor in {
        "mu": mu,
        "sample": sample,
        "log_prob": log_prob,
        "entropy": entropy,
    }.items():
        if tuple(tensor.shape) != expected:
            raise AssertionError(f"{name} shape {tuple(tensor.shape)} != {expected}")
        assert_finite(tensor, name)
    if not torch.all(std > 0):
        raise AssertionError("Gaussian std must be strictly positive")
    if float(log_std.min()) < float(cfg.policy.log_std_min) - 1e-6:
        raise AssertionError("log_std is below configured minimum")
    if float(log_std.max()) > float(cfg.policy.log_std_max) + 1e-6:
        raise AssertionError("log_std is above configured maximum")
    print_pass("mu/sample/log_prob/entropy shapes and Gaussian bounds are correct")

    print_section("IL checkpoint migration")
    from kuavo_train.wrapper.policy.act.ACTConfigWrapper import CustomACTConfigWrapper
    from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper

    det_cfg = CustomACTConfigWrapper.from_pretrained(args.checkpoint)
    det_cfg.device = str(workspace.device)
    deterministic = CustomACTPolicyWrapper.from_pretrained(
        args.checkpoint,
        config=det_cfg,
        strict=True,
    ).to(workspace.device)
    deterministic.eval()
    with torch.no_grad():
        deterministic_mu = deterministic.predict_action_chunk(policy_obs)
    torch.testing.assert_close(mu, deterministic_mu, rtol=1e-5, atol=1e-6)
    print_pass("stochastic ACT initial mean matches deterministic IL ACT output")

    print_section("Scheme-C latent")
    with torch.no_grad():
        latent, encoder_pos = workspace.model.encode_observation(policy_obs)
        adapter_latent = workspace.obs_adapter.encode(batch["obs"])[:, 0]
    if latent.ndim != 3:
        raise AssertionError(f"Policy latent must be [B,S,D], got {tuple(latent.shape)}")
    if latent.shape[0] != batch_size or latent.shape[-1] != workspace.obs_feature_dim:
        raise AssertionError(
            f"Unexpected latent shape {tuple(latent.shape)} for B={batch_size}, D={workspace.obs_feature_dim}"
        )
    if encoder_pos.shape[0] != latent.shape[1]:
        raise AssertionError("encoder positional-token count does not match latent token count")
    torch.testing.assert_close(latent, adapter_latent, rtol=1e-5, atol=1e-6)
    print_pass("Actor and Critic/Dynamics observation frontends produce the same ACT latent")

    print(f"\nDataset sequences available: {len(dataset)}")
    print("SMOKE 02 PASSED")


if __name__ == "__main__":
    main()
