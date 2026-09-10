from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from _common import (
    add_common_args,
    assert_finite,
    load_cfg,
    make_work_dir,
    make_workspace,
    print_pass,
    print_section,
)
from post_rl.data.offline_dataset import OfflineDataset
from post_rl.utils.common import dict_apply


def _assert_endpoint_alignment(workspace, raw: dict, dataset: OfflineDataset, idx: int) -> None:
    obs_idx, next_transition_idx = dataset._endpoint_indices(idx)
    mapped_next_idx = int(workspace.latent_cache.next_indices[next_transition_idx])
    transition_steps = (
        int(workspace.cfg.n_action_steps)
        if bool(workspace.cfg.chunk_as_single_action)
        else 1
    )

    np.testing.assert_array_equal(
        raw["obs"]["state"][0].cpu().numpy(),
        np.asarray(workspace.buffer["state"][obs_idx]),
    )
    np.testing.assert_array_equal(
        raw["next_obs"]["state"][transition_steps - 1].cpu().numpy(),
        np.asarray(workspace.buffer["state"][mapped_next_idx]),
    )

    for key in OfflineDataset.RGB_KEYS:
        np.testing.assert_array_equal(
            raw["obs"][key][0].cpu().numpy(),
            np.asarray(workspace.buffer[key][obs_idx]),
        )
        np.testing.assert_array_equal(
            raw["next_obs"][key][transition_steps - 1].cpu().numpy(),
            np.asarray(workspace.buffer[key][mapped_next_idx]),
        )

    if bool(workspace.cfg.dataset.use_depth):
        for key in OfflineDataset.DEPTH_KEYS:
            np.testing.assert_array_equal(
                raw["obs"][key][0].cpu().numpy(),
                np.asarray(workspace.buffer[key][obs_idx]),
            )
            np.testing.assert_array_equal(
                raw["next_obs"][key][transition_steps - 1].cpu().numpy(),
                np.asarray(workspace.buffer[key][mapped_next_idx]),
            )


def _assert_latent_consistent(
    cached: torch.Tensor,
    direct: torch.Tensor,
    label: str,
    *,
    max_abs_limit: float = 0.1,
    relative_l2_limit: float = 0.02,
) -> None:
    cached = cached.detach().float()
    direct = direct.detach().float()
    if cached.shape != direct.shape:
        raise AssertionError(
            f"{label} latent shape mismatch: {tuple(cached.shape)} vs {tuple(direct.shape)}"
        )

    diff = cached - direct
    max_abs = float(diff.abs().max().item())
    relative_l2 = float(
        torch.linalg.vector_norm(diff).item()
        / max(torch.linalg.vector_norm(direct).item(), 1e-12)
    )
    print(
        f"{label} latent numeric drift: max_abs={max_abs:.6g}, "
        f"relative_l2={relative_l2:.6g}"
    )
    if max_abs > max_abs_limit or relative_l2 > relative_l2_limit:
        raise AssertionError(
            f"{label} cached latent is not numerically consistent with direct encoding: "
            f"max_abs={max_abs:.6g} (limit {max_abs_limit}), "
            f"relative_l2={relative_l2:.6g} (limit {relative_l2_limit})"
        )


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(
            description="Smoke 07: endpoint sampling + frozen ACT latent cache"
        )
    )
    parser.add_argument("--cache-batch-size", type=int, default=64)
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.dataset.endpoint_obs_only = True
    cfg.dataset.use_latent_cache = True
    cfg.dataset.latent_cache_batch_size = int(args.cache_batch_size)

    workspace = make_workspace(cfg, make_work_dir(args, "smoke_07_latent_cache"))
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()

    print_section("build/reuse frozen latent cache")
    workspace._build_act_observation_frontends()
    cache = workspace.latent_cache
    if cache is None:
        raise AssertionError("Frozen latent cache was not built")
    if len(cache.obs) != len(workspace.buffer):
        raise AssertionError("Latent cache length does not match OfflineBuffer")
    if workspace.model._frozen_encoder_pos_embed.numel() == 0:
        raise AssertionError("Frozen policy positional embedding was not prepared")
    print_pass(f"cache ready with shape={cache.obs.shape}")

    print_section("cached endpoints equal direct ACT encoder")
    raw_dataset = OfflineDataset(
        buffer=workspace.buffer,
        horizon=int(cfg.horizon),
        pad_before=int(cfg.dataset.pad_before),
        pad_after=int(cfg.dataset.pad_after),
        sequence_stride=int(cfg.dataset.sequence_stride),
        seed=int(cfg.training.seed),
        val_ratio=float(cfg.dataset.val_ratio),
        max_train_episodes=cfg.dataset.max_train_episodes,
        use_depth=bool(cfg.dataset.use_depth),
    )
    raw = raw_dataset[0]
    cached = workspace.dataset[0]
    if set(cached["obs"]) != {"latent"} or set(cached["next_obs"]) != {"latent"}:
        raise AssertionError("Endpoint dataset should expose latent-only observations")

    _assert_endpoint_alignment(workspace, raw, workspace.dataset, 0)
    print_pass("raw current/next endpoint indices match the cache transition mapping")

    raw_obs = dict_apply(raw["obs"], lambda x: x.unsqueeze(0).to(workspace.device))
    raw_next_obs = dict_apply(
        raw["next_obs"],
        lambda x: x.unsqueeze(0).to(workspace.device),
    )
    transition_steps = (
        int(cfg.n_action_steps) if bool(cfg.chunk_as_single_action) else 1
    )
    with torch.no_grad():
        direct_obs = workspace.obs_adapter.encode(raw_obs, start=0)[:, 0]
        direct_next = workspace.obs_adapter.encode(
            raw_next_obs,
            start=transition_steps - 1,
        )[:, 0]
    cached_obs = cached["obs"]["latent"].unsqueeze(0).to(workspace.device)
    cached_next = cached["next_obs"]["latent"].unsqueeze(0).to(workspace.device)
    cached_obs = workspace.obs_adapter.encode({"latent": cached_obs})[:, 0]
    cached_next = workspace.obs_adapter.encode({"latent": cached_next})[:, 0]

    _assert_latent_consistent(cached_obs, direct_obs, "current")
    _assert_latent_consistent(cached_next, direct_next, "next")
    print_pass("cached current/next endpoints are numerically consistent with direct ACT encoding")

    print_section("cached Critic and Dynamics updates")
    batch = next(iter(workspace.train_dataloader))
    batch = dict_apply(batch, lambda x: x.to(workspace.device, non_blocking=True))
    if "latent" not in batch["obs"] or "latent" not in batch["next_obs"]:
        raise AssertionError("Training batch did not use latent cache")

    workspace._build_critic()
    q_loss, v_loss = workspace.critic.update(batch)
    if not math.isfinite(q_loss) or not math.isfinite(v_loss):
        raise AssertionError(f"Non-finite cached critic loss: q={q_loss}, v={v_loss}")

    workspace._build_dynamics()
    state = workspace.dynamics.obs2latent(batch["obs"])
    next_state = workspace.dynamics.next_obs2latent(batch["next_obs"])
    dyn_loss = workspace.dynamics.learn(
        batch=batch,
        nobs_features=state.reshape(state.shape[0], -1),
        next_nobs_features=next_state.reshape(next_state.shape[0], -1),
    )
    assert_finite(dyn_loss, "cached_dynamics_loss")
    workspace.dynamics.optimize(dyn_loss)
    print_pass("Critic and Dynamics update directly from cached latents")

    print_section("PPO frozen-latent reuse")
    workspace._build_ppo()
    workspace._build_finetune_dataloader()
    if workspace.finetune_sampler is None:
        raise AssertionError("Resumable PPO requires an epoch-addressable finetune sampler")
    finetune_batch = workspace.sample_finetune_batch()
    if "obs" not in finetune_batch or "next_obs" in finetune_batch:
        raise AssertionError("PPO endpoint batch must contain current obs only")
    if set(finetune_batch["obs"]) != {"latent"}:
        raise AssertionError("PPO should reuse latent-only cached observations")

    policy_obs = workspace.obs_adapter.normalize_obs(finetune_batch["obs"])
    policy_obs = {key: value[:, 0] for key, value in policy_obs.items()}
    with torch.no_grad():
        reused_latent, reused_pos = workspace.unio4._policy.encode_observation(policy_obs)
    torch.testing.assert_close(
        reused_latent,
        finetune_batch["obs"]["latent"][:, 0].float(),
        rtol=0.0,
        atol=0.0,
    )
    if reused_pos.shape[0] != reused_latent.shape[1]:
        raise AssertionError("Cached PPO positional embedding/token count mismatch")

    ppo_loss = workspace.unio4.update_distribution(
        finetune_batch,
        workspace.critic,
        is_clip_decay=False,
        is_lr_decay=False,
        is_linear_decay=False,
    )
    if not math.isfinite(ppo_loss):
        raise AssertionError(f"Non-finite cached PPO loss: {ppo_loss}")
    mean_q, mean_reward = workspace._evaluate_dynamics_ope()
    if not math.isfinite(mean_q) or not math.isfinite(mean_reward):
        raise AssertionError(
            f"Non-finite cached Dynamics OPE: mean_q={mean_q}, reward={mean_reward}"
        )
    print_pass("PPO update and Dynamics OPE reuse frozen ACT latents")

    print("\nSMOKE 07 PASSED")


if __name__ == "__main__":
    main()
