from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from analysis_05_terminal_advantage import (  # noqa: E402
    _episode_bounds,
    _load_postrl_policy,
    _processor_fingerprint,
    _resolve_config,
    _resolve_processor_dir,
    _resolve_required_path,
)
from analysis_06_ppo_local_advantage import (  # noqa: E402
    ACTION_GROUPS,
    _select_anchors,
)
from analysis_08_one_step_ppo_replay import (  # noqa: E402
    _aggregate_rows,
    _build_probe_rows,
    _plot_phase_summary,
    _preview_exact_update_batch,
    _probe_policy,
    _restore_rng_state,
    _rng_state,
)
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402
from post_rl.utils.common import dict_apply  # noqa: E402


PHASE_ORDER = (
    "early_0_25",
    "middle_25_50",
    "late_50_75",
    "tail_75_100",
)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sample_batch(
    dataset,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    if batch_size > len(dataset):
        raise ValueError(
            f"finetune_batch_size={batch_size} exceeds dataset size={len(dataset)}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:batch_size].astype(np.int64)
    batch = default_collate([dataset[int(index)] for index in indices])
    batch = dict_apply(
        batch,
        lambda x: x.to(device, non_blocking=True),
    )
    return batch, indices


def _step_decay_args(cfg) -> dict:
    step = 0
    decay_stop_step = int(cfg.unio4.decay_stop_step)
    decay_active = decay_stop_step < 0 or step <= decay_stop_step
    linear_active = bool(cfg.unio4.is_linear_decay) and decay_active
    if linear_active:
        progress = step / max(int(cfg.unio4.bppo_steps), 1)
        bppo_lr_now = float(cfg.unio4.bppo_lr) * (1.0 - progress)
        clip_ratio_now = float(cfg.unio4.clip_ratio) * (1.0 - progress)
    else:
        bppo_lr_now = None
        clip_ratio_now = None
    return {
        "is_clip_decay": bool(cfg.unio4.is_clip_decay) and decay_active,
        "is_lr_decay": bool(cfg.unio4.is_bppo_lr_decay) and decay_active,
        "is_linear_decay": linear_active,
        "bppo_lr_now": bppo_lr_now,
        "clip_ratio_now": clip_ratio_now,
    }


def _repeat_phase_expectation(
    repeat_phase_rows: list[dict],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in repeat_phase_rows:
        grouped[row["phase"]].append(row)

    metrics = (
        "mean_one_step_rmse_all",
        "mean_one_step_rmse_left_joints",
        "mean_one_step_rmse_left_gripper",
        "mean_one_step_rmse_right_joints",
        "mean_one_step_rmse_right_gripper",
        "mean_cos_one_step_terminal_all",
        "mean_cos_one_step_final_postrl_all",
        "mean_cos_one_step_terminal_left_joints",
        "mean_cos_one_step_terminal_left_gripper",
        "mean_cos_one_step_terminal_right_joints",
        "mean_cos_one_step_terminal_right_gripper",
    )

    result = []
    for phase in PHASE_ORDER:
        items = grouped.get(phase, [])
        if not items:
            continue
        out = {
            "phase": phase,
            "repeats": len(items),
        }
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in items],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                out[f"repeat_mean_{metric}"] = float("nan")
                out[f"repeat_std_{metric}"] = float("nan")
                out[f"repeat_positive_fraction_{metric}"] = float("nan")
            else:
                out[f"repeat_mean_{metric}"] = float(finite.mean())
                out[f"repeat_std_{metric}"] = float(
                    finite.std(ddof=0)
                )
                out[f"repeat_positive_fraction_{metric}"] = float(
                    np.mean(finite > 0)
                )
        result.append(out)
    return result


def _enrich_expected_rows_with_repeat_variance(
    rows: list[dict],
    mean_drift: torch.Tensor,
    mean_square_drift: torch.Tensor,
) -> None:
    variance = torch.clamp(
        mean_square_drift - mean_drift.square(),
        min=0.0,
    )
    for group_name, group_slice in ACTION_GROUPS.items():
        group_var = variance[..., group_slice]
        std_rmse = torch.sqrt(
            group_var.mean(dim=(1, 2))
        )
        group_mean = mean_drift[..., group_slice]
        mean_rmse = torch.sqrt(
            group_mean.square().mean(dim=(1, 2))
        )
        snr = mean_rmse / std_rmse.clamp_min(1e-12)
        for index, row in enumerate(rows):
            row[f"repeat_std_rmse_{group_name}"] = float(
                std_rmse[index].item()
            )
            row[f"expected_drift_snr_{group_name}"] = float(
                snr[index].item()
            )


def _aggregate_expected_rows(
    rows: list[dict],
    group_key: str,
) -> list[dict]:
    base = _aggregate_rows(rows, group_key)
    lookup = {
        row[group_key]: row
        for row in base
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)

    for group, items in grouped.items():
        out = lookup[group]
        for group_name in ACTION_GROUPS:
            for metric in (
                "repeat_std_rmse",
                "expected_drift_snr",
            ):
                key = f"{metric}_{group_name}"
                values = np.asarray(
                    [float(row[key]) for row in items],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                out[f"mean_{key}"] = (
                    float(finite.mean())
                    if finite.size
                    else float("nan")
                )
    return base


def _plot_repeat_consistency(
    output_dir: pathlib.Path,
    repeat_expectation_rows: list[dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not repeat_expectation_rows:
        return

    x = np.arange(len(repeat_expectation_rows))
    labels = [row["phase"] for row in repeat_expectation_rows]

    plt.figure(figsize=(10, 5))
    means = [
        row["repeat_mean_mean_cos_one_step_terminal_all"]
        for row in repeat_expectation_rows
    ]
    stds = [
        row["repeat_std_mean_cos_one_step_terminal_all"]
        for row in repeat_expectation_rows
    ]
    plt.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label="whole chunk",
    )
    for metric, label in (
        (
            "mean_cos_one_step_terminal_left_gripper",
            "left gripper",
        ),
        (
            "mean_cos_one_step_terminal_right_gripper",
            "right gripper",
        ),
    ):
        plt.plot(
            x,
            [
                row[f"repeat_mean_{metric}"]
                for row in repeat_expectation_rows
            ],
            marker="o",
            label=label,
        )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("across-repeat mean alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "repeat_alignment_consistency.png",
        dpi=160,
    )
    plt.close()


def _print_report(
    *,
    repeat_update_rows: list[dict],
    repeat_expectation_rows: list[dict],
    expected_phase_rows: list[dict],
    repeats: int,
    cfg,
) -> None:
    losses = np.asarray(
        [row["loss"] for row in repeat_update_rows],
        dtype=np.float64,
    )
    raw_positive = np.asarray(
        [
            row["raw_adv_positive_fraction"]
            for row in repeat_update_rows
        ],
        dtype=np.float64,
    )
    norm_positive = np.asarray(
        [
            row["normalized_adv_positive_fraction"]
            for row in repeat_update_rows
        ],
        dtype=np.float64,
    )

    print("\n=== Multi-Batch One-Step PPO Expectation Diagnostic ===")
    print(f"independent one-step repeats: {repeats}")
    print(
        f"finetune stride: {int(cfg.dataset.finetune_sequence_stride)}"
    )
    print(
        f"finetune batch size: {int(cfg.unio4.finetune_batch_size)}"
    )
    print(
        "loss mean/std: "
        f"{losses.mean():+.8f} / {losses.std(ddof=0):.8f}"
    )
    print(
        "raw advantage positive fraction mean/std: "
        f"{100.0 * raw_positive.mean():.1f}% / "
        f"{100.0 * raw_positive.std(ddof=0):.1f}%"
    )
    print(
        "normalized advantage positive fraction mean/std: "
        f"{100.0 * norm_positive.mean():.1f}% / "
        f"{100.0 * norm_positive.std(ddof=0):.1f}%"
    )

    expected_lookup = {
        row["phase"]: row
        for row in expected_phase_rows
    }
    repeat_lookup = {
        row["phase"]: row
        for row in repeat_expectation_rows
    }

    print(
        "\nExpected drift E[delta action] across repeats "
        "(average vector first, then measure alignment):"
    )
    print(
        "  phase             all_rmse   Ljoint     Lgrip      "
        "Rjoint     Rgrip      Edrift->terminal  Edrift->PostRL"
    )
    for phase in PHASE_ORDER:
        row = expected_lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:17s} "
            f"{row['mean_one_step_rmse_all']:.7f}  "
            f"{row['mean_one_step_rmse_left_joints']:.7f}  "
            f"{row['mean_one_step_rmse_left_gripper']:.7f}  "
            f"{row['mean_one_step_rmse_right_joints']:.7f}  "
            f"{row['mean_one_step_rmse_right_gripper']:.7f}  "
            f"{row['mean_cos_one_step_terminal_all']:+.4f}             "
            f"{row['mean_cos_one_step_final_postrl_all']:+.4f}"
        )

    print(
        "\nAcross-repeat consistency "
        "(mean phase alignment +/- repeat std; positive-repeat fraction):"
    )
    print(
        "  phase             whole->terminal       Lgrip->terminal       "
        "Rgrip->terminal"
    )
    for phase in PHASE_ORDER:
        row = repeat_lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:17s} "
            f"{row['repeat_mean_mean_cos_one_step_terminal_all']:+.4f} "
            f"+/- {row['repeat_std_mean_cos_one_step_terminal_all']:.4f} "
            f"({100.0 * row['repeat_positive_fraction_mean_cos_one_step_terminal_all']:.1f}%)   "
            f"{row['repeat_mean_mean_cos_one_step_terminal_left_gripper']:+.4f} "
            f"+/- {row['repeat_std_mean_cos_one_step_terminal_left_gripper']:.4f} "
            f"({100.0 * row['repeat_positive_fraction_mean_cos_one_step_terminal_left_gripper']:.1f}%)   "
            f"{row['repeat_mean_mean_cos_one_step_terminal_right_gripper']:+.4f} "
            f"+/- {row['repeat_std_mean_cos_one_step_terminal_right_gripper']:.4f} "
            f"({100.0 * row['repeat_positive_fraction_mean_cos_one_step_terminal_right_gripper']:.1f}%)"
        )

    print("\nExpected drift signal-to-repeat-noise ratio:")
    print(
        "  phase             all      Ljoint   Lgrip    Rjoint   Rgrip"
    )
    for phase in PHASE_ORDER:
        row = expected_lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:17s} "
            f"{row['mean_expected_drift_snr_all']:.3f}    "
            f"{row['mean_expected_drift_snr_left_joints']:.3f}    "
            f"{row['mean_expected_drift_snr_left_gripper']:.3f}    "
            f"{row['mean_expected_drift_snr_right_joints']:.3f}    "
            f"{row['mean_expected_drift_snr_right_gripper']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat independent one-step production Offline PPO updates from the same "
            "Base ACT initialization across multiple shuffled finetune batches/seeds, "
            "then estimate the expected deterministic ACT output drift. Analysis-only."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Independent Base-ACT -> one-PPO-step replays.",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
        help="Optional evenly spaced cap over real PPO-stride probe anchors.",
    )
    parser.add_argument("--progress-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("--repeats must be >= 2")
    if args.probe_batch_size < 1:
        raise ValueError("--probe-batch-size must be >= 1")
    if args.progress_bins < 4:
        raise ValueError("--progress-bins must be >= 4")

    stage1_dir = pathlib.Path(
        args.stage1_dir
    ).expanduser().resolve()
    critic_final = stage1_dir / "critic" / "checkpoints" / "final"
    for name in ("Q.pt", "value.pt", "contract.json"):
        path = critic_final / name
        if not path.is_file():
            raise FileNotFoundError(path)

    config_path = _resolve_config(stage1_dir, args.config)
    cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_required_path(
        args.checkpoint,
        cfg.input.get("policy_checkpoint"),
        "Base ACT checkpoint",
    )
    dataset = _resolve_required_path(
        args.dataset,
        cfg.input.get("dataset_path"),
        "Offline dataset",
    )
    latent_cache_dir = _resolve_required_path(
        args.latent_cache_dir,
        None,
        "Frozen latent cache",
    )
    postrl_checkpoint = _resolve_required_path(
        args.postrl_checkpoint,
        None,
        "Post-RL checkpoint",
    )

    cfg.input.policy_checkpoint = str(checkpoint)
    cfg.input.policy_checkpoint_type = "il"
    cfg.input.dataset_path = str(dataset)
    cfg.training.device = str(args.device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    cfg.dataset.use_latent_cache = True
    cfg.dataset.endpoint_obs_only = True
    cfg.dataset.latent_cache_dir = str(latent_cache_dir)
    cfg.critic.load_pretrain = True
    cfg.critic.artifact_dir = str(stage1_dir / "critic")
    if cfg.get("ppo") is not None:
        cfg.ppo.enable_ratio_logging = False
        cfg.ppo.enable_monitoring_csv = False

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "post_training"
        / "outputs"
        / "multi_batch_one_step_ppo_expectation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = (
        int(cfg.training.seed)
        if args.seed is None
        else int(args.seed)
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    workspace = TrainACTWorkspace(
        cfg,
        output_dir=str(output_dir),
    )
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError(
            "Expected pretrained Q/V; analysis_09 must not train the Critic."
        )
    workspace.critic.eval()

    if int(workspace.action_dim) != 16:
        raise RuntimeError(
            "analysis_09 currently expects the sim_task1 16D bimanual action contract."
        )

    episode_starts, episode_ends = _episode_bounds(
        workspace.buffer.episode_ends
    )
    chunk_size = int(cfg.n_action_steps)
    stride = int(cfg.dataset.finetune_sequence_stride)
    anchors, anchor_episode_ids, anchor_starts, anchor_ends = (
        _select_anchors(
            episode_starts,
            episode_ends,
            chunk_size,
            stride,
            args.max_probes,
        )
    )

    episode_lengths = episode_ends - episode_starts
    if np.any(episode_lengths < chunk_size):
        short_ids = np.flatnonzero(
            episode_lengths < chunk_size
        ).tolist()
        raise RuntimeError(
            "Terminal template requires every episode to contain one full "
            f"chunk; short episode ids={short_ids[:10]}"
        )

    terminal_bank_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][
                    int(end) - chunk_size:int(end)
                ],
                dtype=np.float32,
            )
            for end in episode_ends
        ],
        axis=0,
    )
    terminal_raw = torch.from_numpy(
        terminal_bank_raw[anchor_episode_ids]
    ).to(workspace.device)
    terminal_action = (
        workspace.obs_adapter.normalize_action(terminal_raw)
        .detach()
        .cpu()
    )

    postrl_policy, postrl_kind = _load_postrl_policy(
        postrl_checkpoint,
        workspace.device,
        cfg,
    )
    postrl_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    if int(postrl_policy.config.chunk_size) != int(
        workspace.model.config.chunk_size
    ):
        raise RuntimeError(
            "Post-RL/Base ACT chunk mismatch: "
            f"{postrl_policy.config.chunk_size} vs "
            f"{workspace.model.config.chunk_size}"
        )

    il_processor = _resolve_processor_dir(checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_processor_fp = _processor_fingerprint(il_processor)
    postrl_processor_fp = _processor_fingerprint(postrl_processor)
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ. "
            "Refusing multi-batch replay comparison."
        )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    postrl_encoder_sha = workspace._fingerprint_module(
        postrl_encoder
    )
    cache_encoder_sha = str(
        workspace.latent_cache.metadata.get(
            "encoder_sha256",
            "",
        )
    )
    if postrl_encoder_sha != cache_encoder_sha:
        raise RuntimeError(
            "Post-RL encoder does not match the frozen latent cache: "
            f"postrl={postrl_encoder_sha[:12]}..., "
            f"cache={cache_encoder_sha[:12]}..."
        )

    print("Probing fixed Base ACT reference...")
    base_action = _probe_policy(
        workspace,
        workspace.model,
        anchors,
        int(args.probe_batch_size),
    )
    print("Probing fixed final Post-RL reference...")
    final_postrl_action = _probe_policy(
        workspace,
        postrl_policy,
        anchors,
        int(args.probe_batch_size),
    )

    postrl_policy.to("cpu")
    del postrl_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build the exact production finetune dataset once. The temporary PPO
    # instance is discarded; each repeat below constructs a fresh optimizer,
    # current policy, and old/reference policy from Base ACT.
    workspace._build_ppo()
    workspace.unio4._policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    workspace.unio4._old_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )
    workspace._build_finetune_dataloader()
    finetune_dataset = workspace.finetune_dataset
    del workspace.unio4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Base ACT is immutable throughout the experiment. Keeping it on CPU
    # reduces GPU memory while still allowing _build_ppo() to deepcopy it.
    workspace.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    drift_sum = torch.zeros_like(base_action)
    drift_square_sum = torch.zeros_like(base_action)
    repeat_update_rows = []
    repeat_phase_rows = []
    decay_args = _step_decay_args(cfg)
    finetune_batch_size = int(cfg.unio4.finetune_batch_size)

    for repeat in tqdm(
        range(int(args.repeats)),
        desc="Independent one-step PPO replays",
    ):
        repeat_seed = seed + repeat

        workspace._build_ppo()
        workspace.unio4._policy.set_frozen_encoder_pos_embed(
            workspace._frozen_encoder_pos_embed
        )
        workspace.unio4._old_policy.set_frozen_encoder_pos_embed(
            workspace._frozen_encoder_pos_embed
        )

        if repeat == 0:
            check_count = min(len(anchors), 64)
            initial_clone_action = _probe_policy(
                workspace,
                workspace.unio4._policy,
                anchors[:check_count],
                int(args.probe_batch_size),
            )
            max_init_diff = float(
                (
                    initial_clone_action
                    - base_action[:check_count]
                )
                .abs()
                .max()
                .item()
            )
            if max_init_diff > 1e-6:
                raise RuntimeError(
                    "PPO policy clone does not match Base ACT before update: "
                    f"max_abs={max_init_diff:.6g}"
                )
        else:
            max_init_diff = 0.0

        batch, batch_indices = _sample_batch(
            finetune_dataset,
            batch_size=finetune_batch_size,
            seed=repeat_seed,
            device=workspace.device,
        )

        torch.manual_seed(repeat_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(repeat_seed)

        replay_rng = _rng_state()
        batch_preview = _preview_exact_update_batch(
            workspace,
            batch,
        )
        _restore_rng_state(replay_rng)

        log_std_before = (
            workspace.unio4._policy._get_log_std()
            .detach()
            .float()
            .clone()
        )
        loss = workspace.unio4.update_distribution(
            batch=batch,
            critic=workspace.critic,
            **decay_args,
        )
        log_std_after = (
            workspace.unio4._policy._get_log_std()
            .detach()
            .float()
            .clone()
        )

        one_step_action = _probe_policy(
            workspace,
            workspace.unio4._policy,
            anchors,
            int(args.probe_batch_size),
        )
        drift = one_step_action - base_action
        drift_sum += drift
        drift_square_sum += drift.square()

        rows = _build_probe_rows(
            base_action=base_action,
            one_step_action=one_step_action,
            final_postrl_action=final_postrl_action,
            terminal_action=terminal_action,
            anchors=anchors,
            episode_ids=anchor_episode_ids,
            episode_starts=anchor_starts,
            episode_ends=anchor_ends,
            progress_bins=int(args.progress_bins),
        )
        phase_rows = _aggregate_rows(rows, "phase")
        for row in phase_rows:
            row["repeat"] = repeat
            row["repeat_seed"] = repeat_seed
            repeat_phase_rows.append(row)

        repeat_update_rows.append(
            {
                "repeat": repeat,
                "repeat_seed": repeat_seed,
                "loss": float(loss),
                "batch_size": finetune_batch_size,
                "batch_index_min": int(batch_indices.min()),
                "batch_index_max": int(batch_indices.max()),
                "raw_adv_mean": batch_preview["raw_adv_mean"],
                "raw_adv_std": batch_preview["raw_adv_std"],
                "raw_adv_positive_fraction": batch_preview[
                    "raw_adv_positive_fraction"
                ],
                "normalized_adv_mean": batch_preview[
                    "normalized_adv_mean"
                ],
                "normalized_adv_std": batch_preview[
                    "normalized_adv_std"
                ],
                "normalized_adv_positive_fraction": batch_preview[
                    "normalized_adv_positive_fraction"
                ],
                "log_std_mean_delta": float(
                    (log_std_after - log_std_before)
                    .mean()
                    .item()
                ),
                "log_std_max_abs_delta": float(
                    (log_std_after - log_std_before)
                    .abs()
                    .max()
                    .item()
                ),
                "initial_clone_max_abs_action_diff": max_init_diff,
            }
        )

        del workspace.unio4
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    repeat_count = float(args.repeats)
    expected_drift = drift_sum / repeat_count
    mean_square_drift = drift_square_sum / repeat_count
    expected_action = base_action + expected_drift

    expected_rows = _build_probe_rows(
        base_action=base_action,
        one_step_action=expected_action,
        final_postrl_action=final_postrl_action,
        terminal_action=terminal_action,
        anchors=anchors,
        episode_ids=anchor_episode_ids,
        episode_starts=anchor_starts,
        episode_ends=anchor_ends,
        progress_bins=int(args.progress_bins),
    )
    _enrich_expected_rows_with_repeat_variance(
        expected_rows,
        expected_drift,
        mean_square_drift,
    )
    expected_phase_summary = _aggregate_expected_rows(
        expected_rows,
        "phase",
    )
    expected_progress_summary = _aggregate_expected_rows(
        expected_rows,
        "progress_bin",
    )
    repeat_expectation_summary = _repeat_phase_expectation(
        repeat_phase_rows
    )

    _write_csv(
        output_dir / "repeat_update_summary.csv",
        repeat_update_rows,
    )
    _write_csv(
        output_dir / "repeat_phase_summary.csv",
        repeat_phase_rows,
    )
    _write_csv(
        output_dir / "repeat_phase_expectation.csv",
        repeat_expectation_summary,
    )
    _write_csv(
        output_dir / "expected_drift_per_anchor.csv",
        expected_rows,
    )
    _write_csv(
        output_dir / "expected_drift_phase_summary.csv",
        expected_phase_summary,
    )
    _write_csv(
        output_dir / "expected_drift_progress_summary.csv",
        expected_progress_summary,
    )

    _plot_phase_summary(
        output_dir,
        expected_phase_summary,
    )
    _plot_repeat_consistency(
        output_dir,
        repeat_expectation_summary,
    )

    summary = {
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "postrl_checkpoint": str(postrl_checkpoint),
        "postrl_checkpoint_kind": postrl_kind,
        "dataset": str(dataset),
        "latent_cache_dir": str(latent_cache_dir),
        "processor_fingerprint": il_processor_fp,
        "encoder_sha256": cache_encoder_sha,
        "seed": seed,
        "repeats": int(args.repeats),
        "chunk_size": chunk_size,
        "ppo_anchor_stride": stride,
        "ppo_finetune_batch_size": finetune_batch_size,
        "finetune_dataset_size": int(len(finetune_dataset)),
        "probe_anchors": int(len(anchors)),
        "repeat_update_summary": repeat_update_rows,
        "repeat_phase_expectation": repeat_expectation_summary,
        "expected_drift_phase_summary": expected_phase_summary,
        "expected_drift_progress_summary": expected_progress_summary,
        "metric_contract": {
            "independent_repeat": (
                "fresh Base-ACT PPO policy, fresh old/reference policy, fresh "
                "optimizer/scheduler state, one representative shuffled batch "
                "from the production finetune dataset, and one production "
                "update_distribution call"
            ),
            "expected_drift": (
                "vector average of deterministic after-one-step minus Base-ACT "
                "actions across repeats, computed per fixed probe observation "
                "before cosine/alignment metrics"
            ),
            "repeat_consistency": (
                "distribution across independent one-step replays of each "
                "phase-averaged alignment metric"
            ),
            "expected_drift_snr": (
                "RMSE magnitude of the across-repeat mean drift divided by the "
                "across-repeat RMS standard deviation at the same probe/group"
            ),
        },
        "scope_note": (
            "This estimates the expectation of one first-step PPO update over "
            "independent representative shuffled finetune batches and policy "
            "sampling seeds. It does not reconstruct the exact historical Stage-2 "
            "batch order, nor multi-step optimizer momentum, clipping accumulation, "
            "OPE old-policy refreshes, EMA selection, or intermediate checkpoints."
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=True,
        )

    _print_report(
        repeat_update_rows=repeat_update_rows,
        repeat_expectation_rows=repeat_expectation_summary,
        expected_phase_rows=expected_phase_summary,
        repeats=int(args.repeats),
        cfg=cfg,
    )
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
