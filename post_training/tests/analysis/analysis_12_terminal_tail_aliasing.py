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
    _q_components,
    _resolve_config,
    _resolve_required_path,
    _window_starts,
)
from post_rl.training import TrainACTWorkspace  # noqa: E402


DEFAULT_OFFSETS = (-20, -10, -5, 0, 5, 10, 20)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _finite_median(values) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else None


def _fraction(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    if not values:
        return None
    return float(np.mean(values))


def _read_integrity_rows(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "episode_id",
        "left_reopen_frame_offset",
        "right_reopen_frame_offset",
        "last_release_frame_offset",
        "tail_after_last_release",
    }
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def _parse_int(row: dict, key: str) -> int:
    value = row[key]
    if value is None or value == "":
        raise ValueError(f"Missing {key} for episode {row.get('episode_id')}")
    return int(float(value))


def _build_workspace(args):
    stage1_dir = pathlib.Path(args.stage1_dir).expanduser().resolve()
    critic_final = stage1_dir / "critic" / "checkpoints" / "final"
    for name in ("Q.pt", "value.pt", "contract.json"):
        candidate = critic_final / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)

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
        cfg.dataset.get("latent_cache_dir"),
        "Frozen latent cache",
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

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError("Expected pretrained Stage-1 Critic; this diagnostic must not retrain it.")
    workspace.critic.eval()
    workspace.model.eval()

    return workspace, cfg, stage1_dir, checkpoint, dataset, latent_cache_dir, config_path


def _terminal_geometry_rows(
    *,
    integrity_rows: list[dict],
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    chunk_size: int,
    critic_stride: int,
) -> list[dict]:
    if len(integrity_rows) != len(episode_ends):
        raise RuntimeError(
            f"Integrity episodes={len(integrity_rows)} != buffer episodes={len(episode_ends)}"
        )

    rows = []
    for buffer_ep, integrity in enumerate(integrity_rows):
        ep_id = _parse_int(integrity, "episode_id")
        start = int(episode_starts[buffer_ep])
        end = int(episode_ends[buffer_ep])
        length = end - start

        left_release = _parse_int(integrity, "left_reopen_frame_offset")
        right_release = _parse_int(integrity, "right_reopen_frame_offset")
        first_release = min(left_release, right_release)
        last_release = max(left_release, right_release)
        terminal = length - 1
        tail = terminal - last_release
        terminal_anchor = length - chunk_size

        critic_starts = _window_starts(length, chunk_size, critic_stride)
        terminal_window_sampled = bool(
            terminal_anchor >= 0 and np.any(critic_starts == terminal_anchor)
        )
        post_release_hold_steps = max(0, terminal - last_release)
        terminal_chunk_start = max(terminal_anchor, 0)
        hold_steps_inside_terminal_chunk = max(
            0,
            terminal - max(last_release, terminal_chunk_start),
        )
        hold_fraction = (
            hold_steps_inside_terminal_chunk / chunk_size
            if terminal_anchor >= 0
            else float("nan")
        )

        rows.append(
            {
                "buffer_episode": int(buffer_ep),
                "episode_id": int(ep_id),
                "episode_length": int(length),
                "left_release": int(left_release),
                "right_release": int(right_release),
                "first_release": int(first_release),
                "last_release": int(last_release),
                "partial_completion_gap": int(last_release - first_release),
                "terminal_frame": int(terminal),
                "tail_after_last_release": int(tail),
                "chunk_size": int(chunk_size),
                "critic_stride": int(critic_stride),
                "terminal_chunk_anchor": int(terminal_anchor),
                "terminal_anchor_relative_to_last_release": int(
                    terminal_anchor - last_release
                ),
                "post_release_hold_steps": int(post_release_hold_steps),
                "hold_steps_inside_terminal_chunk": int(
                    hold_steps_inside_terminal_chunk
                ),
                "hold_fraction_inside_terminal_chunk": float(hold_fraction),
                "terminal_window_sampled_by_critic": int(terminal_window_sampled),
            }
        )
    return rows


@torch.no_grad()
def _evaluate_probe_batch(
    workspace,
    anchors: np.ndarray,
    episode_ids: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    event_names: list[str],
    relative_offsets: np.ndarray,
    chunk_size: int,
) -> list[dict]:
    device = workspace.device
    latent_np = np.asarray(workspace.latent_cache.obs[anchors], dtype=np.float32)
    latent = torch.from_numpy(latent_np).to(device)
    state_latent = latent.mean(dim=1)

    il = workspace.model.get_action_mean({"latent": latent})
    raw_state_np = np.asarray(workspace.buffer["state"][anchors], dtype=np.float32)
    raw_state = torch.from_numpy(raw_state_np).to(device)
    if raw_state.shape[-1] != il.shape[-1]:
        raise RuntimeError(
            f"Hold counterfactual needs state/action dims to match, got "
            f"{raw_state.shape[-1]} and {il.shape[-1]}"
        )
    hold_raw = raw_state.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()
    hold = workspace.obs_adapter.normalize_action(hold_raw)

    terminal_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][int(end) - chunk_size:int(end)],
                dtype=np.float32,
            )
            for end in episode_ends
        ],
        axis=0,
    )
    terminal = workspace.obs_adapter.normalize_action(
        torch.from_numpy(terminal_raw).to(device)
    )

    v = workspace.critic._value(state_latent).reshape(-1)
    q_il, _, _, _ = _q_components(workspace.critic, latent, il)
    q_hold, _, _, _ = _q_components(workspace.critic, latent, hold)
    q_terminal, _, _, _ = _q_components(workspace.critic, latent, terminal)

    rows = []
    for i, anchor in enumerate(anchors):
        start = int(episode_starts[i])
        end = int(episode_ends[i])
        local = int(anchor - start)
        demo_valid = local + chunk_size <= end - start
        if demo_valid:
            demo_raw_np = np.asarray(
                workspace.buffer["action"][anchor:anchor + chunk_size],
                dtype=np.float32,
            )[None]
            demo = workspace.obs_adapter.normalize_action(
                torch.from_numpy(demo_raw_np).to(device)
            )
            q_demo, _, _, _ = _q_components(
                workspace.critic,
                latent[i:i + 1],
                demo,
            )
            q_demo_value = float(q_demo.item())
        else:
            q_demo_value = float("nan")

        v_value = float(v[i].item())
        q_il_value = float(q_il[i].item())
        q_hold_value = float(q_hold[i].item())
        q_terminal_value = float(q_terminal[i].item())
        rows.append(
            {
                "episode": int(episode_ids[i]),
                "event": event_names[i],
                "relative_offset": int(relative_offsets[i]),
                "anchor": int(anchor),
                "local_anchor": int(local),
                "frames_to_episode_end": int(end - 1 - anchor),
                "demo_chunk_available": int(demo_valid),
                "v": v_value,
                "q_il": q_il_value,
                "q_demo": q_demo_value,
                "q_hold": q_hold_value,
                "q_terminal_template": q_terminal_value,
                "adv_il": q_il_value - v_value,
                "adv_demo": (
                    q_demo_value - v_value
                    if np.isfinite(q_demo_value)
                    else float("nan")
                ),
                "adv_hold": q_hold_value - v_value,
                "adv_terminal_template": q_terminal_value - v_value,
                "q_hold_minus_il": q_hold_value - q_il_value,
                "q_hold_minus_demo": (
                    q_hold_value - q_demo_value
                    if np.isfinite(q_demo_value)
                    else float("nan")
                ),
                "q_terminal_minus_il": q_terminal_value - q_il_value,
                "critic_prefers_hold_over_il": int(q_hold_value > q_il_value),
                "critic_prefers_hold_over_demo": (
                    int(q_hold_value > q_demo_value)
                    if np.isfinite(q_demo_value)
                    else float("nan")
                ),
                "critic_prefers_terminal_over_il": int(
                    q_terminal_value > q_il_value
                ),
                "il_to_hold_rmse": float(
                    torch.sqrt((il[i] - hold[i]).square().mean()).item()
                ),
            }
        )
    return rows


def _aggregate_probe_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["event"], row["relative_offset"])].append(row)

    output = []
    for (event, offset), group in sorted(groups.items()):
        output.append(
            {
                "event": event,
                "relative_offset": int(offset),
                "count": int(len(group)),
                "demo_available_count": int(
                    sum(int(row["demo_chunk_available"]) for row in group)
                ),
                "v_mean": _finite_mean([row["v"] for row in group]),
                "q_il_mean": _finite_mean([row["q_il"] for row in group]),
                "q_demo_mean": _finite_mean([row["q_demo"] for row in group]),
                "q_hold_mean": _finite_mean([row["q_hold"] for row in group]),
                "q_terminal_template_mean": _finite_mean(
                    [row["q_terminal_template"] for row in group]
                ),
                "q_hold_minus_il_mean": _finite_mean(
                    [row["q_hold_minus_il"] for row in group]
                ),
                "q_hold_minus_demo_mean": _finite_mean(
                    [row["q_hold_minus_demo"] for row in group]
                ),
                "q_terminal_minus_il_mean": _finite_mean(
                    [row["q_terminal_minus_il"] for row in group]
                ),
                "hold_over_il_fraction": _fraction(
                    group, "critic_prefers_hold_over_il"
                ),
                "hold_over_demo_fraction": _fraction(
                    group, "critic_prefers_hold_over_demo"
                ),
                "terminal_over_il_fraction": _fraction(
                    group, "critic_prefers_terminal_over_il"
                ),
                "il_to_hold_rmse_mean": _finite_mean(
                    [row["il_to_hold_rmse"] for row in group]
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted terminal-tail aliasing diagnostic. Tests whether the sparse +1 "
            "reward is attached to an H-step chunk dominated by post-release hold, and "
            "whether the trained IQL Critic prefers hold/terminal-like chunks over Base "
            "ACT or demonstrated continuation around partial and true completion events."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--integrity-dir", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/analysis_12_terminal_tail_aliasing",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=list(DEFAULT_OFFSETS),
        help="Frame offsets around first_release, last_release, and terminal_chunk_start.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    integrity_path = (
        pathlib.Path(args.integrity_dir).expanduser().resolve()
        / "episode_integrity.csv"
    )
    if not integrity_path.is_file():
        raise FileNotFoundError(integrity_path)
    integrity_rows = _read_integrity_rows(integrity_path)

    (
        workspace,
        cfg,
        stage1_dir,
        checkpoint,
        dataset,
        latent_cache_dir,
        config_path,
    ) = _build_workspace(args)

    chunk_size = int(cfg.n_action_steps)
    critic_stride = int(cfg.critic.sequence_stride)
    episode_starts, episode_ends = _episode_bounds(workspace.buffer.episode_ends)
    geometry_rows = _terminal_geometry_rows(
        integrity_rows=integrity_rows,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        chunk_size=chunk_size,
        critic_stride=critic_stride,
    )
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    _write_csv(output_dir / "terminal_tail_geometry.csv", geometry_rows)

    probe_specs = []
    for buffer_ep, (integrity, start, end) in enumerate(
        zip(integrity_rows, episode_starts, episode_ends, strict=True)
    ):
        left_release = _parse_int(integrity, "left_reopen_frame_offset")
        right_release = _parse_int(integrity, "right_reopen_frame_offset")
        first_release = min(left_release, right_release)
        last_release = max(left_release, right_release)
        length = int(end - start)
        terminal_anchor = length - chunk_size
        events = {
            "last_release_true": last_release,
            "rewarded_terminal_chunk_start": terminal_anchor,
        }
        if first_release < last_release:
            events["first_release_partial"] = first_release
        for event_name, event_local in events.items():
            for rel in args.offsets:
                local = int(event_local + rel)
                if local < 0 or local >= length:
                    continue
                probe_specs.append(
                    (
                        int(start + local),
                        int(buffer_ep),
                        int(start),
                        int(end),
                        event_name,
                        int(rel),
                    )
                )

    probe_rows = []
    for offset in range(0, len(probe_specs), args.batch_size):
        batch = probe_specs[offset:offset + args.batch_size]
        anchors = np.asarray([item[0] for item in batch], dtype=np.int64)
        episode_ids = np.asarray([item[1] for item in batch], dtype=np.int64)
        starts = np.asarray([item[2] for item in batch], dtype=np.int64)
        ends = np.asarray([item[3] for item in batch], dtype=np.int64)
        event_names = [item[4] for item in batch]
        rel_offsets = np.asarray([item[5] for item in batch], dtype=np.int64)
        probe_rows.extend(
            _evaluate_probe_batch(
                workspace,
                anchors,
                episode_ids,
                starts,
                ends,
                event_names,
                rel_offsets,
                chunk_size,
            )
        )

    _write_csv(output_dir / "critic_event_probe.csv", probe_rows)
    aggregate_rows = _aggregate_probe_rows(probe_rows)
    _write_csv(output_dir / "critic_event_summary.csv", aggregate_rows)

    tail_values = [row["tail_after_last_release"] for row in geometry_rows]
    hold_fractions = [row["hold_fraction_inside_terminal_chunk"] for row in geometry_rows]
    terminal_relative = [
        row["terminal_anchor_relative_to_last_release"] for row in geometry_rows
    ]
    sampled = [
        row["terminal_window_sampled_by_critic"] for row in geometry_rows
    ]

    partial_zero = [
        row
        for row in probe_rows
        if row["event"] == "first_release_partial"
        and row["relative_offset"] == 0
    ]
    last_zero = [
        row
        for row in probe_rows
        if row["event"] == "last_release_true"
        and row["relative_offset"] == 0
    ]
    rewarded_zero = [
        row
        for row in probe_rows
        if row["event"] == "rewarded_terminal_chunk_start"
        and row["relative_offset"] == 0
    ]

    summary = {
        "analysis": "12_terminal_tail_aliasing",
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "latent_cache_dir": str(latent_cache_dir),
        "integrity_csv": str(integrity_path),
        "episodes": int(len(geometry_rows)),
        "chunk_size": int(chunk_size),
        "critic_stride": int(critic_stride),
        "gamma": float(cfg.critic.gamma),
        "terminal_reward_discount_weight_within_final_chunk": float(
            float(cfg.critic.gamma) ** max(chunk_size - 1, 0)
        ),
        "terminal_tail_geometry": {
            "tail_after_last_release_frames": {
                "min": int(np.min(tail_values)),
                "mean": float(np.mean(tail_values)),
                "median": float(np.median(tail_values)),
                "max": int(np.max(tail_values)),
            },
            "terminal_chunk_anchor_relative_to_last_release": {
                "min": int(np.min(terminal_relative)),
                "mean": float(np.mean(terminal_relative)),
                "median": float(np.median(terminal_relative)),
                "max": int(np.max(terminal_relative)),
            },
            "hold_fraction_inside_terminal_chunk": {
                "mean": float(np.mean(hold_fractions)),
                "median": float(np.median(hold_fractions)),
            },
            "terminal_window_sampled_by_critic_fraction": float(np.mean(sampled)),
        },
        "partial_completion_event": {
            "count": int(len(partial_zero)),
            "q_hold_minus_il_mean": _finite_mean(
                [row["q_hold_minus_il"] for row in partial_zero]
            ),
            "q_hold_minus_demo_mean": _finite_mean(
                [row["q_hold_minus_demo"] for row in partial_zero]
            ),
            "hold_over_il_fraction": _fraction(
                partial_zero, "critic_prefers_hold_over_il"
            ),
            "hold_over_demo_fraction": _fraction(
                partial_zero, "critic_prefers_hold_over_demo"
            ),
            "terminal_over_il_fraction": _fraction(
                partial_zero, "critic_prefers_terminal_over_il"
            ),
        },
        "true_completion_event": {
            "count": int(len(last_zero)),
            "q_hold_minus_il_mean": _finite_mean(
                [row["q_hold_minus_il"] for row in last_zero]
            ),
            "hold_over_il_fraction": _fraction(
                last_zero, "critic_prefers_hold_over_il"
            ),
            "terminal_over_il_fraction": _fraction(
                last_zero, "critic_prefers_terminal_over_il"
            ),
        },
        "rewarded_terminal_chunk_start": {
            "count": int(len(rewarded_zero)),
            "q_hold_minus_il_mean": _finite_mean(
                [row["q_hold_minus_il"] for row in rewarded_zero]
            ),
            "q_terminal_minus_il_mean": _finite_mean(
                [row["q_terminal_minus_il"] for row in rewarded_zero]
            ),
            "terminal_over_il_fraction": _fraction(
                rewarded_zero, "critic_prefers_terminal_over_il"
            ),
        },
        "interpretation": {
            "supports_terminal_tail_aliasing_if": (
                "The final rewarded H-step chunk is dominated by post-release hold and "
                "the Critic increasingly ranks hold/terminal-like chunks above Base ACT "
                "or demonstrated continuation at the first-release partial-completion event."
            ),
            "weakens_hypothesis_if": (
                "At first release the Critic consistently ranks Base/demo continuation "
                "above hold and terminal templates even though the terminal chunk is hold-heavy."
            ),
            "caveat": (
                "Q ranking is diagnostic. It does not by itself prove PPO gradients caused "
                "the deployed freeze; Analysis 10/11 address that later step."
            ),
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print("\n=== Analysis 12: terminal-tail aliasing ===")
    geom = summary["terminal_tail_geometry"]
    print(
        "Tail after last release frames min/mean/median/max: "
        f"{geom['tail_after_last_release_frames']['min']} / "
        f"{geom['tail_after_last_release_frames']['mean']:.2f} / "
        f"{geom['tail_after_last_release_frames']['median']:.2f} / "
        f"{geom['tail_after_last_release_frames']['max']}"
    )
    print(
        "Rewarded terminal-chunk start relative to last release "
        "(negative = starts before release): "
        f"{geom['terminal_chunk_anchor_relative_to_last_release']['mean']:.2f} frames mean"
    )
    print(
        "Mean hold fraction inside final H-step chunk: "
        f"{100.0 * geom['hold_fraction_inside_terminal_chunk']['mean']:.1f}%"
    )
    print(
        "Final H-step window actually sampled by Critic: "
        f"{100.0 * geom['terminal_window_sampled_by_critic_fraction']:.1f}% episodes"
    )
    partial = summary["partial_completion_event"]
    print(
        "At first-release partial completion: "
        f"Qhold-QIL={partial['q_hold_minus_il_mean']}, "
        f"hold>IL fraction={partial['hold_over_il_fraction']}, "
        f"hold>demo fraction={partial['hold_over_demo_fraction']}"
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
