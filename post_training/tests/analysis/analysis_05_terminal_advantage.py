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
from safetensors import safe_open
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.policy.stochastic_act_config import StochasticACTConfigWrapper  # noqa: E402
from post_rl.policy.stochastic_act_policy import StochasticACTPolicyWrapper  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


QVA_METRICS = (
    "dataset_return",
    "chunk_reward",
    "v",
    "q_demo",
    "q_il",
    "q_postrl",
    "q_hold",
    "q_terminal_template",
    "q_terminal_shuffled",
    "q_terminal_on_other_state",
    "v_other_state",
    "adv_demo",
    "adv_il",
    "adv_postrl",
    "adv_hold",
    "adv_terminal_template",
    "adv_terminal_shuffled",
    "adv_terminal_on_other_state",
    "q_postrl_minus_il",
    "q_hold_minus_il",
    "q_terminal_template_minus_il",
    "q_terminal_shuffled_minus_il",
    "q_terminal_same_minus_shuffled",
    "postrl_il_action_l2",
    "il_terminal_rmse",
    "postrl_terminal_rmse",
    "postrl_terminal_delta_rmse",
    "demo_motion_l2",
    "il_motion_l2",
    "postrl_motion_l2",
    "terminal_template_motion_l2",
    "q_hybrid_terminal_left_il_right",
    "q_hybrid_il_left_terminal_right",
    "q_hybrid_terminal_left_joints",
    "q_hybrid_terminal_left_gripper",
    "q_hybrid_terminal_right_joints",
    "q_hybrid_terminal_right_gripper",
    "adv_hybrid_terminal_left_il_right",
    "adv_hybrid_il_left_terminal_right",
    "adv_hybrid_terminal_left_joints",
    "adv_hybrid_terminal_left_gripper",
    "adv_hybrid_terminal_right_joints",
    "adv_hybrid_terminal_right_gripper",
    "q_hybrid_terminal_left_il_right_minus_il",
    "q_hybrid_il_left_terminal_right_minus_il",
    "q_hybrid_terminal_left_joints_minus_il",
    "q_hybrid_terminal_left_gripper_minus_il",
    "q_hybrid_terminal_right_joints_minus_il",
    "q_hybrid_terminal_right_gripper_minus_il",
)


def _resolve_config(stage1_dir: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit is not None:
        path = pathlib.Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = (
        stage1_dir / "provenance" / "dynamics_config.yaml",
        stage1_dir / "config.yaml",
        stage1_dir.parent / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not resolve the training config. Checked:\n- "
        + "\n- ".join(str(path) for path in candidates)
        + "\nPass --config explicitly."
    )


def _resolve_required_path(value, fallback, name: str) -> pathlib.Path:
    raw = value if value is not None else fallback
    if raw is None:
        raise ValueError(f"{name} must be supplied by CLI or config")
    path = pathlib.Path(str(raw)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _resolve_processor_dir(checkpoint: pathlib.Path) -> pathlib.Path:
    for candidate in (checkpoint, checkpoint.parent):
        if (
            (candidate / "policy_preprocessor.json").is_file()
            and (candidate / "policy_postprocessor.json").is_file()
        ):
            return candidate
    raise FileNotFoundError(f"Could not resolve ACT processor bundle for {checkpoint}")


def _processor_fingerprint(directory: pathlib.Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    files = sorted(
        [
            *directory.glob("policy_preprocessor*"),
            *directory.glob("policy_postprocessor*"),
        ],
        key=lambda path: path.name,
    )
    for path in files:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_is_stochastic(checkpoint: pathlib.Path) -> bool:
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        return "raw_log_std" in handle.keys()


def _load_postrl_policy(
    checkpoint: pathlib.Path,
    device: torch.device,
    base_cfg,
) -> tuple[StochasticACTPolicyWrapper, str]:
    if _checkpoint_is_stochastic(checkpoint):
        policy_cfg = StochasticACTConfigWrapper.from_pretrained(str(checkpoint))
        policy_cfg.device = str(device)
        policy = StochasticACTPolicyWrapper.from_pretrained(
            str(checkpoint),
            config=policy_cfg,
        )
        kind = "stochastic_postrl"
    else:
        policy_cfg = StochasticACTConfigWrapper.from_il_pretrained(
            str(checkpoint),
            init_log_std=float(base_cfg.policy.init_log_std),
            log_std_min=float(base_cfg.policy.log_std_min),
            log_std_max=float(base_cfg.policy.log_std_max),
        )
        policy_cfg.device = str(device)
        policy = StochasticACTPolicyWrapper.from_il_pretrained(
            str(checkpoint),
            config=policy_cfg,
        )
        kind = "deterministic_export_wrapped_for_mean"
    return policy.to(device).eval(), kind


def _episode_bounds(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1])).astype(np.int64)
    return starts, ends


def _window_starts(length: int, chunk_size: int, stride: int) -> np.ndarray:
    if length < chunk_size:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, length - chunk_size + 1, stride, dtype=np.int64)


def _coverage_rows(
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    chunk_size: int,
    strides: dict[str, int],
) -> list[dict]:
    rows = []
    for episode_id, (start, end) in enumerate(
        zip(episode_starts, episode_ends, strict=True)
    ):
        length = int(end - start)
        terminal_anchor = length - chunk_size
        for name, stride in strides.items():
            local_starts = _window_starts(length, chunk_size, int(stride))
            eligible = terminal_anchor >= 0
            covered = bool(
                eligible and np.any(local_starts == terminal_anchor)
            )
            last_local = int(local_starts[-1]) if len(local_starts) else -1
            rows.append(
                {
                    "episode": int(episode_id),
                    "episode_length": length,
                    "chunk_size": int(chunk_size),
                    "sampler": name,
                    "stride": int(stride),
                    "num_complete_windows": int(len(local_starts)),
                    "terminal_window_eligible": int(eligible),
                    "terminal_window_covered": int(covered),
                    "terminal_anchor_local": int(terminal_anchor),
                    "last_sampled_anchor_local": last_local,
                    "terminal_anchor_gap": (
                        int(terminal_anchor - last_local)
                        if eligible and last_local >= 0
                        else -1
                    ),
                }
            )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _coverage_summary(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sampler"]].append(row)
    summary = {}
    for name, group in grouped.items():
        eligible = [row for row in group if row["terminal_window_eligible"]]
        covered = sum(row["terminal_window_covered"] for row in eligible)
        gaps = [
            row["terminal_anchor_gap"]
            for row in eligible
            if row["terminal_anchor_gap"] >= 0
        ]
        summary[name] = {
            "stride": int(group[0]["stride"]),
            "eligible_episodes": len(eligible),
            "covered_episodes": int(covered),
            "coverage_fraction": (
                float(covered / len(eligible)) if eligible else float("nan")
            ),
            "mean_terminal_anchor_gap": (
                float(np.mean(gaps)) if gaps else float("nan")
            ),
            "max_terminal_anchor_gap": int(max(gaps)) if gaps else -1,
        }
    return summary


def _phase(frame_progress: float, terminal_in_chunk: bool) -> str:
    if terminal_in_chunk:
        return "terminal_chunk"
    if frame_progress < 0.25:
        return "early_0_25"
    if frame_progress < 0.50:
        return "middle_25_50"
    if frame_progress < 0.75:
        return "late_50_75"
    return "tail_75_100"


def _motion_l2(action: torch.Tensor) -> torch.Tensor:
    if action.ndim != 3:
        raise ValueError(f"Expected action chunk [B,H,D], got {tuple(action.shape)}")
    if action.shape[1] <= 1:
        return torch.zeros(action.shape[0], device=action.device)
    delta = action[:, 1:] - action[:, :-1]
    return torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)


def _chunk_rmse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            f"Chunk shapes must match, got {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    return torch.sqrt((left - right).pow(2).mean(dim=(1, 2)))


def _q_components_state(
    critic,
    state: torch.Tensor,
    action: torch.Tensor,
):
    prepared_action = critic._prepare_action(action)
    if critic._is_double_q:
        q1, q2 = critic._Q(state, prepared_action)
        q = torch.minimum(q1, q2)
        gap = (q1 - q2).abs()
    else:
        q = critic._Q(state, prepared_action)
        q1 = q
        q2 = q
        gap = torch.zeros_like(q)
    return q.reshape(-1), q1.reshape(-1), q2.reshape(-1), gap.reshape(-1)


def _q_components(critic, latent: torch.Tensor, action: torch.Tensor):
    state = latent.float().mean(dim=1)
    return _q_components_state(critic, state, action)


def _build_bimanual_hybrids(
    il: torch.Tensor,
    terminal: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if il.shape != terminal.shape:
        raise ValueError(
            f"IL/terminal chunks must match, got {tuple(il.shape)} vs {tuple(terminal.shape)}"
        )
    if il.shape[-1] != 16:
        return {}

    hybrids = {}
    left = il.clone()
    left[:, :, 0:8] = terminal[:, :, 0:8]
    hybrids["hybrid_terminal_left_il_right"] = left

    right = il.clone()
    right[:, :, 8:16] = terminal[:, :, 8:16]
    hybrids["hybrid_il_left_terminal_right"] = right

    left_joints = il.clone()
    left_joints[:, :, 0:7] = terminal[:, :, 0:7]
    hybrids["hybrid_terminal_left_joints"] = left_joints

    left_gripper = il.clone()
    left_gripper[:, :, 7:8] = terminal[:, :, 7:8]
    hybrids["hybrid_terminal_left_gripper"] = left_gripper

    right_joints = il.clone()
    right_joints[:, :, 8:15] = terminal[:, :, 8:15]
    hybrids["hybrid_terminal_right_joints"] = right_joints

    right_gripper = il.clone()
    right_gripper[:, :, 15:16] = terminal[:, :, 15:16]
    hybrids["hybrid_terminal_right_gripper"] = right_gripper
    return hybrids


def _accumulate(
    totals: dict[str, dict[str, float]],
    group: str,
    row: dict,
) -> None:
    acc = totals[group]
    acc["count"] += 1.0
    for key in QVA_METRICS:
        value = float(row[key])
        if np.isfinite(value):
            acc[f"{key}_sum"] += value
            acc[f"{key}_count"] += 1.0


def _mean_rows(totals: dict[str, dict[str, float]], label: str) -> list[dict]:
    rows = []
    for group in sorted(totals):
        acc = totals[group]
        row = {label: group, "count": int(acc["count"])}
        for key in QVA_METRICS:
            count = acc.get(f"{key}_count", 0.0)
            row[key] = (
                acc.get(f"{key}_sum", 0.0) / count
                if count > 0
                else float("nan")
            )
        rows.append(row)
    return rows


def _select_anchors(
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    chunk_size: int,
    max_anchors: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchors = []
    episodes = []
    starts = []
    ends = []
    for episode_id, (start, end) in enumerate(
        zip(episode_starts, episode_ends, strict=True)
    ):
        latest = int(end) - int(chunk_size)
        if latest < int(start):
            continue
        current = np.arange(int(start), latest + 1, dtype=np.int64)
        anchors.append(current)
        episodes.append(np.full(len(current), episode_id, dtype=np.int64))
        starts.append(np.full(len(current), start, dtype=np.int64))
        ends.append(np.full(len(current), end, dtype=np.int64))
    if not anchors:
        raise RuntimeError("No complete chunk anchors exist in the offline dataset")
    anchors = np.concatenate(anchors)
    episodes = np.concatenate(episodes)
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    if max_anchors is not None and len(anchors) > max_anchors:
        if max_anchors < 1:
            raise ValueError("--max-anchors must be >= 1")
        positions = np.linspace(0, len(anchors) - 1, max_anchors, dtype=np.int64)
        anchors = anchors[positions]
        episodes = episodes[positions]
        starts = starts[positions]
        ends = ends[positions]
    return anchors, episodes, starts, ends


def _episode_motion_rows(workspace, starts, ends, chunk_size: int) -> list[dict]:
    rows = []
    for episode_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        raw = np.asarray(
            workspace.buffer["action"][int(start):int(end)],
            dtype=np.float32,
        )
        if len(raw) <= 1:
            continue
        raw_tensor = torch.from_numpy(raw).to(workspace.device)
        with torch.no_grad():
            normalized = workspace.obs_adapter.normalize_action(raw_tensor)
        normalized = normalized.detach().float().cpu().numpy()
        raw_delta = np.linalg.norm(np.diff(raw, axis=0), axis=-1)
        norm_delta = np.linalg.norm(np.diff(normalized, axis=0), axis=-1)

        length = len(raw)
        delta_frame = np.arange(1, length)
        first_half = delta_frame < length * 0.5
        second_half = ~first_half
        last_h = delta_frame >= max(1, length - chunk_size)
        middle = (delta_frame >= length * 0.25) & (delta_frame < length * 0.75)

        def mean(values, mask):
            selected = values[mask]
            return float(selected.mean()) if selected.size else float("nan")

        last_h_mean = mean(norm_delta, last_h)
        middle_mean = mean(norm_delta, middle)
        rows.append(
            {
                "episode": int(episode_id),
                "episode_length": int(length),
                "raw_motion_all": float(raw_delta.mean()),
                "raw_motion_first_half": mean(raw_delta, first_half),
                "raw_motion_second_half": mean(raw_delta, second_half),
                "raw_motion_middle_25_75": mean(raw_delta, middle),
                "raw_motion_last_h": mean(raw_delta, last_h),
                "normalized_motion_all": float(norm_delta.mean()),
                "normalized_motion_first_half": mean(norm_delta, first_half),
                "normalized_motion_second_half": mean(norm_delta, second_half),
                "normalized_motion_middle_25_75": middle_mean,
                "normalized_motion_last_h": last_h_mean,
                "normalized_last_h_over_middle": (
                    last_h_mean / max(middle_mean, 1e-12)
                    if np.isfinite(last_h_mean) and np.isfinite(middle_mean)
                    else float("nan")
                ),
            }
        )
    return rows


@torch.no_grad()
def _evaluate_qva(
    workspace,
    postrl_policy,
    anchors: np.ndarray,
    episode_ids: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    *,
    chunk_size: int,
    batch_size: int,
    output_path: pathlib.Path,
    progress_bins: int,
):
    gamma = float(workspace.cfg.critic.gamma)
    discount = torch.pow(
        torch.tensor(gamma, device=workspace.device, dtype=torch.float32),
        torch.arange(chunk_size, device=workspace.device, dtype=torch.float32),
    )
    phase_totals = defaultdict(lambda: defaultdict(float))
    bin_totals = defaultdict(lambda: defaultdict(float))
    hypothesis = defaultdict(int)
    total_rows = 0

    all_episode_starts, all_episode_ends = _episode_bounds(
        workspace.buffer.episode_ends
    )
    if len(all_episode_ends) < 2:
        raise RuntimeError(
            "Terminal action shuffle requires at least two episodes."
        )
    episode_lengths = all_episode_ends - all_episode_starts
    if np.any(episode_lengths < chunk_size):
        short_ids = np.flatnonzero(episode_lengths < chunk_size).tolist()
        raise RuntimeError(
            "Terminal template requires every analyzed episode to contain one full "
            f"chunk; short episode ids={short_ids[:10]}"
        )
    terminal_bank_raw = np.stack(
        [
            np.asarray(
                workspace.buffer["action"][int(end) - chunk_size:int(end)],
                dtype=np.float32,
            )
            for end in all_episode_ends
        ],
        axis=0,
    )

    fieldnames = [
        "episode",
        "anchor",
        "local_anchor",
        "episode_length",
        "frame_progress",
        "chunk_progress",
        "frames_to_terminal",
        "terminal_in_chunk",
        "second_half",
        "phase",
        *QVA_METRICS,
        "q_demo_gap",
        "q_il_gap",
        "q_postrl_gap",
        "q_hold_gap",
        "q_terminal_template_gap",
    ]

    with output_path.open("w", newline="", buffering=1) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for offset in tqdm(
            range(0, len(anchors), batch_size),
            desc="Terminal / Advantage QVA",
        ):
            batch_anchors = anchors[offset:offset + batch_size]
            batch_episode_ids = episode_ids[offset:offset + batch_size]
            batch_starts = episode_starts[offset:offset + batch_size]
            batch_ends = episode_ends[offset:offset + batch_size]
            batch_n = len(batch_anchors)

            latent_np = np.asarray(
                workspace.latent_cache.obs[batch_anchors],
                dtype=np.float32,
            )
            latent = torch.from_numpy(latent_np).to(workspace.device)

            action_indices = (
                batch_anchors[:, None]
                + np.arange(chunk_size, dtype=np.int64)[None, :]
            )
            demo_raw_np = np.asarray(
                workspace.buffer["action"][action_indices],
                dtype=np.float32,
            )
            reward_np = np.asarray(
                workspace.buffer["reward"][action_indices],
                dtype=np.float32,
            ).reshape(batch_n, chunk_size)
            done_np = np.asarray(
                workspace.buffer["done"][action_indices],
                dtype=bool,
            ).reshape(batch_n, chunk_size)
            dataset_return = np.asarray(
                workspace.buffer["return"][batch_anchors],
                dtype=np.float32,
            ).reshape(batch_n)

            terminal_template_np = terminal_bank_raw[batch_episode_ids]
            other_episode_ids = (
                batch_episode_ids + 1
            ) % len(all_episode_ends)
            terminal_shuffled_np = terminal_bank_raw[other_episode_ids]

            local_anchor_np = batch_anchors - batch_starts
            valid_span_np = np.maximum(
                batch_ends - batch_starts - chunk_size,
                0,
            )
            progress_np = local_anchor_np / np.maximum(valid_span_np, 1)
            other_starts = all_episode_starts[other_episode_ids]
            other_ends = all_episode_ends[other_episode_ids]
            other_spans = np.maximum(
                other_ends - other_starts - chunk_size,
                0,
            )
            other_anchors = other_starts + np.rint(
                progress_np * other_spans
            ).astype(np.int64)
            other_latent_np = np.asarray(
                workspace.latent_cache.obs[other_anchors],
                dtype=np.float32,
            )
            other_state = torch.from_numpy(
                other_latent_np.mean(axis=1)
            ).to(workspace.device)

            demo_raw = torch.from_numpy(demo_raw_np).to(workspace.device)
            terminal_template_raw = torch.from_numpy(
                terminal_template_np
            ).to(workspace.device)
            terminal_shuffled_raw = torch.from_numpy(
                terminal_shuffled_np
            ).to(workspace.device)
            demo = workspace.obs_adapter.normalize_action(demo_raw)
            terminal_template = workspace.obs_adapter.normalize_action(
                terminal_template_raw
            )
            terminal_shuffled = workspace.obs_adapter.normalize_action(
                terminal_shuffled_raw
            )
            hold = demo[:, :1].expand(-1, chunk_size, -1).contiguous()

            il = workspace.model.get_action_mean({"latent": latent})
            postrl = postrl_policy.get_action_mean({"latent": latent})
            value = workspace.critic._value(latent.mean(dim=1)).reshape(-1)
            value_other = workspace.critic._value(other_state).reshape(-1)

            q_demo, _, _, gap_demo = _q_components(workspace.critic, latent, demo)
            q_il, _, _, gap_il = _q_components(workspace.critic, latent, il)
            q_postrl, _, _, gap_postrl = _q_components(
                workspace.critic,
                latent,
                postrl,
            )
            q_hold, _, _, gap_hold = _q_components(workspace.critic, latent, hold)
            q_terminal, _, _, gap_terminal = _q_components(
                workspace.critic,
                latent,
                terminal_template,
            )
            q_terminal_shuffled, _, _, _ = _q_components(
                workspace.critic,
                latent,
                terminal_shuffled,
            )
            q_terminal_other_state, _, _, _ = _q_components_state(
                workspace.critic,
                other_state,
                terminal_template,
            )

            hybrid_actions = _build_bimanual_hybrids(
                il,
                terminal_template,
            )
            hybrid_q = {}
            for name, action in hybrid_actions.items():
                q, _, _, _ = _q_components(
                    workspace.critic,
                    latent,
                    action,
                )
                hybrid_q[name] = q

            chunk_reward = (
                torch.from_numpy(reward_np).to(workspace.device) * discount
            ).sum(dim=1)
            demo_motion = _motion_l2(demo)
            il_motion = _motion_l2(il)
            postrl_motion = _motion_l2(postrl)
            terminal_motion = _motion_l2(terminal_template)
            postrl_il_action_l2 = torch.linalg.vector_norm(
                (postrl - il).reshape(batch_n, -1),
                dim=1,
            )
            il_terminal_rmse = _chunk_rmse(il, terminal_template)
            postrl_terminal_rmse = _chunk_rmse(
                postrl,
                terminal_template,
            )
            postrl_terminal_delta_rmse = (
                postrl_terminal_rmse - il_terminal_rmse
            )

            nan_vector = torch.full_like(q_il, float("nan"))
            tensors = {
                "v": value,
                "q_demo": q_demo,
                "q_il": q_il,
                "q_postrl": q_postrl,
                "q_hold": q_hold,
                "q_terminal_template": q_terminal,
                "q_terminal_shuffled": q_terminal_shuffled,
                "q_terminal_on_other_state": q_terminal_other_state,
                "v_other_state": value_other,
                "chunk_reward": chunk_reward,
                "postrl_il_action_l2": postrl_il_action_l2,
                "il_terminal_rmse": il_terminal_rmse,
                "postrl_terminal_rmse": postrl_terminal_rmse,
                "postrl_terminal_delta_rmse": postrl_terminal_delta_rmse,
                "demo_motion_l2": demo_motion,
                "il_motion_l2": il_motion,
                "postrl_motion_l2": postrl_motion,
                "terminal_template_motion_l2": terminal_motion,
                "q_demo_gap": gap_demo,
                "q_il_gap": gap_il,
                "q_postrl_gap": gap_postrl,
                "q_hold_gap": gap_hold,
                "q_terminal_template_gap": gap_terminal,
            }
            for name in (
                "hybrid_terminal_left_il_right",
                "hybrid_il_left_terminal_right",
                "hybrid_terminal_left_joints",
                "hybrid_terminal_left_gripper",
                "hybrid_terminal_right_joints",
                "hybrid_terminal_right_gripper",
            ):
                tensors[f"q_{name}"] = hybrid_q.get(name, nan_vector)

            values = {
                key: tensor.detach().float().cpu().numpy().reshape(-1)
                for key, tensor in tensors.items()
            }

            for local in range(batch_n):
                start = int(batch_starts[local])
                end = int(batch_ends[local])
                anchor = int(batch_anchors[local])
                length = end - start
                local_anchor = anchor - start
                frame_progress = local_anchor / max(length - 1, 1)
                terminal_anchor = max(length - chunk_size, 0)
                chunk_progress = local_anchor / max(terminal_anchor, 1)
                terminal_in_chunk = bool(done_np[local].any())
                phase = _phase(frame_progress, terminal_in_chunk)
                second_half = frame_progress >= 0.5

                row = {
                    "episode": int(batch_episode_ids[local]),
                    "anchor": anchor,
                    "local_anchor": local_anchor,
                    "episode_length": length,
                    "frame_progress": frame_progress,
                    "chunk_progress": chunk_progress,
                    "frames_to_terminal": int(end - 1 - anchor),
                    "terminal_in_chunk": int(terminal_in_chunk),
                    "second_half": int(second_half),
                    "phase": phase,
                    "dataset_return": float(dataset_return[local]),
                    **{key: float(array[local]) for key, array in values.items()},
                }
                row["adv_demo"] = row["q_demo"] - row["v"]
                row["adv_il"] = row["q_il"] - row["v"]
                row["adv_postrl"] = row["q_postrl"] - row["v"]
                row["adv_hold"] = row["q_hold"] - row["v"]
                row["adv_terminal_template"] = (
                    row["q_terminal_template"] - row["v"]
                )
                row["adv_terminal_shuffled"] = (
                    row["q_terminal_shuffled"] - row["v"]
                )
                row["adv_terminal_on_other_state"] = (
                    row["q_terminal_on_other_state"] - row["v_other_state"]
                )
                row["q_postrl_minus_il"] = row["q_postrl"] - row["q_il"]
                row["q_hold_minus_il"] = row["q_hold"] - row["q_il"]
                row["q_terminal_template_minus_il"] = (
                    row["q_terminal_template"] - row["q_il"]
                )
                row["q_terminal_shuffled_minus_il"] = (
                    row["q_terminal_shuffled"] - row["q_il"]
                )
                row["q_terminal_same_minus_shuffled"] = (
                    row["q_terminal_template"] - row["q_terminal_shuffled"]
                )

                for hybrid_name in (
                    "hybrid_terminal_left_il_right",
                    "hybrid_il_left_terminal_right",
                    "hybrid_terminal_left_joints",
                    "hybrid_terminal_left_gripper",
                    "hybrid_terminal_right_joints",
                    "hybrid_terminal_right_gripper",
                ):
                    q_key = f"q_{hybrid_name}"
                    adv_key = f"adv_{hybrid_name}"
                    delta_key = f"q_{hybrid_name}_minus_il"
                    row[adv_key] = row[q_key] - row["v"]
                    row[delta_key] = row[q_key] - row["q_il"]

                writer.writerow(row)
                _accumulate(phase_totals, phase, row)
                bin_id = min(
                    int(frame_progress * progress_bins),
                    progress_bins - 1,
                )
                _accumulate(bin_totals, f"{bin_id:02d}", row)
                total_rows += 1

                if second_half:
                    hypothesis["second_half_count"] += 1
                    hypothesis["second_half_postrl_q_gt_il"] += int(
                        row["q_postrl"] > row["q_il"]
                    )
                    hypothesis["second_half_hold_q_gt_il"] += int(
                        row["q_hold"] > row["q_il"]
                    )
                    hypothesis["second_half_terminal_q_gt_il"] += int(
                        row["q_terminal_template"] > row["q_il"]
                    )
                    hypothesis["second_half_postrl_adv_positive"] += int(
                        row["adv_postrl"] > 0
                    )
                    hypothesis["second_half_hold_adv_positive"] += int(
                        row["adv_hold"] > 0
                    )
                    hypothesis["second_half_terminal_adv_positive"] += int(
                        row["adv_terminal_template"] > 0
                    )
                    hypothesis["second_half_terminal_shuffled_q_gt_il"] += int(
                        row["q_terminal_shuffled"] > row["q_il"]
                    )
                    hypothesis["second_half_terminal_shuffled_adv_positive"] += int(
                        row["adv_terminal_shuffled"] > 0
                    )
                    hypothesis["second_half_terminal_same_q_gt_shuffled"] += int(
                        row["q_terminal_template"] > row["q_terminal_shuffled"]
                    )
                    hypothesis["second_half_terminal_other_state_adv_positive"] += int(
                        row["adv_terminal_on_other_state"] > 0
                    )
                    hypothesis["second_half_postrl_closer_terminal"] += int(
                        row["postrl_terminal_delta_rmse"] < 0
                    )

                    for hybrid_name in (
                        "hybrid_terminal_left_il_right",
                        "hybrid_il_left_terminal_right",
                        "hybrid_terminal_left_joints",
                        "hybrid_terminal_left_gripper",
                        "hybrid_terminal_right_joints",
                        "hybrid_terminal_right_gripper",
                    ):
                        q_value = row[f"q_{hybrid_name}"]
                        if np.isfinite(q_value):
                            hypothesis[f"second_half_{hybrid_name}_q_gt_il"] += int(
                                q_value > row["q_il"]
                            )

    return (
        _mean_rows(phase_totals, "phase"),
        _mean_rows(bin_totals, "progress_bin"),
        dict(hypothesis),
        total_rows,
    )


def _fractions(counts: dict) -> dict:
    denom = max(int(counts.get("second_half_count", 0)), 1)
    result = dict(counts)
    for key, value in list(counts.items()):
        if key.startswith("second_half_") and key != "second_half_count":
            result[f"{key}_fraction"] = float(value / denom)
    return result


def _plot_progress(path: pathlib.Path, progress_rows: list[dict]) -> None:
    if not progress_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = (np.arange(len(progress_rows), dtype=np.float64) + 0.5) / len(progress_rows)

    plt.figure(figsize=(10, 5))
    for key in (
        "dataset_return",
        "v",
        "q_demo",
        "q_il",
        "q_postrl",
        "q_hold",
        "q_terminal_template",
    ):
        plt.plot(x, [row[key] for row in progress_rows], label=key)
    plt.xlabel("episode frame progress")
    plt.ylabel("value / Q")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path / "q_value_vs_progress.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for key in (
        "adv_demo",
        "adv_il",
        "adv_postrl",
        "adv_hold",
        "adv_terminal_template",
    ):
        plt.plot(x, [row[key] for row in progress_rows], label=key)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("episode frame progress")
    plt.ylabel("advantage = Q - V")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path / "advantage_vs_progress.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for key in (
        "demo_motion_l2",
        "il_motion_l2",
        "postrl_motion_l2",
        "terminal_template_motion_l2",
    ):
        plt.plot(x, [row[key] for row in progress_rows], label=key)
    plt.xlabel("episode frame progress")
    plt.ylabel("mean normalized action step delta L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "action_motion_vs_progress.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for key in (
        "q_terminal_template_minus_il",
        "q_terminal_shuffled_minus_il",
        "q_postrl_minus_il",
    ):
        plt.plot(x, [row[key] for row in progress_rows], label=key)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("episode frame progress")
    plt.ylabel("Q difference relative to IL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "terminal_shuffle_q_vs_progress.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        x,
        [row["il_terminal_rmse"] for row in progress_rows],
        label="IL -> terminal RMSE",
    )
    plt.plot(
        x,
        [row["postrl_terminal_rmse"] for row in progress_rows],
        label="Post-RL -> terminal RMSE",
    )
    plt.plot(
        x,
        [row["postrl_terminal_delta_rmse"] for row in progress_rows],
        label="Post-RL minus IL terminal RMSE",
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("episode frame progress")
    plt.ylabel("normalized action chunk RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "terminal_manifold_distance_vs_progress.png", dpi=160)
    plt.close()

    hybrid_keys = (
        "q_hybrid_terminal_left_il_right_minus_il",
        "q_hybrid_il_left_terminal_right_minus_il",
        "q_hybrid_terminal_left_joints_minus_il",
        "q_hybrid_terminal_left_gripper_minus_il",
        "q_hybrid_terminal_right_joints_minus_il",
        "q_hybrid_terminal_right_gripper_minus_il",
    )
    if any(
        np.isfinite(row[key])
        for row in progress_rows
        for key in hybrid_keys
    ):
        plt.figure(figsize=(10, 5))
        for key in hybrid_keys:
            plt.plot(x, [row[key] for row in progress_rows], label=key)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("episode frame progress")
        plt.ylabel("hybrid Q - IL Q")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(path / "bimanual_hybrid_q_vs_progress.png", dpi=160)
        plt.close()


def _print_report(summary: dict) -> None:
    print("\n=== Terminal / Advantage Diagnostic ===")
    print(f"chunk_size: {summary['chunk_size']}")
    print("Sampler terminal-window coverage:")
    for name, data in summary["coverage"].items():
        fraction = data["coverage_fraction"]
        print(
            f"  {name:20s} stride={data['stride']:<4d} "
            f"covered={data['covered_episodes']}/{data['eligible_episodes']} "
            f"({100.0 * fraction:.1f}%) "
            f"mean_gap={data['mean_terminal_anchor_gap']:.3f}"
        )

    motion = summary["motion"]
    print("\nDemonstration action motion:")
    print(
        "  mean normalized last-H / middle ratio: "
        f"{motion['mean_normalized_last_h_over_middle']:.4f}"
    )
    if motion["mean_normalized_last_h_over_middle"] < 0.5:
        print(
            "  [FLAG] Demonstration tail is substantially more stationary than the "
            "middle of the episode."
        )

    h = summary["hypothesis"]
    n = max(int(h.get("second_half_count", 0)), 1)
    print("\nSecond-half counterfactual critic preferences:")
    for label, key in (
        ("Post-RL Q > IL Q", "second_half_postrl_q_gt_il"),
        ("Hold Q > IL Q", "second_half_hold_q_gt_il"),
        ("Terminal-template Q > IL Q", "second_half_terminal_q_gt_il"),
        ("Post-RL advantage > 0", "second_half_postrl_adv_positive"),
        ("Hold advantage > 0", "second_half_hold_adv_positive"),
        ("Terminal-template advantage > 0", "second_half_terminal_adv_positive"),
    ):
        count = int(h.get(key, 0))
        print(f"  {label:32s}: {count}/{n} ({100.0 * count / n:.1f}%)")

    print("\nTerminal action shuffle / state-shuffle:")
    for label, key in (
        ("Shuffled terminal Q > IL Q", "second_half_terminal_shuffled_q_gt_il"),
        (
            "Shuffled terminal advantage > 0",
            "second_half_terminal_shuffled_adv_positive",
        ),
        (
            "Same terminal Q > shuffled terminal Q",
            "second_half_terminal_same_q_gt_shuffled",
        ),
        (
            "Terminal action advantage > 0 on other state",
            "second_half_terminal_other_state_adv_positive",
        ),
    ):
        count = int(h.get(key, 0))
        print(f"  {label:42s}: {count}/{n} ({100.0 * count / n:.1f}%)")

    closer = int(h.get("second_half_postrl_closer_terminal", 0))
    print("\nActor distance to terminal-action manifold:")
    print(
        "  Post-RL closer to terminal than IL: "
        f"{closer}/{n} ({100.0 * closer / n:.1f}%)"
    )

    if summary["bimanual_hybrid_supported"]:
        print("\nBimanual terminal-component counterfactuals (Q > IL Q):")
        for label, key in (
            (
                "terminal LEFT + IL right",
                "second_half_hybrid_terminal_left_il_right_q_gt_il",
            ),
            (
                "IL left + terminal RIGHT",
                "second_half_hybrid_il_left_terminal_right_q_gt_il",
            ),
            (
                "terminal left joints only",
                "second_half_hybrid_terminal_left_joints_q_gt_il",
            ),
            (
                "terminal left gripper only",
                "second_half_hybrid_terminal_left_gripper_q_gt_il",
            ),
            (
                "terminal right joints only",
                "second_half_hybrid_terminal_right_joints_q_gt_il",
            ),
            (
                "terminal right gripper only",
                "second_half_hybrid_terminal_right_gripper_q_gt_il",
            ),
        ):
            count = int(h.get(key, 0))
            print(f"  {label:32s}: {count}/{n} ({100.0 * count / n:.1f}%)")

    phase_lookup = {row["phase"]: row for row in summary["phase_summary"]}
    print("\nMean Q - IL Q by phase:")
    for phase in (
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
        "terminal_chunk",
    ):
        row = phase_lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:16s} "
            f"postrl={row['q_postrl_minus_il']:+.5f} "
            f"hold={row['q_hold_minus_il']:+.5f} "
            f"terminal_template={row['q_terminal_template_minus_il']:+.5f}"
        )

    print("\nPost-RL terminal-distance delta by phase (negative = closer than IL):")
    for phase in (
        "early_0_25",
        "middle_25_50",
        "late_50_75",
        "tail_75_100",
        "terminal_chunk",
    ):
        row = phase_lookup.get(phase)
        if row is None:
            continue
        print(
            f"  {phase:16s} "
            f"delta_rmse={row['postrl_terminal_delta_rmse']:+.6f}"
        )

    print(
        "\nInterpretation: the premature-terminal hypothesis is supported if terminal/hold-like "
        "actions become positively advantaged before the true terminal region, especially "
        "when Q ranks them above the IL continuation action. These are critic diagnostics, "
        "not proof that the actor explicitly recognizes a task-end flag."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Terminal / Advantage diagnostic for ARBiM. Reuses the trained "
            "offline buffer, frozen ACT latent cache, IQL Q/V weights, Base ACT, and "
            "Post-RL policy weights without changing or retraining production code."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override Base/IL ACT checkpoint.")
    parser.add_argument("--dataset", default=None, help="Override Offline RL Zarr path.")
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--progress-bins", type=int, default=20)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.progress_bins < 4:
        raise ValueError("--progress-bins must be >= 4")

    stage1_dir = pathlib.Path(args.stage1_dir).expanduser().resolve()
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

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "post_training" / "outputs" / "terminal_advantage_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace.buffer = workspace._load_buffer()
    workspace._build_main_dataloaders()
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError("Expected a pretrained Critic; diagnostic must not train Q/V.")
    workspace.critic.eval()

    postrl_policy, postrl_kind = _load_postrl_policy(
        postrl_checkpoint,
        workspace.device,
        cfg,
    )
    if int(postrl_policy.config.chunk_size) != int(workspace.model.config.chunk_size):
        raise RuntimeError(
            "Post-RL/Base ACT chunk mismatch: "
            f"{postrl_policy.config.chunk_size} vs {workspace.model.config.chunk_size}"
        )

    il_processor = _resolve_processor_dir(checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_processor_fp = _processor_fingerprint(il_processor)
    postrl_processor_fp = _processor_fingerprint(postrl_processor)
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ. Refusing Q/V comparison."
        )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    postrl_encoder_sha = workspace._fingerprint_module(postrl_encoder)
    cache_encoder_sha = str(workspace.latent_cache.metadata.get("encoder_sha256", ""))
    if postrl_encoder_sha != cache_encoder_sha:
        raise RuntimeError(
            "Post-RL encoder does not match the frozen training cache: "
            f"postrl={postrl_encoder_sha[:12]}..., cache={cache_encoder_sha[:12]}..."
        )
    postrl_policy.set_frozen_encoder_pos_embed(
        workspace._frozen_encoder_pos_embed
    )

    chunk_size = int(cfg.n_action_steps)
    episode_starts, episode_ends = _episode_bounds(workspace.buffer.episode_ends)
    strides = {
        "dataset": int(cfg.dataset.sequence_stride),
        "critic": int(cfg.critic.sequence_stride),
        "ppo_finetune": int(cfg.dataset.finetune_sequence_stride),
        "diagnostic_stride1": 1,
    }
    coverage_rows = _coverage_rows(
        episode_starts,
        episode_ends,
        chunk_size,
        strides,
    )
    coverage = _coverage_summary(coverage_rows)
    _write_csv(output_dir / "terminal_window_coverage.csv", coverage_rows)

    motion_rows = _episode_motion_rows(
        workspace,
        episode_starts,
        episode_ends,
        chunk_size,
    )
    _write_csv(output_dir / "episode_action_motion.csv", motion_rows)
    motion_ratios = np.asarray(
        [
            row["normalized_last_h_over_middle"]
            for row in motion_rows
            if np.isfinite(row["normalized_last_h_over_middle"])
        ],
        dtype=np.float64,
    )
    motion_summary = {
        "episodes": len(motion_rows),
        "mean_normalized_last_h_over_middle": (
            float(motion_ratios.mean()) if motion_ratios.size else float("nan")
        ),
        "median_normalized_last_h_over_middle": (
            float(np.median(motion_ratios)) if motion_ratios.size else float("nan")
        ),
    }

    anchors, anchor_episode_ids, anchor_starts, anchor_ends = _select_anchors(
        episode_starts,
        episode_ends,
        chunk_size,
        args.max_anchors,
    )
    phase_summary, progress_summary, hypothesis, analyzed_rows = _evaluate_qva(
        workspace,
        postrl_policy,
        anchors,
        anchor_episode_ids,
        anchor_starts,
        anchor_ends,
        chunk_size=chunk_size,
        batch_size=int(args.batch_size),
        output_path=output_dir / "qva_per_anchor.csv",
        progress_bins=int(args.progress_bins),
    )
    _write_csv(output_dir / "qva_phase_summary.csv", phase_summary)
    _write_csv(output_dir / "qva_progress_summary.csv", progress_summary)
    _plot_progress(output_dir, progress_summary)

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
        "chunk_size": chunk_size,
        "configured_strides": strides,
        "coverage": coverage,
        "motion": motion_summary,
        "hypothesis": _fractions(hypothesis),
        "bimanual_hybrid_supported": int(workspace.action_dim) == 16,
        "phase_summary": phase_summary,
        "progress_summary": progress_summary,
        "analyzed_anchors": int(analyzed_rows),
        "counterfactual_actions": {
            "demo": "dataset action chunk at the same anchor",
            "il": "Base ACT deterministic mean on the same cached observation latent",
            "postrl": "Post-RL deterministic mean on the same cached observation latent",
            "hold": "repeat the first demonstration action across the whole chunk",
            "terminal_template": "reuse the final H-action chunk from the same episode",
            "terminal_shuffled": "reuse the final H-action chunk from the next episode while keeping the current state",
            "terminal_on_other_state": "apply the current episode terminal chunk to a progress-matched state from the next episode",
            "bimanual_hybrids": (
                "for 16D [left7,left_gripper,right7,right_gripper], replace one arm, "
                "one arm's joints, or one gripper in the Base ACT chunk with the terminal template"
            ),
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    _print_report(summary)
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
