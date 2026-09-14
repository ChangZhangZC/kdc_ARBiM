from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from post_rl.data.sampler import get_val_mask  # noqa: E402
from post_rl.dynamics.core.ensemble_dynamics_for_batch import (  # noqa: E402
    EnsembleDynamics_batch,
)
from post_rl.dynamics.models.dynamics_model import EnsembleDynamicsModel  # noqa: E402
from post_rl.dynamics.utils.termination_fns import get_termination_fn  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "post_training" / "configs" / "rl" / "offline_rl.yaml"


@dataclass
class OneStepAccumulator:
    model_sse: np.ndarray
    elite_sse: float = 0.0
    zero_delta_sse: float = 0.0
    shuffled_sse: float = 0.0
    zero_action_sse: float = 0.0
    count: int = 0
    shuffled_count: int = 0

    @classmethod
    def create(cls, ensemble_size: int) -> "OneStepAccumulator":
        return cls(model_sse=np.zeros(ensemble_size, dtype=np.float64))


def _resolve_config(stage1_dir: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        path = pathlib.Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    provenance = stage1_dir / "provenance" / "dynamics_config.yaml"
    return provenance if provenance.is_file() else DEFAULT_CONFIG


def _resolve_required_path(value: str | None, fallback, name: str) -> pathlib.Path:
    raw = value if value is not None else fallback
    if raw is None:
        raise ValueError(f"{name} must be supplied either by CLI or config")
    path = pathlib.Path(str(raw)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _select_evenly(indices: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    positions = np.linspace(0, len(indices) - 1, num=max_samples, dtype=np.int64)
    return np.unique(indices[positions])


def _validation_anchors(
    episode_ends: np.ndarray,
    val_ratio: float,
    seed: int,
    transition_steps: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if transition_steps < 1:
        raise ValueError("transition_steps must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    val_mask = get_val_mask(len(episode_ends), val_ratio=val_ratio, seed=seed)
    if not np.any(val_mask):
        raise RuntimeError(
            f"Validation split is empty for val_ratio={val_ratio}. "
            "Use --val-ratio with a positive value."
        )

    starts = np.concatenate(([0], episode_ends[:-1])).astype(np.int64)
    anchors = []
    episode_ids = []
    for episode_id, (start, end) in enumerate(zip(starts, episode_ends, strict=True)):
        if not val_mask[episode_id]:
            continue
        # A complete chunk uses actions [t, t + transition_steps - 1]. The
        # final next observation is read from the stored next_* view of the
        # last action, so t=end-transition_steps remains a valid terminal chunk.
        latest = int(end) - int(transition_steps)
        if latest < int(start):
            continue
        current = np.arange(int(start), latest + 1, stride, dtype=np.int64)
        anchors.append(current)
        episode_ids.append(np.full(len(current), episode_id, dtype=np.int64))

    if not anchors:
        raise RuntimeError("No complete validation transitions were found")
    return np.concatenate(anchors), np.concatenate(episode_ids)


def _to_time_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array))
    if tensor.ndim < 2:
        raise ValueError(f"Expected batched array, got shape={tuple(tensor.shape)}")
    return tensor.unsqueeze(1).to(device, non_blocking=True)


def _gather_obs(buffer, indices: np.ndarray, *, next_view: bool, device: torch.device) -> dict:
    prefix = "next_" if next_view else ""
    obs = {
        "state": _to_time_tensor(np.asarray(buffer[f"{prefix}state"][indices]), device),
    }
    for key in buffer.RGB_KEYS:
        obs[key] = _to_time_tensor(np.asarray(buffer[f"{prefix}{key}"][indices]), device)
    if getattr(buffer, "_use_depth", False):
        for key in buffer.DEPTH_KEYS:
            obs[key] = _to_time_tensor(np.asarray(buffer[f"{prefix}{key}"][indices]), device)
    return obs


def _gather_action_chunks(buffer, anchors: np.ndarray, chunk_size: int, device: torch.device) -> torch.Tensor:
    offsets = np.arange(chunk_size, dtype=np.int64)[None, :]
    indices = anchors[:, None] + offsets
    actions = np.asarray(buffer["action"][indices], dtype=np.float32)
    return torch.from_numpy(actions).to(device, non_blocking=True)


def _encode_obs(workspace: TrainACTWorkspace, obs: dict) -> torch.Tensor:
    return workspace.obs_adapter.encode(obs, start=0, track_grad=False)[:, 0]


def _predict_all(dynamics: EnsembleDynamics_batch, state: torch.Tensor, action: torch.Tensor):
    mean, logvar = dynamics.model(state, action)
    if dynamics.predict_delta:
        next_mean = mean + state.unsqueeze(0)
    else:
        next_mean = mean
    return next_mean, logvar


def _elite_mean(next_mean: torch.Tensor, elite_indices: torch.Tensor) -> torch.Tensor:
    return next_mean.index_select(0, elite_indices).mean(dim=0)


def _sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, prediction.ndim))
    return ((prediction - target) ** 2).mean(dim=reduce_dims)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _build_dynamics(workspace: TrainACTWorkspace, cfg, checkpoint_dir: pathlib.Path):
    model_action_dim = int(workspace.action_dim) * int(cfg.n_action_steps)
    model = EnsembleDynamicsModel(
        obs_dim=int(workspace.obs_feature_dim),
        action_dim=model_action_dim,
        hidden_dims=list(cfg.dynamics.dynamics_hidden_dims),
        num_ensemble=int(cfg.dynamics.n_ensemble),
        num_elites=int(cfg.dynamics.n_elites),
        weight_decays=list(cfg.dynamics.dynamics_weight_decay),
        with_reward=False,
        device=workspace.device,
        cfg=cfg,
    )
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    dynamics = EnsembleDynamics_batch(
        model=model,
        optim=optimizer,
        terminal_fn=get_termination_fn(cfg.task_name),
        env=None,
        obs_adapter=workspace.obs_adapter,
        cfg=cfg,
        action_dim=model_action_dim,
        gamma=float(cfg.critic.gamma),
        device=workspace.device,
        chunk_as_single_action=True,
        n_action_steps=int(cfg.n_action_steps),
        prediction_mode="full",
    )
    workspace._validate_artifact_contract(str(checkpoint_dir), "Dynamics")
    dynamics.load(str(checkpoint_dir))
    dynamics.model.eval()
    return dynamics


def _check_checkpoint(dynamics: EnsembleDynamics_batch) -> dict:
    model = dynamics._model()
    nonfinite = []
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            nonfinite.append(name)
    elites = [int(v) for v in model.elites.detach().cpu().tolist()]
    if len(elites) != int(model.num_elites):
        raise AssertionError(f"Expected {model.num_elites} elites, got {elites}")
    if len(set(elites)) != len(elites) or any(v < 0 or v >= model.num_ensemble for v in elites):
        raise AssertionError(f"Invalid elite indices: {elites}")
    if nonfinite:
        raise AssertionError(f"Dynamics checkpoint contains NaN/Inf tensors: {nonfinite}")
    return {
        "num_ensemble": int(model.num_ensemble),
        "num_elites": int(model.num_elites),
        "elites": elites,
        "nonfinite_tensors": nonfinite,
    }


@torch.no_grad()
def _evaluate_one_step(
    workspace: TrainACTWorkspace,
    dynamics: EnsembleDynamics_batch,
    anchors: np.ndarray,
    batch_size: int,
    chunk_size: int,
):
    model = dynamics._model()
    elite_indices = model.elites.detach().to(workspace.device, dtype=torch.long)
    accumulator = OneStepAccumulator.create(model.num_ensemble)
    disagreements: list[float] = []
    actual_errors: list[float] = []
    aleatoric_vars: list[float] = []

    for start in tqdm(range(0, len(anchors), batch_size), desc="One-step Dynamics eval"):
        batch_anchors = anchors[start:start + batch_size]
        final_action_indices = batch_anchors + chunk_size - 1
        obs = _gather_obs(workspace.buffer, batch_anchors, next_view=False, device=workspace.device)
        next_obs = _gather_obs(
            workspace.buffer,
            final_action_indices,
            next_view=True,
            device=workspace.device,
        )
        state = _encode_obs(workspace, obs)
        target_next = _encode_obs(workspace, next_obs)
        raw_action = _gather_action_chunks(workspace.buffer, batch_anchors, chunk_size, workspace.device)
        action = workspace.obs_adapter.normalize_action(raw_action)

        next_mean, logvar = _predict_all(dynamics, state, action)
        target_ensemble = target_next.unsqueeze(0)
        model_mse = ((next_mean - target_ensemble) ** 2).mean(dim=(2, 3))
        accumulator.model_sse += model_mse.sum(dim=1).cpu().numpy()

        elite_prediction = _elite_mean(next_mean, elite_indices)
        elite_error = _sample_mse(elite_prediction, target_next)
        zero_error = _sample_mse(state, target_next)
        accumulator.elite_sse += float(elite_error.sum().item())
        accumulator.zero_delta_sse += float(zero_error.sum().item())
        accumulator.count += int(len(batch_anchors))

        zero_action = torch.zeros_like(action)
        zero_next, _ = _predict_all(dynamics, state, zero_action)
        zero_prediction = _elite_mean(zero_next, elite_indices)
        accumulator.zero_action_sse += float(
            _sample_mse(zero_prediction, target_next).sum().item()
        )

        if len(batch_anchors) > 1:
            shuffled_action = torch.roll(action, shifts=1, dims=0)
            shuffled_next, _ = _predict_all(dynamics, state, shuffled_action)
            shuffled_prediction = _elite_mean(shuffled_next, elite_indices)
            accumulator.shuffled_sse += float(
                _sample_mse(shuffled_prediction, target_next).sum().item()
            )
            accumulator.shuffled_count += int(len(batch_anchors))

        # Raw ensemble disagreement is calculated from deterministic predicted
        # means. Do not use compute_model_uncertainty(), because its public
        # return is multiplied by penalty_coef and can be identically zero.
        disagreement = next_mean.var(dim=0, unbiased=False).mean(dim=(1, 2))
        predicted_var = torch.exp(logvar).mean(dim=(0, 2, 3))
        disagreements.extend(disagreement.cpu().tolist())
        actual_errors.extend(elite_error.cpu().tolist())
        aleatoric_vars.extend(predicted_var.cpu().tolist())

    if accumulator.count == 0:
        raise RuntimeError("One-step evaluation accumulated zero samples")

    per_model = accumulator.model_sse / accumulator.count
    elite_mse = accumulator.elite_sse / accumulator.count
    zero_delta_mse = accumulator.zero_delta_sse / accumulator.count
    zero_action_mse = accumulator.zero_action_sse / accumulator.count
    shuffled_mse = (
        accumulator.shuffled_sse / accumulator.shuffled_count
        if accumulator.shuffled_count > 0
        else float("nan")
    )
    result = {
        "samples": accumulator.count,
        "per_model_mse": [float(v) for v in per_model],
        "elite_mean_mse": float(elite_mse),
        "zero_delta_mse": float(zero_delta_mse),
        "elite_over_zero_delta": float(elite_mse / max(zero_delta_mse, 1e-12)),
        "shuffled_action_mse": float(shuffled_mse),
        "zero_normalized_action_mse": float(zero_action_mse),
        "shuffled_over_correct": float(shuffled_mse / max(elite_mse, 1e-12)),
        "zero_action_over_correct": float(zero_action_mse / max(elite_mse, 1e-12)),
        "disagreement_error_pearson": _pearson(disagreements, actual_errors),
        "aleatoric_error_pearson": _pearson(aleatoric_vars, actual_errors),
        "mean_raw_ensemble_disagreement": float(np.mean(disagreements)),
        "mean_predicted_variance": float(np.mean(aleatoric_vars)),
    }
    return result


@torch.no_grad()
def _evaluate_rollout(
    workspace: TrainACTWorkspace,
    dynamics: EnsembleDynamics_batch,
    anchors: np.ndarray,
    batch_size: int,
    chunk_size: int,
    rollout_steps: int,
):
    model = dynamics._model()
    elite_indices = model.elites.detach().to(workspace.device, dtype=torch.long)
    rollout_sse = np.zeros(rollout_steps, dtype=np.float64)
    teacher_sse = np.zeros(rollout_steps, dtype=np.float64)
    zero_sse = np.zeros(rollout_steps, dtype=np.float64)
    count = np.zeros(rollout_steps, dtype=np.int64)

    for start in tqdm(range(0, len(anchors), batch_size), desc="Multi-step Dynamics eval"):
        batch_anchors = anchors[start:start + batch_size]
        obs = _gather_obs(workspace.buffer, batch_anchors, next_view=False, device=workspace.device)
        initial_state = _encode_obs(workspace, obs)
        rollout_state = initial_state
        teacher_state = initial_state

        for step in range(rollout_steps):
            chunk_anchors = batch_anchors + step * chunk_size
            final_action_indices = chunk_anchors + chunk_size - 1
            raw_action = _gather_action_chunks(
                workspace.buffer,
                chunk_anchors,
                chunk_size,
                workspace.device,
            )
            action = workspace.obs_adapter.normalize_action(raw_action)
            next_obs = _gather_obs(
                workspace.buffer,
                final_action_indices,
                next_view=True,
                device=workspace.device,
            )
            target_next = _encode_obs(workspace, next_obs)

            rollout_all, _ = _predict_all(dynamics, rollout_state, action)
            rollout_state = _elite_mean(rollout_all, elite_indices)
            rollout_error = _sample_mse(rollout_state, target_next)

            teacher_all, _ = _predict_all(dynamics, teacher_state, action)
            teacher_prediction = _elite_mean(teacher_all, elite_indices)
            teacher_error = _sample_mse(teacher_prediction, target_next)
            teacher_state = target_next

            zero_error = _sample_mse(initial_state, target_next)
            rollout_sse[step] += float(rollout_error.sum().item())
            teacher_sse[step] += float(teacher_error.sum().item())
            zero_sse[step] += float(zero_error.sum().item())
            count[step] += len(batch_anchors)

    rows = []
    for step in range(rollout_steps):
        denom = max(int(count[step]), 1)
        rollout_mse = rollout_sse[step] / denom
        teacher_mse = teacher_sse[step] / denom
        zero_mse = zero_sse[step] / denom
        rows.append(
            {
                "rollout_step": step + 1,
                "environment_frames": (step + 1) * chunk_size,
                "samples": int(count[step]),
                "rollout_mse": float(rollout_mse),
                "teacher_forced_mse": float(teacher_mse),
                "zero_delta_from_start_mse": float(zero_mse),
                "rollout_over_teacher": float(rollout_mse / max(teacher_mse, 1e-12)),
                "rollout_over_zero": float(rollout_mse / max(zero_mse, 1e-12)),
            }
        )
    return rows


def _write_one_step_csv(path: pathlib.Path, one_step: dict, elites: list[int]) -> None:
    elite_set = set(elites)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "is_elite", "mse"])
        writer.writeheader()
        for model_id, mse in enumerate(one_step["per_model_mse"]):
            writer.writerow({"model": model_id, "is_elite": model_id in elite_set, "mse": mse})


def _write_rollout_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_report(checkpoint: dict, one_step: dict, rollout: list[dict]) -> None:
    print("\n=== Dynamics checkpoint ===")
    print(f"ensemble={checkpoint['num_ensemble']} elites={checkpoint['num_elites']} indices={checkpoint['elites']}")
    print("NaN/Inf tensors: 0")

    print("\n=== One-step held-out prediction ===")
    for model_id, mse in enumerate(one_step["per_model_mse"]):
        marker = "*" if model_id in checkpoint["elites"] else " "
        print(f"{marker} model {model_id}: mse={mse:.6f}")
    print(f"elite mean prediction MSE : {one_step['elite_mean_mse']:.6f}")
    print(f"zero-delta baseline MSE   : {one_step['zero_delta_mse']:.6f}")
    print(f"elite / zero-delta        : {one_step['elite_over_zero_delta']:.3f}")

    print("\n=== Action sensitivity ===")
    print(f"correct action MSE         : {one_step['elite_mean_mse']:.6f}")
    print(f"shuffled action MSE        : {one_step['shuffled_action_mse']:.6f}")
    print(f"zero normalized action MSE : {one_step['zero_normalized_action_mse']:.6f}")
    print(f"shuffled / correct         : {one_step['shuffled_over_correct']:.3f}")
    print(f"zero-action / correct      : {one_step['zero_action_over_correct']:.3f}")

    print("\n=== Uncertainty diagnostics ===")
    print(f"ensemble disagreement -> error Pearson : {one_step['disagreement_error_pearson']:.4f}")
    print(f"predicted variance -> error Pearson     : {one_step['aleatoric_error_pearson']:.4f}")

    print("\n=== Deterministic elite-mean rollout ===")
    print("step | frames | rollout_mse | teacher_mse | zero_from_start | rollout/teacher")
    for row in rollout:
        print(
            f"{row['rollout_step']:>4} | {row['environment_frames']:>6} | "
            f"{row['rollout_mse']:.6f} | {row['teacher_forced_mse']:.6f} | "
            f"{row['zero_delta_from_start_mse']:.6f} | {row['rollout_over_teacher']:.3f}"
        )

    checks = {
        "beats_zero_delta": one_step["elite_mean_mse"] < one_step["zero_delta_mse"],
        "uses_action_information": one_step["shuffled_action_mse"] > one_step["elite_mean_mse"],
        "finite_uncertainty_correlation": math.isfinite(one_step["disagreement_error_pearson"]),
    }
    print("\n=== Diagnostic checks ===")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'WARN'}] {name}")
    if rollout:
        growth = rollout[-1]["rollout_mse"] / max(rollout[0]["rollout_mse"], 1e-12)
        print(f"multi-step rollout error growth (last/first): {growth:.3f}x")
        if growth > 10.0:
            print("[WARN] rollout error grows by >10x; inspect before using long-horizon OPE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ARBiM ACT-latent Dynamics ensemble before Stage 2 PPO."
    )
    parser.add_argument("--stage1-dir", required=True, help="Combined Stage-1 artifact directory.")
    parser.add_argument("--checkpoint", default=None, help="Override ACT IL checkpoint path.")
    parser.add_argument("--dataset", default=None, help="Override Offline RL Zarr path.")
    parser.add_argument("--config", default=None, help="Resolved Stage-1 YAML. Defaults to provenance/dynamics_config.yaml.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--rollout-samples", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--stride", type=int, default=None, help="Anchor stride; defaults to one ACT chunk.")
    parser.add_argument("--val-ratio", type=float, default=None)
    args = parser.parse_args()

    stage1_dir = pathlib.Path(args.stage1_dir).expanduser().resolve()
    dynamics_final = stage1_dir / "dynamics" / "checkpoints" / "final"
    critic_contract = stage1_dir / "critic" / "checkpoints" / "final" / "contract.json"
    dynamics_contract = dynamics_final / "contract.json"
    dynamics_weights = dynamics_final / "dynamics.pth"
    for path in (critic_contract, dynamics_contract, dynamics_weights):
        if not path.exists():
            raise FileNotFoundError(path)
    if critic_contract.read_text() != dynamics_contract.read_text():
        raise RuntimeError("Critic and Dynamics contract.json files differ")

    config_path = _resolve_config(stage1_dir, args.config)
    cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_required_path(args.checkpoint, cfg.input.get("policy_checkpoint"), "ACT checkpoint")
    dataset = _resolve_required_path(args.dataset, cfg.input.get("dataset_path"), "Offline dataset")
    cfg.input.policy_checkpoint = str(checkpoint)
    cfg.input.policy_checkpoint_type = "il"
    cfg.input.dataset_path = str(dataset)
    cfg.training.device = str(args.device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    cfg.dataset.use_latent_cache = False

    workspace_output = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "post_training" / "outputs" / "dynamics_evaluation"
    )
    workspace_output.mkdir(parents=True, exist_ok=True)

    workspace = TrainACTWorkspace(cfg, output_dir=str(workspace_output))
    workspace.buffer = workspace._load_buffer()
    workspace._build_act_observation_frontends()
    dynamics = _build_dynamics(workspace, cfg, dynamics_final)
    workspace.dynamics = dynamics

    checkpoint_info = _check_checkpoint(dynamics)
    chunk_size = int(cfg.n_action_steps)
    if chunk_size != int(workspace.model.config.chunk_size):
        raise RuntimeError(
            f"Config chunk={chunk_size} != ACT chunk={workspace.model.config.chunk_size}"
        )
    stride = int(args.stride or chunk_size)
    val_ratio = float(cfg.dataset.val_ratio if args.val_ratio is None else args.val_ratio)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    anchors, _ = _validation_anchors(
        workspace.buffer.episode_ends,
        val_ratio=val_ratio,
        seed=int(cfg.training.seed),
        transition_steps=chunk_size,
        stride=stride,
    )
    anchors = _select_evenly(anchors, int(args.max_samples))

    rollout_transition_steps = chunk_size * int(args.rollout_steps)
    rollout_anchors, _ = _validation_anchors(
        workspace.buffer.episode_ends,
        val_ratio=val_ratio,
        seed=int(cfg.training.seed),
        transition_steps=rollout_transition_steps,
        stride=stride,
    )
    rollout_anchors = _select_evenly(rollout_anchors, int(args.rollout_samples))

    print(f"Config: {config_path}")
    print(f"ACT checkpoint: {checkpoint}")
    print(f"Offline dataset: {dataset}")
    print(f"Dynamics checkpoint: {dynamics_weights}")
    print(f"Chunk size: {chunk_size}")
    print(f"Validation ratio: {val_ratio}")
    print(f"One-step anchors: {len(anchors)}")
    print(f"Rollout anchors: {len(rollout_anchors)}")

    one_step = _evaluate_one_step(
        workspace,
        dynamics,
        anchors,
        batch_size=int(args.batch_size),
        chunk_size=chunk_size,
    )
    rollout = _evaluate_rollout(
        workspace,
        dynamics,
        rollout_anchors,
        batch_size=int(args.batch_size),
        chunk_size=chunk_size,
        rollout_steps=int(args.rollout_steps),
    )

    summary = {
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "chunk_size": chunk_size,
        "validation_ratio": val_ratio,
        "checkpoint_info": checkpoint_info,
        "one_step": one_step,
        "rollout": rollout,
    }
    with (workspace_output / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    _write_one_step_csv(workspace_output / "one_step_models.csv", one_step, checkpoint_info["elites"])
    _write_rollout_csv(workspace_output / "rollout_metrics.csv", rollout)
    _print_report(checkpoint_info, one_step, rollout)

    print(f"\nSaved: {workspace_output / 'summary.json'}")
    print(f"Saved: {workspace_output / 'one_step_models.csv'}")
    print(f"Saved: {workspace_output / 'rollout_metrics.csv'}")


if __name__ == "__main__":
    main()
