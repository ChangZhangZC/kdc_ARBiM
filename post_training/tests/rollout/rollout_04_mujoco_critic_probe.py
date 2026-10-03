from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
ANALYSIS_DIR = REPO_ROOT / "post_training" / "tests" / "analysis"
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from analysis_05_terminal_advantage import (  # noqa: E402
    _processor_fingerprint,
    _resolve_config,
    _resolve_processor_dir,
    _resolve_required_path,
)
from kuavo_deploy.config import load_kuavo_config  # noqa: E402
from kuavo_deploy.src.scripts.script_auto_test import ArmMove  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


def _stat_tensor(stats: dict, feature: str, name: str, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        stats[feature][name],
        device=ref.device,
        dtype=ref.dtype,
    )


def _physical_state_from_normalized_obs(
    observation: dict,
    stats: dict,
) -> torch.Tensor:
    state = observation["observation.state"].float()
    if state.ndim != 2:
        raise RuntimeError(f"Expected processed state [B,D], got {tuple(state.shape)}")
    state_mean = _stat_tensor(stats, "observation.state", "mean", state)
    state_std = _stat_tensor(stats, "observation.state", "std", state)
    return state * state_std + state_mean


def _physical_action_from_normalized(
    action: torch.Tensor,
    stats: dict,
) -> torch.Tensor:
    action_mean = _stat_tensor(stats, "action", "mean", action)
    action_std = _stat_tensor(stats, "action", "std", action)
    return action * action_std + action_mean


def _hold_chunk_from_normalized_obs(
    observation: dict,
    stats: dict,
    horizon: int,
) -> torch.Tensor:
    state = observation["observation.state"].float()
    physical_state = _physical_state_from_normalized_obs(observation, stats)
    action_mean = _stat_tensor(stats, "action", "mean", state)
    action_std = _stat_tensor(stats, "action", "std", state)
    if state.shape[-1] != action_mean.numel():
        raise RuntimeError(
            f"Hold counterfactual requires state/action dims to match, got "
            f"{state.shape[-1]} and {action_mean.numel()}"
        )
    normalized_hold = (physical_state - action_mean) / (action_std + 1e-8)
    return normalized_hold.unsqueeze(1).expand(-1, horizon, -1).clone()


@torch.no_grad()
def _critic_values(critic, state: torch.Tensor, actions: dict[str, torch.Tensor]) -> dict:
    v = critic.value(state).reshape(-1)
    out = {"v": float(v[0].item())}
    for name, action in actions.items():
        q = critic.minQ(state, action).reshape(-1)
        out[f"q_{name}"] = float(q[0].item())
        out[f"a_{name}"] = float((q - v)[0].item())
    return out


def _chunk_rmse(a: torch.Tensor, b: torch.Tensor, slc: slice = slice(None)) -> float:
    d = a[..., slc] - b[..., slc]
    return float(torch.sqrt(d.square().mean()).item())


def _chunk_step_rmse(chunk: torch.Tensor, slc: slice = slice(None)) -> float:
    if chunk.shape[1] < 2:
        return 0.0
    d = chunk[:, 1:, slc] - chunk[:, :-1, slc]
    return float(torch.sqrt(d.square().mean()).item())


class CriticProbePolicy:
    """Execute one ACT policy while scoring fresh Base/Post-RL/hold chunks at each state."""

    def __init__(
        self,
        *,
        base_policy,
        postrl_policy,
        critic,
        stats: dict,
        control_policy: str,
        csv_path: pathlib.Path,
        chunk_path: pathlib.Path,
        probe_every: int,
        partial_completion_step: int | None,
    ) -> None:
        if control_policy not in {"base", "postrl"}:
            raise ValueError("control_policy must be base or postrl")
        self.base_policy = base_policy
        self.postrl_policy = postrl_policy
        self.critic = critic
        self.stats = stats
        self.control_policy = control_policy
        self.config = (
            postrl_policy.config if control_policy == "postrl" else base_policy.config
        )
        self.probe_every = max(int(probe_every), 1)
        self.partial_completion_step = partial_completion_step

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.chunk_path = chunk_path
        self._file = csv_path.open("w", newline="", buffering=1)
        self._writer = None
        self._episode = -1
        self._step = 0
        self._has_steps = False
        self._prev_executed_norm = None
        self._chunks = {
            "episode": [],
            "step": [],
            "base": [],
            "postrl": [],
            "hold": [],
            "left_rl_right_base": [],
            "left_base_right_rl": [],
        }
        self._episode_results = []
        self._probe_rows = []

    def eval(self):
        self.base_policy.eval()
        self.postrl_policy.eval()
        self.critic.eval()
        return self

    def to(self, device):
        self.base_policy.to(device)
        self.postrl_policy.to(device)
        self.critic.to(device)
        return self

    def reset(self):
        self.base_policy.reset()
        self.postrl_policy.reset()
        if self._episode < 0:
            self._episode = 0
        elif self._has_steps:
            self._episode += 1
        self._step = 0
        self._has_steps = False
        self._prev_executed_norm = None

    @torch.inference_mode()
    def select_action(self, observation):
        control = (
            self.postrl_policy if self.control_policy == "postrl" else self.base_policy
        )

        # This is the action actually returned to the deployment loop. It can come
        # from ACT's execution queue, so keep it separate from the fresh full chunks
        # used for the Critic counterfactual.
        action = control.select_action(observation)
        executed_norm_t = action[0].detach().float()
        executed_norm = executed_norm_t.cpu().numpy()
        executed_step_delta = (
            float("nan")
            if self._prev_executed_norm is None
            else float(np.linalg.norm(executed_norm - self._prev_executed_norm))
        )

        if self._step % self.probe_every == 0:
            base_chunk = self.base_policy.predict_action_chunk(observation)
            postrl_chunk = self.postrl_policy.predict_action_chunk(observation)
            if base_chunk.shape != postrl_chunk.shape:
                raise RuntimeError(
                    f"Base/Post-RL chunk mismatch: {tuple(base_chunk.shape)} vs "
                    f"{tuple(postrl_chunk.shape)}"
                )
            if base_chunk.ndim != 3 or base_chunk.shape[0] != 1:
                raise RuntimeError(
                    f"Expected fresh ACT chunk [1,H,D], got {tuple(base_chunk.shape)}"
                )
            if base_chunk.shape[-1] != 16:
                raise RuntimeError("Critic probe expects the sim_task1 16D action contract.")

            hold_chunk = _hold_chunk_from_normalized_obs(
                observation, self.stats, int(base_chunk.shape[1])
            )
            left_rl_right_base = base_chunk.clone()
            left_rl_right_base[..., 0:8] = postrl_chunk[..., 0:8]
            left_base_right_rl = base_chunk.clone()
            left_base_right_rl[..., 8:16] = postrl_chunk[..., 8:16]

            # observation is already passed through the deployment preprocessor.
            # Bypass IQLCritic's raw-observation normalizer and encode directly.
            state = self.critic.obs_encoder(observation)
            values = _critic_values(
                self.critic,
                state,
                {
                    "base": base_chunk,
                    "postrl": postrl_chunk,
                    "hold": hold_chunk,
                    "left_rl_right_base": left_rl_right_base,
                    "left_base_right_rl": left_base_right_rl,
                },
            )

            state_norm_t = observation["observation.state"][0].detach().float()
            state_phys_t = _physical_state_from_normalized_obs(
                observation, self.stats
            )[0].detach().float()
            executed_phys_t = _physical_action_from_normalized(
                executed_norm_t, self.stats
            ).detach().float()
            fresh_control_chunk = (
                postrl_chunk if self.control_policy == "postrl" else base_chunk
            )
            fresh_control_first = fresh_control_chunk[0, 0].detach().float()
            executed_to_fresh = float(
                torch.linalg.vector_norm(executed_norm_t - fresh_control_first).item()
            )

            row = {
                "episode": int(self._episode),
                "step": int(self._step),
                "control_policy": self.control_policy,
                "after_partial_completion": (
                    int(
                        self.partial_completion_step is not None
                        and self._step >= self.partial_completion_step
                    )
                ),
                "executed_step_delta_l2": executed_step_delta,
                "executed_to_fresh_control_l2": executed_to_fresh,
                **values,
                "q_postrl_minus_base": values["q_postrl"] - values["q_base"],
                "q_hold_minus_base": values["q_hold"] - values["q_base"],
                "q_left_rl_right_base_minus_base": (
                    values["q_left_rl_right_base"] - values["q_base"]
                ),
                "q_left_base_right_rl_minus_base": (
                    values["q_left_base_right_rl"] - values["q_base"]
                ),
                "critic_prefers_hold_over_base": int(
                    values["q_hold"] > values["q_base"]
                ),
                "critic_prefers_postrl_over_base": int(
                    values["q_postrl"] > values["q_base"]
                ),
                "postrl_to_base_rmse": _chunk_rmse(postrl_chunk, base_chunk),
                "postrl_to_hold_rmse": _chunk_rmse(postrl_chunk, hold_chunk),
                "base_to_hold_rmse": _chunk_rmse(base_chunk, hold_chunk),
                "base_chunk_step_rmse": _chunk_step_rmse(base_chunk),
                "postrl_chunk_step_rmse": _chunk_step_rmse(postrl_chunk),
                "left_postrl_to_base_rmse": _chunk_rmse(
                    postrl_chunk, base_chunk, slice(0, 8)
                ),
                "right_postrl_to_base_rmse": _chunk_rmse(
                    postrl_chunk, base_chunk, slice(8, 16)
                ),
                "left_postrl_to_hold_rmse": _chunk_rmse(
                    postrl_chunk, hold_chunk, slice(0, 8)
                ),
                "right_postrl_to_hold_rmse": _chunk_rmse(
                    postrl_chunk, hold_chunk, slice(8, 16)
                ),
                "left_base_chunk_step_rmse": _chunk_step_rmse(
                    base_chunk, slice(0, 8)
                ),
                "right_base_chunk_step_rmse": _chunk_step_rmse(
                    base_chunk, slice(8, 16)
                ),
                "left_postrl_chunk_step_rmse": _chunk_step_rmse(
                    postrl_chunk, slice(0, 8)
                ),
                "right_postrl_chunk_step_rmse": _chunk_step_rmse(
                    postrl_chunk, slice(8, 16)
                ),
            }
            for index, value in enumerate(state_norm_t.cpu().tolist()):
                row[f"state_norm_{index}"] = float(value)
            for index, value in enumerate(state_phys_t.cpu().tolist()):
                row[f"state_phys_{index}"] = float(value)
            for index, value in enumerate(executed_norm_t.cpu().tolist()):
                row[f"executed_action_norm_{index}"] = float(value)
            for index, value in enumerate(executed_phys_t.cpu().tolist()):
                row[f"executed_action_phys_{index}"] = float(value)

            if self._writer is None:
                self._writer = csv.DictWriter(self._file, fieldnames=list(row))
                self._writer.writeheader()
            self._writer.writerow(row)
            self._file.flush()
            self._probe_rows.append(row)

            for name, chunk in (
                ("base", base_chunk),
                ("postrl", postrl_chunk),
                ("hold", hold_chunk),
                ("left_rl_right_base", left_rl_right_base),
                ("left_base_right_rl", left_base_right_rl),
            ):
                self._chunks[name].append(chunk[0].detach().float().cpu().numpy())
            self._chunks["episode"].append(int(self._episode))
            self._chunks["step"].append(int(self._step))

            print(
                f"[critic-probe] ep={self._episode} step={self._step} "
                f"V={values['v']:+.4f} "
                f"Qbase={values['q_base']:+.4f} "
                f"Qrl={values['q_postrl']:+.4f} "
                f"Qhold={values['q_hold']:+.4f} "
                f"dQhold={values['q_hold'] - values['q_base']:+.4f} "
                f"exec_delta={executed_step_delta:.6g} "
                f"queue_to_fresh={executed_to_fresh:.6g}"
            )

        self._prev_executed_norm = executed_norm.copy()
        self._step += 1
        self._has_steps = True
        return action

    def record_episode_result(self, episode: int, result: int) -> None:
        self._episode_results.append({
            "episode": int(episode),
            "task_success": int(result == 1),
            "steps": int(self._step),
        })

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()
        if self._chunks["step"]:
            np.savez_compressed(
                self.chunk_path,
                **{
                    key: np.asarray(value)
                    for key, value in self._chunks.items()
                },
            )
        summary_path = self.csv_path.parent / "episode_summary.csv"
        if self._episode_results:
            with summary_path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(self._episode_results[0])
                )
                writer.writeheader()
                writer.writerows(self._episode_results)
        probe_summary_path = self.csv_path.parent / "critic_probe_summary.json"
        if self._probe_rows:
            metric_keys = (
                "q_postrl_minus_base",
                "q_hold_minus_base",
                "q_left_rl_right_base_minus_base",
                "q_left_base_right_rl_minus_base",
                "postrl_to_base_rmse",
                "postrl_to_hold_rmse",
                "base_to_hold_rmse",
                "postrl_chunk_step_rmse",
                "executed_step_delta_l2",
                "executed_to_fresh_control_l2",
            )

            def summarize(rows):
                summary = {"count": int(len(rows))}
                for key in metric_keys:
                    values = np.asarray(
                        [float(row[key]) for row in rows], dtype=np.float64
                    )
                    values = values[np.isfinite(values)]
                    summary[f"{key}_mean"] = (
                        float(values.mean()) if values.size else None
                    )
                    summary[f"{key}_median"] = (
                        float(np.median(values)) if values.size else None
                    )
                summary["hold_over_base_fraction"] = float(
                    np.mean(
                        [row["critic_prefers_hold_over_base"] for row in rows]
                    )
                )
                summary["postrl_over_base_fraction"] = float(
                    np.mean(
                        [row["critic_prefers_postrl_over_base"] for row in rows]
                    )
                )
                return summary

            before = [
                row for row in self._probe_rows
                if not row["after_partial_completion"]
            ]
            after = [
                row for row in self._probe_rows
                if row["after_partial_completion"]
            ]
            probe_summary = {
                "all": summarize(self._probe_rows),
                "before_partial_completion": summarize(before) if before else None,
                "after_partial_completion": summarize(after) if after else None,
                "interpretation_note": (
                    "Same-state Q ranking is diagnostic only. Q(hold)>Q(Base) on "
                    "live states does not by itself prove PPO training caused the "
                    "deployed hold fixed point."
                ),
            }
            with probe_summary_path.open("w") as file:
                json.dump(probe_summary, file, indent=2, allow_nan=False)

        print(f"Critic probe CSV: {self.csv_path}")
        print(f"Fresh counterfactual chunks: {self.chunk_path}")
        print(f"Episode summary: {summary_path}")
        print(f"Critic probe aggregate summary: {probe_summary_path}")


def _build_critic_workspace(
    *,
    stage1_dir: pathlib.Path,
    rl_config: str | None,
    base_checkpoint: str | None,
    dataset: str | None,
    device: torch.device,
    output_dir: pathlib.Path,
):
    config_path = _resolve_config(stage1_dir, rl_config)
    cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_required_path(
        base_checkpoint, cfg.input.get("policy_checkpoint"), "Base ACT checkpoint"
    )
    if dataset is not None:
        cfg.input.dataset_path = str(
            _resolve_required_path(dataset, None, "Offline dataset")
        )
    cfg.input.policy_checkpoint = str(checkpoint)
    cfg.input.policy_checkpoint_type = "il"
    cfg.training.device = str(device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    # Live probe has normalized RGB/state, not a dataset latent cache. Build the
    # ordinary IQLCritic; CachedIQLCritic has the same learned Q/V parameters.
    cfg.dataset.use_latent_cache = False
    cfg.dataset.endpoint_obs_only = False
    cfg.critic.load_pretrain = True
    cfg.critic.artifact_dir = str(stage1_dir / "critic")

    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError("MuJoCo Critic Probe requires pretrained Stage-1 Q/V.")
    workspace.critic.eval()
    if int(workspace.action_dim) != 16:
        raise RuntimeError("MuJoCo Critic Probe expects sim_task1 16D action.")
    return workspace, cfg, config_path, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Kuavo MuJoCo with Base or Post-RL ACT control while scoring fresh "
            "Base/Post-RL/hold and left/right hybrid action chunks with the Stage-1 IQL Critic."
        )
    )
    parser.add_argument("--config", required=True, help="Kuavo simulation evaluation YAML.")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--postrl-checkpoint", required=True, help="Exported deterministic Post-RL ACT.")
    parser.add_argument("--rl-config", default=None)
    parser.add_argument("--checkpoint", default=None, help="Base ACT checkpoint override.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--control-policy", choices=("postrl", "base"), default="postrl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=1)
    parser.add_argument("--partial-completion-step", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/mujoco_critic_probe",
    )
    args = parser.parse_args()
    if args.probe_every < 1:
        raise ValueError("--probe-every must be >= 1")

    sim_config_path = pathlib.Path(args.config).expanduser().resolve()
    sim_config = load_kuavo_config(sim_config_path)
    if args.device is not None:
        sim_config.inference.device = args.device
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("--episodes must be >= 1")
        sim_config.inference.eval_episodes = int(args.episodes)
    device = torch.device(sim_config.inference.device)

    stage1_dir = pathlib.Path(args.stage1_dir).expanduser().resolve()
    postrl_checkpoint = pathlib.Path(args.postrl_checkpoint).expanduser().resolve()
    if not (postrl_checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(postrl_checkpoint / "model.safetensors")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace, rl_cfg, rl_config_path, base_checkpoint = _build_critic_workspace(
        stage1_dir=stage1_dir,
        rl_config=args.rl_config,
        base_checkpoint=args.checkpoint,
        dataset=args.dataset,
        device=device,
        output_dir=output_dir,
    )

    base_fp = _processor_fingerprint(_resolve_processor_dir(base_checkpoint))
    postrl_fp = _processor_fingerprint(_resolve_processor_dir(postrl_checkpoint))
    if base_fp != postrl_fp:
        raise RuntimeError("Base ACT and Post-RL deployment processor bundles differ.")

    sim_config.inference.policy_type = "act"
    control_checkpoint = (
        postrl_checkpoint if args.control_policy == "postrl" else base_checkpoint
    )
    sim_config.inference.pretrained_path = str(control_checkpoint)
    sim_config.inference.method = f"{sim_config.inference.method}_critic_probe"
    sim_config.inference.timestamp = f"{sim_config.inference.timestamp}_{stamp}"

    with (output_dir / "run_meta.json").open("w") as file:
        json.dump({
            "sim_config": str(sim_config_path),
            "rl_config": str(rl_config_path),
            "stage1_dir": str(stage1_dir),
            "base_checkpoint": str(base_checkpoint),
            "postrl_checkpoint": str(postrl_checkpoint),
            "control_policy": args.control_policy,
            "processor_fingerprint": base_fp,
            "chunk_size": int(rl_cfg.n_action_steps),
            "critic_contract": "fresh full normalized ACT chunk [H,D]",
            "partial_completion_step": args.partial_completion_step,
        }, file, indent=2)

    _arm = ArmMove(sim_config)
    from kuavo_deploy.src.eval import sim_auto_test as sim_eval

    base_setup_policy = sim_eval.setup_policy
    base_run_single_episode = sim_eval.run_single_episode
    base_policy = base_setup_policy(base_checkpoint, "act", sim_config.inference, device)
    postrl_policy = base_setup_policy(postrl_checkpoint, "act", sim_config.inference, device)
    if int(base_policy.config.chunk_size) != int(rl_cfg.n_action_steps):
        raise RuntimeError(
            f"Deploy ACT chunk={base_policy.config.chunk_size} but RL config "
            f"n_action_steps={rl_cfg.n_action_steps}"
        )

    probe = CriticProbePolicy(
        base_policy=base_policy,
        postrl_policy=postrl_policy,
        critic=workspace.critic,
        stats=workspace.stats,
        control_policy=args.control_policy,
        csv_path=output_dir / "critic_probe_steps.csv",
        chunk_path=output_dir / "counterfactual_chunks.npz",
        probe_every=args.probe_every,
        partial_completion_step=args.partial_completion_step,
    ).eval().to(device)

    def _setup(_pretrained_path, policy_type, cfg, device=device):
        if policy_type != "act":
            raise ValueError("Critic probe supports ACT only.")
        return probe

    def _run_single_episode(config, policy, preprocessor, postprocessor, episode, output_directory):
        result = base_run_single_episode(
            config, policy, preprocessor, postprocessor, episode, output_directory
        )
        probe.record_episode_result(episode, result)
        return result

    sim_eval.setup_policy = _setup
    sim_eval.run_single_episode = _run_single_episode
    try:
        print(f"Control policy: {args.control_policy}")
        print("Critic scores fresh full chunks; it does not score only the queued executed action.")
        print(f"Probe output: {output_dir}")
        sim_eval.kuavo_eval_autotest(sim_config)
    finally:
        sim_eval.setup_policy = base_setup_policy
        sim_eval.run_single_episode = base_run_single_episode
        probe.close()


if __name__ == "__main__":
    main()
