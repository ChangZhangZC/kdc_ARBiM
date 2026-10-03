from __future__ import annotations

import argparse
import csv
import inspect
import json
import pathlib
import sys
from typing import Any

import hydra
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
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from analysis_05_terminal_advantage import (  # noqa: E402
    _load_postrl_policy,
    _processor_fingerprint,
    _resolve_config,
    _resolve_processor_dir,
    _resolve_required_path,
)
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


def _parse_slice(text: str) -> slice:
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("slice must be START:END")
    return slice(int(parts[0]), int(parts[1]))


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _tensor_stats(delta: torch.Tensor, group: slice) -> dict[str, float]:
    x = delta[..., group].reshape(delta.shape[0], -1).float()
    return {
        "mae": float(x.abs().mean().item()),
        "rmse": float(torch.sqrt(x.square().mean()).item()),
        "max_abs": float(x.abs().max().item()),
    }


def _stats_tensor(stats: dict, feature: str, name: str, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        stats[feature][name],
        device=ref.device,
        dtype=ref.dtype,
    )


def _hold_chunk_from_policy_obs(
    policy_obs: dict[str, torch.Tensor],
    stats: dict,
    *,
    horizon: int,
    action_dim: int,
) -> torch.Tensor | None:
    state = policy_obs.get(OBS_STATE)
    if state is None:
        return None
    state = torch.as_tensor(state)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[-1] != action_dim:
        return None

    # The env runner receives/constructs the same normalized policy batch used by
    # StochasticACTPolicyWrapper. Convert normalized observation.state back to raw
    # robot coordinates, then normalize those coordinates with the ACTION stats.
    state_mean = _stats_tensor(stats, OBS_STATE, "mean", state)
    state_std = _stats_tensor(stats, OBS_STATE, "std", state)
    action_mean = _stats_tensor(stats, ACTION, "mean", state)
    action_std = _stats_tensor(stats, ACTION, "std", state)
    raw_state = state * (state_std + 1e-8) + state_mean
    hold_step = (raw_state - action_mean) / (action_std + 1e-8)
    return hold_step[:, None, :].expand(-1, horizon, -1).contiguous()


class CriticProbePolicy:
    """Deterministic Post-RL policy wrapper that logs live Q/V/A at each replan.

    The wrapped env runner still executes the Post-RL deterministic mean. The
    diagnostic only adds counterfactual Critic evaluations for the exact same
    observation passed to the policy.
    """

    def __init__(
        self,
        *,
        active_policy,
        base_policy,
        critic,
        stats: dict,
        output_csv: pathlib.Path,
        left_slice: slice,
        right_slice: slice,
        print_every: int,
    ) -> None:
        self._active_policy = active_policy
        self._base_policy = base_policy
        self._critic = critic
        self._stats = stats
        self._output_csv = output_csv
        self._left_slice = left_slice
        self._right_slice = right_slice
        self._print_every = max(1, int(print_every))
        self._probe_call = 0
        self._header_written = output_csv.is_file() and output_csv.stat().st_size > 0

    def __getattr__(self, name: str):
        return getattr(self._active_policy, name)

    def eval(self):
        self._active_policy.eval()
        self._base_policy.eval()
        return self

    @property
    def probe_calls(self) -> int:
        return self._probe_call

    def _append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._output_csv.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)

    @torch.no_grad()
    def _record(
        self,
        policy_obs: dict[str, torch.Tensor],
        executed_chunk: torch.Tensor,
    ) -> None:
        if not isinstance(policy_obs, dict):
            raise TypeError(
                "Critic probe expects the env runner to call the ACT policy "
                "with a dict observation batch."
            )
        if "latent" in policy_obs:
            raise RuntimeError(
                "Live MuJoCo probe received cached latent input. It needs the current "
                "normalized policy observation so a physical hold counterfactual can "
                "be constructed."
            )

        base_chunk = self._base_policy.get_action_mean(policy_obs)
        rl_chunk = self._active_policy.get_action_mean(policy_obs)
        executed_chunk = torch.as_tensor(
            executed_chunk,
            device=rl_chunk.device,
            dtype=rl_chunk.dtype,
        )
        if executed_chunk.shape != rl_chunk.shape:
            raise ValueError(
                f"Executed policy chunk shape {tuple(executed_chunk.shape)} != "
                f"deterministic Post-RL chunk {tuple(rl_chunk.shape)}"
            )

        latent, _ = self._base_policy.encode_observation(policy_obs)
        critic_state = latent.mean(dim=1)
        value = self._critic.value(critic_state).reshape(-1)
        horizon = int(rl_chunk.shape[1])
        action_dim = int(rl_chunk.shape[2])
        hold_chunk = _hold_chunk_from_policy_obs(
            policy_obs,
            self._stats,
            horizon=horizon,
            action_dim=action_dim,
        )

        actions = {
            "base": base_chunk,
            "postrl": rl_chunk,
        }
        if hold_chunk is not None:
            actions["hold"] = hold_chunk
        if action_dim >= max(self._left_slice.stop, self._right_slice.stop):
            left_rl_right_base = base_chunk.clone()
            left_rl_right_base[..., self._left_slice] = rl_chunk[..., self._left_slice]
            left_base_right_rl = base_chunk.clone()
            left_base_right_rl[..., self._right_slice] = rl_chunk[..., self._right_slice]
            actions["left_rl_right_base"] = left_rl_right_base
            actions["left_base_right_rl"] = left_base_right_rl

        q_values = {
            name: self._critic.minQ(critic_state, action).reshape(-1)
            for name, action in actions.items()
        }
        advantages = {name: q - value for name, q in q_values.items()}

        batch_size = int(rl_chunk.shape[0])
        rows = []
        delta_rl_base = rl_chunk - base_chunk
        for sample in range(batch_size):
            q_sample = {
                name: float(q[sample].item())
                for name, q in q_values.items()
            }
            best_action = max(q_sample, key=q_sample.get)
            row = {
                "probe_call": self._probe_call,
                "sample": sample,
                "value": float(value[sample].item()),
                "q_base": q_sample["base"],
                "a_base": float(advantages["base"][sample].item()),
                "q_postrl": q_sample["postrl"],
                "a_postrl": float(advantages["postrl"][sample].item()),
                "q_hold": q_sample.get("hold", float("nan")),
                "a_hold": (
                    float(advantages["hold"][sample].item())
                    if "hold" in advantages
                    else float("nan")
                ),
                "q_left_rl_right_base": q_sample.get(
                    "left_rl_right_base",
                    float("nan"),
                ),
                "q_left_base_right_rl": q_sample.get(
                    "left_base_right_rl",
                    float("nan"),
                ),
                "q_postrl_minus_base": q_sample["postrl"] - q_sample["base"],
                "q_hold_minus_base": (
                    q_sample.get("hold", float("nan")) - q_sample["base"]
                ),
                "critic_best_counterfactual": best_action,
            }
            for label, group in (
                ("all", slice(0, action_dim)),
                ("left", self._left_slice),
                ("right", self._right_slice),
            ):
                stats = _tensor_stats(
                    delta_rl_base[sample:sample + 1],
                    group,
                )
                row[f"postrl_base_{label}_mae"] = stats["mae"]
                row[f"postrl_base_{label}_rmse"] = stats["rmse"]
                row[f"postrl_base_{label}_max_abs"] = stats["max_abs"]
            if hold_chunk is not None:
                delta_hold = (
                    rl_chunk[sample:sample + 1]
                    - hold_chunk[sample:sample + 1]
                )
                row["postrl_hold_all_rmse"] = _tensor_stats(
                    delta_hold,
                    slice(0, action_dim),
                )["rmse"]
            else:
                row["postrl_hold_all_rmse"] = float("nan")
            rows.append(row)

        self._append_rows(rows)
        if self._probe_call % self._print_every == 0:
            first = rows[0]
            print(
                "[critic-probe] "
                f"replan={self._probe_call} "
                f"V={first['value']:.4f} "
                f"Qbase={first['q_base']:.4f} "
                f"Qrl={first['q_postrl']:.4f} "
                f"Qhold={first['q_hold']:.4f} "
                f"best={first['critic_best_counterfactual']} "
                f"RL-Base-RMSE={first['postrl_base_all_rmse']:.5f}"
            )
        self._probe_call += 1

    @torch.no_grad()
    def get_action_mean(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action = self._active_policy.get_action_mean(batch)
        self._record(batch, action)
        return action

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action = self._active_policy.predict_action_chunk(batch)
        self._record(batch, action)
        return action


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the configured MuJoCo/env_runner with the deterministic Post-RL "
            "ACT policy while probing the frozen Stage-1 IQL Critic on Base, "
            "Post-RL, hold, and bimanual counterfactual action chunks at every "
            "ACT replan."
        )
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override Base/IL ACT checkpoint.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-times", type=int, default=1)
    parser.add_argument("--eval-env-num", type=int, default=None)
    parser.add_argument(
        "--left-action-slice",
        type=_parse_slice,
        default=slice(0, 8),
    )
    parser.add_argument(
        "--right-action-slice",
        type=_parse_slice,
        default=slice(8, 16),
    )
    parser.add_argument("--print-every", type=int, default=1)
    args = parser.parse_args()

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
    postrl_checkpoint = _resolve_required_path(
        args.postrl_checkpoint,
        None,
        "Post-RL checkpoint",
    )
    cfg.input.policy_checkpoint = str(checkpoint)
    cfg.input.policy_checkpoint_type = "il"
    cfg.training.device = str(args.device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    cfg.critic.load_pretrain = True
    cfg.critic.artifact_dir = str(stage1_dir / "critic")

    if not bool(cfg.chunk_as_single_action):
        raise RuntimeError(
            "MuJoCo critic probe currently requires chunk_as_single_action=true "
            "so Q(s,a) has the same whole-ACT-chunk semantics used by the "
            "trained Stage-1 Critic."
        )

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "post_training" / "outputs" / "mujoco_critic_probe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = TrainACTWorkspace(cfg, output_dir=str(output_dir))
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError(
            "Expected pretrained Stage-1 Q/V; probe must not train the Critic."
        )
    workspace.critic.eval()

    postrl_policy, postrl_kind = _load_postrl_policy(
        postrl_checkpoint,
        workspace.device,
        cfg,
    )
    il_processor = _resolve_processor_dir(checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_processor_fp = _processor_fingerprint(il_processor)
    postrl_processor_fp = _processor_fingerprint(postrl_processor)
    if il_processor_fp != postrl_processor_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor bundles differ; refusing live Q/V "
            "comparison."
        )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(workspace.device).eval()
    if workspace._fingerprint_module(postrl_encoder) != workspace._encoder_sha256:
        raise RuntimeError(
            "Post-RL encoder differs from the Stage-1/Base encoder contract."
        )

    task_cfg = cfg.get("task")
    env_runner_cfg = None if task_cfg is None else task_cfg.get("env_runner")
    if env_runner_cfg is None:
        raise RuntimeError(
            "Resolved config has no task.env_runner. Use the same resolved "
            "Stage-2 config that you normally use for MuJoCo evaluation, or "
            "pass it with --config."
        )
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        output_dir=str(output_dir),
    )

    probe_csv = output_dir / "live_critic_probe.csv"
    if probe_csv.exists():
        probe_csv.unlink()
    probe_policy = CriticProbePolicy(
        active_policy=postrl_policy,
        base_policy=workspace.model,
        critic=workspace.critic,
        stats=workspace.stats,
        output_csv=probe_csv,
        left_slice=args.left_action_slice,
        right_slice=args.right_action_slice,
        print_every=args.print_every,
    ).eval()

    try:
        run_params = inspect.signature(env_runner.run).parameters
    except (TypeError, ValueError):
        run_params = {}
    kwargs = {}
    if "eval_env_num" in run_params:
        kwargs["eval_env_num"] = int(
            args.eval_env_num
            if args.eval_env_num is not None
            else cfg.ppo.eval_env_num
        )

    results = []
    for episode in range(int(args.eval_times)):
        print(
            f"\n=== MuJoCo Critic Probe rollout "
            f"{episode + 1}/{args.eval_times} ==="
        )
        result = env_runner.run(probe_policy, **kwargs)
        results.append(_jsonable(result))

    if probe_policy.probe_calls == 0:
        raise RuntimeError(
            "env_runner completed without calling get_action_mean/"
            "predict_action_chunk on the probe wrapper. This runner bypasses "
            "the standard ACT policy API; instrument its policy call site before "
            "interpreting Critic results."
        )

    summary = {
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base_checkpoint": str(checkpoint),
        "postrl_checkpoint": str(postrl_checkpoint),
        "postrl_checkpoint_kind": postrl_kind,
        "processor_fingerprint": il_processor_fp,
        "probe_calls": probe_policy.probe_calls,
        "chunk_as_single_action": bool(cfg.chunk_as_single_action),
        "chunk_size": int(cfg.n_action_steps),
        "left_action_slice": [
            args.left_action_slice.start,
            args.left_action_slice.stop,
        ],
        "right_action_slice": [
            args.right_action_slice.start,
            args.right_action_slice.stop,
        ],
        "rollout_results": results,
        "interpretation_contract": {
            "value": (
                "V(s) from the frozen Stage-1 IQL value network at the exact "
                "live policy observation."
            ),
            "q_base": "Q(s, deterministic Base-ACT whole action chunk).",
            "q_postrl": (
                "Q(s, deterministic Post-RL whole action chunk actually "
                "executed by the wrapper)."
            ),
            "q_hold": (
                "Q(s, current observation.state repeated as a position-hold "
                "chunk after converting state normalization to action "
                "normalization)."
            ),
            "counterfactuals": (
                "Swap only the configured left or right action dimensions "
                "between Base and Post-RL chunks."
            ),
            "causal_limit": (
                "Critic misranking on a live state diagnoses the learned Q "
                "surface; by itself it does not prove PPO gradients caused the "
                "final Actor drift. Analysis 10/11 test that training mechanism."
            ),
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=True)
    print(f"\nProbe rows: {probe_policy.probe_calls}; saved to {probe_csv}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
