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
ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from analysis_05_terminal_advantage import (  # noqa: E402
    _processor_fingerprint,
    _resolve_config,
    _resolve_processor_dir,
)
from kuavo_deploy.config import load_kuavo_config  # noqa: E402
from kuavo_deploy.src.scripts.script_auto_test import ArmMove  # noqa: E402
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402


def _parse_slice(text: str) -> slice:
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("slice must be START:END")
    return slice(int(parts[0]), int(parts[1]))


def _stats_tensor(
    stats: dict,
    feature: str,
    name: str,
    ref: torch.Tensor,
) -> torch.Tensor:
    return torch.as_tensor(
        stats[feature][name],
        device=ref.device,
        dtype=ref.dtype,
    )


def _state_raw_and_hold(
    observation: dict[str, torch.Tensor],
    stats: dict,
    *,
    horizon: int,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if OBS_STATE not in observation:
        raise KeyError(
            f"MuJoCo policy observation has no {OBS_STATE!r}; "
            "cannot build hold chunk."
        )
    state = torch.as_tensor(observation[OBS_STATE])
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[-1] != action_dim:
        raise ValueError(
            "Critic probe assumes state/action coordinates match "
            "for the hold counterfactual; "
            f"state={tuple(state.shape)}, action_dim={action_dim}."
        )

    state_mean = _stats_tensor(
        stats,
        OBS_STATE,
        "mean",
        state,
    )
    state_std = _stats_tensor(
        stats,
        OBS_STATE,
        "std",
        state,
    )
    action_mean = _stats_tensor(
        stats,
        ACTION,
        "mean",
        state,
    )
    action_std = _stats_tensor(
        stats,
        ACTION,
        "std",
        state,
    )
    raw_state = (
        state * (state_std + 1e-8)
        + state_mean
    )
    hold_step = (
        (raw_state - action_mean)
        / (action_std + 1e-8)
    )
    hold = (
        hold_step[:, None, :]
        .expand(-1, horizon, -1)
        .contiguous()
    )
    return raw_state, hold


def _physical_action(
    normalized_action: torch.Tensor,
    stats: dict,
) -> torch.Tensor:
    mean = _stats_tensor(
        stats,
        ACTION,
        "mean",
        normalized_action,
    )
    std = _stats_tensor(
        stats,
        ACTION,
        "std",
        normalized_action,
    )
    return (
        normalized_action
        * (std + 1e-8)
        + mean
    )


def _group_rmse(
    delta: torch.Tensor,
    group: slice,
) -> float:
    value = delta[..., group].float()
    return float(
        torch.sqrt(
            value.square().mean()
        ).item()
    )


class MuJoCoCriticProbePolicy:
    """Execute Post-RL while probing Stage-1 Q/V on every live state.

    The returned action is exactly the deployed Post-RL policy's
    select_action output. Counterfactual full chunks are fresh
    deterministic plans on the same normalized observation and
    never affect the simulator trajectory.
    """

    def __init__(
        self,
        *,
        postrl_policy,
        base_policy,
        critic_workspace,
        csv_path: pathlib.Path,
        left_slice: slice,
        right_slice: slice,
        log_every: int,
    ) -> None:
        self.postrl_policy = postrl_policy
        self.base_policy = base_policy
        self.workspace = critic_workspace
        self.critic = critic_workspace.critic
        self.stats = critic_workspace.stats
        self.config = postrl_policy.config
        self.csv_path = csv_path
        self.left_slice = left_slice
        self.right_slice = right_slice
        self.log_every = max(
            int(log_every),
            1,
        )

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._file = csv_path.open(
            "w",
            newline="",
            buffering=1,
        )
        self._writer = None
        self._episode = -1
        self._step = 0
        self._has_steps = False
        self._rows = 0

    @property
    def rows(self) -> int:
        return self._rows

    def eval(self):
        self.postrl_policy.eval()
        self.base_policy.eval()
        return self

    def to(self, device):
        self.postrl_policy.to(device)
        self.base_policy.to(device)
        return self

    def reset(self):
        self.postrl_policy.reset()
        self.base_policy.reset()
        if self._episode < 0:
            self._episode = 0
        elif self._has_steps:
            self._episode += 1
        self._step = 0
        self._has_steps = False

    def _write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=list(row),
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        self._rows += 1

    @torch.inference_mode()
    def select_action(
        self,
        observation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # This is the only action returned to the MuJoCo loop.
        executed_action = (
            self.postrl_policy.select_action(
                observation
            )
        )

        # Fresh counterfactual plans on the exact current observation.
        # predict_action_chunk does not alter either ACT action queue.
        base_chunk = (
            self.base_policy.predict_action_chunk(
                observation
            )
        )
        postrl_chunk = (
            self.postrl_policy.predict_action_chunk(
                observation
            )
        )
        if base_chunk.shape != postrl_chunk.shape:
            raise RuntimeError(
                "Base/Post-RL fresh chunk shape mismatch: "
                f"{tuple(base_chunk.shape)} vs "
                f"{tuple(postrl_chunk.shape)}"
            )
        if postrl_chunk.ndim != 3:
            raise RuntimeError(
                "Expected ACT chunk [B,H,D], got "
                f"{tuple(postrl_chunk.shape)}"
            )

        batch_size, horizon, action_dim = (
            postrl_chunk.shape
        )
        if batch_size != 1:
            raise RuntimeError(
                "Kuavo MuJoCo critic probe currently expects "
                "one live environment; got policy batch "
                f"{batch_size}."
            )
        if horizon != int(
            self.workspace.cfg.n_action_steps
        ):
            raise RuntimeError(
                "Deploy/RL chunk mismatch: fresh policy chunk "
                f"has H={horizon}, Stage-1 Critic expects "
                f"H={self.workspace.cfg.n_action_steps}."
            )

        # Deployment preprocessor already normalized observation into
        # the LeRobot policy feature contract. Reuse the frozen Base
        # encoder directly; do not normalize the observation again.
        latent, _ = (
            self.workspace.model.encode_observation(
                observation
            )
        )
        critic_state = latent.mean(dim=1)
        value = self.critic.value(
            critic_state
        ).reshape(-1)

        raw_state, hold_chunk = (
            _state_raw_and_hold(
                observation,
                self.stats,
                horizon=horizon,
                action_dim=action_dim,
            )
        )

        actions = {
            "base": base_chunk,
            "postrl": postrl_chunk,
            "hold": hold_chunk,
        }
        if action_dim >= max(
            self.left_slice.stop,
            self.right_slice.stop,
        ):
            left_rl_right_base = (
                base_chunk.clone()
            )
            left_rl_right_base[
                ...,
                self.left_slice,
            ] = postrl_chunk[
                ...,
                self.left_slice,
            ]
            left_base_right_rl = (
                base_chunk.clone()
            )
            left_base_right_rl[
                ...,
                self.right_slice,
            ] = postrl_chunk[
                ...,
                self.right_slice,
            ]
            actions[
                "left_rl_right_base"
            ] = left_rl_right_base
            actions[
                "left_base_right_rl"
            ] = left_base_right_rl

        q = {
            name: self.critic.minQ(
                critic_state,
                action,
            ).reshape(-1)
            for name, action in actions.items()
        }
        advantage = {
            name: q_value - value
            for name, q_value in q.items()
        }
        q_scalar = {
            name: float(value_[0].item())
            for name, value_ in q.items()
        }
        best = max(
            q_scalar,
            key=q_scalar.get,
        )

        executed_action = torch.as_tensor(
            executed_action,
            device=postrl_chunk.device,
            dtype=postrl_chunk.dtype,
        )
        if (
            executed_action.ndim != 2
            or executed_action.shape
            != (1, action_dim)
        ):
            raise RuntimeError(
                "Expected deployed action "
                f"[1,{action_dim}], got "
                f"{tuple(executed_action.shape)}"
            )

        executed_phys = _physical_action(
            executed_action,
            self.stats,
        )
        fresh_first_phys = _physical_action(
            postrl_chunk[:, 0],
            self.stats,
        )
        base_first_phys = _physical_action(
            base_chunk[:, 0],
            self.stats,
        )
        postrl_phys = _physical_action(
            postrl_chunk,
            self.stats,
        )
        base_phys = _physical_action(
            base_chunk,
            self.stats,
        )
        hold_phys = _physical_action(
            hold_chunk,
            self.stats,
        )

        row = {
            "episode": self._episode,
            "step": self._step,
            "value": float(
                value[0].item()
            ),
            "q_base": q_scalar["base"],
            "a_base": float(
                advantage["base"][0].item()
            ),
            "q_postrl": q_scalar["postrl"],
            "a_postrl": float(
                advantage["postrl"][0].item()
            ),
            "q_hold": q_scalar["hold"],
            "a_hold": float(
                advantage["hold"][0].item()
            ),
            "q_left_rl_right_base": (
                q_scalar.get(
                    "left_rl_right_base",
                    float("nan"),
                )
            ),
            "q_left_base_right_rl": (
                q_scalar.get(
                    "left_base_right_rl",
                    float("nan"),
                )
            ),
            "q_postrl_minus_base": (
                q_scalar["postrl"]
                - q_scalar["base"]
            ),
            "q_hold_minus_base": (
                q_scalar["hold"]
                - q_scalar["base"]
            ),
            "critic_best_counterfactual": best,
            "postrl_base_chunk_rmse_phys": (
                _group_rmse(
                    postrl_phys - base_phys,
                    slice(0, action_dim),
                )
            ),
            "postrl_base_left_rmse_phys": (
                _group_rmse(
                    postrl_phys - base_phys,
                    self.left_slice,
                )
            ),
            "postrl_base_right_rmse_phys": (
                _group_rmse(
                    postrl_phys - base_phys,
                    self.right_slice,
                )
            ),
            "postrl_hold_chunk_rmse_phys": (
                _group_rmse(
                    postrl_phys - hold_phys,
                    slice(0, action_dim),
                )
            ),
            "base_hold_chunk_rmse_phys": (
                _group_rmse(
                    base_phys - hold_phys,
                    slice(0, action_dim),
                )
            ),
            "executed_vs_fresh_postrl_first_l2_phys": (
                float(
                    torch.linalg.vector_norm(
                        executed_phys
                        - fresh_first_phys
                    ).item()
                )
            ),
            "fresh_postrl_first_vs_state_l2_phys": (
                float(
                    torch.linalg.vector_norm(
                        fresh_first_phys
                        - raw_state
                    ).item()
                )
            ),
            "fresh_base_first_vs_state_l2_phys": (
                float(
                    torch.linalg.vector_norm(
                        base_first_phys
                        - raw_state
                    ).item()
                )
            ),
        }

        for index in range(action_dim):
            row[f"state_{index}"] = float(
                raw_state[0, index].item()
            )
            row[f"executed_action_{index}"] = float(
                executed_phys[0, index].item()
            )
            row[f"fresh_base_first_{index}"] = float(
                base_first_phys[0, index].item()
            )
            row[f"fresh_postrl_first_{index}"] = float(
                fresh_first_phys[0, index].item()
            )

        self._write(row)
        if self._step % self.log_every == 0:
            print(
                "[critic-probe] "
                f"ep={self._episode} "
                f"step={self._step} "
                f"V={row['value']:.4f} "
                f"Qbase={row['q_base']:.4f} "
                f"Qrl={row['q_postrl']:.4f} "
                f"Qhold={row['q_hold']:.4f} "
                f"best={best} "
                "RL-Hold-RMSE="
                f"{row['postrl_hold_chunk_rmse_phys']:.5f}"
            )

        self._step += 1
        self._has_steps = True
        return executed_action

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()
        print(
            f"Live Critic probe CSV: "
            f"{self.csv_path}"
        )


def _build_critic_workspace(
    *,
    stage1_dir: pathlib.Path,
    stage1_config: pathlib.Path | None,
    base_checkpoint: pathlib.Path,
    device: str,
    output_dir: pathlib.Path,
):
    config_path = _resolve_config(
        stage1_dir,
        (
            None
            if stage1_config is None
            else str(stage1_config)
        ),
    )
    cfg = OmegaConf.load(config_path)
    cfg.input.policy_checkpoint = str(
        base_checkpoint
    )
    cfg.input.policy_checkpoint_type = "il"
    cfg.training.device = str(device)
    cfg.use_wandb = False
    cfg.eval = False
    cfg.training.debug = False
    cfg.critic.load_pretrain = True
    cfg.critic.artifact_dir = str(
        stage1_dir / "critic"
    )

    if not bool(cfg.chunk_as_single_action):
        raise RuntimeError(
            "Live Critic probe currently requires "
            "chunk_as_single_action=true so Q(s,a) uses the "
            "same whole-chunk semantics as Stage-1 training."
        )

    workspace = TrainACTWorkspace(
        cfg,
        output_dir=str(
            output_dir / "critic_workspace"
        ),
    )
    workspace._build_act_observation_frontends()
    workspace._build_critic()
    if not workspace._load_critic_if_needed():
        raise RuntimeError(
            "Expected pretrained Stage-1 Critic; "
            "probe must not train Q/V."
        )
    workspace.critic.eval()
    return workspace, config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard Kuavo MuJoCo ACT evaluation "
            "with deterministic Post-RL control and probe "
            "the frozen Stage-1 Critic on Base/Post-RL/hold "
            "counterfactual chunks for every live simulator "
            "observation."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Kuavo MuJoCo deployment YAML.",
    )
    parser.add_argument(
        "--stage1-dir",
        required=True,
    )
    parser.add_argument(
        "--stage1-config",
        default=None,
        help=(
            "Optional resolved Offline-RL config override."
        ),
    )
    parser.add_argument(
        "--il-checkpoint",
        required=True,
        help="Deterministic Base ACT checkpoint.",
    )
    parser.add_argument(
        "--postrl-checkpoint",
        required=True,
        help=(
            "Deterministic exported Post-RL ACT checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "post_training/outputs/"
            "mujoco_critic_probe"
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )
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
    args = parser.parse_args()

    base_checkpoint = pathlib.Path(
        args.il_checkpoint
    ).expanduser().resolve()
    postrl_checkpoint = pathlib.Path(
        args.postrl_checkpoint
    ).expanduser().resolve()

    for label, checkpoint in (
        ("Base ACT", base_checkpoint),
        ("Post-RL", postrl_checkpoint),
    ):
        if (
            not (
                checkpoint / "config.json"
            ).is_file()
            or not (
                checkpoint / "model.safetensors"
            ).is_file()
        ):
            raise FileNotFoundError(
                f"{label} deterministic checkpoint "
                f"is incomplete: {checkpoint}"
            )

    base_processor = _resolve_processor_dir(
        base_checkpoint
    )
    postrl_processor = _resolve_processor_dir(
        postrl_checkpoint
    )
    base_fp = _processor_fingerprint(
        base_processor
    )
    postrl_fp = _processor_fingerprint(
        postrl_processor
    )
    if base_fp != postrl_fp:
        raise RuntimeError(
            "Base ACT and Post-RL processor "
            "bundles differ."
        )

    deploy_config_path = pathlib.Path(
        args.config
    ).expanduser().resolve()
    config = load_kuavo_config(
        deploy_config_path
    )
    if args.device is not None:
        config.inference.device = args.device
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError(
                "--episodes must be >= 1"
            )
        config.inference.eval_episodes = int(
            args.episodes
        )
    config.inference.policy_type = "act"
    config.inference.pretrained_path = str(
        postrl_checkpoint
    )

    stamp = dt.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    config.inference.method = (
        f"{config.inference.method}"
        "_postrl_critic_probe"
    )
    config.inference.timestamp = (
        f"{config.inference.timestamp}_{stamp}"
    )
    output_dir = (
        pathlib.Path(args.output_dir)
        .expanduser()
        .resolve()
        / stamp
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage1_dir = pathlib.Path(
        args.stage1_dir
    ).expanduser().resolve()
    stage1_config = (
        None
        if args.stage1_config is None
        else pathlib.Path(
            args.stage1_config
        ).expanduser().resolve()
    )
    (
        workspace,
        resolved_stage1_config,
    ) = _build_critic_workspace(
        stage1_dir=stage1_dir,
        stage1_config=stage1_config,
        base_checkpoint=base_checkpoint,
        device=str(
            config.inference.device
        ),
        output_dir=output_dir,
    )

    # Initialize the same ROS/control environment used by the
    # existing rollout diagnostics.
    _arm = ArmMove(config)
    from kuavo_deploy.src.eval import (
        sim_auto_test as sim_eval,
    )

    device = torch.device(
        config.inference.device
    )
    base_setup_policy = sim_eval.setup_policy
    base_policy = base_setup_policy(
        base_checkpoint,
        "act",
        config.inference,
        device,
    )
    postrl_policy = base_setup_policy(
        postrl_checkpoint,
        "act",
        config.inference,
        device,
    )

    postrl_encoder = ACTCriticEncoder(
        postrl_policy.model,
        copy_model=False,
    ).to(device).eval()
    if (
        workspace._fingerprint_module(
            postrl_encoder
        )
        != workspace._encoder_sha256
    ):
        raise RuntimeError(
            "Deployed Post-RL encoder differs from "
            "the Stage-1/Base encoder contract."
        )
    if int(
        postrl_policy.config.chunk_size
    ) != int(
        workspace.cfg.n_action_steps
    ):
        raise RuntimeError(
            "Post-RL deploy/RL chunk mismatch: "
            f"deploy={postrl_policy.config.chunk_size}, "
            f"critic={workspace.cfg.n_action_steps}."
        )

    csv_path = (
        output_dir / "live_critic_probe.csv"
    )
    probe_policy = MuJoCoCriticProbePolicy(
        postrl_policy=postrl_policy,
        base_policy=base_policy,
        critic_workspace=workspace,
        csv_path=csv_path,
        left_slice=args.left_action_slice,
        right_slice=args.right_action_slice,
        log_every=args.log_every,
    ).eval().to(device)

    def _setup(
        _pretrained_path,
        policy_type,
        cfg,
        device=device,
    ):
        if policy_type != "act":
            raise ValueError(
                "MuJoCo Critic probe supports ACT only."
            )
        return probe_policy

    with (
        output_dir / "run_meta.json"
    ).open("w") as file:
        json.dump(
            {
                "deploy_config": str(
                    deploy_config_path
                ),
                "resolved_stage1_config": str(
                    resolved_stage1_config
                ),
                "stage1_dir": str(
                    stage1_dir
                ),
                "base_checkpoint": str(
                    base_checkpoint
                ),
                "postrl_checkpoint": str(
                    postrl_checkpoint
                ),
                "processor_fingerprint": (
                    base_fp
                ),
                "device": str(
                    config.inference.device
                ),
                "episodes": int(
                    config.inference.eval_episodes
                ),
                "chunk_size": int(
                    workspace.cfg.n_action_steps
                ),
                "executed_policy": (
                    "postrl_select_action"
                ),
                "counterfactual_policy_chunks": [
                    "fresh_base",
                    "fresh_postrl",
                    "hold",
                    "left_postrl_right_base",
                    "left_base_right_postrl",
                ],
                "causal_limit": (
                    "This probe diagnoses the learned Critic "
                    "ranking on live states. It does not by "
                    "itself prove that PPO gradients caused "
                    "final Actor drift; Analysis 10 and 11 "
                    "test that mechanism."
                ),
            },
            file,
            indent=2,
        )

    sim_eval.setup_policy = _setup
    try:
        print(
            "Processor bundles are bit-identical: "
            f"{base_fp[:12]}..."
        )
        print(
            "Control policy: deterministic Post-RL; "
            "Base/Hold are Critic-only counterfactuals."
        )
        print(
            f"Per-step Q/V/A diagnostics: "
            f"{csv_path}"
        )
        sim_eval.kuavo_eval_autotest(
            config
        )
    finally:
        sim_eval.setup_policy = base_setup_policy
        probe_policy.close()

    if probe_policy.rows == 0:
        raise RuntimeError(
            "MuJoCo rollout produced zero "
            "Critic-probe rows."
        )
    print(
        f"Saved {probe_policy.rows} live "
        f"Critic-probe rows to: {csv_path}"
    )


if __name__ == "__main__":
    main()
