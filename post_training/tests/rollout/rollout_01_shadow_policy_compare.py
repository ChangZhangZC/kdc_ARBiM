from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "third_party" / "lerobot" / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from kuavo_deploy.config import load_kuavo_config  # noqa: E402
from kuavo_deploy.src.scripts.script_auto_test import ArmMove  # noqa: E402


def _resolve_processor_dir(checkpoint: pathlib.Path) -> pathlib.Path:
    checked = []
    for candidate in (checkpoint, checkpoint.parent):
        required = (
            candidate / "policy_preprocessor.json",
            candidate / "policy_postprocessor.json",
        )
        checked.extend(str(path) for path in required)
        if all(path.is_file() for path in required):
            return candidate
    raise FileNotFoundError(
        "Could not resolve processor bundle for checkpoint. Checked:\n- "
        + "\n- ".join(checked)
    )


def _processor_fingerprint(processor_dir: pathlib.Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        [
            *processor_dir.glob("policy_preprocessor*"),
            *processor_dir.glob("policy_postprocessor*"),
        ],
        key=lambda path: path.name,
    )
    for path in files:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_action_norm(processor_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with (processor_dir / "policy_preprocessor.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    normalizer = None
    for step in config.get("steps", []):
        if step.get("registry_name") == "normalizer_processor" or str(
            step.get("class", "")
        ).endswith("NormalizerProcessorStep"):
            normalizer = step
            break
    if normalizer is None or not normalizer.get("state_file"):
        raise KeyError("policy_preprocessor.json does not define a normalizer state_file")
    state = load_file(str(processor_dir / normalizer["state_file"]), device="cpu")
    for key in ("action.mean", "action.std"):
        if key not in state:
            raise KeyError(f"Normalizer state is missing {key!r}")
    mean = state["action.mean"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    std = state["action.std"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    if np.any(std <= 0):
        raise ValueError("action.std must be strictly positive")
    return mean, std


def _group_metrics(delta: np.ndarray) -> dict[str, float]:
    if delta.size != 16:
        return {
            "left_arm_policy_delta_l2": float("nan"),
            "left_gripper_policy_abs_delta": float("nan"),
            "right_arm_policy_delta_l2": float("nan"),
            "right_gripper_policy_abs_delta": float("nan"),
        }
    return {
        "left_arm_policy_delta_l2": float(np.linalg.norm(delta[0:7])),
        "left_gripper_policy_abs_delta": float(abs(delta[7])),
        "right_arm_policy_delta_l2": float(np.linalg.norm(delta[8:15])),
        "right_gripper_policy_abs_delta": float(abs(delta[15])),
    }


class ShadowComparePolicy:
    """Execute one deterministic ACT while shadow-running the other on identical inputs."""

    def __init__(
        self,
        primary,
        reference,
        *,
        primary_name: str,
        reference_name: str,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        csv_path: pathlib.Path,
        log_every: int,
    ) -> None:
        self.primary = primary
        self.reference = reference
        self.primary_name = primary_name
        self.reference_name = reference_name
        self.config = primary.config
        self.action_mean = np.asarray(action_mean, dtype=np.float64)
        self.action_std = np.asarray(action_std, dtype=np.float64)
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_every = max(int(log_every), 1)
        self._file = self.csv_path.open("w", newline="", buffering=1)
        self._writer = None
        self._episode = -1
        self._step = 0
        self._has_steps = False
        self._prev_primary = None
        self._prev_reference = None
        self._policy_delta_l2 = []
        self._primary_step_delta_l2 = []
        self._reference_step_delta_l2 = []

    def eval(self):
        self.primary.eval()
        self.reference.eval()
        return self

    def to(self, device):
        self.primary.to(device)
        self.reference.to(device)
        return self

    def reset(self):
        self.primary.reset()
        self.reference.reset()
        if self._episode < 0:
            self._episode = 0
        elif self._has_steps:
            self._episode += 1
        self._step = 0
        self._has_steps = False
        self._prev_primary = None
        self._prev_reference = None

    @torch.inference_mode()
    def select_action(self, observation):
        primary_action = self.primary.select_action(observation)
        reference_action = self.reference.select_action(observation)
        if tuple(primary_action.shape) != tuple(reference_action.shape):
            raise RuntimeError(
                "Primary/reference action shape mismatch: "
                f"{tuple(primary_action.shape)} vs {tuple(reference_action.shape)}"
            )
        if primary_action.ndim != 2 or primary_action.shape[0] != 1:
            raise RuntimeError(
                f"Expected deployed ACT action [1,D], got {tuple(primary_action.shape)}"
            )

        primary_norm = primary_action[0].detach().float().cpu().numpy().astype(np.float64)
        reference_norm = reference_action[0].detach().float().cpu().numpy().astype(np.float64)
        if primary_norm.shape != self.action_mean.shape:
            raise RuntimeError(
                f"Action dim {primary_norm.shape} != normalizer {self.action_mean.shape}"
            )
        primary_phys = primary_norm * self.action_std + self.action_mean
        reference_phys = reference_norm * self.action_std + self.action_mean
        policy_delta = primary_phys - reference_phys
        policy_delta_l2 = float(np.linalg.norm(policy_delta))
        policy_delta_mae = float(np.mean(np.abs(policy_delta)))
        policy_delta_max = float(np.max(np.abs(policy_delta)))
        primary_step_delta = (
            float("nan")
            if self._prev_primary is None
            else float(np.linalg.norm(primary_phys - self._prev_primary))
        )
        reference_step_delta = (
            float("nan")
            if self._prev_reference is None
            else float(np.linalg.norm(reference_phys - self._prev_reference))
        )
        row = {
            "episode": self._episode,
            "step": self._step,
            "executed_policy": self.primary_name,
            "shadow_policy": self.reference_name,
            "policy_delta_l2": policy_delta_l2,
            "policy_delta_mae": policy_delta_mae,
            "policy_delta_max_abs": policy_delta_max,
            "executed_step_delta_l2": primary_step_delta,
            "shadow_step_delta_l2": reference_step_delta,
            **_group_metrics(policy_delta),
        }
        for index, value in enumerate(primary_phys):
            row[f"executed_action_{index}"] = float(value)
        for index, value in enumerate(reference_phys):
            row[f"shadow_action_{index}"] = float(value)
        for index, value in enumerate(policy_delta):
            row[f"policy_delta_{index}"] = float(value)

        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

        self._policy_delta_l2.append(policy_delta_l2)
        if math.isfinite(primary_step_delta):
            self._primary_step_delta_l2.append(primary_step_delta)
        if math.isfinite(reference_step_delta):
            self._reference_step_delta_l2.append(reference_step_delta)
        if self._step % self.log_every == 0:
            print(
                f"[shadow] episode={self._episode} step={self._step} "
                f"policy_delta_l2={policy_delta_l2:.6g} "
                f"{self.primary_name}_step_delta={primary_step_delta:.6g} "
                f"{self.reference_name}_step_delta={reference_step_delta:.6g}"
            )

        self._prev_primary = primary_phys.copy()
        self._prev_reference = reference_phys.copy()
        self._step += 1
        self._has_steps = True
        return primary_action

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()
        print(f"\nShadow comparison CSV: {self.csv_path}")
        if self._policy_delta_l2:
            print(
                "Mean executed-vs-shadow policy L2: "
                f"{float(np.mean(self._policy_delta_l2)):.6g}"
            )
        if self._primary_step_delta_l2:
            print(
                f"Mean {self.primary_name} step-to-step L2: "
                f"{float(np.mean(self._primary_step_delta_l2)):.6g}"
            )
        if self._reference_step_delta_l2:
            print(
                f"Mean {self.reference_name} step-to-step L2: "
                f"{float(np.mean(self._reference_step_delta_l2)):.6g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the normal Kuavo simulator rollout with one deterministic ACT policy "
            "executed and the other shadow-run on the exact same preprocessed observations. "
            "Use this to diagnose Post-RL fixed-point/OOD behavior without letting the shadow "
            "policy affect the robot trajectory."
        )
    )
    parser.add_argument("--config", required=True, help="Kuavo deployment YAML")
    parser.add_argument("--il-checkpoint", required=True, help="Deterministic IL ACT checkpoint")
    parser.add_argument(
        "--postrl-checkpoint",
        required=True,
        help="Exported deterministic Post-RL ACT checkpoint (epochbest), not stochastic best_ope",
    )
    parser.add_argument(
        "--execute",
        choices=("postrl", "il"),
        default="postrl",
        help="Policy that controls the simulator; the other policy is shadow-only.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="post_training/outputs/shadow_policy_compare",
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    il_checkpoint = pathlib.Path(args.il_checkpoint).expanduser().resolve()
    postrl_checkpoint = pathlib.Path(args.postrl_checkpoint).expanduser().resolve()
    for label, path in (("IL", il_checkpoint), ("Post-RL", postrl_checkpoint)):
        if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"{label} deterministic checkpoint is incomplete: {path}")

    il_processor = _resolve_processor_dir(il_checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_fp = _processor_fingerprint(il_processor)
    postrl_fp = _processor_fingerprint(postrl_processor)
    if il_fp != postrl_fp:
        raise RuntimeError(
            "IL and exported Post-RL processor bundles differ. Shadow comparison requires "
            "the same preprocessing/postprocessing contract so only policy weights differ."
        )
    action_mean, action_std = _load_action_norm(il_processor)

    config = load_kuavo_config(pathlib.Path(args.config).expanduser().resolve())
    if args.device is not None:
        config.inference.device = args.device
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("--episodes must be >= 1")
        config.inference.eval_episodes = int(args.episodes)
    config.inference.policy_type = "act"

    if args.execute == "postrl":
        primary_checkpoint = postrl_checkpoint
        reference_checkpoint = il_checkpoint
        primary_name = "postrl"
        reference_name = "il"
    else:
        primary_checkpoint = il_checkpoint
        reference_checkpoint = postrl_checkpoint
        primary_name = "il"
        reference_name = "postrl"
    config.inference.pretrained_path = str(primary_checkpoint)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_method = str(config.inference.method)
    original_timestamp = str(config.inference.timestamp)
    config.inference.method = f"{original_method}_shadow_compare"
    config.inference.timestamp = f"{original_timestamp}_{stamp}"

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shadow_action_comparison.csv"
    metadata = {
        "il_checkpoint": str(il_checkpoint),
        "postrl_checkpoint": str(postrl_checkpoint),
        "execute": args.execute,
        "processor_fingerprint": il_fp,
        "config": str(pathlib.Path(args.config).expanduser().resolve()),
        "device": str(config.inference.device),
        "episodes": int(config.inference.eval_episodes),
        "action_dim": int(action_mean.size),
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    # Match the normal deployment entry point: ArmMove initializes the ROS node and signals.
    _arm = ArmMove(config)
    from kuavo_deploy.src.eval import sim_auto_test as sim_eval

    device = torch.device(config.inference.device)
    base_setup_policy = sim_eval.setup_policy
    primary = base_setup_policy(primary_checkpoint, "act", config.inference, device)
    reference = base_setup_policy(reference_checkpoint, "act", config.inference, device)
    comparator = ShadowComparePolicy(
        primary,
        reference,
        primary_name=primary_name,
        reference_name=reference_name,
        action_mean=action_mean,
        action_std=action_std,
        csv_path=csv_path,
        log_every=args.log_every,
    ).eval().to(device)

    def _shadow_setup_policy(_pretrained_path, policy_type, cfg, device=device):
        if policy_type != "act":
            raise ValueError("Shadow comparison supports ACT only")
        return comparator

    sim_eval.setup_policy = _shadow_setup_policy
    try:
        print(f"Processor bundles are bit-identical: {il_fp[:12]}...")
        print(f"Executing {primary_name}; shadow-running {reference_name}")
        print(f"Per-step diagnostics: {csv_path}")
        sim_eval.kuavo_eval_autotest(config)
    finally:
        sim_eval.setup_policy = base_setup_policy
        comparator.close()


if __name__ == "__main__":
    main()
