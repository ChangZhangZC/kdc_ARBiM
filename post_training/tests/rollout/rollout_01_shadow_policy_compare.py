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
    for candidate in (checkpoint, checkpoint.parent):
        required = (
            candidate / "policy_preprocessor.json",
            candidate / "policy_postprocessor.json",
        )
        if all(path.is_file() for path in required):
            return candidate
    raise FileNotFoundError(f"Could not resolve processor bundle for {checkpoint}")


def _processor_fingerprint(processor_dir: pathlib.Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        [*processor_dir.glob("policy_preprocessor*"), *processor_dir.glob("policy_postprocessor*")],
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
    normalizer = next(
        (
            step
            for step in config.get("steps", [])
            if step.get("registry_name") == "normalizer_processor"
            or str(step.get("class", "")).endswith("NormalizerProcessorStep")
        ),
        None,
    )
    if normalizer is None or not normalizer.get("state_file"):
        raise KeyError("policy_preprocessor.json does not define a normalizer state_file")
    state = load_file(str(processor_dir / normalizer["state_file"]), device="cpu")
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


class ILShadowPostRLPolicy:
    """Execute deterministic IL while shadow-running deterministic Post-RL."""

    def __init__(
        self,
        il_policy,
        postrl_policy,
        *,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        csv_path: pathlib.Path,
        log_every: int,
    ) -> None:
        self.il_policy = il_policy
        self.postrl_policy = postrl_policy
        self.config = il_policy.config
        self.action_mean = np.asarray(action_mean, dtype=np.float64)
        self.action_std = np.asarray(action_std, dtype=np.float64)
        self.log_every = max(int(log_every), 1)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self._file = csv_path.open("w", newline="", buffering=1)
        self._writer = None
        self._episode = -1
        self._step = 0
        self._has_steps = False
        self._prev_il = None
        self._prev_postrl = None
        self._policy_delta_l2 = []
        self._il_step_delta_l2 = []
        self._postrl_step_delta_l2 = []

    def eval(self):
        self.il_policy.eval()
        self.postrl_policy.eval()
        return self

    def to(self, device):
        self.il_policy.to(device)
        self.postrl_policy.to(device)
        return self

    def reset(self):
        self.il_policy.reset()
        self.postrl_policy.reset()
        if self._episode < 0:
            self._episode = 0
        elif self._has_steps:
            self._episode += 1
        self._step = 0
        self._has_steps = False
        self._prev_il = None
        self._prev_postrl = None

    @torch.inference_mode()
    def select_action(self, observation):
        il_action = self.il_policy.select_action(observation)
        postrl_action = self.postrl_policy.select_action(observation)
        if tuple(il_action.shape) != tuple(postrl_action.shape):
            raise RuntimeError(
                f"IL/Post-RL action shape mismatch: {tuple(il_action.shape)} vs {tuple(postrl_action.shape)}"
            )
        if il_action.ndim != 2 or il_action.shape[0] != 1:
            raise RuntimeError(f"Expected deployed ACT action [1,D], got {tuple(il_action.shape)}")

        il_norm = il_action[0].detach().float().cpu().numpy().astype(np.float64)
        postrl_norm = postrl_action[0].detach().float().cpu().numpy().astype(np.float64)
        if il_norm.shape != self.action_mean.shape:
            raise RuntimeError(f"Action dim {il_norm.shape} != normalizer {self.action_mean.shape}")
        il_phys = il_norm * self.action_std + self.action_mean
        postrl_phys = postrl_norm * self.action_std + self.action_mean
        policy_delta = postrl_phys - il_phys
        policy_delta_l2 = float(np.linalg.norm(policy_delta))
        il_step_delta = (
            float("nan")
            if self._prev_il is None
            else float(np.linalg.norm(il_phys - self._prev_il))
        )
        postrl_step_delta = (
            float("nan")
            if self._prev_postrl is None
            else float(np.linalg.norm(postrl_phys - self._prev_postrl))
        )

        row = {
            "episode": self._episode,
            "step": self._step,
            "executed_policy": "il",
            "shadow_policy": "postrl",
            "policy_delta_l2": policy_delta_l2,
            "policy_delta_mae": float(np.mean(np.abs(policy_delta))),
            "policy_delta_max_abs": float(np.max(np.abs(policy_delta))),
            "il_step_delta_l2": il_step_delta,
            "postrl_step_delta_l2": postrl_step_delta,
            **_group_metrics(policy_delta),
        }
        for index, value in enumerate(il_phys):
            row[f"il_action_{index}"] = float(value)
        for index, value in enumerate(postrl_phys):
            row[f"postrl_action_{index}"] = float(value)
        for index, value in enumerate(policy_delta):
            row[f"policy_delta_{index}"] = float(value)

        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

        self._policy_delta_l2.append(policy_delta_l2)
        if math.isfinite(il_step_delta):
            self._il_step_delta_l2.append(il_step_delta)
        if math.isfinite(postrl_step_delta):
            self._postrl_step_delta_l2.append(postrl_step_delta)
        if self._step % self.log_every == 0:
            print(
                f"[shadow] episode={self._episode} step={self._step} "
                f"control=il policy_delta_l2={policy_delta_l2:.6g} "
                f"il_step_delta={il_step_delta:.6g} postrl_step_delta={postrl_step_delta:.6g}"
            )

        self._prev_il = il_phys.copy()
        self._prev_postrl = postrl_phys.copy()
        self._step += 1
        self._has_steps = True
        return il_action

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()
        print(f"\nIL-control shadow comparison CSV: {self.csv_path}")
        if self._policy_delta_l2:
            print(f"Mean Post-RL-vs-IL policy L2: {float(np.mean(self._policy_delta_l2)):.6g}")
        if self._il_step_delta_l2:
            print(f"Mean IL step-to-step L2: {float(np.mean(self._il_step_delta_l2)):.6g}")
        if self._postrl_step_delta_l2:
            print(f"Mean Post-RL shadow step-to-step L2: {float(np.mean(self._postrl_step_delta_l2)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pure IL-controlled Kuavo rollout and shadow-run exported Post-RL ACT on the "
            "same observations. The Post-RL policy never affects the trajectory."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default="post_training/outputs/il_shadow_postrl")
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
        raise RuntimeError("IL and Post-RL processor bundles differ")
    action_mean, action_std = _load_action_norm(il_processor)

    config_path = pathlib.Path(args.config).expanduser().resolve()
    config = load_kuavo_config(config_path)
    if args.device is not None:
        config.inference.device = args.device
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("--episodes must be >= 1")
        config.inference.eval_episodes = int(args.episodes)
    config.inference.policy_type = "act"
    config.inference.pretrained_path = str(il_checkpoint)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.inference.method = f"{config.inference.method}_il_shadow_postrl"
    config.inference.timestamp = f"{config.inference.timestamp}_{stamp}"
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shadow_action_comparison.csv"
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": str(config_path),
                "il_checkpoint": str(il_checkpoint),
                "postrl_checkpoint": str(postrl_checkpoint),
                "executed_policy": "il",
                "shadow_policy": "postrl",
                "processor_fingerprint": il_fp,
                "device": str(config.inference.device),
                "episodes": int(config.inference.eval_episodes),
            },
            file,
            indent=2,
            sort_keys=True,
        )

    _arm = ArmMove(config)
    from kuavo_deploy.src.eval import sim_auto_test as sim_eval

    device = torch.device(config.inference.device)
    base_setup_policy = sim_eval.setup_policy
    il_policy = base_setup_policy(il_checkpoint, "act", config.inference, device)
    postrl_policy = base_setup_policy(postrl_checkpoint, "act", config.inference, device)
    comparator = ILShadowPostRLPolicy(
        il_policy,
        postrl_policy,
        action_mean=action_mean,
        action_std=action_std,
        csv_path=csv_path,
        log_every=args.log_every,
    ).eval().to(device)

    def _setup(_pretrained_path, policy_type, cfg, device=device):
        if policy_type != "act":
            raise ValueError("Shadow comparison supports ACT only")
        return comparator

    sim_eval.setup_policy = _setup
    try:
        print(f"Processor bundles are bit-identical: {il_fp[:12]}...")
        print("Control policy: IL; shadow policy: Post-RL")
        print(f"Per-step diagnostics: {csv_path}")
        sim_eval.kuavo_eval_autotest(config)
    finally:
        sim_eval.setup_policy = base_setup_policy
        comparator.close()


if __name__ == "__main__":
    main()
