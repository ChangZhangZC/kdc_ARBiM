from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
from collections import deque

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


class SwitchComparePolicy:
    """Start under Post-RL control, shadow IL, then hand control to IL once."""

    def __init__(
        self,
        postrl_policy,
        il_policy,
        *,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        csv_path: pathlib.Path,
        log_every: int,
        switch_on_freeze: bool,
        switch_step: int | None,
        freeze_min_step: int,
        freeze_window: int,
        freeze_step_delta_max: float,
        freeze_policy_delta_min: float,
    ) -> None:
        if freeze_window < 1:
            raise ValueError("freeze_window must be >= 1")
        if freeze_min_step < 0:
            raise ValueError("freeze_min_step must be >= 0")
        if freeze_step_delta_max < 0 or freeze_policy_delta_min < 0:
            raise ValueError("freeze thresholds must be non-negative")
        if switch_step is not None and switch_step < 0:
            raise ValueError("switch_step must be >= 0")
        if not switch_on_freeze and switch_step is None:
            raise ValueError("Enable --switch-on-freeze or provide --switch-step")

        self.postrl_policy = postrl_policy
        self.il_policy = il_policy
        self.config = postrl_policy.config
        self.action_mean = np.asarray(action_mean, dtype=np.float64)
        self.action_std = np.asarray(action_std, dtype=np.float64)
        self.log_every = max(int(log_every), 1)
        self.switch_on_freeze = bool(switch_on_freeze)
        self.switch_step = switch_step
        self.freeze_min_step = int(freeze_min_step)
        self.freeze_window = int(freeze_window)
        self.freeze_step_delta_max = float(freeze_step_delta_max)
        self.freeze_policy_delta_min = float(freeze_policy_delta_min)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self._file = csv_path.open("w", newline="", buffering=1)
        self._writer = None
        self._episode = -1
        self._step = 0
        self._has_steps = False
        self._control_policy = "postrl"
        self._switched = False
        self._switch_events: list[dict] = []
        self._prev_postrl = None
        self._prev_il = None
        self._postrl_delta_window = deque(maxlen=self.freeze_window)
        self._policy_delta_window = deque(maxlen=self.freeze_window)

    def eval(self):
        self.postrl_policy.eval()
        self.il_policy.eval()
        return self

    def to(self, device):
        self.postrl_policy.to(device)
        self.il_policy.to(device)
        return self

    def reset(self):
        self.postrl_policy.reset()
        self.il_policy.reset()
        if self._episode < 0:
            self._episode = 0
        elif self._has_steps:
            self._episode += 1
        self._step = 0
        self._has_steps = False
        self._control_policy = "postrl"
        self._switched = False
        self._prev_postrl = None
        self._prev_il = None
        self._postrl_delta_window.clear()
        self._policy_delta_window.clear()

    def _maybe_switch(self, postrl_step_delta: float, policy_delta_l2: float):
        if self._switched:
            return False, "", float("nan"), float("nan")
        if math.isfinite(postrl_step_delta):
            self._postrl_delta_window.append(postrl_step_delta)
            self._policy_delta_window.append(policy_delta_l2)

        rolling_step_delta = (
            float(np.mean(self._postrl_delta_window))
            if self._postrl_delta_window
            else float("nan")
        )
        rolling_policy_delta = (
            float(np.mean(self._policy_delta_window))
            if self._policy_delta_window
            else float("nan")
        )

        reason = ""
        if self.switch_step is not None and self._step >= self.switch_step:
            reason = "fixed_step"
        elif (
            self.switch_on_freeze
            and self._step >= self.freeze_min_step
            and len(self._postrl_delta_window) == self.freeze_window
            and rolling_step_delta <= self.freeze_step_delta_max
            and rolling_policy_delta >= self.freeze_policy_delta_min
        ):
            reason = "freeze_detected"

        if not reason:
            return False, "", rolling_step_delta, rolling_policy_delta

        self._control_policy = "il"
        self._switched = True
        event = {
            "episode": self._episode,
            "step": self._step,
            "reason": reason,
            "rolling_postrl_step_delta_l2": rolling_step_delta,
            "rolling_policy_delta_l2": rolling_policy_delta,
        }
        self._switch_events.append(event)
        print(
            "\n[SWITCH] "
            f"episode={self._episode} step={self._step}: Post-RL -> IL "
            f"reason={reason}, rolling_postrl_step_delta={rolling_step_delta:.6g}, "
            f"rolling_policy_delta={rolling_policy_delta:.6g}\n"
        )
        return True, reason, rolling_step_delta, rolling_policy_delta

    @torch.inference_mode()
    def select_action(self, observation):
        postrl_action = self.postrl_policy.select_action(observation)
        il_action = self.il_policy.select_action(observation)
        if tuple(postrl_action.shape) != tuple(il_action.shape):
            raise RuntimeError(
                f"Post-RL/IL action shape mismatch: {tuple(postrl_action.shape)} vs {tuple(il_action.shape)}"
            )
        postrl_norm = postrl_action[0].detach().float().cpu().numpy().astype(np.float64)
        il_norm = il_action[0].detach().float().cpu().numpy().astype(np.float64)
        postrl_phys = postrl_norm * self.action_std + self.action_mean
        il_phys = il_norm * self.action_std + self.action_mean
        policy_delta = postrl_phys - il_phys
        policy_delta_l2 = float(np.linalg.norm(policy_delta))
        postrl_step_delta = (
            float("nan")
            if self._prev_postrl is None
            else float(np.linalg.norm(postrl_phys - self._prev_postrl))
        )
        il_step_delta = (
            float("nan")
            if self._prev_il is None
            else float(np.linalg.norm(il_phys - self._prev_il))
        )

        switched_now, reason, rolling_step_delta, rolling_policy_delta = self._maybe_switch(
            postrl_step_delta, policy_delta_l2
        )
        if self._control_policy == "postrl":
            executed_action, executed_phys = postrl_action, postrl_phys
        else:
            executed_action, executed_phys = il_action, il_phys

        row = {
            "episode": self._episode,
            "step": self._step,
            "control_policy": self._control_policy,
            "switch_event": int(switched_now),
            "switch_reason": reason,
            "rolling_postrl_step_delta_l2": rolling_step_delta,
            "rolling_policy_delta_l2": rolling_policy_delta,
            "policy_delta_l2": policy_delta_l2,
            "policy_delta_mae": float(np.mean(np.abs(policy_delta))),
            "policy_delta_max_abs": float(np.max(np.abs(policy_delta))),
            "postrl_step_delta_l2": postrl_step_delta,
            "il_step_delta_l2": il_step_delta,
            **_group_metrics(policy_delta),
        }
        for index, value in enumerate(postrl_phys):
            row[f"postrl_action_{index}"] = float(value)
        for index, value in enumerate(il_phys):
            row[f"il_action_{index}"] = float(value)
        for index, value in enumerate(executed_phys):
            row[f"executed_action_{index}"] = float(value)
        for index, value in enumerate(policy_delta):
            row[f"policy_delta_{index}"] = float(value)

        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

        if self._step % self.log_every == 0:
            print(
                f"[switch] episode={self._episode} step={self._step} "
                f"control={self._control_policy} policy_delta_l2={policy_delta_l2:.6g} "
                f"postrl_step_delta={postrl_step_delta:.6g} il_step_delta={il_step_delta:.6g}"
            )
        self._prev_postrl = postrl_phys.copy()
        self._prev_il = il_phys.copy()
        self._step += 1
        self._has_steps = True
        return executed_action

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()
        print(f"\nSwitch comparison CSV: {self.csv_path}")
        if self._switch_events:
            print("Switch events:")
            for event in self._switch_events:
                print(
                    f"  episode={event['episode']} step={event['step']} reason={event['reason']} "
                    f"postrl_delta={event['rolling_postrl_step_delta_l2']:.6g} "
                    f"policy_delta={event['rolling_policy_delta_l2']:.6g}"
                )
        else:
            print("Switch events: none")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Causal recovery test: Post-RL controls first while IL runs in shadow; "
            "then control switches once to IL at a fixed step or after freeze detection."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--switch-on-freeze", action="store_true")
    parser.add_argument("--switch-step", type=int, default=None)
    parser.add_argument("--freeze-min-step", type=int, default=80)
    parser.add_argument("--freeze-window", type=int, default=10)
    parser.add_argument("--freeze-step-delta-max", type=float, default=0.02)
    parser.add_argument("--freeze-policy-delta-min", type=float, default=0.04)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default="post_training/outputs/switch_policy_compare")
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
    config.inference.pretrained_path = str(postrl_checkpoint)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.inference.method = f"{config.inference.method}_switch_compare"
    config.inference.timestamp = f"{config.inference.timestamp}_{stamp}"
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "switch_action_comparison.csv"
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": str(config_path),
                "il_checkpoint": str(il_checkpoint),
                "postrl_checkpoint": str(postrl_checkpoint),
                "processor_fingerprint": il_fp,
                "switch_on_freeze": bool(args.switch_on_freeze),
                "switch_step": args.switch_step,
                "freeze_min_step": args.freeze_min_step,
                "freeze_window": args.freeze_window,
                "freeze_step_delta_max": args.freeze_step_delta_max,
                "freeze_policy_delta_min": args.freeze_policy_delta_min,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    _arm = ArmMove(config)
    from kuavo_deploy.src.eval import sim_auto_test as sim_eval

    device = torch.device(config.inference.device)
    base_setup_policy = sim_eval.setup_policy
    postrl_policy = base_setup_policy(postrl_checkpoint, "act", config.inference, device)
    il_policy = base_setup_policy(il_checkpoint, "act", config.inference, device)
    comparator = SwitchComparePolicy(
        postrl_policy,
        il_policy,
        action_mean=action_mean,
        action_std=action_std,
        csv_path=csv_path,
        log_every=args.log_every,
        switch_on_freeze=args.switch_on_freeze,
        switch_step=args.switch_step,
        freeze_min_step=args.freeze_min_step,
        freeze_window=args.freeze_window,
        freeze_step_delta_max=args.freeze_step_delta_max,
        freeze_policy_delta_min=args.freeze_policy_delta_min,
    ).eval().to(device)

    def _setup(_pretrained_path, policy_type, cfg, device=device):
        if policy_type != "act":
            raise ValueError("Switch comparison supports ACT only")
        return comparator

    sim_eval.setup_policy = _setup
    try:
        print(f"Processor bundles are bit-identical: {il_fp[:12]}...")
        if args.switch_step is not None:
            print(f"Fixed switch: Post-RL -> IL at step {args.switch_step}")
        if args.switch_on_freeze:
            print(
                "Freeze switch: "
                f"min_step={args.freeze_min_step}, window={args.freeze_window}, "
                f"postrl_step_delta<={args.freeze_step_delta_max}, "
                f"policy_delta>={args.freeze_policy_delta_min}"
            )
        print(f"Per-step diagnostics: {csv_path}")
        sim_eval.kuavo_eval_autotest(config)
    finally:
        sim_eval.setup_policy = base_setup_policy
        comparator.close()


if __name__ == "__main__":
    main()
