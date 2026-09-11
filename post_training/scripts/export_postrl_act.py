from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import tempfile
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

if not LEROBOT_SRC.is_dir():
    raise RuntimeError(
        "LeRobot submodule is not initialized. "
        "Run `git submodule update --init --recursive`."
    )

for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.chdir(REPO_ROOT)

import lerobot_patches.custom_patches  # noqa: F401,E402
from kuavo_train.wrapper.policy.act.ACTConfigWrapper import (  # noqa: E402
    CustomACTConfigWrapper,
)
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import (  # noqa: E402
    CustomACTPolicyWrapper,
)
from post_rl.policy.stochastic_act_config import (  # noqa: E402
    StochasticACTConfigWrapper,
)
from post_rl.policy.stochastic_act_policy import (  # noqa: E402
    StochasticACTPolicyWrapper,
)


def _validate_path_component(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be a single directory name, got {value!r}")
    return value


def _resolve_processor_dir(checkpoint_dir: Path) -> Path:
    required = ("policy_preprocessor.json", "policy_postprocessor.json")
    for candidate in (checkpoint_dir, checkpoint_dir.parent):
        if all((candidate / name).is_file() for name in required):
            return candidate
    searched = "\n- ".join(str(path) for path in (checkpoint_dir, checkpoint_dir.parent))
    raise FileNotFoundError(
        "Could not find policy_preprocessor.json and policy_postprocessor.json in:\n- "
        f"{searched}"
    )


def _processor_files(processor_dir: Path) -> list[Path]:
    files = {
        path
        for pattern in ("policy_preprocessor*", "policy_postprocessor*")
        for path in processor_dir.glob(pattern)
        if path.is_file()
    }
    required = {
        processor_dir / "policy_preprocessor.json",
        processor_dir / "policy_postprocessor.json",
    }
    if not required.issubset(files):
        missing = sorted(str(path) for path in required - files)
        raise FileNotFoundError(f"Missing ACT processor files: {missing}")
    return sorted(files)


def _to_deterministic_config(
    stochastic_config: StochasticACTConfigWrapper,
) -> CustomACTConfigWrapper:
    kwargs = {
        field.name: copy.deepcopy(getattr(stochastic_config, field.name))
        for field in fields(CustomACTConfigWrapper)
        if field.init and hasattr(stochastic_config, field.name)
    }
    return CustomACTConfigWrapper(**kwargs)


def _build_dummy_observation(config, device: torch.device) -> dict[str, torch.Tensor]:
    observation = {}
    for key, feature in config.input_features.items():
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape is None:
            raise RuntimeError(f"Input feature {key!r} does not define a shape")
        observation[key] = torch.zeros(
            (1, *[int(dim) for dim in shape]),
            dtype=torch.float32,
            device=device,
        )
    if not observation:
        raise RuntimeError("ACT config contains no input features")
    return observation


def _assert_identical_state_dicts(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "Deterministic ACT state_dict mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in expected:
        lhs = expected[key].detach().cpu()
        rhs = actual[key].detach().cpu()
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or not torch.equal(lhs, rhs):
            max_abs = float((lhs.float() - rhs.float()).abs().max().item())
            raise RuntimeError(
                f"Parameter mismatch after conversion for {key}: "
                f"shape {tuple(lhs.shape)} vs {tuple(rhs.shape)}, "
                f"dtype {lhs.dtype} vs {rhs.dtype}, max_abs_diff={max_abs:.6g}"
            )


def export_postrl_checkpoint(
    checkpoint: str | Path,
    task: str,
    method: str,
    *,
    output_root: str | Path = "outputs/train",
    run_name: str | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Post-RL checkpoint directory not found: {checkpoint_dir}")
    for filename in ("config.json", "model.safetensors"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(
                f"Post-RL checkpoint is missing {filename}: {checkpoint_dir / filename}"
            )

    task = _validate_path_component(task, "task")
    method = _validate_path_component(method, "method")
    processor_dir = _resolve_processor_dir(checkpoint_dir)
    processor_files = _processor_files(processor_dir)

    torch_device = torch.device(device)
    source_config = StochasticACTConfigWrapper.from_pretrained(checkpoint_dir)
    serialized_device = getattr(source_config, "device", None)
    source_config.device = str(torch_device)
    source_policy = StochasticACTPolicyWrapper.from_pretrained(
        checkpoint_dir,
        config=source_config,
    ).to(torch_device)
    source_policy.eval()

    source_state = source_policy.state_dict()
    if "raw_log_std" not in source_state:
        raise RuntimeError(
            "Expected a stochastic Post-RL checkpoint containing raw_log_std, but it was not found."
        )
    deterministic_state = {
        key: value.detach().clone()
        for key, value in source_state.items()
        if key != "raw_log_std"
    }

    deterministic_config = _to_deterministic_config(source_config)
    deterministic_config.device = str(torch_device)
    deterministic_policy = CustomACTPolicyWrapper(deterministic_config).to(torch_device)
    deterministic_policy.load_state_dict(deterministic_state, strict=True)
    deterministic_policy.eval()
    _assert_identical_state_dicts(deterministic_state, deterministic_policy.state_dict())

    dummy_observation = _build_dummy_observation(source_config, torch_device)
    with torch.inference_mode():
        source_mean = source_policy.get_action_mean(dummy_observation)
        deterministic_mean = deterministic_policy.predict_action_chunk(dummy_observation)
    diff = (source_mean - deterministic_mean).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    if not torch.allclose(source_mean, deterministic_mean, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            "Stochastic ACT mean and deterministic ACT output differ before serialization: "
            f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}"
        )

    output_root = Path(output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    method_dir = output_root / task / method
    method_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = _validate_path_component(run_name, "run_name")
    run_dir = method_dir / run_name
    if run_dir.exists():
        raise FileExistsError(f"Output run directory already exists: {run_dir}")

    temp_run_dir = Path(tempfile.mkdtemp(prefix=".postrl_export_", dir=method_dir))
    try:
        for source_file in processor_files:
            shutil.copy2(source_file, temp_run_dir / source_file.name)

        epoch_dir = temp_run_dir / "epochbest"
        epoch_dir.mkdir(parents=True, exist_ok=False)
        deterministic_policy.config.device = (
            serialized_device if serialized_device is not None else str(torch_device)
        )
        deterministic_policy.save_pretrained(epoch_dir)

        verify_config = CustomACTConfigWrapper.from_pretrained(epoch_dir)
        verify_config.device = str(torch_device)
        reloaded_policy = CustomACTPolicyWrapper.from_pretrained(
            epoch_dir,
            config=verify_config,
            strict=True,
        ).to(torch_device)
        reloaded_policy.eval()
        if "raw_log_std" in reloaded_policy.state_dict():
            raise RuntimeError("Exported deterministic ACT still contains raw_log_std")
        _assert_identical_state_dicts(
            deterministic_state,
            reloaded_policy.state_dict(),
        )

        with torch.inference_mode():
            reloaded_mean = reloaded_policy.predict_action_chunk(dummy_observation)
        serialized_diff = (source_mean - reloaded_mean).abs()
        serialized_max_abs_diff = float(serialized_diff.max().item())
        serialized_mean_abs_diff = float(serialized_diff.mean().item())
        if not torch.allclose(source_mean, reloaded_mean, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                "Post-RL mean and serialized deterministic ACT output differ: "
                f"max_abs_diff={serialized_max_abs_diff:.6g}, "
                f"mean_abs_diff={serialized_mean_abs_diff:.6g}"
            )

        temp_run_dir.rename(run_dir)
    except Exception:
        shutil.rmtree(temp_run_dir, ignore_errors=True)
        raise

    final_epoch_dir = run_dir / "epochbest"
    print(f"Exported deterministic ACT checkpoint: {final_epoch_dir}")
    print(f"Processor files: {run_dir}")
    print(
        "Action mean verification: "
        f"max_abs_diff={serialized_max_abs_diff:.6g}, "
        f"mean_abs_diff={serialized_mean_abs_diff:.6g}"
    )
    return {
        "run_dir": run_dir,
        "checkpoint_dir": final_epoch_dir,
        "processor_source_dir": processor_dir,
        "max_abs_diff": serialized_max_abs_diff,
        "mean_abs_diff": serialized_mean_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a stochastic Post-RL ACT checkpoint as a deterministic Kuavo ACT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the Post-RL stochastic ACT checkpoint directory, normally "
            "offline_ppo/last or offline_ppo/checkpoints/final/policy."
        ),
    )
    parser.add_argument("--task", required=True, help="Output task directory name.")
    parser.add_argument("--method", required=True, help="Output method directory name.")
    parser.add_argument(
        "--output-root",
        default="outputs/train",
        help="Output root. Default: outputs/train",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only for conversion verification. Default: cpu",
    )
    args = parser.parse_args()

    export_postrl_checkpoint(
        checkpoint=args.checkpoint,
        task=args.task,
        method=args.method,
        output_root=args.output_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
