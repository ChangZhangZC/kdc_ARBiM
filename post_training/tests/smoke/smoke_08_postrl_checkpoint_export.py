from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
SCRIPTS_DIR = REPO_ROOT / "post_training" / "scripts"

if not LEROBOT_SRC.is_dir():
    raise RuntimeError(
        "LeRobot submodule is not initialized. Run `git submodule update --init --recursive`."
    )

for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC, SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: F401,E402
from export_postrl_act import export_postrl_checkpoint  # noqa: E402
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import (  # noqa: E402
    CustomACTPolicyWrapper,
)
from post_rl.policy.stochastic_act_policy import (  # noqa: E402
    StochasticACTPolicyWrapper,
)


def _print_pass(message: str) -> None:
    print(f"[PASS] {message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke 08: stochastic Post-RL ACT -> deterministic Kuavo ACT export"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to a stochastic Post-RL ACT checkpoint, normally "
            "offline_ppo/last or offline_ppo/checkpoints/final/policy."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for conversion verification.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional output root. A temporary directory is used by default.",
    )
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    if args.work_dir:
        output_root = pathlib.Path(args.work_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = pathlib.Path(tempfile.mkdtemp(prefix="arbim_smoke08_export_"))

    result = export_postrl_checkpoint(
        checkpoint=checkpoint,
        task="smoke_task",
        method="smoke_postrl",
        output_root=output_root,
        run_name="run_smoke08",
        device=args.device,
    )

    run_dir = pathlib.Path(result["run_dir"])
    exported = pathlib.Path(result["checkpoint_dir"])

    expected_files = (
        run_dir / "policy_preprocessor.json",
        run_dir / "policy_postprocessor.json",
        exported / "config.json",
        exported / "model.safetensors",
    )
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing exported files: {missing}")
    _print_pass("Kuavo ACT output layout is complete")

    processor_sidecars = [
        *run_dir.glob("policy_preprocessor_step_*.safetensors"),
        *run_dir.glob("policy_postprocessor_step_*.safetensors"),
    ]
    if not processor_sidecars:
        raise AssertionError("No processor safetensor sidecar files were copied")
    _print_pass(f"processor sidecars copied: {len(processor_sidecars)}")

    source = StochasticACTPolicyWrapper.from_pretrained(checkpoint)
    target = CustomACTPolicyWrapper.from_pretrained(exported, strict=True)

    source_keys = set(source.state_dict())
    target_keys = set(target.state_dict())
    if "raw_log_std" not in source_keys:
        raise AssertionError("Source checkpoint does not contain raw_log_std")
    if target_keys != source_keys - {"raw_log_std"}:
        raise AssertionError(
            "Exported deterministic key set mismatch: "
            f"missing={sorted((source_keys - {'raw_log_std'}) - target_keys)}, "
            f"unexpected={sorted(target_keys - (source_keys - {'raw_log_std'}))}"
        )
    _print_pass("deterministic key set equals stochastic key set minus raw_log_std")

    target_state = target.state_dict()
    for key, source_tensor in source.state_dict().items():
        if key == "raw_log_std":
            continue
        target_tensor = target_state[key]
        torch.testing.assert_close(
            source_tensor.detach().cpu(),
            target_tensor.detach().cpu(),
            rtol=0.0,
            atol=0.0,
        )
    _print_pass("all shared parameters are bit-identical")

    max_abs_diff = float(result["max_abs_diff"])
    mean_abs_diff = float(result["mean_abs_diff"])
    if max_abs_diff > 1e-6:
        raise AssertionError(
            f"Exported deterministic action mean drift is too large: {max_abs_diff}"
        )
    _print_pass(
        f"action mean preserved: max_abs_diff={max_abs_diff:.6g}, "
        f"mean_abs_diff={mean_abs_diff:.6g}"
    )

    print("\nSMOKE 08 PASSED")


if __name__ == "__main__":
    main()
