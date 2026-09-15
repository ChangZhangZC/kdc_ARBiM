from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
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
from post_rl.policy.stochastic_act_config import (  # noqa: E402
    StochasticACTConfigWrapper,
)
from post_rl.policy.stochastic_act_policy import (  # noqa: E402
    StochasticACTPolicyWrapper,
)


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _default_output_dir(resume_dir: Path) -> Path:
    if resume_dir.parent.name != "checkpoints":
        raise ValueError(
            "Cannot infer best_ope output directory. Pass --output explicitly, "
            "or provide a resume checkpoint under offline_ppo/checkpoints/<name>."
        )
    return resume_dir.parent.parent / "best_ope"


def _copy_processors(template_dir: Path, output_dir: Path) -> None:
    copied = []
    for pattern in ("policy_preprocessor*", "policy_postprocessor*"):
        for src in template_dir.glob(pattern):
            if src.is_file():
                shutil.copy2(src, output_dir / src.name)
                copied.append(src.name)
    required = {"policy_preprocessor.json", "policy_postprocessor.json"}
    missing = sorted(required.difference(copied))
    if missing:
        raise FileNotFoundError(
            f"Template checkpoint is missing processor files: {missing}"
        )


def _best_step_from_state(state: dict, best_mean_q: float) -> int | None:
    history = state.get("ope_history")
    ppo_state = state.get("ppo", {})
    offline_hparams = ppo_state.get("offline_hparams", {})
    eval_step = offline_hparams.get("eval_step")
    if not history or eval_step is None:
        return None
    best_index = min(
        range(len(history)),
        key=lambda idx: abs(float(history[idx]) - float(best_mean_q)),
    )
    return int(best_index * int(eval_step))


def materialize_best_ope(
    resume_checkpoint: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, object]:
    resume_dir = Path(resume_checkpoint).expanduser().resolve()
    if not resume_dir.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_dir}")

    training_state_path = resume_dir / "training_state.pt"
    template_dir = resume_dir / "policy"
    required = (
        training_state_path,
        template_dir / "config.json",
        template_dir / "model.safetensors",
        template_dir / "policy_preprocessor.json",
        template_dir / "policy_postprocessor.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete resume checkpoint {resume_dir}; missing: {missing}"
        )

    output_dir = (
        _default_output_dir(resume_dir)
        if output is None
        else Path(output).expanduser().resolve()
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Remove or rename it first."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    state = _load_torch(training_state_path)
    ppo_state = state.get("ppo")
    if not isinstance(ppo_state, dict) or "old_policy" not in ppo_state:
        raise KeyError("training_state.pt does not contain ppo.old_policy")
    if state.get("best_mean_q") is None:
        raise KeyError("training_state.pt does not contain best_mean_q")

    best_state = ppo_state["old_policy"]
    best_mean_q = float(state["best_mean_q"])
    best_step = _best_step_from_state(state, best_mean_q)

    cfg = StochasticACTConfigWrapper.from_pretrained(template_dir)
    cfg.device = "cpu"
    policy = StochasticACTPolicyWrapper.from_pretrained(
        template_dir,
        config=cfg,
    ).cpu()
    policy.load_state_dict(best_state, strict=True)
    policy.eval()
    if "raw_log_std" not in policy.state_dict():
        raise RuntimeError(
            "ppo.old_policy is not a stochastic ACT policy: raw_log_std is missing."
        )

    tmp_dir = Path(
        tempfile.mkdtemp(prefix=".best_ope_", dir=output_dir.parent)
    )
    try:
        policy.save_pretrained(tmp_dir)
        _copy_processors(template_dir, tmp_dir)

        verify_cfg = StochasticACTConfigWrapper.from_pretrained(tmp_dir)
        verify_cfg.device = "cpu"
        verify_policy = StochasticACTPolicyWrapper.from_pretrained(
            tmp_dir,
            config=verify_cfg,
        ).cpu()
        saved_state = verify_policy.state_dict()

        if set(best_state) != set(saved_state):
            missing_keys = sorted(set(best_state) - set(saved_state))
            unexpected_keys = sorted(set(saved_state) - set(best_state))
            raise RuntimeError(
                "Serialized best_ope key mismatch: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        for key, expected in best_state.items():
            expected_cpu = expected.detach().cpu()
            actual_cpu = saved_state[key].detach().cpu()
            if not torch.equal(expected_cpu, actual_cpu):
                max_abs = float(
                    (expected_cpu.float() - actual_cpu.float()).abs().max().item()
                )
                raise RuntimeError(
                    f"Serialized best_ope tensor mismatch for {key}: "
                    f"max_abs_diff={max_abs:.6g}"
                )

        with open(tmp_dir / "best_ope_score.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "mean_q"])
            writer.writerow([
                "" if best_step is None else int(best_step),
                f"{best_mean_q:.6f}",
            ])

        metadata = {
            "artifact_type": "postrl_best_ope_policy",
            "policy_kind": "stochastic_act_postrl",
            "model_format": "safetensors",
            "contains_raw_log_std": True,
            "processors_included": True,
            "best_ope_step": best_step,
            "best_mean_q": best_mean_q,
            "policy_source": "ppo.old_policy",
            "source_training_state": str(training_state_path),
        }
        with open(tmp_dir / "best_ope_meta.json", "w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

        tmp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"Materialized stochastic best-OPE policy: {output_dir}")
    print(f"best_mean_q={best_mean_q:.6f}, best_step={best_step}")
    print("Serialization verification: exact tensor match")
    return {
        "output_dir": output_dir,
        "best_mean_q": best_mean_q,
        "best_step": best_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize ppo.old_policy from a legacy PPO training_state.pt as a "
            "complete stochastic ACT best_ope pretrained bundle."
        )
    )
    parser.add_argument(
        "--resume-checkpoint",
        required=True,
        help=(
            "Full PPO resume checkpoint directory containing training_state.pt and "
            "policy/, normally stage2/offline_ppo/checkpoints/final."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output best_ope directory. By default this is inferred as the sibling "
            "stage2/offline_ppo/best_ope directory."
        ),
    )
    args = parser.parse_args()
    materialize_best_ope(
        args.resume_checkpoint,
        output=args.output,
    )


if __name__ == "__main__":
    main()
