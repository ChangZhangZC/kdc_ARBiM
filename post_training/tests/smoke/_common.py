from __future__ import annotations

import argparse
import gc
import hashlib
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

if not LEROBOT_SRC.is_dir():
    raise RuntimeError(
        "LeRobot submodule is not initialized. Run `git submodule update --init --recursive`."
    )

for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.chdir(REPO_ROOT)

import lerobot_patches.custom_patches  # noqa: F401,E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from post_rl.data.offline_dataset import OfflineDataset  # noqa: E402
from post_rl.training import TrainACTWorkspace  # noqa: E402
from post_rl.utils.common import dict_apply  # noqa: E402

CONFIG_PATH = REPO_ROOT / "post_training" / "configs" / "rl" / "offline_rl.yaml"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    require_dataset: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", required=True, help="Path to the ACT IL checkpoint.")
    if require_dataset:
        parser.add_argument("--dataset", required=True, help="Path to offline_dataset.zarr.")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used by the smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional explicit ACT/RL chunk size override. Leave unset to test YAML defaults.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional root directory for smoke-test outputs. A temporary directory is used otherwise.",
    )
    return parser


def load_cfg(
    args,
    *,
    chunk_as_single_action: bool | None = None,
    ratio_mode: str | None = None,
    adv_mode: str | None = None,
):
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.input.policy_checkpoint = str(pathlib.Path(args.checkpoint).expanduser().resolve())
    cfg.input.policy_checkpoint_type = "il"
    if hasattr(args, "dataset"):
        cfg.input.dataset_path = str(pathlib.Path(args.dataset).expanduser().resolve())
    cfg.training.device = str(args.device)
    cfg.use_wandb = False
    cfg.training.use_ema = False
    cfg.dataloader.batch_size = int(args.batch_size)
    cfg.dataloader.num_workers = 0
    cfg.dataloader.persistent_workers = False
    cfg.val_dataloader.batch_size = int(args.batch_size)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.unio4.finetune_batch_size = int(args.batch_size)
    cfg.ppo.enable_ratio_logging = True
    cfg.ppo.ratio_log_every_updates = 1
    cfg.ppo.ratio_plot_on_final_flush = False

    if args.chunk_size is not None:
        chunk_size = int(args.chunk_size)
        cfg.act_chunk_size = chunk_size
        cfg.rl_chunk_size = chunk_size
        cfg.n_action_steps = chunk_size
        cfg.horizon = chunk_size
    if chunk_as_single_action is not None:
        cfg.chunk_as_single_action = bool(chunk_as_single_action)
    if ratio_mode is not None:
        cfg.offline_chunk_ratio_mode = str(ratio_mode)
    if adv_mode is not None:
        cfg.offline_chunk_adv_mode = str(adv_mode)
    return cfg


def make_work_dir(args, label: str) -> str:
    if args.work_dir:
        path = pathlib.Path(args.work_dir).expanduser().resolve() / label
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return tempfile.mkdtemp(prefix=f"arbim_{label}_")


def make_workspace(cfg, output_dir: str) -> TrainACTWorkspace:
    return TrainACTWorkspace(cfg, output_dir=output_dir)


def build_real_batch(workspace: TrainACTWorkspace, batch_size: int):
    workspace.buffer = workspace._load_buffer()
    workspace._build_act_observation_frontends()
    dataset = OfflineDataset(
        buffer=workspace.buffer,
        horizon=int(workspace.cfg.horizon),
        pad_before=int(workspace.cfg.dataset.pad_before),
        pad_after=int(workspace.cfg.dataset.pad_after),
        sequence_stride=int(workspace.cfg.dataset.sequence_stride),
        seed=int(workspace.cfg.training.seed),
        val_ratio=0.0,
        max_train_episodes=workspace.cfg.dataset.max_train_episodes,
        use_depth=bool(workspace.cfg.dataset.use_depth),
    )
    if len(dataset) == 0:
        raise RuntimeError("Smoke-test dataset has zero valid sequences.")
    actual_batch_size = min(max(int(batch_size), 1), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch = dict_apply(
        batch,
        lambda x: x.to(workspace.device, non_blocking=True),
    )
    return dataset, batch


def policy_obs_from_batch(workspace: TrainACTWorkspace, batch: dict) -> dict:
    normalized = workspace.obs_adapter.normalize_obs(batch["obs"])
    return {key: value[:, 0] for key, value in normalized.items()}


def assert_finite(value, name: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            assert_finite(item, f"{name}[{idx}]")
        return
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise AssertionError(f"{name} is empty")
        if not torch.isfinite(value.float()).all():
            raise AssertionError(f"{name} contains NaN or Inf")


def snapshot_params(module: torch.nn.Module, *, trainable_only: bool = False) -> dict[str, torch.Tensor]:
    result = {}
    for name, param in module.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        result[name] = param.detach().cpu().clone()
    return result


def changed_param_names(before: dict[str, torch.Tensor], module: torch.nn.Module) -> list[str]:
    current = dict(module.named_parameters())
    changed = []
    for name, old in before.items():
        new = current[name].detach().cpu()
        if not torch.equal(old, new):
            changed.append(name)
    return changed


def fingerprint_params(module: torch.nn.Module, *, trainable: bool | None = None) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        if trainable is not None and bool(param.requires_grad) != trainable:
            continue
        tensor = param.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def max_param_diff(
    first: dict[str, torch.Tensor],
    second_module: torch.nn.Module,
) -> float:
    second = dict(second_module.named_parameters())
    max_diff = 0.0
    for name, tensor in first.items():
        diff = (tensor - second[name].detach().cpu()).abs().max().item()
        max_diff = max(max_diff, float(diff))
    return max_diff


def assert_close_to_one(ratio: torch.Tensor, *, atol: float = 1e-5) -> None:
    assert_finite(ratio, "ratio")
    max_error = float((ratio.detach() - 1.0).abs().max().item())
    if max_error > atol:
        raise AssertionError(f"Initial PPO ratio is not ~1; max |ratio-1|={max_error:.6g}")


def print_pass(message: str) -> None:
    print(f"[PASS] {message}")


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
