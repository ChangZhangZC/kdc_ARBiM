import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
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

import lerobot_patches.custom_patches

import hydra
import torch
import torch.distributed as dist

from post_rl.training import TrainACTWorkspace


def setup_ddp() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../configs/rl", config_name="offline_rl")
def main(cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        setup_ddp()

    workspace = None
    try:
        workspace = TrainACTWorkspace(cfg)
        workspace.run()
    finally:
        if workspace is not None:
            if workspace.wandb_run is not None:
                try:
                    workspace.wandb_run.finish()
                except Exception:
                    pass
            workspace.cleanup_shared_memory()
        cleanup_ddp()


if __name__ == "__main__":
    main()
