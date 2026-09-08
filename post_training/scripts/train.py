import os

import hydra

from train_ddp import TrainACTWorkspace


@hydra.main(version_base=None, config_path="../configs/rl", config_name="offline_rl")
def main(cfg):
    if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise RuntimeError(
            "train.py is the single-process entry point. "
            "Use train_ddp.py with torchrun for distributed training."
        )

    workspace = None
    try:
        workspace = TrainACTWorkspace(cfg)
        workspace.run()
    finally:
        if workspace is not None:
            wandb_run = getattr(workspace, "wandb_run", None)
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
            workspace.cleanup_shared_memory()


if __name__ == "__main__":
    main()
