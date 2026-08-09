"""LingBot launch model wrapper.

`CustomLingbotModelWrapper` focuses on command composition and environment
normalization for torchrun-based LingBot training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from .LingbotConfigWrapper import CustomLingbotConfigWrapper


@dataclass
class CustomLingbotModelWrapper:
    config: CustomLingbotConfigWrapper

    def _infer_nproc_per_node(self) -> int:
        env_cuda_visible = (self.config.env or {}).get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_cuda_visible:
            return len([x for x in env_cuda_visible.split(",") if x.strip()])

        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible:
            return len([x for x in cuda_visible.split(",") if x.strip()])

        # Fallback: query GPU count, default 1 if nvidia-smi not available.
        rc = os.popen("nvidia-smi -L 2>/dev/null | wc -l").read().strip()
        try:
            n = int(rc)
            return max(n, 1)
        except ValueError:
            return 1

    def build_command(self, repo_root: Path) -> tuple[Path, list[str], dict[str, str]]:
        lingbot_root = self.config.resolve_lingbot_root(repo_root)
        lerobot_root = self.config.resolve_lerobot_root(repo_root)
        config_path = self.config.resolve_config_path(repo_root)

        # 1) Prefer internal entry path in kuavo_data_challenge
        internal_entry = (repo_root / self.config.train_entry).resolve()
        # 2) Fallback to lingbot-vla repo entry
        external_entry = (lingbot_root / self.config.train_entry).resolve()

        if internal_entry.is_file():
            entry_abs = internal_entry
            workdir = repo_root
        elif external_entry.is_file():
            entry_abs = external_entry
            workdir = lingbot_root
        else:
            raise FileNotFoundError(
                f"Training entry not found. Tried:\n- {internal_entry}\n- {external_entry}"
            )

        nproc_per_node = self._infer_nproc_per_node()
        cmd = [
            "torchrun",
            f"--nnodes={self.config.nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--node-rank={self.config.node_rank}",
            f"--master-addr={self.config.master_addr}",
            f"--master-port={self.config.master_port}",
            str(entry_abs),
            str(config_path),
        ]

        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Ensure current repo and compatible LeRobot are imported before any installed package.
        env["PYTHONPATH"] = ":".join([str(repo_root), str(lerobot_root)])
        env.update(self.config.env)

        return workdir, cmd, env