"""LingBot launch configuration wrapper.

This wrapper intentionally follows the `pi05` folder pattern:
- LingbotConfigWrapper.py
- LingbotModelWrapper.py
- LingbotPolicyWrapper.py

Unlike PI05, this wrapper standardizes *training launch* for the external
LingBot-VLA codebase instead of implementing a LeRobot policy forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class CustomLingbotConfigWrapper:
    # Path of external LingBot-VLA repository.
    lingbot_root: str = ""
    # Path of LeRobot repo compatible with LingBot (`lerobot/common/*` layout).
    # If empty, kuavo internal third_party/lerobot is preferred.
    lerobot_root: str = ""

    # Relative training entry inside lingbot_root.
    # Prefer internal entry in kuavo_data_challenge; fallback to lingbot repo entry.
    train_entry: str = "kuavo_train/lingbot/tasks/vla/train_lingbotvla.py"

    # YAML config used by LingBot task script.
    config_path: str = "configs/policy/lingbot/robotwin_load20000h.yaml"

    # Distributed launch settings.
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "0.0.0.0"
    master_port: int = 62500

    # Whether to only print command without running.
    dry_run: bool = False

    # Optional environment variables injected before launch.
    env: dict[str, str] = field(default_factory=dict)

    def resolve_lingbot_root(self, repo_root: Path) -> Path:
        """Resolve lingbot-vla root with deterministic fallback order."""
        if self.lingbot_root:
            p = Path(self.lingbot_root).expanduser().resolve()
            if p.is_dir():
                return p

        env_root = os.getenv("LINGBOT_ROOT")
        if env_root:
            p = Path(env_root).expanduser().resolve()
            if p.is_dir():
                return p

        candidates = [
            (repo_root / "../lingbot-vla").resolve(),
            (repo_root / "../../lingbot-vla").resolve(),
            Path("/home/yunxi/lmy/VLA/lingbot-vla"),
        ]
        for p in candidates:
            if p.is_dir():
                return p

        raise FileNotFoundError(
            "Cannot find LingBot-VLA repo. Set `LINGBOT_ROOT` or `lingbot_root`."
        )

    def resolve_lerobot_root(self, repo_root: Path) -> Path:
        """Resolve lerobot root (prefer kuavo third_party/lerobot/src)."""
        def _is_valid_root(p: Path) -> bool:
            # Accept either ".../src" (contains lerobot package) or repo root.
            return (p / "lerobot").is_dir() or (p / "src" / "lerobot").is_dir()

        def _normalize_root(p: Path) -> Path:
            # Prefer the path that should be directly appended into PYTHONPATH.
            if (p / "lerobot").is_dir():
                return p
            if (p / "src" / "lerobot").is_dir():
                return p / "src"
            return p

        if self.lerobot_root:
            p = Path(self.lerobot_root).expanduser().resolve()
            if _is_valid_root(p):
                return _normalize_root(p)

        candidates = [
            (repo_root / "third_party/lerobot/src").resolve(),  # kuavo internal (preferred)
            (repo_root / "third_party/lerobot").resolve(),
            (repo_root / "../lerobot").resolve(),               # sibling fallback
            (repo_root / "../../lerobot").resolve(),
        ]
        for p in candidates:
            if _is_valid_root(p):
                return _normalize_root(p)

        env_root = os.getenv("LEROBOT_ROOT")
        if env_root:
            p = Path(env_root).expanduser().resolve()
            if _is_valid_root(p):
                return _normalize_root(p)

        raise FileNotFoundError(
            "Cannot find compatible LeRobot repo. "
            "Set `LEROBOT_ROOT` or `policy.lerobot_root`."
        )

    def resolve_config_path(self, repo_root: Path) -> Path:
        """Resolve config path; prefer absolute then repo-relative."""
        p = Path(self.config_path).expanduser()
        if p.is_absolute() and p.is_file():
            return p
        p2 = (repo_root / p).resolve()
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"Config file not found: {self.config_path}")
