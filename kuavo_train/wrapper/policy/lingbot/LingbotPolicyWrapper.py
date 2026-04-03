"""LingBot policy wrapper.

This wrapper provides a single `launch()` interface aligned with the wrapper
pattern used under `kuavo_train/wrapper/policy/pi05`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

from .LingbotConfigWrapper import CustomLingbotConfigWrapper
from .LingbotModelWrapper import CustomLingbotModelWrapper


@dataclass
class CustomLingbotPolicyWrapper:
    name = "custom_lingbot"
    config: CustomLingbotConfigWrapper

    def launch(self, repo_root: Path, extra_args: list[str] | None = None) -> int:
        model_wrapper = CustomLingbotModelWrapper(self.config)
        workdir, cmd, env = model_wrapper.build_command(repo_root)

        cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible:
            nproc = len([x for x in cuda_visible.split(",") if x.strip()])
            cmd = [
                f"--nproc-per-node={nproc}" if part.startswith("--nproc-per-node=") else part
                for part in cmd
            ]

        if extra_args:
            cmd.extend(extra_args)

        pretty = " ".join(shlex.quote(x) for x in cmd)
        pythonpath_parts = [part for part in env.get("PYTHONPATH", "").split(":") if part]
        lerobot_root = pythonpath_parts[1] if len(pythonpath_parts) > 1 else ""
        print(f"[INFO] LingBot workdir: {workdir}")
        print(f"[INFO] LingBot lerobot_root: {lerobot_root}")
        print(f"[INFO] LingBot launch: {pretty}")

        if self.config.dry_run:
            return 0

        # Keep behavior consistent with previous shell launcher by tee-ing to repo log.
        log_path = repo_root / "log_lingbot_train.txt"
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n[CMD] {pretty}\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
            return proc.wait()
