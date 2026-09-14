from __future__ import annotations

from .checkpoint_compat_workspace import TrainACTWorkspace as _CheckpointCompatTrainACTWorkspace


class TrainACTWorkspace(_CheckpointCompatTrainACTWorkspace):
    """Top-level Stage 1 / Stage 2 execution gates for post-training."""

    def _stage_flags(self) -> tuple[bool, bool]:
        stages = self.cfg.get("stages")
        if stages is None:
            return True, True
        return (
            bool(stages.get("run_stage1", True)),
            bool(stages.get("run_stage2", True)),
        )

    def _validate_stage_gates(self) -> None:
        run_stage1, run_stage2 = self._stage_flags()
        eval_only = bool(self.cfg.eval)

        if not run_stage1 and not run_stage2:
            raise ValueError(
                "At least one stage must be enabled: stages.run_stage1=true "
                "and/or stages.run_stage2=true."
            )
        if eval_only and not run_stage2:
            raise ValueError("eval=true requires stages.run_stage2=true.")
        if run_stage2 and not eval_only and int(self.cfg.unio4.bppo_steps) <= 0:
            raise ValueError(
                "stages.run_stage2=true requires unio4.bppo_steps > 0. "
                "For Stage-1-only runs, set stages.run_stage2=false."
            )

        resume_cfg = self.cfg.get("resume")
        full_resume = bool(resume_cfg is not None and resume_cfg.get("checkpoint_dir"))
        if not run_stage2 and full_resume:
            raise ValueError(
                "resume.checkpoint_dir is a full PPO resume and requires "
                "stages.run_stage2=true."
            )
        if run_stage1:
            return

        stage1_resume = bool(self.cfg.unio4.get("stage1_resume_dir"))
        explicit_artifacts = bool(
            self.cfg.critic.get("artifact_dir")
            and self.cfg.dynamics.get("artifact_dir")
        )
        if not (full_resume or stage1_resume or explicit_artifacts):
            raise ValueError(
                "stages.run_stage1=false requires existing Stage 1 artifacts. "
                "Set unio4.stage1_resume_dir, provide critic.artifact_dir and "
                "dynamics.artifact_dir, or use resume.checkpoint_dir."
            )

    def run(self):
        self._validate_stage_gates()
        run_stage1, run_stage2 = self._stage_flags()

        if not run_stage1:
            self.cfg.critic.load_pretrain = True
            self.cfg.dynamics.load_pretrain = True

        if self.rank == 0:
            print(
                "Post-RL stage gates: "
                f"stage1={'on' if run_stage1 else 'off'}, "
                f"stage2={'on' if run_stage2 else 'off'}"
            )

        return super().run()

    def _train_critic(self, dataloader) -> None:
        run_stage1, _ = self._stage_flags()
        if not run_stage1:
            raise RuntimeError(
                "Stage 1 is disabled, but the Critic could not be loaded from "
                "the configured Stage 1 artifacts."
            )
        return super()._train_critic(dataloader)

    def _train_dynamics(self) -> None:
        run_stage1, _ = self._stage_flags()
        if not run_stage1:
            raise RuntimeError(
                "Stage 1 is disabled, but Dynamics could not be loaded from "
                "the configured Stage 1 artifacts."
            )
        return super()._train_dynamics()

    def _build_ppo(self) -> None:
        _, run_stage2 = self._stage_flags()
        if not run_stage2:
            return
        return super()._build_ppo()

    def _build_ema(self) -> None:
        _, run_stage2 = self._stage_flags()
        if not run_stage2:
            self.ema = None
            self.ema_model = None
            return
        return super()._build_ema()

    def _build_finetune_dataloader(self) -> None:
        _, run_stage2 = self._stage_flags()
        if not run_stage2:
            return
        return super()._build_finetune_dataloader()

    def _train_offline_ppo(self) -> None:
        _, run_stage2 = self._stage_flags()
        if not run_stage2:
            return
        return super()._train_offline_ppo()
