from __future__ import annotations

import copy
import os

from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .performance_workspace import TrainACTWorkspace as _PerformanceTrainACTWorkspace


class TrainACTWorkspace(_PerformanceTrainACTWorkspace):
    """Adapt post-RL loading to the Kuavo ACT checkpoint directory layout."""

    def _resolve_policy_processor_dir(self) -> str:
        checkpoint = os.path.abspath(str(self.cfg.input.policy_checkpoint))
        required = ("policy_preprocessor.json", "policy_postprocessor.json")
        candidates = (checkpoint, os.path.dirname(checkpoint))

        for candidate in candidates:
            if all(os.path.isfile(os.path.join(candidate, name)) for name in required):
                return candidate

        searched = "\n- ".join(candidates)
        raise FileNotFoundError(
            "Could not find ACT policy processor files. Expected both "
            "policy_preprocessor.json and policy_postprocessor.json in either:\n- "
            f"{searched}"
        )

    def _load_policy_processors(self):
        processor_dir = self._resolve_policy_processor_dir()
        self.policy_processor_dir = processor_dir
        self.policy_preprocessor = PolicyProcessorPipeline.from_pretrained(
            processor_dir,
            config_filename="policy_preprocessor.json",
        )
        self.policy_postprocessor = PolicyProcessorPipeline.from_pretrained(
            processor_dir,
            config_filename="policy_postprocessor.json",
        )
        normalizers = [
            step
            for step in self.policy_preprocessor.steps
            if isinstance(step, NormalizerProcessorStep)
        ]
        if len(normalizers) != 1:
            raise RuntimeError(
                "Expected exactly one NormalizerProcessorStep in ACT preprocessor, "
                f"got {len(normalizers)}."
            )
        stats = copy.deepcopy(normalizers[0].stats)
        if not stats:
            raise RuntimeError("ACT preprocessor does not contain normalization stats.")
        if self.rank == 0 and processor_dir != os.path.abspath(str(self.cfg.input.policy_checkpoint)):
            print(f"Using ACT policy processors from run directory: {processor_dir}")
        return stats

    def _build_finetune_dataloader(self) -> None:
        super()._build_finetune_dataloader()
        if self.finetune_sampler is not None:
            return

        cfg = self.cfg
        kwargs = self._dataloader_kwargs(
            cfg.dataloader,
            batch_size=int(cfg.unio4.finetune_batch_size),
            shuffle=False,
        )
        self.finetune_sampler = DistributedSampler(
            self.finetune_dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=int(cfg.training.seed),
            drop_last=False,
        )
        self.finetune_dataloader = DataLoader(
            self.finetune_dataset,
            sampler=self.finetune_sampler,
            **kwargs,
        )
        self._finetune_iter = None
        self._finetune_batch_in_epoch = 0
