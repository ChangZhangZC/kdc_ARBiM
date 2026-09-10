from __future__ import annotations

import copy
import os

from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline

from .contract_workspace import TrainACTWorkspace as _ContractTrainACTWorkspace


class TrainACTWorkspace(_ContractTrainACTWorkspace):
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
