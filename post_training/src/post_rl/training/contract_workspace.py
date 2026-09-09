import hashlib
import json
import os
from collections.abc import Mapping

import numpy as np
import torch
from lerobot.processor import NormalizerProcessorStep

from .resumable_workspace import TrainACTWorkspace as _ResumableTrainACTWorkspace


class TrainACTWorkspace(_ResumableTrainACTWorkspace):
    """Final Scheme-C contract guards layered on the resumable workspace."""

    def _validate_scheme_c_contract(self, cfg) -> None:
        super()._validate_scheme_c_contract(cfg)
        errors = []
        if not bool(cfg.dynamics.predict_delta):
            errors.append("dynamics.predict_delta must be true")
        if int(cfg.dataset.pad_before) != 0:
            errors.append("dataset.pad_before must be 0")
        if int(cfg.dataset.pad_after) != 0:
            errors.append("dataset.pad_after must be 0")
        if errors:
            raise ValueError(
                "Invalid Scheme C post-RL config:\n- " + "\n- ".join(errors)
            )

    def _build_act_observation_frontends(self) -> None:
        super()._build_act_observation_frontends()
        normalizers = [
            step
            for step in self.policy_preprocessor.steps
            if isinstance(step, NormalizerProcessorStep)
        ]
        if len(normalizers) != 1:
            raise RuntimeError(
                "Expected exactly one NormalizerProcessorStep while building the "
                "normalization contract."
            )
        normalizer = normalizers[0]
        normalizer_contract = {
            "stats": self.stats,
            "norm_map": {
                str(key): str(value)
                for key, value in normalizer.norm_map.items()
            },
            "features": {
                str(key): {
                    "type": str(feature.type),
                    "shape": list(feature.shape),
                }
                for key, feature in normalizer.features.items()
            },
        }
        self._normalizer_sha256 = self._fingerprint_stats(normalizer_contract)
        if self.rank == 0:
            print(
                "ACT normalization contract ready: "
                f"normalizer_sha256={self._normalizer_sha256[:12]}..."
            )

    @staticmethod
    def _fingerprint_stats(stats) -> str:
        digest = hashlib.sha256()

        def update(value, path: str) -> None:
            digest.update(path.encode("utf-8"))
            if torch.is_tensor(value):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(b"torch")
                digest.update(str(array.shape).encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(array.tobytes())
                return
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest.update(b"numpy")
                digest.update(str(array.shape).encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(array.tobytes())
                return
            if isinstance(value, np.generic):
                update(value.item(), path)
                return
            if isinstance(value, Mapping):
                digest.update(b"mapping")
                for key in sorted(value, key=lambda item: str(item)):
                    update(value[key], f"{path}/{key}")
                return
            if isinstance(value, (list, tuple)):
                digest.update(type(value).__name__.encode("utf-8"))
                for index, item in enumerate(value):
                    update(item, f"{path}/{index}")
                return
            if value is None or isinstance(value, (bool, int, float, str)):
                digest.update(type(value).__name__.encode("utf-8"))
                digest.update(repr(value).encode("utf-8"))
                return
            raise TypeError(
                f"Unsupported normalization-stat type at {path}: {type(value).__name__}"
            )

        update(stats, "stats")
        return digest.hexdigest()

    def _artifact_contract(self) -> dict:
        contract = super()._artifact_contract()
        normalizer_sha256 = getattr(self, "_normalizer_sha256", None)
        if normalizer_sha256 is None:
            raise RuntimeError("ACT normalization stats must be loaded first.")
        contract.update(
            {
                "normalizer_sha256": normalizer_sha256,
                "predict_delta": bool(self.cfg.dynamics.predict_delta),
                "pad_before": int(self.cfg.dataset.pad_before),
                "pad_after": int(self.cfg.dataset.pad_after),
            }
        )
        return contract

    def _validate_artifact_contract(self, directory: str, label: str) -> None:
        super()._validate_artifact_contract(directory, label)
        path = os.path.join(directory, "contract.json")
        with open(path, "r") as file:
            stored = json.load(file)
        current = self._artifact_contract()
        keys = (
            "normalizer_sha256",
            "predict_delta",
            "pad_before",
            "pad_after",
        )
        mismatch = {
            key: (stored.get(key), current.get(key))
            for key in keys
            if stored.get(key) != current.get(key)
        }
        if mismatch:
            raise RuntimeError(f"{label} contract mismatch: {mismatch}")
