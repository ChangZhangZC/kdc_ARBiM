import copy
import hashlib
import json
import os
from collections.abc import Mapping

import hydra
import numpy as np
import torch
from lerobot.processor import NormalizerProcessorStep

from post_rl.algorithms.offline_ppo import BehaviorProximalPolicyOptimization
from post_rl.critic.iql_critic import IQLCritic
from post_rl.dynamics.core.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from post_rl.dynamics.models.dynamics_model import EnsembleDynamicsModel
from post_rl.dynamics.trainer import train_dynamics
from post_rl.dynamics.utils.termination_fns import get_termination_fn
from .resumable_workspace import TrainACTWorkspace as _ResumableTrainACTWorkspace


class TrainACTWorkspace(_ResumableTrainACTWorkspace):
    """Final Scheme-C contract guards layered on the resumable workspace."""

    def _validate_scheme_c_contract(self, cfg) -> None:
        checkpoint_act_chunk_size = int(self.model.config.chunk_size)
        act_chunk_size = int(cfg.act_chunk_size)
        rl_chunk_size = int(cfg.rl_chunk_size)
        n_action_steps = int(cfg.n_action_steps)
        errors = []

        if min(act_chunk_size, rl_chunk_size, n_action_steps) < 1:
            errors.append("act_chunk_size, rl_chunk_size, and n_action_steps must be >= 1")
        if act_chunk_size != checkpoint_act_chunk_size:
            errors.append(
                f"act_chunk_size={act_chunk_size} must match the loaded ACT checkpoint "
                f"chunk_size={checkpoint_act_chunk_size}"
            )
        if rl_chunk_size != act_chunk_size:
            errors.append(
                f"rl_chunk_size={rl_chunk_size} must equal act_chunk_size={act_chunk_size}"
            )
        if n_action_steps != rl_chunk_size:
            errors.append(
                f"n_action_steps={n_action_steps} must equal rl_chunk_size={rl_chunk_size}"
            )
        if int(cfg.horizon) < n_action_steps:
            errors.append(
                f"horizon={cfg.horizon} must be >= n_action_steps={n_action_steps}"
            )
        if int(cfg.n_obs_steps) != 1:
            errors.append("n_obs_steps must be 1 in Scheme C V1")

        ratio_mode = str(cfg.offline_chunk_ratio_mode)
        adv_mode = str(cfg.offline_chunk_adv_mode)
        if ratio_mode not in {"scalar", "per_step"}:
            errors.append(
                "offline_chunk_ratio_mode must be one of: scalar, per_step"
            )
        if adv_mode not in {
            "scalar_iql",
            "per_step_vdelta",
            "chunk_vdelta_scalar",
            "chunk_vdelta_gae",
        }:
            errors.append(
                "offline_chunk_adv_mode must be one of: scalar_iql, "
                "per_step_vdelta, chunk_vdelta_scalar, chunk_vdelta_gae"
            )

        if str(cfg.dynamics_type) != "mlp":
            errors.append("dynamics_type must be 'mlp' in V1")
        if bool(cfg.predict_r):
            errors.append("predict_r must be false in V1")
        if str(cfg.dynamics.latent_mode) != "transformer_encoder":
            errors.append("dynamics.latent_mode must be 'transformer_encoder' in V1")
        if str(cfg.dynamics.prediction_mode) != "full":
            errors.append("dynamics.prediction_mode must be 'full' in V1")
        if not bool(cfg.dynamics.predict_delta):
            errors.append("dynamics.predict_delta must be true in V1")
        if not bool(cfg.dynamics.fix_encoder):
            errors.append("dynamics.fix_encoder must be true in V1")
        if str(cfg.critic.latent_readout) != "mean":
            errors.append("critic.latent_readout must be 'mean' in V1")
        if not bool(cfg.critic.is_iql):
            errors.append("critic.is_iql must be true in V1")
        if not bool(cfg.critic.is_share_encoder):
            errors.append("critic.is_share_encoder must be true in V1")
        if not bool(cfg.critic.fix_encoder):
            errors.append("critic.fix_encoder must be true in V1")
        if not bool(cfg.unio4.fix_encoder):
            errors.append("unio4.fix_encoder must be true in V1")
        if bool(cfg.unio4.use_gae):
            errors.append("unio4.use_gae must be false in V1")
        if int(cfg.dataset.pad_before) != 0:
            errors.append("dataset.pad_before must be 0 in V1")
        if int(cfg.dataset.pad_after) != 0:
            errors.append("dataset.pad_after must be 0 in V1")
        if int(cfg.dynamics.ope_rollout_length) < 1:
            errors.append("dynamics.ope_rollout_length must be >= 1")
        if int(cfg.critic.sequence_stride) < 1:
            errors.append("critic.sequence_stride must be >= 1")
        if int(cfg.unio4.bppo_steps) < 0:
            errors.append("unio4.bppo_steps must be >= 0")
        if int(cfg.unio4.eval_step) < 1:
            errors.append("unio4.eval_step must be >= 1")
        if int(cfg.unio4.checkpoint_every_steps) < 0:
            errors.append("unio4.checkpoint_every_steps must be >= 0")

        model_use_depth = bool(getattr(self.model.config, "use_depth", False))
        if bool(cfg.dataset.use_depth) != model_use_depth:
            errors.append(
                "dataset.use_depth must match the loaded ACT checkpoint use_depth setting"
            )
        if len(cfg.dynamics.dynamics_weight_decay) != (
            len(cfg.dynamics.dynamics_hidden_dims) + 1
        ):
            errors.append(
                "dynamics_weight_decay must have len(dynamics_hidden_dims)+1 values"
            )

        if errors:
            raise ValueError(
                "Invalid Scheme C post-RL config:\n- " + "\n- ".join(errors)
            )

    def _build_critic(self) -> None:
        cfg = self.cfg
        self.critic = IQLCritic(
            device=self.device,
            obs_encoder=self.critic_encoder,
            stats=self.stats,
            action_dim=self.action_dim,
            feature_dim=self.obs_feature_dim,
            q_hidden_dim=cfg.critic.q_hidden_dim,
            q_depth=cfg.critic.q_depth,
            q_lr=cfg.critic.q_lr,
            v_hidden_dim=cfg.critic.v_hidden_dim,
            v_depth=cfg.critic.v_depth,
            v_lr=cfg.critic.v_lr,
            omega=cfg.critic.omega,
            gamma=cfg.critic.gamma,
            tau=cfg.critic.tau,
            target_update_freq=cfg.critic.target_update_freq,
            is_double_q=cfg.critic.is_double_q,
            is_share_encoder=cfg.critic.is_share_encoder,
            fix_encoder=cfg.critic.fix_encoder,
            encoder_update_with="value",
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            chunk_as_single_action=cfg.chunk_as_single_action,
            use_action_embed=cfg.use_action_embed,
            use_conv_action_embed=cfg.use_conv_action_embed,
            conv_hidden_dims=list(cfg.conv_hidden_dims),
            conv_latent_cz=cfg.conv_latent_cz,
            conv_kernel_size=cfg.conv_kernel_size,
            conv_n_groups=cfg.conv_n_groups,
            action_recon_beta=cfg.action_recon_beta,
            q_layer_norm=cfg.critic.q_layer_norm,
            action_embed_layer_norm=cfg.critic.action_embed_layer_norm,
            action_scale_norm=cfg.critic.action_scale_norm,
        ).to(self.device)

    def _build_dynamics(self) -> None:
        cfg = self.cfg
        env = getattr(self.env_runner, "env", None) if self.env_runner is not None else None
        work_dir = os.path.join(self.get_dynamics_artifact_dir(), "work")
        chunk_as_single_action = bool(cfg.chunk_as_single_action)
        effective_action_steps = int(cfg.n_action_steps) if chunk_as_single_action else 1

        if self.rank == 0:
            self.dynamics = train_dynamics(
                env=env,
                obs_adapter=self.obs_adapter,
                dynamics_save_path=work_dir,
                cfg=cfg,
                feature_dim=self.obs_feature_dim,
                action_dim=self.action_dim,
                chunk_as_single_action=chunk_as_single_action,
                n_action_steps=cfg.n_action_steps,
                n_obs_steps=cfg.n_obs_steps,
                device=self.device,
            )
        else:
            model_action_dim = self.action_dim * effective_action_steps
            model_cfg = copy.deepcopy(cfg)
            if not chunk_as_single_action:
                model_cfg.n_action_steps = 1
            dynamics_model = EnsembleDynamicsModel(
                obs_dim=self.obs_feature_dim,
                action_dim=model_action_dim,
                hidden_dims=cfg.dynamics.dynamics_hidden_dims,
                num_ensemble=cfg.dynamics.n_ensemble,
                num_elites=cfg.dynamics.n_elites,
                weight_decays=cfg.dynamics.dynamics_weight_decay,
                with_reward=cfg.predict_r,
                device=self.device,
                cfg=model_cfg,
            )
            dynamics_optim = hydra.utils.instantiate(
                cfg.optimizer,
                params=dynamics_model.parameters(),
            )
            self.dynamics = EnsembleDynamics_batch(
                model=dynamics_model,
                optim=dynamics_optim,
                terminal_fn=get_termination_fn(cfg.task_name),
                env=env,
                obs_adapter=self.obs_adapter,
                cfg=cfg,
                action_dim=model_action_dim,
                gamma=cfg.critic.gamma,
                device=self.device,
                chunk_as_single_action=chunk_as_single_action,
                n_action_steps=cfg.n_action_steps,
                prediction_mode=cfg.dynamics.prediction_mode,
            )
        if self.is_ddp:
            torch.distributed.barrier()

    def _build_ppo(self) -> None:
        cfg = self.cfg
        self.unio4 = BehaviorProximalPolicyOptimization(
            policy=self.model,
            device=self.device,
            obs_adapter=self.obs_adapter,
            policy_lr=cfg.unio4.bppo_lr,
            clip_ratio=cfg.unio4.clip_ratio,
            entropy_weight=cfg.unio4.entropy_weight,
            decay=cfg.unio4.decay,
            temperature=cfg.unio4.temperature,
            fix_encoder=cfg.unio4.fix_encoder,
            dynamics=self.dynamics,
            cfg=cfg,
        )
        self.unio4.set_ratio_log_dir(
            os.path.join(self.get_ppo_artifact_dir(), "ratio_logs")
        )
        if self.rank == 0:
            trainable = sum(
                param.numel()
                for param in self.unio4._policy.parameters()
                if param.requires_grad
            )
            total = sum(param.numel() for param in self.unio4._policy.parameters())
            print(f"PPO Actor ready: trainable={trainable:,}, total={total:,}")

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
        expected_modes = {
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
            "VISUAL": "MEAN_STD",
            "RGB": "MEAN_STD",
            "DEPTH": "MIN_MAX",
        }
        mode_errors = []
        for key, feature in normalizer.features.items():
            feature_type = getattr(feature.type, "value", str(feature.type))
            expected = expected_modes.get(str(feature_type))
            if expected is None:
                continue
            actual_mode = normalizer.norm_map.get(feature.type)
            actual = getattr(actual_mode, "value", str(actual_mode))
            if str(actual) != expected:
                mode_errors.append(
                    f"{key}: {feature_type} requires {expected}, got {actual}"
                )
        if mode_errors:
            raise RuntimeError(
                "ACT normalization mapping is incompatible with the post-RL "
                "normalization adapter:\n- " + "\n- ".join(mode_errors)
            )

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
                "act_chunk_size": int(self.cfg.act_chunk_size),
                "rl_chunk_size": int(self.cfg.rl_chunk_size),
                "chunk_as_single_action": bool(self.cfg.chunk_as_single_action),
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
            "rl_chunk_size",
            "chunk_as_single_action",
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
