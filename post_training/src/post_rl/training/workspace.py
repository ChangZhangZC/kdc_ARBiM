import copy
import csv
import fcntl
import hashlib
import inspect
import json
import os
import random
from copy import deepcopy

import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from termcolor import cprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
from post_rl.algorithms.offline_ppo import BehaviorProximalPolicyOptimization
from post_rl.critic.iql_critic import IQLCritic
from post_rl.critic.networks import ACTCriticEncoder
from post_rl.data.offline_buffer import OfflineBuffer
from post_rl.data.offline_dataset import OfflineDataset
from post_rl.data.shared_memory_utils import (
    get_shared_memory_data,
    setup_shared_memory_dataset,
)
from post_rl.dynamics.core.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from post_rl.dynamics.models.dynamics_model import EnsembleDynamicsModel
from post_rl.dynamics.trainer import train_dynamics
from post_rl.dynamics.utils.act_obs_adapter import ACTObservationAdapter
from post_rl.dynamics.utils.termination_fns import get_termination_fn
from post_rl.policy.stochastic_act_config import StochasticACTConfigWrapper
from post_rl.policy.stochastic_act_policy import StochasticACTPolicyWrapper
from post_rl.utils.common import dict_apply
from post_rl.utils.ema import EMAModel


class _NoOpWandb:
    def log(self, *args, **kwargs):
        pass


def init_wandb_run(cfg: OmegaConf, output_dir: str):
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    init_timeout = int(logging_cfg.pop("init_timeout", 120))
    retry_init_timeout = int(
        logging_cfg.pop("retry_init_timeout", max(init_timeout * 2, 300))
    )
    settings_cfg = logging_cfg.pop("settings", {}) or {}

    def _init(timeout: int):
        settings = dict(settings_cfg)
        settings["init_timeout"] = timeout
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(**settings),
            **logging_cfg,
        )

    try:
        return _init(init_timeout)
    except wandb.errors.CommError as exc:
        if "timeout" not in str(exc).lower():
            raise
        cprint(
            f"[WandB] init timed out after {init_timeout}s; retrying with "
            f"{retry_init_timeout}s",
            "yellow",
        )
        return _init(retry_init_timeout)


class TrainACTWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.shm_manager = None
        self.wandb_run = None
        self.env_runner = None
        self.critic = None
        self.dynamics = None
        self.unio4 = None
        self.ema = None
        self.ema_model = None

        self.is_ddp = dist.is_available() and dist.is_initialized()
        if self.is_ddp:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            seed = int(cfg.training.seed) + self.rank
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device(cfg.training.device)
            seed = int(cfg.training.seed)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        stochastic_cfg = StochasticACTConfigWrapper.from_il_pretrained(
            cfg.input.policy_checkpoint,
            init_log_std=cfg.policy.init_log_std,
            log_std_min=cfg.policy.log_std_min,
            log_std_max=cfg.policy.log_std_max,
        )
        stochastic_cfg.device = str(self.device)
        self.model = StochasticACTPolicyWrapper.from_il_pretrained(
            cfg.input.policy_checkpoint,
            config=stochastic_cfg,
        ).to(self.device)
        self.model.eval()
        self._validate_scheme_c_contract(cfg)

        self.global_step = 0
        self._finetune_iter = None
        self._finetune_epoch = 0
        self._encoder_sha256 = None

        if self.rank == 0:
            cprint(
                f"Loaded Base ACT checkpoint: {cfg.input.policy_checkpoint}",
                "green",
            )
            cprint(
                f"Workspace device={self.device}, world_size={self.world_size}",
                "green",
            )

    @property
    def output_dir(self) -> str:
        if self._output_dir is not None:
            return self._output_dir
        return HydraConfig.get().runtime.output_dir

    def _validate_scheme_c_contract(self, cfg) -> None:
        act_chunk_size = int(self.model.config.chunk_size)
        errors = []

        if int(cfg.n_obs_steps) != 1:
            errors.append("n_obs_steps must be 1")
        if not bool(cfg.chunk_as_single_action):
            errors.append("chunk_as_single_action must be true")
        if int(cfg.n_action_steps) != act_chunk_size:
            errors.append(
                f"n_action_steps={cfg.n_action_steps} must equal ACT chunk_size={act_chunk_size}"
            )
        if int(cfg.horizon) != int(cfg.n_action_steps):
            errors.append(
                f"horizon={cfg.horizon} must equal n_action_steps={cfg.n_action_steps}"
            )
        if str(cfg.dynamics_type) != "mlp":
            errors.append("dynamics_type must be 'mlp'")
        if bool(cfg.predict_r):
            errors.append("predict_r must be false")
        if str(cfg.dynamics.latent_mode) != "transformer_encoder":
            errors.append("dynamics.latent_mode must be 'transformer_encoder'")
        if str(cfg.dynamics.prediction_mode) != "full":
            errors.append("dynamics.prediction_mode must be 'full'")
        if not bool(cfg.dynamics.fix_encoder):
            errors.append("dynamics.fix_encoder must be true")
        if str(cfg.critic.latent_readout) != "mean":
            errors.append("critic.latent_readout must be 'mean'")
        if not bool(cfg.critic.is_iql):
            errors.append("critic.is_iql must be true")
        if not bool(cfg.critic.is_share_encoder):
            errors.append("critic.is_share_encoder must be true")
        if not bool(cfg.critic.fix_encoder):
            errors.append("critic.fix_encoder must be true")
        if not bool(cfg.unio4.fix_encoder):
            errors.append("unio4.fix_encoder must be true")
        if bool(cfg.unio4.use_gae):
            errors.append("unio4.use_gae must be false")
        if int(cfg.dynamics.ope_rollout_length) < 1:
            errors.append("dynamics.ope_rollout_length must be >= 1")
        if int(cfg.critic.sequence_stride) < 1:
            errors.append("critic.sequence_stride must be >= 1")
        model_use_depth = bool(getattr(self.model.config, "use_depth", False))
        if bool(cfg.dataset.use_depth) != model_use_depth:
            errors.append(
                "dataset.use_depth must match the Base ACT checkpoint use_depth setting"
            )
        if len(cfg.dynamics.dynamics_weight_decay) != (
            len(cfg.dynamics.dynamics_hidden_dims) + 1
        ):
            errors.append(
                "dynamics_weight_decay must have len(dynamics_hidden_dims)+1 values"
            )

        if errors:
            raise ValueError("Invalid Scheme C post-RL config:\n- " + "\n- ".join(errors))

    def _apply_debug_overrides(self, cfg) -> None:
        if not bool(cfg.training.debug):
            return
        cfg.training.num_critic_epochs = 1
        cfg.dynamics.dynamics_max_epochs = 1
        cfg.dynamics.max_epochs_since_update = 1
        cfg.unio4.bppo_steps = min(int(cfg.unio4.bppo_steps), 10)
        cfg.dataloader.num_workers = 0
        cfg.dataloader.persistent_workers = False
        cfg.val_dataloader.num_workers = 0
        cfg.val_dataloader.persistent_workers = False
        cfg.use_wandb = False

    def get_stage1_artifact_dir(self) -> str:
        path = self.cfg.unio4.get("stage1_resume_dir", None)
        return str(path) if path else self.output_dir

    def get_critic_artifact_dir(self) -> str:
        path = self.cfg.critic.get("artifact_dir", None)
        return str(path) if path else os.path.join(self.get_stage1_artifact_dir(), "critic")

    def get_dynamics_artifact_dir(self) -> str:
        path = self.cfg.dynamics.get("artifact_dir", None)
        return str(path) if path else os.path.join(self.get_stage1_artifact_dir(), "dynamics")

    def get_ppo_artifact_dir(self) -> str:
        path = self.cfg.unio4.get("artifact_dir", None)
        return str(path) if path else os.path.join(self.output_dir, "ppo")

    def get_global_best_dir(self) -> str:
        path = self.cfg.unio4.get("global_best_dir", None)
        return str(path) if path else os.path.join(self.get_ppo_artifact_dir(), "best")

    def get_global_best_ema_dir(self) -> str:
        path = self.cfg.unio4.get("global_best_ema_dir", None)
        return str(path) if path else os.path.join(self.get_ppo_artifact_dir(), "best_ema")

    def _load_policy_processors(self):
        checkpoint = self.cfg.input.policy_checkpoint
        self.policy_preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_preprocessor.json",
        )
        self.policy_postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
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
        return stats

    def _build_act_observation_frontends(self) -> None:
        self.stats = self._load_policy_processors()
        self.critic_encoder = ACTCriticEncoder(
            self.model.model,
            copy_model=True,
        ).to(self.device)
        dynamics_encoder = ACTCriticEncoder(
            self.model.model,
            copy_model=True,
        ).to(self.device)
        self.obs_adapter = ACTObservationAdapter(
            encoder=dynamics_encoder,
            stats=self.stats,
            n_obs_steps=self.cfg.n_obs_steps,
            device=self.device,
            fix_encoder=self.cfg.dynamics.fix_encoder,
        )
        self.action_dim = int(self.model.config.action_feature.shape[0])
        self.obs_feature_dim = int(self.critic_encoder.output_dim)
        self._encoder_sha256 = self._fingerprint_module(self.critic_encoder)

        if self.rank == 0:
            print(
                "ACT Scheme C encoder ready: "
                f"latent_dim={self.obs_feature_dim}, action_dim={self.action_dim}, "
                f"encoder_sha256={self._encoder_sha256[:12]}..."
            )

    @staticmethod
    def _fingerprint_module(module: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _artifact_contract(self) -> dict:
        if self._encoder_sha256 is None:
            raise RuntimeError("ACT observation frontend must be built first.")
        return {
            "scheme": "act_transformer_encoder_v1",
            "encoder_sha256": self._encoder_sha256,
            "act_chunk_size": int(self.model.config.chunk_size),
            "horizon": int(self.cfg.horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
            "action_dim": int(self.action_dim),
            "latent_dim": int(self.obs_feature_dim),
            "use_depth": bool(self.cfg.dataset.use_depth),
            "policy_checkpoint": str(self.cfg.input.policy_checkpoint),
        }

    def _write_artifact_contract(self, directory: str) -> None:
        with open(os.path.join(directory, "contract.json"), "w") as file:
            json.dump(self._artifact_contract(), file, indent=2, sort_keys=True)

    def _validate_artifact_contract(self, directory: str, label: str) -> None:
        path = os.path.join(directory, "contract.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} contract metadata not found: {path}. Retrain or migrate the artifact."
            )
        with open(path, "r") as file:
            stored = json.load(file)
        current = self._artifact_contract()
        keys = (
            "scheme",
            "encoder_sha256",
            "act_chunk_size",
            "horizon",
            "n_action_steps",
            "action_dim",
            "latent_dim",
            "use_depth",
        )
        mismatch = {
            key: (stored.get(key), current.get(key))
            for key in keys
            if stored.get(key) != current.get(key)
        }
        if mismatch:
            raise RuntimeError(f"{label} contract mismatch: {mismatch}")

    def _dataloader_kwargs(self, cfg, **overrides) -> dict:
        kwargs = OmegaConf.to_container(cfg, resolve=True)
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        kwargs.pop("sampler", None)
        if int(kwargs.get("num_workers", 0)) == 0 and kwargs.get(
            "persistent_workers", False
        ):
            raise ValueError("persistent_workers=true requires num_workers>0")
        return kwargs

    def _per_rank_batch_size(self, global_batch_size: int, name: str) -> int:
        global_batch_size = int(global_batch_size)
        if not self.is_ddp:
            return global_batch_size
        if global_batch_size < self.world_size:
            raise ValueError(f"{name}={global_batch_size} must be >= world_size={self.world_size}")
        if global_batch_size % self.world_size != 0:
            raise ValueError(
                f"{name}={global_batch_size} must be divisible by world_size={self.world_size}"
            )
        return global_batch_size // self.world_size

    def _load_buffer(self) -> OfflineBuffer:
        cfg = self.cfg
        use_shared_memory = (
            self.is_ddp
            and bool(cfg.use_shared_memory)
            and os.environ.get("DISABLE_SHARED_MEMORY", "").lower() != "true"
        )
        info_file = os.path.join(self.output_dir, "shared_memory_info_path.txt")

        if use_shared_memory and self.rank == 0:
            try:
                info_path, self.shm_manager = setup_shared_memory_dataset(
                    cfg.input.dataset_path,
                    keys=None,
                )
                with open(info_file, "w") as file:
                    file.write(info_path)
            except Exception as exc:
                cprint(
                    f"Shared-memory setup failed ({exc}); falling back to Zarr loading.",
                    "yellow",
                )
                with open(info_file, "w") as file:
                    file.write("DISABLED")

        if self.is_ddp:
            dist.barrier()

        buffer = OfflineBuffer(
            device=self.device,
            gamma=cfg.critic.gamma,
            use_depth=cfg.dataset.use_depth,
        )

        if use_shared_memory:
            with open(info_file, "r") as file:
                info_path = file.read().strip()
            if info_path == "DISABLED":
                use_shared_memory = False
            else:
                shared_data, attached_manager = get_shared_memory_data(info_path)
                if self.rank != 0:
                    self.shm_manager = attached_manager
                dataset_data = dict(shared_data["data"])
                dataset_data["episode_ends"] = np.asarray(
                    shared_data["meta"]["episode_ends"],
                    dtype=np.int64,
                )
                buffer.load_dataset(dataset_data)

        if not use_shared_memory:
            buffer.load_zarr(cfg.input.dataset_path)

        if cfg.dataset.reward_scaling == "none":
            buffer.compute_return()
        else:
            buffer.reward_normalize(
                scaling=cfg.dataset.reward_scaling,
                fixed_scale=cfg.dataset.fixed_reward_scale,
            )
        return buffer

    def _build_main_dataloaders(self) -> None:
        cfg = self.cfg
        self.dataset = OfflineDataset(
            buffer=self.buffer,
            horizon=cfg.horizon,
            pad_before=cfg.dataset.pad_before,
            pad_after=cfg.dataset.pad_after,
            sequence_stride=cfg.dataset.sequence_stride,
            seed=cfg.training.seed,
            val_ratio=cfg.dataset.val_ratio,
            max_train_episodes=cfg.dataset.max_train_episodes,
            use_depth=cfg.dataset.use_depth,
        )
        self.val_dataset = self.dataset.get_validation_dataset()

        if self.is_ddp:
            self.train_sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            train_kwargs = self._dataloader_kwargs(
                cfg.dataloader,
                batch_size=self._per_rank_batch_size(
                    cfg.dataloader.batch_size,
                    "dataloader.batch_size",
                ),
                shuffle=False,
                drop_last=True,
            )
            self.train_dataloader = DataLoader(
                self.dataset,
                sampler=self.train_sampler,
                **train_kwargs,
            )
        else:
            self.train_sampler = None
            self.train_dataloader = DataLoader(
                self.dataset,
                **self._dataloader_kwargs(cfg.dataloader),
            )

        if len(self.val_dataset) == 0:
            raise RuntimeError("Dynamics training requires a non-empty validation split.")

        if self.is_ddp:
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            val_kwargs = self._dataloader_kwargs(
                cfg.val_dataloader,
                batch_size=self._per_rank_batch_size(
                    cfg.val_dataloader.batch_size,
                    "val_dataloader.batch_size",
                ),
                shuffle=False,
                drop_last=False,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                sampler=self.val_sampler,
                **val_kwargs,
            )
        else:
            self.val_sampler = None
            self.val_dataloader = DataLoader(
                self.val_dataset,
                **self._dataloader_kwargs(
                    cfg.val_dataloader,
                    shuffle=False,
                    drop_last=False,
                ),
            )

    def _build_critic_dataset(self):
        cfg = self.cfg
        sequence_stride = int(cfg.critic.sequence_stride)
        dataset = OfflineDataset(
            buffer=self.buffer,
            horizon=cfg.horizon,
            pad_before=cfg.dataset.pad_before,
            pad_after=cfg.dataset.pad_after,
            sequence_stride=sequence_stride,
            seed=cfg.training.seed,
            val_ratio=0.0,
            max_train_episodes=cfg.dataset.max_train_episodes,
            use_depth=cfg.dataset.use_depth,
        )
        kwargs = self._dataloader_kwargs(cfg.dataloader, drop_last=True)
        if self.is_ddp:
            kwargs["batch_size"] = self._per_rank_batch_size(
                cfg.dataloader.batch_size,
                "dataloader.batch_size",
            )
            kwargs["shuffle"] = False
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        else:
            sampler = None
            dataloader = DataLoader(dataset, **kwargs)
        self.critic_sampler = sampler
        if self.rank == 0:
            print(f"Critic dataset: {len(dataset)} samples (stride={sequence_stride})")
        return dataset, dataloader

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
            is_share_encoder=True,
            fix_encoder=True,
            encoder_update_with="value",
            n_obs_steps=1,
            n_action_steps=cfg.n_action_steps,
            chunk_as_single_action=True,
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

    def _wrap_critic_ddp(self) -> None:
        if not self.is_ddp:
            return
        self.critic._Q = DDP(
            self.critic._Q,
            device_ids=[self.local_rank],
            find_unused_parameters=True,
        )
        self.critic._value = DDP(
            self.critic._value,
            device_ids=[self.local_rank],
            find_unused_parameters=True,
        )
        self.critic._q_optimizer = torch.optim.Adam(
            self.critic._Q.parameters(),
            lr=self.cfg.critic.q_lr,
        )
        self.critic._v_optimizer = torch.optim.Adam(
            self.critic._value.parameters(),
            lr=self.cfg.critic.v_lr,
        )

    def _unwrap_critic_ddp(self) -> None:
        if hasattr(self.critic._Q, "module"):
            self.critic._Q = self.critic._Q.module
        if hasattr(self.critic._value, "module"):
            self.critic._value = self.critic._value.module

    def _save_critic_checkpoint(self, name: str) -> None:
        if self.rank != 0:
            return
        directory = os.path.join(
            self.get_critic_artifact_dir(),
            "checkpoints",
            name,
        )
        os.makedirs(directory, exist_ok=True)
        self.critic.save(
            q_path=os.path.join(directory, "Q.pt"),
            v_path=os.path.join(directory, "value.pt"),
            encoder_path=os.path.join(directory, "encoder.pt"),
        )
        self._write_artifact_contract(directory)

    def _load_critic_if_needed(self) -> bool:
        if not bool(self.cfg.critic.load_pretrain):
            return False
        directory = os.path.join(
            self.get_critic_artifact_dir(),
            "checkpoints",
            "final",
        )
        self._validate_artifact_contract(directory, "Critic")
        self.critic.load(
            q_path=os.path.join(directory, "Q.pt"),
            v_path=os.path.join(directory, "value.pt"),
            encoder_path=os.path.join(directory, "encoder.pt"),
        )
        self.critic.eval()
        if self.rank == 0:
            print(f"Loaded Critic from {directory}")
        return True

    def _train_critic(self, dataloader) -> None:
        cfg = self.cfg
        self._wrap_critic_ddp()
        artifact_dir = self.get_critic_artifact_dir()
        metrics_path = os.path.join(artifact_dir, "metrics.csv")
        if self.rank == 0:
            os.makedirs(artifact_dir, exist_ok=True)
            write_header = not os.path.exists(metrics_path)
        else:
            write_header = False

        for epoch in range(int(cfg.training.num_critic_epochs)):
            if self.is_ddp and self.critic_sampler is not None:
                self.critic_sampler.set_epoch(epoch)
            iterator = (
                tqdm.tqdm(
                    dataloader,
                    desc=f"Training Critic epoch {epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                )
                if self.rank == 0
                else dataloader
            )
            q_sum = 0.0
            v_sum = 0.0
            count = 0
            for batch in iterator:
                batch = dict_apply(
                    batch,
                    lambda x: x.to(self.device, non_blocking=True),
                )
                q_loss, v_loss = self.critic.update(batch)
                q_sum += float(q_loss)
                v_sum += float(v_loss)
                count += 1

            metrics = torch.tensor(
                [q_sum, v_sum, float(count)],
                device=self.device,
                dtype=torch.float64,
            )
            if self.is_ddp:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            denom = max(float(metrics[2].item()), 1.0)
            q_mean = float(metrics[0].item() / denom)
            v_mean = float(metrics[1].item() / denom)

            if self.rank == 0:
                print(f"Critic epoch {epoch}: Q loss={q_mean:.6f}, Value loss={v_mean:.6f}")
                with open(metrics_path, "a", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=["epoch", "Q_loss", "value_loss"],
                    )
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(
                        {"epoch": epoch + 1, "Q_loss": q_mean, "value_loss": v_mean}
                    )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"critic/Q_loss": q_mean, "critic/value_loss": v_mean}
                    )
                save_every = int(cfg.critic.save_every_epochs)
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    self._save_critic_checkpoint(f"epoch_{epoch + 1:04d}")

        self._save_critic_checkpoint("final")
        if self.is_ddp:
            dist.barrier()
        self._unwrap_critic_ddp()
        self.critic.eval()

    def _build_dynamics(self) -> None:
        cfg = self.cfg
        env = getattr(self.env_runner, "env", None) if self.env_runner is not None else None
        work_dir = os.path.join(self.get_dynamics_artifact_dir(), "work")
        if self.rank == 0:
            self.dynamics = train_dynamics(
                env=env,
                obs_adapter=self.obs_adapter,
                dynamics_save_path=work_dir,
                cfg=cfg,
                feature_dim=self.obs_feature_dim,
                action_dim=self.action_dim,
                chunk_as_single_action=True,
                n_action_steps=cfg.n_action_steps,
                n_obs_steps=1,
                device=self.device,
            )
        else:
            model_action_dim = self.action_dim * int(cfg.n_action_steps)
            dynamics_model = EnsembleDynamicsModel(
                obs_dim=self.obs_feature_dim,
                action_dim=model_action_dim,
                hidden_dims=cfg.dynamics.dynamics_hidden_dims,
                num_ensemble=cfg.dynamics.n_ensemble,
                num_elites=cfg.dynamics.n_elites,
                weight_decays=cfg.dynamics.dynamics_weight_decay,
                with_reward=False,
                device=self.device,
                cfg=cfg,
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
                chunk_as_single_action=True,
                n_action_steps=cfg.n_action_steps,
                prediction_mode="full",
            )
        if self.is_ddp:
            dist.barrier()

    def _wrap_dynamics_ddp(self) -> None:
        if not self.is_ddp:
            return
        self.dynamics.model = DDP(
            self.dynamics.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        self.dynamics.optim = hydra.utils.instantiate(
            self.cfg.optimizer,
            params=self.dynamics.model.parameters(),
        )

    def _unwrap_dynamics_ddp(self) -> None:
        if hasattr(self.dynamics.model, "module"):
            self.dynamics.model = self.dynamics.model.module

    def _save_dynamics_checkpoint(self, name: str) -> None:
        if self.rank != 0:
            return
        directory = os.path.join(
            self.get_dynamics_artifact_dir(),
            "checkpoints",
            name,
        )
        os.makedirs(directory, exist_ok=True)
        self.dynamics.save(directory)
        self._write_artifact_contract(directory)

    def _load_dynamics_if_needed(self) -> bool:
        if not bool(self.cfg.dynamics.load_pretrain):
            return False
        directory = os.path.join(
            self.get_dynamics_artifact_dir(),
            "checkpoints",
            "final",
        )
        self._validate_artifact_contract(directory, "Dynamics")
        self.dynamics.load(directory)
        self.dynamics.model.eval()
        if self.rank == 0:
            print(f"Loaded Dynamics from {directory}")
        return True

    def _dynamics_validation_metrics(self) -> list[float]:
        cfg = self.cfg
        model = self.dynamics._model()
        local_sum = torch.zeros(
            model.num_ensemble,
            device=self.device,
            dtype=torch.float64,
        )
        local_count = torch.zeros(1, device=self.device, dtype=torch.float64)

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = dict_apply(
                    batch,
                    lambda x: x.to(self.device, non_blocking=True),
                )
                state = self.dynamics.obs2latent(batch["obs"])
                next_state = self.dynamics.next_obs2latent(batch["next_obs"])
                batch_size = state.shape[0]
                inputs, targets = self.dynamics.format_samples_for_training(
                    batch,
                    state.reshape(batch_size, -1),
                    next_state.reshape(batch_size, -1),
                )
                losses = torch.as_tensor(
                    self.dynamics.validate(inputs, targets),
                    device=self.device,
                    dtype=torch.float64,
                )
                local_sum += losses * batch_size
                local_count += batch_size

        if self.is_ddp:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        if local_count.item() <= 0:
            raise RuntimeError("Dynamics validation produced zero samples.")
        return (local_sum / local_count).cpu().tolist()

    def _train_dynamics(self) -> None:
        cfg = self.cfg
        self._wrap_dynamics_ddp()
        artifact_dir = self.get_dynamics_artifact_dir()
        metrics_path = os.path.join(artifact_dir, "metrics.csv")
        if self.rank == 0:
            os.makedirs(artifact_dir, exist_ok=True)
            write_header = not os.path.exists(metrics_path)
        else:
            write_header = False
        wandb_logger = self.wandb_run if self.wandb_run is not None else _NoOpWandb()

        for epoch in range(int(cfg.dynamics.dynamics_max_epochs)):
            if self.is_ddp and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            iterator = (
                tqdm.tqdm(
                    self.train_dataloader,
                    desc=f"Training Dynamics epoch {epoch}",
                    leave=False,
                    mininterval=cfg.dynamics.tqdm_interval_sec,
                )
                if self.rank == 0
                else self.train_dataloader
            )
            loss_sum = 0.0
            count = 0
            for batch in iterator:
                batch = dict_apply(
                    batch,
                    lambda x: x.to(self.device, non_blocking=True),
                )
                state = self.dynamics.obs2latent(batch["obs"])
                next_state = self.dynamics.next_obs2latent(batch["next_obs"])
                batch_size = state.shape[0]
                loss = self.dynamics.learn(
                    batch=batch,
                    nobs_features=state.reshape(batch_size, -1),
                    next_nobs_features=next_state.reshape(batch_size, -1),
                )
                self.dynamics.optimize(loss)
                loss_sum += float(loss.detach().item())
                count += 1

            train_metrics = torch.tensor(
                [loss_sum, float(count)],
                device=self.device,
                dtype=torch.float64,
            )
            if self.is_ddp:
                dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            train_loss = float(
                train_metrics[0].item() / max(train_metrics[1].item(), 1.0)
            )

            holdout_losses = self._dynamics_validation_metrics()
            should_stop = torch.zeros(1, device=self.device, dtype=torch.int32)
            if self.rank == 0:
                model = self.dynamics._model()
                holdout_loss = float(
                    np.sort(holdout_losses)[: model.num_elites].mean()
                )
                stop = self.dynamics._update_holdout_and_log(
                    new_holdout_losses=holdout_losses,
                    train_loss=train_loss,
                    wandb=wandb_logger,
                    epoch=epoch + 1,
                    max_epochs_since_update=cfg.dynamics.max_epochs_since_update,
                    max_epochs=cfg.dynamics.dynamics_max_epochs,
                )
                with open(metrics_path, "a", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=["epoch", "train_loss", "holdout_loss"],
                    )
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "holdout_loss": holdout_loss,
                        }
                    )
                print(
                    f"Dynamics epoch {epoch + 1}: train={train_loss:.6f}, "
                    f"holdout={holdout_loss:.6f}"
                )
                save_every = int(cfg.dynamics.save_every_epochs)
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    self._save_dynamics_checkpoint(f"epoch_{epoch + 1:04d}")
                if stop:
                    should_stop.fill_(1)

            if self.is_ddp:
                dist.broadcast(should_stop, src=0)
            if should_stop.item():
                break

        if self.is_ddp:
            dist.barrier()
        self._unwrap_dynamics_ddp()
        if self.rank == 0:
            self.dynamics.post_well_learned()
            self._save_dynamics_checkpoint("best")
            self._save_dynamics_checkpoint("final")
        if self.is_ddp:
            dist.barrier()
        final_dir = os.path.join(
            self.get_dynamics_artifact_dir(),
            "checkpoints",
            "final",
        )
        self.dynamics.load(final_dir)
        self.dynamics.model.eval()

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
            fix_encoder=True,
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

    def _build_ema(self) -> None:
        if not bool(self.cfg.training.use_ema):
            self.ema = None
            self.ema_model = None
            return
        self.ema_model = deepcopy(self.unio4._policy).to(self.device)
        self.ema = EMAModel(
            model=self.ema_model,
            update_after_step=self.cfg.ema.update_after_step,
            inv_gamma=self.cfg.ema.inv_gamma,
            power=self.cfg.ema.power,
            min_value=self.cfg.ema.min_value,
            max_value=self.cfg.ema.max_value,
        )

    def _build_finetune_dataloader(self) -> None:
        cfg = self.cfg
        stride = int(cfg.dataset.finetune_sequence_stride)
        dataset = OfflineDataset(
            buffer=self.buffer,
            horizon=cfg.horizon,
            pad_before=cfg.dataset.pad_before,
            pad_after=cfg.dataset.pad_after,
            sequence_stride=stride,
            seed=cfg.training.seed,
            val_ratio=0.0,
            max_train_episodes=cfg.dataset.max_train_episodes,
            use_depth=cfg.dataset.use_depth,
        )
        kwargs = self._dataloader_kwargs(
            cfg.dataloader,
            batch_size=int(cfg.unio4.finetune_batch_size),
        )
        if self.is_ddp:
            kwargs["batch_size"] = self._per_rank_batch_size(
                cfg.unio4.finetune_batch_size,
                "unio4.finetune_batch_size",
            )
            kwargs["shuffle"] = False
            self.finetune_sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            self.finetune_dataloader = DataLoader(
                dataset,
                sampler=self.finetune_sampler,
                **kwargs,
            )
        else:
            self.finetune_sampler = None
            self.finetune_dataloader = DataLoader(dataset, **kwargs)
        self.finetune_dataset = dataset
        self._finetune_iter = None
        self._finetune_epoch = 0
        self.unio4.set_old_policy()
        if self.rank == 0:
            print(
                f"Finetune dataset: {len(dataset)} samples "
                f"(stride={stride}, batch_size={cfg.unio4.finetune_batch_size})"
            )

    def sample_finetune_batch(self):
        if self._finetune_iter is None:
            if self.finetune_sampler is not None:
                self.finetune_sampler.set_epoch(self._finetune_epoch)
            self._finetune_iter = iter(self.finetune_dataloader)
        try:
            batch = next(self._finetune_iter)
        except StopIteration:
            self._finetune_epoch += 1
            if self.finetune_sampler is not None:
                self.finetune_sampler.set_epoch(self._finetune_epoch)
            self._finetune_iter = iter(self.finetune_dataloader)
            batch = next(self._finetune_iter)
        return dict_apply(
            batch,
            lambda x: x.to(self.device, non_blocking=True),
        )

    def _save_policy_bundle(self, policy, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        policy.save_pretrained(directory)
        self.policy_preprocessor.save_pretrained(
            directory,
            config_filename="policy_preprocessor.json",
        )
        self.policy_postprocessor.save_pretrained(
            directory,
            config_filename="policy_postprocessor.json",
        )

    @staticmethod
    def _read_score(path: str) -> float:
        if not os.path.isfile(path):
            return float("-inf")
        return float(np.asarray(np.loadtxt(path, delimiter=",")).reshape(-1)[0])

    def _maybe_update_best(self, score: float, directory: str, policy) -> tuple[float, bool]:
        if self.rank != 0:
            return score, False
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        score_path = os.path.join(directory, "best_score.csv")
        lock_path = os.path.join(os.path.dirname(directory), ".best.lock")
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            best = self._read_score(score_path)
            updated = score > best
            if updated:
                os.makedirs(directory, exist_ok=True)
                self._save_policy_bundle(policy, directory)
                np.savetxt(score_path, [score], fmt="%f", delimiter=",")
                best = score
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return best, updated

    @torch.no_grad()
    def eval_policy(self, policy=None, eval_times: int | None = None) -> dict:
        if self.is_ddp:
            dist.barrier()
        if self.rank != 0:
            if self.is_ddp:
                dist.barrier()
            return {"test_mean_score": 0.0, "mean_returns": 0.0}
        if self.env_runner is None:
            raise RuntimeError("Evaluation requires task.env_runner.")

        if policy is None:
            if self.unio4 is not None:
                policy = self.unio4._policy
            else:
                policy = self.model
        policy.eval()
        eval_times = int(eval_times or self.cfg.unio4.eval_times)
        try:
            run_params = inspect.signature(self.env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        kwargs = {}
        if "eval_env_num" in run_params:
            kwargs["eval_env_num"] = int(self.cfg.ppo.eval_env_num)

        scores = []
        returns = []
        for _ in range(eval_times):
            result = self.env_runner.run(policy, **kwargs)
            scores.append(float(result["test_mean_score"]))
            returns.append(float(result["mean_returns"]))
        output = {
            "test_mean_score": float(np.mean(scores)),
            "mean_returns": float(np.mean(returns)),
        }
        if self.is_ddp:
            dist.barrier()
        return output

    @torch.no_grad()
    def _evaluate_dynamics_ope(self) -> tuple[float, float]:
        batch = self.sample_finetune_batch()
        mean_q, mean_reward = self.dynamics.rollout(
            policy=self.unio4._policy,
            Q=self.critic.minQ,
            iql=self.critic,
            batch=batch,
            rollout_length=int(self.cfg.dynamics.ope_rollout_length),
            is_iql=True,
            use_gae=False,
            first_action=False,
        )
        return float(mean_q.detach().cpu().item()), float(mean_reward)

    def _train_offline_ppo(self) -> None:
        cfg = self.cfg
        steps = int(cfg.unio4.bppo_steps)
        if steps <= 0:
            return
        self.critic.eval()
        self.dynamics.model.eval()
        self.unio4._policy.eval()
        self.unio4._old_policy.eval()
        ppo_dir = self.get_ppo_artifact_dir()
        os.makedirs(ppo_dir, exist_ok=True)

        run_env_eval = self.env_runner is not None
        if run_env_eval:
            initial_eval = self.eval_policy(self.unio4._policy)
            if self.rank == 0:
                self._maybe_update_best(
                    initial_eval["test_mean_score"],
                    self.get_global_best_dir(),
                    self.unio4._policy,
                )

        best_mean_q, initial_reward = self._evaluate_dynamics_ope()
        opes = [best_mean_q]
        if self.rank == 0:
            print(
                f"Initial Dynamics OPE: mean_q={best_mean_q:.6f}, "
                f"mean_reward={initial_reward:.6f}"
            )

        iterator = (
            tqdm.tqdm(
                range(steps),
                desc="BPPO updating",
                mininterval=cfg.training.tqdm_interval_sec,
            )
            if self.rank == 0
            else range(steps)
        )
        decay_stop_step = int(cfg.unio4.decay_stop_step)

        for step in iterator:
            if self.is_ddp:
                dist.barrier()
            decay_active = decay_stop_step < 0 or step <= decay_stop_step
            linear_active = bool(cfg.unio4.is_linear_decay) and decay_active
            if linear_active:
                progress = step / max(steps, 1)
                bppo_lr_now = cfg.unio4.bppo_lr * (1.0 - progress)
                clip_ratio_now = cfg.unio4.clip_ratio * (1.0 - progress)
            else:
                bppo_lr_now = None
                clip_ratio_now = None

            batch = self.sample_finetune_batch()
            loss = self.unio4.update_distribution(
                batch=batch,
                critic=self.critic,
                is_clip_decay=bool(cfg.unio4.is_clip_decay) and decay_active,
                is_lr_decay=bool(cfg.unio4.is_bppo_lr_decay) and decay_active,
                is_linear_decay=linear_active,
                bppo_lr_now=bppo_lr_now,
                clip_ratio_now=clip_ratio_now,
            )
            if self.ema is not None:
                self.ema.step(self.unio4._policy)
            self.global_step += 1

            if self.rank == 0 and self.wandb_run is not None:
                self.wandb_run.log({"dpg_loss": loss})

            if run_env_eval and self.global_step % int(cfg.unio4.eval_freq) == 0:
                policy = (
                    self.ema_model
                    if bool(cfg.unio4.use_ema_eval) and self.ema_model is not None
                    else self.unio4._policy
                )
                eval_data = self.eval_policy(policy)
                if self.rank == 0:
                    self._maybe_update_best(
                        eval_data["test_mean_score"],
                        self.get_global_best_dir(),
                        policy,
                    )
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "current_bppo_scores": eval_data["test_mean_score"],
                                "mean_returns": eval_data["mean_returns"],
                            }
                        )

            if self.global_step % int(cfg.unio4.eval_step) == 0:
                current_mean_q, mean_reward = self._evaluate_dynamics_ope()
                if current_mean_q > best_mean_q and bool(cfg.unio4.is_update_old_policy):
                    best_mean_q = current_mean_q
                    self.unio4.set_old_policy()
                    if self.rank == 0:
                        print(
                            "Updated PPO old policy: "
                            f"step={self.global_step}, mean_q={current_mean_q:.6f}"
                        )
                opes.append(current_mean_q)
                if self.rank == 0:
                    print(
                        f"Dynamics OPE: step={self.global_step}, "
                        f"mean_q={current_mean_q:.6f}, best_q={best_mean_q:.6f}"
                    )
                    np.savetxt(
                        os.path.join(ppo_dir, "each_ope_score.csv"),
                        opes,
                        fmt="%f",
                        delimiter=",",
                    )
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "current_mean_qs": current_mean_q,
                                "ope_mean_reward": mean_reward,
                            }
                        )

        if self.rank == 0:
            np.savetxt(
                os.path.join(ppo_dir, "last_ope_score.csv"),
                opes,
                fmt="%f",
                delimiter=",",
            )
            self._save_policy_bundle(
                self.unio4._policy,
                os.path.join(ppo_dir, "last"),
            )
            self.unio4.flush_ratio_logs(force=True)
        if self.is_ddp:
            dist.barrier()

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self._apply_debug_overrides(cfg)
        self._validate_scheme_c_contract(cfg)
        self.cfg = cfg

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(self.output_dir, "config.yaml"))
        if self.is_ddp:
            dist.barrier()

        self.buffer = self._load_buffer()
        self._build_main_dataloaders()
        self._build_act_observation_frontends()

        task_cfg = cfg.get("task")
        env_runner_cfg = None if task_cfg is None else task_cfg.get("env_runner")
        if env_runner_cfg is not None:
            self.env_runner = hydra.utils.instantiate(
                env_runner_cfg,
                output_dir=self.output_dir,
            )

        if bool(cfg.use_wandb) and self.rank == 0:
            cfg.logging.name = str(cfg.logging.name)
            self.wandb_run = init_wandb_run(cfg, self.output_dir)

        _, critic_dataloader = self._build_critic_dataset()
        self._build_critic()
        if not self._load_critic_if_needed():
            self._train_critic(critic_dataloader)
        self.critic.eval()

        self._build_dynamics()
        if not self._load_dynamics_if_needed():
            self._train_dynamics()
        self.dynamics.model.eval()

        if int(cfg.unio4.bppo_steps) > 0 or bool(cfg.eval):
            self._build_ppo()
            self._build_ema()

        if bool(cfg.eval):
            if self.env_runner is None:
                raise RuntimeError("Evaluation requires task.env_runner.")
            policy = (
                self.ema_model
                if bool(cfg.unio4.use_ema_eval) and self.ema_model is not None
                else self.unio4._policy
            )
            return self.eval_policy(policy)["test_mean_score"]

        if int(cfg.unio4.bppo_steps) > 0:
            self._build_finetune_dataloader()
            self._train_offline_ppo()
        return None

    def cleanup_shared_memory(self) -> None:
        if self.shm_manager is None:
            return
        try:
            self.shm_manager.cleanup()
        finally:
            self.shm_manager = None
            if self.rank == 0:
                info_file = os.path.join(
                    self.output_dir,
                    "shared_memory_info_path.txt",
                )
                if os.path.exists(info_file):
                    os.remove(info_file)
