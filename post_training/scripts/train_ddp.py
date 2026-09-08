import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

if not LEROBOT_SRC.is_dir():
    raise RuntimeError(
        "LeRobot submodule is not initialized. "
        "Run `git submodule update --init --recursive`."
    )

for path in (REPO_ROOT, POST_TRAINING_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.chdir(REPO_ROOT)

import lerobot_patches.custom_patches

import copy
import csv
import fcntl
import inspect
import random
import time
import warnings
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
from post_rl.dynamics.core.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from post_rl.dynamics.models.dynamics_model import EnsembleDynamicsModel
from post_rl.dynamics.trainer import train_dynamics
from post_rl.dynamics.utils.act_obs_adapter import ACTObservationAdapter
from post_rl.dynamics.utils.termination_fns import get_termination_fn
from post_rl.policy.stochastic_act_config import StochasticACTConfigWrapper
from post_rl.policy.stochastic_act_policy import StochasticACTPolicyWrapper
from post_rl.utils.common import dict_apply
from post_rl.utils.ema import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)
warnings.filterwarnings("ignore")
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"


def init_wandb_run(cfg: OmegaConf, output_dir: str):
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    init_timeout = int(logging_cfg.pop("init_timeout", 120))
    retry_init_timeout = int(
        logging_cfg.pop("retry_init_timeout", max(init_timeout * 2, 300))
    )
    settings_cfg = logging_cfg.pop("settings", {}) or {}

    def _wandb_init(timeout: int):
        settings = dict(settings_cfg)
        settings["init_timeout"] = timeout
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(**settings),
            **logging_cfg,
        )

    try:
        return _wandb_init(init_timeout)
    except wandb.errors.CommError as exc:
        if "timeout" not in str(exc).lower():
            raise
        cprint(
            f"[WandB] init timed out after {init_timeout}s, "
            f"retrying with {retry_init_timeout}s",
            "yellow",
        )
        return _wandb_init(retry_init_timeout)


class _NoOpWandb:
    def log(self, *args, **kwargs):
        pass


def setup_ddp() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class TrainACTWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.shm_manager = None

        self.is_ddp = dist.is_available() and dist.is_initialized()
        if self.is_ddp:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            seed = cfg.training.seed + self.rank
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device(cfg.training.device)
            seed = cfg.training.seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        stochastic_cfg = StochasticACTConfigWrapper.from_il_pretrained(
            cfg.input.policy_checkpoint,
            init_log_std=getattr(cfg.policy, "init_log_std", None),
            log_std_min=getattr(cfg.policy, "log_std_min", None),
            log_std_max=getattr(cfg.policy, "log_std_max", None),
        )
        stochastic_cfg.device = str(self.device)

        self.model = StochasticACTPolicyWrapper.from_il_pretrained(
            cfg.input.policy_checkpoint,
            config=stochastic_cfg,
        )
        self.model.to(self.device)
        self.model.eval()

        self.model_module = self.model
        self.ema_model = None
        self.unio4 = None
        self.critic = None
        self.dynamics = None
        self.obs_adapter = None
        self.env_runner = None

        self.global_step = 0
        self.epoch = 0
        self._train_iter = None
        self._finetune_iter = None

        if self.rank == 0:
            cprint(
                f"Loaded Base ACT checkpoint: {cfg.input.policy_checkpoint}",
                "green",
            )
            cprint(
                f"Workspace device={self.device}, world_size={self.world_size}",
                "green",
            )

    def get_stage1_artifact_dir(self) -> str:
        stage1_dir = self.cfg.unio4.get(
            "stage1_resume_dir",
            None,
        )
        if stage1_dir:
            return stage1_dir
        return self.output_dir

    def get_critic_artifact_dir(self) -> str:
        explicit_dir = self.cfg.critic.get(
            "artifact_dir",
            None,
        )
        if explicit_dir:
            return explicit_dir

        return os.path.join(
            self.get_stage1_artifact_dir(),
            "critic",
        )

    def get_dynamics_artifact_dir(self) -> str:
        explicit_dir = self.cfg.dynamics.get(
            "artifact_dir",
            None,
        )
        if explicit_dir:
            return explicit_dir

        return os.path.join(
            self.get_stage1_artifact_dir(),
            "dynamics",
        )

    def get_ppo_artifact_dir(self) -> str:
        explicit_dir = self.cfg.unio4.get(
            "artifact_dir",
            None,
        )
        if explicit_dir:
            return explicit_dir

        return os.path.join(
            self.output_dir,
            "ppo",
        )

    def get_global_best_dir(self) -> str:
        explicit_dir = self.cfg.unio4.get("global_best_dir", None)
        if explicit_dir:
            return explicit_dir
        return os.path.join(self.get_ppo_artifact_dir(), "best")

    def get_global_best_score_path(self) -> str:
        return os.path.join(self.get_global_best_dir(), "best_score.csv")

    def get_global_best_lock_path(self) -> str:
        best_dir = self.get_global_best_dir()
        return os.path.join(os.path.dirname(best_dir), ".global_best.lock")

    @staticmethod
    def _read_best_score(score_path: str) -> float:
        if not os.path.exists(score_path):
            return float("-inf")
        best_score = np.loadtxt(score_path, delimiter=",")
        if isinstance(best_score, np.ndarray):
            return float(np.asarray(best_score).reshape(-1)[0])
        return float(best_score)

    def _maybe_update_best(
        self,
        score: float,
        best_dir: str,
        best_score_path: str,
        lock_path: str,
        save_fn,
        eval_name: str,
    ) -> tuple[float, bool]:
        os.makedirs(os.path.dirname(best_dir), exist_ok=True)
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            best_saved_score = self._read_best_score(best_score_path)
            is_updated = score > best_saved_score

            if is_updated:
                os.makedirs(best_dir, exist_ok=True)
                save_fn(best_dir)
                np.savetxt(
                    best_score_path,
                    [score],
                    fmt="%f",
                    delimiter=",",
                )
                meta_path = os.path.join(best_dir, "best_meta.txt")
                with open(meta_path, "w") as f:
                    f.write(f"score: {score}\n")
                    f.write(f"eval_name: {eval_name}\n")
                    f.write(f"source_run_dir: {self.output_dir}\n")
                    f.write(f"seed: {self.cfg.training.seed}\n")
                    f.write(f"bppo_lr: {self.cfg.unio4.bppo_lr}\n")
            else:
                score = best_saved_score

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return score, is_updated

    def maybe_update_global_best(self, score: float) -> tuple[float, bool]:
        if self.unio4 is None:
            raise RuntimeError("PPO must be initialized before saving best policy.")

        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_dir(),
            best_score_path=self.get_global_best_score_path(),
            lock_path=self.get_global_best_lock_path(),
            save_fn=lambda path: self._save_policy_bundle(self.unio4._policy, path),
            eval_name="Current Policy Eval",
        )

    def maybe_update_global_best_ema(self, score: float):
        if self.ema_model is None:
            return float("-inf"), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_ema_dir(),
            best_score_path=self.get_global_best_ema_score_path(),
            lock_path=self.get_global_best_ema_lock_path(),
            save_fn=lambda path: self._save_policy_bundle(self.ema_model, path),
            eval_name="EMA Policy Eval",
        )

    @property
    def output_dir(self) -> str:
        if self._output_dir is not None:
            return self._output_dir
        return HydraConfig.get().runtime.output_dir

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
                "Expected exactly one NormalizerProcessorStep in "
                f"ACT preprocessor, got {len(normalizers)}."
            )

        stats = copy.deepcopy(normalizers[0].stats)
        if not stats:
            raise RuntimeError(
                "ACT preprocessor does not contain normalization stats."
            )

        return stats

    def _build_act_observation_frontends(self, cfg) -> None:
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
            n_obs_steps=cfg.n_obs_steps,
            device=self.device,
            fix_encoder=cfg.dynamics.fix_encoder,
        )

        self.action_dim = self.model.config.action_feature.shape[0]
        self.obs_feature_dim = self.critic_encoder.output_dim
        if self.rank == 0:
            print(
                f"ACT observation frontend ready: "
                f"feature_dim={self.obs_feature_dim}, "
                f"action_dim={self.action_dim}"
            )

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.max_train_steps = 10
            verbose = True
        else:
            verbose = False

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            config_path = os.path.join(self.output_dir, "config.yaml")
            OmegaConf.save(cfg, config_path)

        if self.is_ddp:
            dist.barrier()

        use_shared_memory = (
            self.is_ddp
            and cfg.get("use_shared_memory", False)
            and os.environ.get("DISABLE_SHARED_MEMORY", "").lower() != "true"
        )

        self.shm_manager = None
        shared_info_file = os.path.join(
            self.output_dir,
            "shared_memory_info_path.txt",
        )

        if use_shared_memory and self.rank == 0:
            try:
                from shared_memory_utils import setup_shared_memory_dataset

                info_path, self.shm_manager = setup_shared_memory_dataset(
                    cfg.input.dataset_path,
                    keys=None,
                )
                with open(shared_info_file, "w") as f:
                    f.write(info_path)
                print(
                    f"[Rank 0] Shared memory setup complete: {info_path}"
                )
            except Exception as exc:
                print(
                    f"[Rank 0] Shared memory setup failed: {exc}. "
                    "Falling back to regular loading."
                )
                with open(shared_info_file, "w") as f:
                    f.write("DISABLED")

        if self.is_ddp:
            dist.barrier()

        buffer = OfflineBuffer(
            device=self.device,
            gamma=cfg.critic.gamma,
            use_depth=cfg.dataset.use_depth,
        )

        if use_shared_memory:
            with open(shared_info_file, "r") as f:
                info_path = f.read().strip()

            if info_path == "DISABLED":
                use_shared_memory = False
            else:
                from shared_memory_utils import get_shared_memory_data

                shared_data, attached_manager = get_shared_memory_data(
                    info_path
                )
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

        reward_scaling = cfg.dataset.get(
            "reward_scaling",
            "none",
        )
        if reward_scaling == "none":
            buffer.compute_return()
        else:
            buffer.reward_normalize(
                scaling=reward_scaling,
                fixed_scale=cfg.dataset.get(
                    "fixed_reward_scale",
                    0.1,
                ),
            )

        dataset = OfflineDataset(
            buffer=buffer,
            horizon=cfg.horizon,
            pad_before=cfg.dataset.get("pad_before", 0),
            pad_after=cfg.dataset.get("pad_after", 0),
            sequence_stride=cfg.dataset.get("sequence_stride", 1),
            seed=cfg.training.seed,
            val_ratio=cfg.dataset.get("val_ratio", 0.0),
            max_train_episodes=cfg.dataset.get(
                "max_train_episodes",
                None,
            ),
            use_depth=cfg.dataset.use_depth,
        )

        self.buffer = buffer
        self.dataset = dataset
        self.val_dataset = dataset.get_validation_dataset()

        def _safe_dataloader_cfg(
            base_cfg,
            batch_size=None,
            shuffle=None,
            drop_last=None,
        ):
            dataloader_cfg = OmegaConf.to_container(
                base_cfg,
                resolve=True,
            )
            if batch_size is not None:
                dataloader_cfg["batch_size"] = batch_size
            if shuffle is not None:
                dataloader_cfg["shuffle"] = shuffle
            if drop_last is not None:
                dataloader_cfg["drop_last"] = drop_last

            dataloader_cfg["pin_memory"] = False
            dataloader_cfg["persistent_workers"] = False
            dataloader_cfg["num_workers"] = min(
                dataloader_cfg.get("num_workers", 8),
                2,
            )
            dataloader_cfg.pop("sampler", None)
            return dataloader_cfg

        if self.is_ddp:
            global_batch_size = cfg.dataloader.batch_size
            per_gpu_batch_size = global_batch_size // self.world_size

            if per_gpu_batch_size < 1:
                per_gpu_batch_size = 1
                if self.rank == 0:
                    print(
                        f"Warning: batch_size={global_batch_size} < "
                        f"world_size={self.world_size}; using 1 per GPU."
                    )

            self.train_sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            train_cfg = _safe_dataloader_cfg(
                cfg.dataloader,
                batch_size=per_gpu_batch_size,
                shuffle=False,
                drop_last=True,
            )
            self.train_dataloader = DataLoader(
                dataset,
                sampler=self.train_sampler,
                **train_cfg,
            )

            if self.rank == 0:
                print(
                    f"DDP batch size: {per_gpu_batch_size} per GPU, "
                    f"{per_gpu_batch_size * self.world_size} effective."
                )
        else:
            self.train_sampler = None
            self.train_dataloader = DataLoader(
                dataset,
                **_safe_dataloader_cfg(cfg.dataloader),
            )

        if len(self.val_dataset) > 0:
            if self.is_ddp:
                global_val_batch_size = cfg.val_dataloader.batch_size
                per_gpu_val_batch_size = max(
                    1,
                    global_val_batch_size // self.world_size,
                )
                self.val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False,
                )
                val_cfg = _safe_dataloader_cfg(
                    cfg.val_dataloader,
                    batch_size=per_gpu_val_batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    sampler=self.val_sampler,
                    **val_cfg,
                )
            else:
                self.val_sampler = None
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    **_safe_dataloader_cfg(
                        cfg.val_dataloader,
                        shuffle=False,
                        drop_last=False,
                    ),
                )
        else:
            self.val_sampler = None
            self.val_dataloader = None

        if self.rank == 0 and verbose:
            print(f"Dataset transitions: {len(buffer)}")
            print(f"Train sequences: {len(dataset)}")
            print(f"Validation sequences: {len(self.val_dataset)}")

        self._build_act_observation_frontends(cfg)

        if self.is_ddp:
            dist.barrier()

        task_cfg = cfg.get("task")
        env_runner_cfg = None if task_cfg is None else task_cfg.get("env_runner")
        if env_runner_cfg is not None:
            self.env_runner = hydra.utils.instantiate(
                env_runner_cfg,
                output_dir=self.output_dir,
            )

        self.wandb_run = None
        if cfg.get("use_wandb", False) and self.rank == 0:
            cfg.logging.name = str(cfg.logging.name)
            self.wandb_run = init_wandb_run(
                cfg,
                self.output_dir,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True,
            )

        self.critic_dataset, critic_dataloader = (
            self._build_critic_dataset(cfg)
        )

        self._build_critic(cfg)

        critic_loaded = self._load_critic_if_needed(cfg)

        if not critic_loaded:
            self._train_critic(
                cfg,
                critic_dataloader,
                self.wandb_run,
            )

        self.q_eval = self.critic.minQ

        self._build_dynamics(cfg)

        dynamics_loaded = self._load_dynamics_if_needed(cfg)

        if not dynamics_loaded:
            self._train_dynamics(
                cfg,
                self.wandb_run,
            )
        else:
            self.dynamics.model.eval()

        bppo_steps = int(
            cfg.unio4.get(
                "bppo_steps",
                0,
            )
        )

        if bppo_steps > 0 or cfg.get("eval", False):
            self._build_ppo(cfg)
            self._build_ema(cfg)

        if cfg.get("eval", False):
            if self.env_runner is None:
                raise RuntimeError("Evaluation requires task.env_runner.")
            if cfg.unio4.idql_eval:
                log_data = self.unio4_eval(
                    idql_eval=True,
                    dynamics=self.dynamics,
                    first_action=cfg.unio4.first_action,
                    get_np=True,
                    use_gae=cfg.unio4.use_gae,
                    iql=self.critic,
                    Q=self.q_eval,
                    repeat_num=128,
                    eval_times=cfg.unio4.eval_times,
                )
            else:
                log_data = self.eval(eval_times=cfg.unio4.eval_times)
            return log_data["test_mean_score"]

        if bppo_steps > 0:
            self._build_finetune_dataloader(cfg)
            self._train_offline_ppo(cfg)

    def _build_critic_dataset(self, cfg):
        if not cfg.chunk_as_single_action:
            return self.dataset, self.train_dataloader

        critic_dataset = OfflineDataset(
            buffer=self.buffer,
            horizon=cfg.horizon,
            pad_before=cfg.dataset.get("pad_before", 0),
            pad_after=cfg.dataset.get("pad_after", 0),
            sequence_stride=cfg.n_action_steps,
            seed=cfg.training.seed,
            val_ratio=0.0,
            max_train_episodes=cfg.dataset.get("max_train_episodes", None),
            use_depth=cfg.dataset.use_depth,
        )

        dataloader_cfg = OmegaConf.to_container(
            cfg.dataloader,
            resolve=True,
        )
        dataloader_cfg["pin_memory"] = False
        dataloader_cfg["persistent_workers"] = False
        dataloader_cfg["num_workers"] = min(
            dataloader_cfg.get("num_workers", 8),
            2,
        )
        dataloader_cfg["drop_last"] = True
        dataloader_cfg.pop("sampler", None)

        if self.is_ddp:
            global_batch_size = dataloader_cfg["batch_size"]
            dataloader_cfg["batch_size"] = max(
                1,
                global_batch_size // self.world_size,
            )
            dataloader_cfg["shuffle"] = False
            sampler = DistributedSampler(
                critic_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            dataloader = DataLoader(
                critic_dataset,
                sampler=sampler,
                **dataloader_cfg,
            )
        else:
            sampler = None
            dataloader = DataLoader(
                critic_dataset,
                **dataloader_cfg,
            )

        self.critic_sampler = sampler

        if self.rank == 0:
            print(
                f"Critic dataset: {len(critic_dataset)} samples "
                f"(stride={cfg.n_action_steps})"
            )

        return critic_dataset, dataloader

    def _build_critic(self, cfg) -> None:
        self.critic = IQLCritic(
            device=self.device,
            obs_encoder=self.critic_encoder,
            stats=self.stats,
            action_dim=self.action_dim,
            feature_dim=cfg.critic.get(
                "feature_dim",
                self.obs_feature_dim,
            ),
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
            encoder_update_with=cfg.critic.get(
                "encoder_update_with",
                "value",
            ),
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            chunk_as_single_action=cfg.chunk_as_single_action,
            use_action_embed=cfg.get("use_action_embed", False),
            use_conv_action_embed=cfg.get(
                "use_conv_action_embed",
                False,
            ),
            conv_hidden_dims=cfg.get(
                "conv_hidden_dims",
                [128, 256],
            ),
            conv_latent_cz=cfg.get("conv_latent_cz", 32),
            conv_kernel_size=cfg.get("conv_kernel_size", 5),
            conv_n_groups=cfg.get("conv_n_groups", 8),
            action_recon_beta=cfg.get(
                "action_recon_beta",
                0.5,
            ),
            q_layer_norm=cfg.critic.get(
                "q_layer_norm",
                False,
            ),
            action_embed_layer_norm=cfg.critic.get(
                "action_embed_layer_norm",
                False,
            ),
            action_scale_norm=cfg.critic.get(
                "action_scale_norm",
                False,
            ),
        )

        self.critic.to(self.device)

    def _wrap_critic_ddp(self, cfg) -> None:
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
            lr=cfg.critic.q_lr,
        )
        self.critic._v_optimizer = torch.optim.Adam(
            self.critic._value.parameters(),
            lr=cfg.critic.v_lr,
        )

    def _save_critic_checkpoint(
        self,
        checkpoint_name: str,
    ) -> None:
        if self.rank != 0:
            return

        save_dir = os.path.join(
            self.get_critic_artifact_dir(),
            "checkpoints",
            checkpoint_name,
        )
        os.makedirs(save_dir, exist_ok=True)

        encoder_path = (
            os.path.join(save_dir, "encoder.pt")
            if self.critic.is_share_encoder
            else None
        )
        self.critic.save(
            q_path=os.path.join(save_dir, "Q.pt"),
            v_path=os.path.join(save_dir, "value.pt"),
            encoder_path=encoder_path,
        )

    def _train_critic(
        self,
        cfg,
        critic_dataloader,
        wandb_run=None,
    ) -> None:
        self._wrap_critic_ddp(cfg)

        metrics_path = os.path.join(
            self.get_critic_artifact_dir(),
            "metrics.csv",
        )

        if self.rank == 0:
            os.makedirs(
                self.get_critic_artifact_dir(),
                exist_ok=True,
            )
            write_header = not os.path.exists(metrics_path)
        else:
            write_header = False

        save_every = int(
            cfg.critic.get("save_every_epochs", 0)
        )

        for local_epoch_idx in range(
            cfg.training.num_critic_epochs
        ):
            if self.is_ddp:
                dist.barrier()

            if (
                self.is_ddp
                and hasattr(critic_dataloader, "sampler")
                and hasattr(
                    critic_dataloader.sampler,
                    "set_epoch",
                )
            ):
                critic_dataloader.sampler.set_epoch(
                    local_epoch_idx
                )

            q_train_losses = []
            v_train_losses = []

            if self.rank == 0:
                tepoch = tqdm.tqdm(
                    critic_dataloader,
                    desc=f"Training Critic epoch {local_epoch_idx}",
                    leave=False,
                    mininterval=cfg.training.get(
                        "tqdm_interval_sec",
                        1.0,
                    ),
                )
            else:
                tepoch = critic_dataloader

            for batch in tepoch:
                batch = dict_apply(
                    batch,
                    lambda x: x.to(
                        self.device,
                        non_blocking=True,
                    ),
                )
                q_loss, value_loss = self.critic.update(
                    batch=batch
                )
                q_train_losses.append(float(q_loss))
                v_train_losses.append(float(value_loss))

            if self.is_ddp:
                q_loss_tensor = torch.tensor(
                    q_train_losses,
                    device=self.device,
                )
                v_loss_tensor = torch.tensor(
                    v_train_losses,
                    device=self.device,
                )

                gathered_q = [
                    torch.zeros_like(q_loss_tensor)
                    for _ in range(self.world_size)
                ]
                gathered_v = [
                    torch.zeros_like(v_loss_tensor)
                    for _ in range(self.world_size)
                ]

                dist.all_gather(
                    gathered_q,
                    q_loss_tensor,
                )
                dist.all_gather(
                    gathered_v,
                    v_loss_tensor,
                )

                q_loss_mean = torch.cat(
                    gathered_q
                ).mean().item()
                v_loss_mean = torch.cat(
                    gathered_v
                ).mean().item()
            else:
                q_loss_mean = float(
                    np.mean(q_train_losses)
                )
                v_loss_mean = float(
                    np.mean(v_train_losses)
                )

            if self.rank == 0:
                print(
                    f"Critic epoch {local_epoch_idx}: "
                    f"Q loss={q_loss_mean:.6f}, "
                    f"Value loss={v_loss_mean:.6f}"
                )

                with open(
                    metrics_path,
                    "a",
                    newline="",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "Q_loss",
                            "value_loss",
                        ],
                    )
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(
                        {
                            "epoch": local_epoch_idx + 1,
                            "Q_loss": q_loss_mean,
                            "value_loss": v_loss_mean,
                        }
                    )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "critic/Q_loss": q_loss_mean,
                            "critic/value_loss": v_loss_mean,
                        }
                    )

            if (
                save_every > 0
                and (local_epoch_idx + 1) % save_every == 0
            ):
                self._save_critic_checkpoint(
                    f"epoch_{local_epoch_idx + 1:04d}"
                )

            if self.is_ddp:
                dist.barrier()

        self._save_critic_checkpoint("final")

        if self.is_ddp:
            dist.barrier()

        self._unwrap_critic_ddp()
        self.critic.eval()

    def _load_critic_if_needed(self, cfg) -> bool:
        if not cfg.critic.get("load_pretrain", False):
            return False

        final_dir = os.path.join(
            self.get_critic_artifact_dir(),
            "checkpoints",
            "final",
        )
        q_path = os.path.join(final_dir, "Q.pt")
        v_path = os.path.join(final_dir, "value.pt")
        encoder_path = (
            os.path.join(final_dir, "encoder.pt")
            if self.critic.is_share_encoder
            else None
        )

        if not os.path.isfile(q_path):
            raise FileNotFoundError(
                f"Critic Q artifact not found: {q_path}"
            )
        if not os.path.isfile(v_path):
            raise FileNotFoundError(
                f"Critic Value artifact not found: {v_path}"
            )
        if encoder_path is not None and not os.path.isfile(encoder_path):
            raise FileNotFoundError(
                f"Critic encoder artifact not found: {encoder_path}"
            )

        self.critic.load(
            q_path=q_path,
            v_path=v_path,
            encoder_path=encoder_path,
        )
        self.critic.eval()

        if self.rank == 0:
            print(f"Loaded Critic from {final_dir}")

        return True

    def _unwrap_critic_ddp(self) -> None:
        if hasattr(self.critic._Q, "module"):
            self.critic._Q = self.critic._Q.module
        if hasattr(self.critic._value, "module"):
            self.critic._value = self.critic._value.module

    def _build_dynamics(self, cfg) -> None:
        prediction_mode = cfg.dynamics.get(
            "prediction_mode",
            "last",
        )

        if cfg.chunk_as_single_action and prediction_mode != "full":
            raise ValueError(
                "chunk_as_single_action=True requires "
                "dynamics.prediction_mode='full'."
            )

        if cfg.dynamics_type == "diffusion":
            raise NotImplementedError(
                "ARBiM V1 supports Ensemble MLP Dynamics only; "
                "Diffusion Dynamics is intentionally out of scope for V1."
            )

        env_runner = getattr(self, "env_runner", None)
        env = getattr(env_runner, "env", None)
        work_dir = os.path.join(
            self.get_dynamics_artifact_dir(),
            "work",
        )

        if self.rank == 0:
            self.dynamics = train_dynamics(
                env=env,
                obs_adapter=self.obs_adapter,
                dynamics_save_path=work_dir,
                cfg=cfg,
                feature_dim=self.obs_feature_dim,
                action_dim=self.action_dim,
                chunk_as_single_action=cfg.chunk_as_single_action,
                n_action_steps=cfg.n_action_steps,
                n_obs_steps=cfg.n_obs_steps,
                device=self.device,
            )
        else:
            model_action_dim = self.action_dim
            if cfg.chunk_as_single_action:
                model_action_dim *= cfg.n_action_steps

            output_obs_dim = self.obs_feature_dim
            if prediction_mode == "full":
                output_obs_dim *= cfg.n_obs_steps

            dynamics_model = EnsembleDynamicsModel(
                obs_dim=output_obs_dim,
                action_dim=model_action_dim,
                hidden_dims=cfg.dynamics.dynamics_hidden_dims,
                num_ensemble=cfg.dynamics.n_ensemble,
                num_elites=cfg.dynamics.n_elites,
                weight_decays=cfg.dynamics.dynamics_weight_decay,
                with_reward=cfg.predict_r,
                device=self.device,
                cfg=cfg,
            )

            if cfg.dynamics.fix_encoder:
                dynamics_params = dynamics_model.parameters()
            else:
                dynamics_params = list(dynamics_model.parameters()) + list(
                    self.obs_adapter.encoder.parameters()
                )
            dynamics_optim = hydra.utils.instantiate(
                cfg.optimizer,
                params=dynamics_params,
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
                chunk_as_single_action=cfg.chunk_as_single_action,
                n_action_steps=cfg.n_action_steps,
                prediction_mode=prediction_mode,
            )

        if self.is_ddp:
            dist.barrier()

    def _load_dynamics_if_needed(self, cfg) -> bool:
        if not cfg.dynamics.get("load_pretrain", False):
            return False

        final_dir = os.path.join(
            self.get_dynamics_artifact_dir(),
            "checkpoints",
            "final",
        )
        model_path = os.path.join(
            final_dir,
            "dynamics.pth",
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Dynamics artifact not found: {model_path}"
            )

        self.dynamics.load(final_dir)

        if self.rank == 0:
            print(f"Loaded Dynamics from {final_dir}")

        return True

    def _wrap_dynamics_ddp(self, cfg) -> None:
        if not self.is_ddp:
            return

        self.dynamics.model = DDP(
            self.dynamics.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

        if cfg.dynamics.fix_encoder:
            dynamics_params = self.dynamics.model.parameters()
        else:
            dynamics_params = list(self.dynamics.model.parameters()) + list(
                self.obs_adapter.encoder.parameters()
            )
        self.dynamics.optim = hydra.utils.instantiate(
            cfg.optimizer,
            params=dynamics_params,
        )

    def _unwrap_dynamics_ddp(self) -> None:
        if hasattr(self.dynamics.model, "module"):
            self.dynamics.model = self.dynamics.model.module

    def _save_dynamics_checkpoint(
        self,
        checkpoint_name: str,
    ) -> None:
        if self.rank != 0:
            return

        save_dir = os.path.join(
            self.get_dynamics_artifact_dir(),
            "checkpoints",
            checkpoint_name,
        )
        os.makedirs(save_dir, exist_ok=True)
        self.dynamics.save(save_dir)

    def _train_dynamics(
        self,
        cfg,
        wandb_run=None,
    ) -> None:
        if self.val_dataloader is None:
            raise RuntimeError(
                "Dynamics training requires a validation split."
            )

        self._wrap_dynamics_ddp(cfg)

        artifact_dir = self.get_dynamics_artifact_dir()
        metrics_path = os.path.join(
            artifact_dir,
            "metrics.csv",
        )

        if self.rank == 0:
            os.makedirs(artifact_dir, exist_ok=True)
            write_header = not os.path.exists(metrics_path)
        else:
            write_header = False

        prediction_mode = cfg.dynamics.get(
            "prediction_mode",
            "last",
        )
        save_every = int(
            cfg.dynamics.get(
                "save_every_epochs",
                0,
            )
        )
        wandb_logger = (
            wandb_run
            if wandb_run is not None
            else _NoOpWandb()
        )

        for local_epoch_idx in range(
            cfg.dynamics.dynamics_max_epochs
        ):
            if self.is_ddp:
                dist.barrier()
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(
                        local_epoch_idx
                    )

            if self.rank == 0:
                tepoch = tqdm.tqdm(
                    self.train_dataloader,
                    desc=(
                        f"Training Dynamics epoch "
                        f"{local_epoch_idx}"
                    ),
                    leave=False,
                    mininterval=cfg.dynamics.get(
                        "tqdm_interval_sec",
                        1.0,
                    ),
                )
            else:
                tepoch = self.train_dataloader

            dynamics_losses = []

            for batch in tepoch:
                batch = dict_apply(
                    batch,
                    lambda x: x.to(
                        self.device,
                        non_blocking=True,
                    ),
                )

                nobs_features = self.dynamics.obs2latent(
                    batch["obs"]
                )

                if cfg.chunk_as_single_action:
                    next_nobs_features = (
                        self.dynamics.next_obs2latent(
                            batch["next_obs"]
                        )
                    )
                else:
                    next_nobs_features = (
                        self.dynamics.obs2latent(
                            batch["next_obs"]
                        )
                    )

                if prediction_mode == "full":
                    batch_size = nobs_features.shape[0]
                    train_nobs = nobs_features.reshape(
                        batch_size,
                        -1,
                    )
                    train_next_nobs = (
                        next_nobs_features.reshape(
                            batch_size,
                            -1,
                        )
                    )
                else:
                    train_nobs = nobs_features[:, -1]
                    train_next_nobs = (
                        next_nobs_features[:, -1]
                    )

                dynamics_loss = self.dynamics.learn(
                    batch=batch,
                    nobs_features=train_nobs,
                    next_nobs_features=train_next_nobs,
                )
                self.dynamics.optimize(dynamics_loss)

                dynamics_losses.append(
                    float(dynamics_loss.detach().item())
                )

            local_train_loss = float(
                np.mean(dynamics_losses)
            )

            train_loss_tensor = torch.tensor(
                local_train_loss,
                device=self.device,
                dtype=torch.float32,
            )

            if self.is_ddp:
                dist.all_reduce(
                    train_loss_tensor,
                    op=dist.ReduceOp.SUM,
                )
                train_loss_tensor /= self.world_size

            train_loss = train_loss_tensor.item()

            should_stop = torch.zeros(
                1,
                device=self.device,
                dtype=torch.int32,
            )

            if self.rank == 0:
                holdout_batches = []

                with torch.no_grad():
                    for val_batch in self.val_dataloader:
                        val_batch = dict_apply(
                            val_batch,
                            lambda x: x.to(
                                self.device,
                                non_blocking=True,
                            ),
                        )

                        val_nobs = self.dynamics.obs2latent(
                            val_batch["obs"]
                        )

                        if cfg.chunk_as_single_action:
                            val_next_nobs = (
                                self.dynamics.next_obs2latent(
                                    val_batch["next_obs"]
                                )
                            )
                        else:
                            val_next_nobs = (
                                self.dynamics.obs2latent(
                                    val_batch["next_obs"]
                                )
                            )

                        if prediction_mode == "full":
                            batch_size = val_nobs.shape[0]
                            val_input = val_nobs.reshape(
                                batch_size,
                                -1,
                            )
                            val_target = val_next_nobs.reshape(
                                batch_size,
                                -1,
                            )
                        else:
                            val_input = val_nobs[:, -1]
                            val_target = val_next_nobs[:, -1]

                        inputs, targets = (
                            self.dynamics.format_samples_for_training(
                                val_batch,
                                val_input,
                                val_target,
                            )
                        )

                        holdout_batches.append(
                            self.dynamics.validate(
                                inputs,
                                targets,
                            )
                        )

                holdout_losses = np.asarray(
                    holdout_batches,
                    dtype=np.float64,
                ).mean(axis=0).tolist()

                model = (
                    self.dynamics.model.module
                    if hasattr(
                        self.dynamics.model,
                        "module",
                    )
                    else self.dynamics.model
                )

                holdout_loss = float(
                    np.sort(holdout_losses)[
                        :model.num_elites
                    ].mean()
                )

                stop = self.dynamics._update_holdout_and_log(
                    new_holdout_losses=holdout_losses,
                    train_loss=train_loss,
                    wandb=wandb_logger,
                    epoch=local_epoch_idx + 1,
                    max_epochs_since_update=(
                        cfg.dynamics.max_epochs_since_update
                    ),
                    max_epochs=(
                        cfg.dynamics.dynamics_max_epochs
                    ),
                )

                with open(
                    metrics_path,
                    "a",
                    newline="",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "train_loss",
                            "holdout_loss",
                        ],
                    )

                    if write_header:
                        writer.writeheader()
                        write_header = False

                    writer.writerow(
                        {
                            "epoch": local_epoch_idx + 1,
                            "train_loss": train_loss,
                            "holdout_loss": holdout_loss,
                        }
                    )

                print(
                    f"Dynamics epoch {local_epoch_idx + 1}: "
                    f"train={train_loss:.6f}, "
                    f"holdout={holdout_loss:.6f}"
                )

                if (
                    save_every > 0
                    and (local_epoch_idx + 1) % save_every == 0
                ):
                    self._save_dynamics_checkpoint(
                        f"epoch_{local_epoch_idx + 1:04d}"
                    )

                if stop:
                    should_stop.fill_(1)

            if self.is_ddp:
                dist.broadcast(
                    should_stop,
                    src=0,
                )

            if should_stop.item():
                break

            if self.is_ddp:
                dist.barrier()

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

    def _build_ppo(self, cfg) -> None:
        self.unio4 = BehaviorProximalPolicyOptimization(
            policy=self.model,
            device=self.device,
            obs_adapter=self.obs_adapter,
            policy_lr=cfg.unio4.bppo_lr,
            clip_ratio=cfg.unio4.clip_ratio,
            entropy_weight=cfg.unio4.entropy_weight,
            decay=cfg.unio4.decay,
            omega=cfg.unio4.omega,
            batch_size=cfg.unio4.bppo_batch_size,
            is_iql=cfg.critic.is_iql,
            temperature=cfg.unio4.get(
                "temperature",
                None,
            ),
            ratio_strategy=cfg.unio4.get(
                "ratio_strategy",
                "scalar",
            ),
            fix_encoder=cfg.unio4.get(
                "fix_encoder",
                True,
            ),
            cfg=cfg,
        )

        ratio_log_dir = os.path.join(
            self.get_ppo_artifact_dir(),
            "ratio_logs",
        )
        self.unio4.set_ratio_log_dir(
            ratio_log_dir
        )

        if self.rank == 0:
            trainable_params = sum(
                p.numel()
                for p in self.unio4._policy.parameters()
                if p.requires_grad
            )
            total_params = sum(
                p.numel()
                for p in self.unio4._policy.parameters()
            )

            print(
                f"PPO Actor ready: "
                f"trainable={trainable_params:,}, "
                f"total={total_params:,}"
            )

    def _build_finetune_dataloader(self, cfg) -> None:
        if cfg.chunk_as_single_action:
            sequence_stride = cfg.dataset.get(
                "finetune_sequence_stride",
                cfg.n_action_steps,
            )
            finetune_dataset = OfflineDataset(
                buffer=self.buffer,
                horizon=cfg.horizon,
                pad_before=cfg.dataset.get("pad_before", 0),
                pad_after=cfg.dataset.get("pad_after", 0),
                sequence_stride=sequence_stride,
                seed=cfg.training.seed,
                val_ratio=0.0,
                max_train_episodes=cfg.dataset.get(
                    "max_train_episodes",
                    None,
                ),
                use_depth=cfg.dataset.use_depth,
            )
        else:
            sequence_stride = cfg.dataset.get(
                "sequence_stride",
                1,
            )
            finetune_dataset = self.dataset

        finetune_batch_size = int(
            cfg.unio4.get(
                "finetune_batch_size",
                cfg.dataloader.batch_size,
            )
        )

        dataloader_cfg = OmegaConf.to_container(
            cfg.dataloader,
            resolve=True,
        )
        dataloader_cfg["batch_size"] = finetune_batch_size
        dataloader_cfg["pin_memory"] = False
        dataloader_cfg["persistent_workers"] = False
        dataloader_cfg["num_workers"] = min(
            dataloader_cfg.get("num_workers", 8),
            2,
        )
        dataloader_cfg.pop("sampler", None)

        self.finetune_dataset = finetune_dataset

        if self.is_ddp:
            per_gpu_batch_size = max(1, finetune_batch_size // self.world_size)
            dataloader_cfg["batch_size"] = per_gpu_batch_size
            dataloader_cfg["shuffle"] = False
            self.finetune_sampler = DistributedSampler(
                finetune_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            self.finetune_dataloader = DataLoader(
                finetune_dataset,
                sampler=self.finetune_sampler,
                **dataloader_cfg,
            )
        else:
            dataloader_cfg["batch_size"] = finetune_batch_size
            self.finetune_sampler = None
            self.finetune_dataloader = DataLoader(
                finetune_dataset,
                **dataloader_cfg,
            )

        self._finetune_iter = None
        self._finetune_epoch = 0

        if self.rank == 0:
            if self.is_ddp:
                print(
                    f"Finetune dataset: {len(finetune_dataset)} samples "
                    f"(stride={sequence_stride}, batch_size={finetune_batch_size}, "
                    f"per_gpu={dataloader_cfg['batch_size']})"
                )
            else:
                print(
                    f"Finetune dataset: {len(finetune_dataset)} samples "
                    f"(stride={sequence_stride}, batch_size={finetune_batch_size})"
                )

        self.unio4.set_old_policy()

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

    @torch.no_grad()
    def eval(
        self,
        eval_times: int = 1,
        policy_override=None,
        eval_name: str = "Eval",
    ):
        if self.is_ddp:
            dist.barrier()

        if self.rank != 0:
            if self.is_ddp:
                dist.barrier()
            return {
                "test_mean_score": 0.0,
                "mean_returns": 0.0,
            }

        if self.env_runner is None:
            raise RuntimeError("Evaluation requires task.env_runner.")

        if policy_override is not None:
            policy = policy_override
        elif self.cfg.training.use_ema:
            policy = self.ema_model
        else:
            policy = self.model

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, "eval_env_num", 1)
        try:
            run_params = inspect.signature(self.env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        run_kwargs = {
            key: value
            for key, value in {"eval_env_num": eval_env_num}.items()
            if key in run_params
        }
        log_data = {
            "test_mean_score": [],
            "mean_returns": [],
        }

        for _ in range(eval_times):
            runner_log = self.env_runner.run(policy, **run_kwargs)
            log_data["test_mean_score"].append(
                runner_log["test_mean_score"]
            )
            log_data["mean_returns"].append(
                runner_log["mean_returns"]
            )

            cprint(
                f"---------------- {eval_name} Results --------------",
                "magenta",
            )
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", "magenta")

        log_data["test_mean_score"] = float(
            np.mean(log_data["test_mean_score"])
        )
        log_data["mean_returns"] = float(
            np.mean(log_data["mean_returns"])
        )

        if self.is_ddp:
            dist.barrier()

        return log_data

    @torch.no_grad()
    def unio4_eval(
        self,
        idql_eval: bool = False,
        dynamics=None,
        first_action: bool = False,
        get_np: bool = True,
        use_gae: bool = True,
        iql=None,
        Q=None,
        repeat_num: int = 100,
        eval_times: int = 1,
        eval_name: str = "IDQL Eval",
    ):
        if self.is_ddp:
            dist.barrier()
        if self.rank != 0:
            if self.is_ddp:
                dist.barrier()
            return {"test_mean_score": 0.0, "mean_returns": 0.0}

        if self.env_runner is None:
            raise RuntimeError("IDQL evaluation requires task.env_runner.")

        policy = self.unio4._policy
        if self.cfg.training.use_ema and self.cfg.unio4.use_ema_eval:
            policy = self.ema_model

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, "eval_env_num", 1)
        idql_run_params = {}
        if idql_eval:
            try:
                idql_run_params = inspect.signature(
                    self.env_runner.idql_run
                ).parameters
            except (TypeError, ValueError):
                idql_run_params = {}
        try:
            run_params = inspect.signature(self.env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}

        log_data = {"test_mean_score": [], "mean_returns": []}
        for _ in range(eval_times):
            if idql_eval:
                idql_kwargs = {
                    key: value
                    for key, value in {
                        "dynamics": dynamics,
                        "first_action": first_action,
                        "get_np": get_np,
                        "use_gae": use_gae,
                        "iql": iql,
                        "Q": Q,
                        "repeat_num": repeat_num,
                        "eval_env_num": eval_env_num,
                    }.items()
                    if key in idql_run_params
                }
                runner_log = self.env_runner.idql_run(
                    policy,
                    **idql_kwargs,
                )
            else:
                run_kwargs = {
                    key: value
                    for key, value in {"eval_env_num": eval_env_num}.items()
                    if key in run_params
                }
                runner_log = self.env_runner.run(policy, **run_kwargs)

            log_data["test_mean_score"].append(runner_log["test_mean_score"])
            log_data["mean_returns"].append(runner_log["mean_returns"])
            cprint(f"---------------- {eval_name} Results --------------", "magenta")
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", "magenta")

        log_data["test_mean_score"] = float(np.mean(log_data["test_mean_score"]))
        log_data["mean_returns"] = float(np.mean(log_data["mean_returns"]))

        if self.is_ddp:
            dist.barrier()
        return log_data

    def _train_offline_ppo(self, cfg) -> None:
        bppo_steps = int(cfg.unio4.bppo_steps)
        if bppo_steps <= 0:
            return

        self.critic.eval()
        self.dynamics.model.eval()
        self.unio4._policy.eval()
        self.unio4._old_policy.eval()

        ppo_dir = self.get_ppo_artifact_dir()
        os.makedirs(ppo_dir, exist_ok=True)

        eval_step = int(cfg.unio4.get("eval_step", 100))
        if eval_step < 1:
            raise ValueError("unio4.eval_step must be >= 1.")

        env_eval_freq = int(cfg.unio4.get("eval_freq", 1000))
        if env_eval_freq < 1:
            raise ValueError("unio4.eval_freq must be >= 1.")

        update_old_policy = bool(cfg.unio4.get("is_update_old_policy", True))
        run_env_eval = self.env_runner is not None
        run_idql_eval = bool(cfg.unio4.get("idql_eval", False))

        if run_idql_eval and not run_env_eval:
            raise RuntimeError("unio4.idql_eval=True requires task.env_runner.")

        if run_env_eval:
            if run_idql_eval:
                idql_log_data = self.unio4_eval(
                    idql_eval=True,
                    dynamics=self.dynamics,
                    first_action=cfg.unio4.first_action,
                    get_np=True,
                    use_gae=cfg.unio4.use_gae,
                    iql=self.critic,
                    Q=self.q_eval,
                    repeat_num=128,
                    eval_times=cfg.unio4.eval_times,
                )
            else:
                idql_log_data = None

            normal_log_data = self.eval(
                eval_times=cfg.unio4.eval_times,
            )

            if run_idql_eval:
                best_bppo_score = idql_log_data["test_mean_score"]
            else:
                best_bppo_score = normal_log_data["test_mean_score"]

            scores = [best_bppo_score]
            normal_scores = [normal_log_data["test_mean_score"]]
            idql_scores = []
            if run_idql_eval:
                idql_scores.append(idql_log_data["test_mean_score"])

            if self.rank == 0:
                _, is_updated = self.maybe_update_global_best(
                    best_bppo_score
                )
                if is_updated:
                    print("------------saved best model----------------")
        else:
            idql_log_data = None
            normal_log_data = None
            scores = []
            normal_scores = []
            idql_scores = []

        best_mean_q, initial_reward = self._evaluate_dynamics_ope(cfg)

        print(
            f"Initial Dynamics OPE: "
            f"mean_q={best_mean_q:.6f}, "
            f"mean_reward={initial_reward:.6f}"
        )
        opes = [best_mean_q]

        if self.rank == 0 and self.wandb_run is not None:
            init_log_data = {"current_mean_qs": best_mean_q}
            if run_env_eval:
                init_log_data["current_bppo_scores"] = best_bppo_score
                init_log_data["normal_eval_scores"] = normal_log_data[
                    "test_mean_score"
                ]
            if run_idql_eval:
                init_log_data["idql_eval_scores"] = idql_log_data[
                    "test_mean_score"
                ]
            self.wandb_run.log(init_log_data)

        if self.rank == 0:
            iterator = tqdm.tqdm(
                range(bppo_steps),
                desc="BPPO updating",
                mininterval=cfg.training.get(
                    "tqdm_interval_sec",
                    1.0,
                ),
            )
        else:
            iterator = range(bppo_steps)

        for step in iterator:
            if self.is_ddp:
                dist.barrier()
            if cfg.unio4.get("is_linear_decay", False):
                progress = step / bppo_steps
                bppo_lr_now = (
                    cfg.unio4.bppo_lr
                    * (1.0 - progress)
                )
                clip_ratio_now = (
                    cfg.unio4.clip_ratio
                    * (1.0 - progress)
                )
            else:
                bppo_lr_now = None
                clip_ratio_now = None

            if step > 200:
                cfg.unio4.is_clip_decay = False
                cfg.unio4.is_bppo_lr_decay = False

            batch = self.sample_finetune_batch()

            policy_loss = self.unio4.update_distribution(
                batch=batch,
                critic=self.critic,
                is_clip_decay=cfg.unio4.get("is_clip_decay", False),
                is_lr_decay=cfg.unio4.get("is_bppo_lr_decay", False),
                is_linear_decay=cfg.unio4.get("is_linear_decay", False),
                bppo_lr_now=bppo_lr_now,
                clip_ratio_now=clip_ratio_now,
                dynamics=self.dynamics,
                use_gae=cfg.unio4.get("use_gae", True),
                gamma=cfg.critic.gamma,
                lamda=cfg.ppo.get("lamda", 0.95),
            )

            if self.ema is not None:
                self.ema.step(self.unio4._policy)

            self.global_step += 1

            if self.wandb_run is not None:
                self.wandb_run.log({"dpg_loss": policy_loss})

            if run_env_eval and self.global_step % env_eval_freq == 0:
                if run_idql_eval:
                    idql_log_data = self.unio4_eval(
                        idql_eval=True,
                        dynamics=self.dynamics,
                        first_action=cfg.unio4.first_action,
                        get_np=True,
                        use_gae=cfg.unio4.use_gae,
                        iql=self.critic,
                        Q=self.q_eval,
                        repeat_num=128,
                        eval_times=cfg.unio4.eval_times,
                    )
                    idql_current_score = idql_log_data["test_mean_score"]
                    idql_scores.append(idql_current_score)

                normal_log_data = self.eval(eval_times=cfg.unio4.eval_times)
                normal_current_score = normal_log_data["test_mean_score"]
                normal_scores.append(normal_current_score)

                if run_idql_eval:
                    current_bppo_score = idql_current_score
                else:
                    current_bppo_score = normal_current_score

                scores.append(current_bppo_score)

                if self.rank == 0:
                    _, is_updated = self.maybe_update_global_best(current_bppo_score)
                    if is_updated:
                        print("------------saved best model----------------")
                    else:
                        score_dir = os.path.join(
                            ppo_dir,
                            f"score_{step}",
                        )
                        os.makedirs(score_dir, exist_ok=True)
                        self._save_policy_bundle(
                            self.unio4._policy,
                            score_dir,
                        )
                        print(f"------------saved {current_bppo_score} model----------------")

                    if self.wandb_run is not None:
                        eval_log_data = {
                            "current_bppo_scores": current_bppo_score,
                            "normal_eval_scores": normal_current_score,
                        }
                        if run_idql_eval:
                            eval_log_data["idql_eval_scores"] = idql_current_score
                        self.wandb_run.log(eval_log_data)

            if self.global_step % eval_step == 0:
                current_mean_q, mean_reward = self._evaluate_dynamics_ope(cfg)
                if self.rank == 0 and self.wandb_run is not None:
                    self.wandb_run.log(
                        {"current_mean_qs": current_mean_q}
                    )

                if (
                    update_old_policy
                    and current_mean_q > best_mean_q
                ):
                    best_mean_q = current_mean_q
                    self.unio4.set_old_policy()

                    if self.rank == 0:
                        print(
                            "Updated PPO old policy: "
                            f"step={self.global_step}, "
                            f"mean_q={current_mean_q:.6f}"
                        )

                if self.rank == 0:
                    print(
                        f"Dynamics OPE: "
                        f"step={self.global_step}, "
                        f"mean_q={current_mean_q:.6f}, "
                        f"best_q={best_mean_q:.6f}"
                    )

                opes.append(current_mean_q)
                if self.rank == 0:
                    np.savetxt(
                        os.path.join(ppo_dir, "each_ope_score.csv"),
                        opes,
                        fmt="%f",
                        delimiter=",",
                    )

            if self.rank == 0 and scores:
                np.savetxt(
                    os.path.join(ppo_dir, "each_scores.csv"),
                    scores,
                    fmt="%f",
                    delimiter=",",
                )

        if self.rank == 0:
            np.savetxt(
                os.path.join(ppo_dir, "last_ope_score.csv"),
                opes,
                fmt="%f",
                delimiter=",",
            )
            if run_idql_eval and idql_scores:
                np.savetxt(
                    os.path.join(ppo_dir, "last_idql_eval_scores.csv"),
                    idql_scores,
                    fmt="%f",
                    delimiter=",",
                )
            if normal_scores:
                np.savetxt(
                    os.path.join(ppo_dir, "last_normal_eval_scores.csv"),
                    normal_scores,
                    fmt="%f",
                    delimiter=",",
                )

        self._save_ppo_final_artifacts()

        if self.is_ddp:
            dist.barrier()

    @torch.no_grad()
    def _evaluate_dynamics_ope(
        self,
        cfg,
    ) -> tuple[float, float]:
        if not self.critic.is_share_encoder:
            raise ValueError(
                "ACT Dynamics OPE V1 requires "
                "critic.is_share_encoder=True."
            )

        rollout_length = int(
            cfg.unio4.get(
                "rollout_length",
                1,
            )
        )

        if (
            cfg.chunk_as_single_action
            and rollout_length != 1
        ):
            raise ValueError(
                "chunk_as_single_action=True requires "
                "unio4.rollout_length=1 for ACT V1."
            )

        batch = self.sample_finetune_batch()

        mean_q, mean_reward = self.dynamics.rollout(
            policy=self.unio4._policy,
            Q=self.q_eval,
            iql=self.critic,
            batch=batch,
            rollout_length=rollout_length,
            is_iql=cfg.critic.is_iql,
            use_gae=cfg.unio4.use_gae,
            first_action=cfg.dynamics.first_action,
        )

        mean_q = float(
            mean_q.detach().cpu().item()
        )
        mean_reward = float(mean_reward)

        return mean_q, mean_reward

    def _build_ema(self, cfg) -> None:
        self.ema = None
        self.ema_model = None
        if not cfg.training.get("use_ema", True):
            return

        self.ema_model = deepcopy(self.unio4._policy).to(self.device)
        self.ema = EMAModel(
            model=self.ema_model,
            update_after_step=cfg.ema.get("update_after_step", 0),
            inv_gamma=cfg.ema.get("inv_gamma", 1.0),
            power=cfg.ema.get("power", 0.75),
            min_value=cfg.ema.get("min_value", 0.0),
            max_value=cfg.ema.get("max_value", 0.9999),
        )

    def _save_policy_bundle(self, policy, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        policy.save_pretrained(save_dir)
        self.policy_preprocessor.save_pretrained(
            save_dir,
            config_filename="policy_preprocessor.json",
        )
        self.policy_postprocessor.save_pretrained(
            save_dir,
            config_filename="policy_postprocessor.json",
        )

    def _save_ppo_final_artifacts(self) -> None:
        if self.rank != 0:
            return
        ppo_dir = self.get_ppo_artifact_dir()
        self._save_policy_bundle(
            self.unio4._policy,
            os.path.join(ppo_dir, "last"),
        )
        self.unio4.flush_ratio_logs(force=True)
        print(f"Saved final PPO actor to {os.path.join(ppo_dir, 'last')}")

    def get_global_best_ema_dir(self) -> str:
        return self.cfg.unio4.get(
            "global_best_ema_dir",
            None,
        ) or os.path.join(self.output_dir, "best_ema")

    def get_global_best_ema_score_path(self) -> str:
        return os.path.join(self.get_global_best_ema_dir(), "best_score.csv")

    def get_global_best_ema_lock_path(self) -> str:
        best_dir = self.get_global_best_ema_dir()
        return os.path.join(os.path.dirname(best_dir), ".global_best_ema.lock")

    def cleanup_shared_memory(self) -> None:
        if self.shm_manager is None:
            return
        try:
            self.shm_manager.cleanup()
            self.shm_manager = None
            if self.rank == 0:
                info_file = os.path.join(
                    self.output_dir,
                    "shared_memory_info_path.txt",
                )
                if os.path.exists(info_file):
                    os.remove(info_file)
        except Exception as exc:
            print(f"[Rank {self.rank}] Shared memory cleanup warning: {exc}")


@hydra.main(version_base=None, config_path="../configs/rl", config_name="offline_rl")
def main(cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        setup_ddp()

    workspace = None
    try:
        workspace = TrainACTWorkspace(cfg)
        workspace.run()
    finally:
        if workspace is not None:
            if getattr(workspace, "wandb_run", None) is not None:
                try:
                    workspace.wandb_run.finish()
                except Exception:
                    pass
            workspace.cleanup_shared_memory()
        cleanup_ddp()


if __name__ == "__main__":
    main()
