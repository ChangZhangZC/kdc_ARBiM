import copy
import json
import os
import random

import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from post_rl.data.offline_dataset import OfflineDataset
from post_rl.policy.stochastic_act_config import StochasticACTConfigWrapper
from post_rl.policy.stochastic_act_policy import StochasticACTPolicyWrapper
from post_rl.utils.common import dict_apply

from .workspace import TrainACTWorkspace as _CoreTrainACTWorkspace


class TrainACTWorkspace(_CoreTrainACTWorkspace):
    """Offline RL workspace with explicit IL/RL policy starts and full PPO resume."""

    RESUME_VERSION = 1

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        cfg = copy.deepcopy(cfg)
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

        self._resume_checkpoint_dir = None
        self._resume_metadata = None
        (
            self._policy_checkpoint_type,
            self._policy_source_checkpoint,
        ) = self._resolve_policy_source(cfg)
        cfg.input.policy_checkpoint = self._policy_source_checkpoint

        if self._policy_checkpoint_type == "il":
            stochastic_cfg = StochasticACTConfigWrapper.from_il_pretrained(
                self._policy_source_checkpoint,
                init_log_std=cfg.policy.init_log_std,
                log_std_min=cfg.policy.log_std_min,
                log_std_max=cfg.policy.log_std_max,
            )
            stochastic_cfg.device = str(self.device)
            self.model = StochasticACTPolicyWrapper.from_il_pretrained(
                self._policy_source_checkpoint,
                config=stochastic_cfg,
            ).to(self.device)
        else:
            stochastic_cfg = StochasticACTConfigWrapper.from_pretrained(
                self._policy_source_checkpoint,
            )
            stochastic_cfg.device = str(self.device)
            self.model = StochasticACTPolicyWrapper.from_pretrained(
                self._policy_source_checkpoint,
                config=stochastic_cfg,
            ).to(self.device)

        self.model.eval()
        self._validate_scheme_c_contract(cfg)

        self.global_step = 0
        self._finetune_iter = None
        self._finetune_epoch = 0
        self._finetune_batch_in_epoch = 0
        self._encoder_sha256 = None
        self._resume_loaded = False
        self._best_mean_q = None
        self._ope_history = []

        if self.rank == 0:
            label = {
                "il": "IL checkpoint",
                "rl": "RL warm-start checkpoint",
                "resume": "PPO resume policy",
            }[self._policy_checkpoint_type]
            cprint(
                f"Loaded {label}: {self._policy_source_checkpoint}",
                "green",
            )
            cprint(
                f"Workspace device={self.device}, world_size={self.world_size}",
                "green",
            )

    def _resolve_policy_source(self, cfg) -> tuple[str, str]:
        resume_cfg = cfg.get("resume")
        resume_dir = None if resume_cfg is None else resume_cfg.get("checkpoint_dir")
        if resume_dir:
            resume_dir = os.path.abspath(os.path.expanduser(str(resume_dir)))
            metadata_path = os.path.join(resume_dir, "resume_meta.json")
            policy_dir = os.path.join(resume_dir, "policy")
            required = [
                metadata_path,
                os.path.join(resume_dir, "training_state.pt"),
                os.path.join(resume_dir, "contract.json"),
                policy_dir,
            ]
            missing = [path for path in required if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete PPO resume checkpoint {resume_dir}; missing: {missing}"
                )
            with open(metadata_path, "r") as file:
                metadata = json.load(file)
            if int(metadata.get("version", -1)) != self.RESUME_VERSION:
                raise RuntimeError(
                    f"Unsupported PPO resume version {metadata.get('version')}; "
                    f"expected {self.RESUME_VERSION}."
                )
            saved_world_size = int(metadata.get("world_size", 1))
            if saved_world_size != self.world_size:
                raise RuntimeError(
                    "Full PPO resume requires the same DDP world_size: "
                    f"saved={saved_world_size}, current={self.world_size}. "
                    "Use input.policy_checkpoint_type=rl for a new warm-start run "
                    "when changing GPU count."
                )
            self._resume_checkpoint_dir = resume_dir
            self._resume_metadata = metadata
            return "resume", policy_dir

        checkpoint_type = str(
            cfg.input.get("policy_checkpoint_type", "il")
        ).lower()
        if checkpoint_type not in {"il", "rl"}:
            raise ValueError(
                "input.policy_checkpoint_type must be 'il' or 'rl'."
            )
        return checkpoint_type, str(cfg.input.policy_checkpoint)

    def _validate_scheme_c_contract(self, cfg) -> None:
        super()._validate_scheme_c_contract(cfg)
        if int(cfg.unio4.bppo_steps) < 0:
            raise ValueError("unio4.bppo_steps must be >= 0")
        if int(cfg.unio4.eval_step) < 1:
            raise ValueError("unio4.eval_step must be >= 1")
        if int(cfg.unio4.checkpoint_every_steps) < 0:
            raise ValueError("unio4.checkpoint_every_steps must be >= 0")

    def _apply_debug_overrides(self, cfg) -> None:
        if self._resume_metadata is None:
            return super()._apply_debug_overrides(cfg)
        if not bool(cfg.training.debug):
            return
        cfg.dataloader.num_workers = 0
        cfg.dataloader.persistent_workers = False
        cfg.val_dataloader.num_workers = 0
        cfg.val_dataloader.persistent_workers = False
        cfg.use_wandb = False

    def _resolve_resume_artifact_dir(self, prefix: str) -> str:
        metadata = self._resume_metadata
        if metadata is None or self._resume_checkpoint_dir is None:
            raise RuntimeError("Resume metadata is not available.")

        relpath = metadata.get(f"{prefix}_artifact_relpath")
        if relpath:
            candidate = os.path.normpath(
                os.path.join(self._resume_checkpoint_dir, relpath)
            )
            if os.path.isdir(candidate):
                return candidate

        absolute = metadata.get(f"{prefix}_artifact_dir")
        if absolute and os.path.isdir(absolute):
            return str(absolute)
        raise FileNotFoundError(
            f"Cannot locate saved {prefix} artifacts for PPO resume."
        )

    def _apply_resume_config(self, cfg) -> None:
        if self._resume_metadata is not None:
            cfg.critic.load_pretrain = True
            cfg.dynamics.load_pretrain = True
            cfg.critic.artifact_dir = self._resolve_resume_artifact_dir("critic")
            cfg.dynamics.artifact_dir = self._resolve_resume_artifact_dir("dynamics")
            return
        if cfg.unio4.get("stage1_resume_dir"):
            cfg.critic.load_pretrain = True
            cfg.dynamics.load_pretrain = True

    def _validate_resume_dataset(self) -> None:
        if self._resume_metadata is None:
            return
        expected_size = int(self._resume_metadata["dataset_size"])
        expected_episodes = int(self._resume_metadata["episode_count"])
        actual_size = len(self.buffer)
        actual_episodes = len(self.buffer.episode_ends)
        if actual_size != expected_size or actual_episodes != expected_episodes:
            raise RuntimeError(
                "PPO resume dataset contract mismatch: "
                f"saved size/episodes={expected_size}/{expected_episodes}, "
                f"current={actual_size}/{actual_episodes}."
            )

    def _wrap_critic_ddp(self) -> None:
        super()._wrap_critic_ddp()
        if not self.is_ddp:
            return
        q_model = self.critic._Q.module
        self.critic._target_Q.load_state_dict(q_model.state_dict())
        self.critic._target_Q.requires_grad_(False)
        self.critic._target_Q.eval()

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
            shuffle=False,
        )
        if self.is_ddp:
            kwargs["batch_size"] = self._per_rank_batch_size(
                cfg.unio4.finetune_batch_size,
                "unio4.finetune_batch_size",
            )

        self.finetune_sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=int(cfg.training.seed),
            drop_last=self.is_ddp,
        )
        self.finetune_dataloader = DataLoader(
            dataset,
            sampler=self.finetune_sampler,
            **kwargs,
        )
        self.finetune_dataset = dataset
        self._finetune_iter = None
        self._finetune_epoch = 0
        self._finetune_batch_in_epoch = 0
        self.unio4.set_old_policy()
        if self.rank == 0:
            print(
                f"Finetune dataset: {len(dataset)} samples "
                f"(stride={stride}, batch_size={cfg.unio4.finetune_batch_size})"
            )

    def _new_finetune_iterator(self):
        self.finetune_sampler.set_epoch(self._finetune_epoch)
        return iter(self.finetune_dataloader)

    def _prime_finetune_iterator(self) -> None:
        iterator = self._new_finetune_iterator()
        for _ in range(int(self._finetune_batch_in_epoch)):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Saved finetune dataloader cursor exceeds the current epoch length."
                ) from exc
        self._finetune_iter = iterator

    def sample_finetune_batch(self):
        if self._finetune_iter is None:
            self._finetune_iter = self._new_finetune_iterator()
        try:
            batch = next(self._finetune_iter)
            self._finetune_batch_in_epoch += 1
        except StopIteration:
            self._finetune_epoch += 1
            self._finetune_batch_in_epoch = 0
            self._finetune_iter = self._new_finetune_iterator()
            try:
                batch = next(self._finetune_iter)
            except StopIteration as exc:
                raise RuntimeError("Finetune dataloader is empty.") from exc
            self._finetune_batch_in_epoch = 1
        return dict_apply(
            batch,
            lambda x: x.to(self.device, non_blocking=True),
        )

    @staticmethod
    def _torch_load(path: str, map_location):
        try:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _rank_runtime_state(self) -> dict:
        cuda_rng_state = None
        if self.device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(self.device)
        return {
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng_state,
            "finetune_epoch": int(self._finetune_epoch),
            "finetune_batch_in_epoch": int(self._finetune_batch_in_epoch),
        }

    def _restore_rank_runtime_state(self, state: dict) -> None:
        self._finetune_epoch = int(state["finetune_epoch"])
        self._finetune_batch_in_epoch = int(state["finetune_batch_in_epoch"])
        self._prime_finetune_iterator()

        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if self.device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(
                state["cuda_rng_state"].cpu(),
                device=self.device,
            )

    def _resume_checkpoint_path(self, name: str) -> str:
        return os.path.join(
            self.get_ppo_artifact_dir(),
            "checkpoints",
            name,
        )

    def _save_resume_checkpoint(self, name: str) -> None:
        directory = self._resume_checkpoint_path(name)
        if self.rank == 0:
            os.makedirs(directory, exist_ok=True)
            policy_dir = os.path.join(directory, "policy")
            self._save_policy_bundle(self.unio4._policy, policy_dir)

            shared_state = {
                "ppo": self.unio4.training_state_dict(),
                "global_step": int(self.global_step),
                "best_mean_q": self._best_mean_q,
                "ope_history": list(self._ope_history),
                "ema_enabled": self.ema is not None,
            }
            if self.ema is not None:
                shared_state.update(
                    {
                        "ema_model": self.ema_model.state_dict(),
                        "ema_optimization_step": int(self.ema.optimization_step),
                        "ema_decay": float(self.ema.decay),
                    }
                )
            torch.save(
                shared_state,
                os.path.join(directory, "training_state.pt"),
            )

            critic_dir = os.path.abspath(self.get_critic_artifact_dir())
            dynamics_dir = os.path.abspath(self.get_dynamics_artifact_dir())
            metadata = {
                "version": self.RESUME_VERSION,
                "world_size": int(self.world_size),
                "global_step": int(self.global_step),
                "dataset_path": str(self.cfg.input.dataset_path),
                "dataset_size": int(len(self.buffer)),
                "episode_count": int(len(self.buffer.episode_ends)),
                "critic_artifact_dir": critic_dir,
                "dynamics_artifact_dir": dynamics_dir,
                "critic_artifact_relpath": os.path.relpath(critic_dir, directory),
                "dynamics_artifact_relpath": os.path.relpath(dynamics_dir, directory),
            }
            with open(os.path.join(directory, "resume_meta.json"), "w") as file:
                json.dump(metadata, file, indent=2, sort_keys=True)
            self._write_artifact_contract(directory)

        if self.is_ddp:
            dist.barrier()
        torch.save(
            self._rank_runtime_state(),
            os.path.join(directory, f"rank_{self.rank:04d}_state.pt"),
        )
        if self.is_ddp:
            dist.barrier()

    def _load_resume_checkpoint(self, restore_data_cursor: bool) -> None:
        if self._resume_checkpoint_dir is None:
            return
        self._validate_artifact_contract(
            self._resume_checkpoint_dir,
            "PPO resume",
        )
        self._validate_resume_dataset()

        shared_state = self._torch_load(
            os.path.join(self._resume_checkpoint_dir, "training_state.pt"),
            map_location=self.device,
        )
        self.unio4.load_training_state_dict(shared_state["ppo"])
        self.global_step = int(shared_state["global_step"])
        self._best_mean_q = (
            None
            if shared_state.get("best_mean_q") is None
            else float(shared_state["best_mean_q"])
        )
        self._ope_history = [
            float(value) for value in shared_state.get("ope_history", [])
        ]

        if int(getattr(self.unio4, "iteration", -1)) != self.global_step:
            raise RuntimeError(
                "PPO resume state mismatch: iteration does not equal global_step."
            )
        if self.global_step > int(self.cfg.unio4.bppo_steps):
            raise RuntimeError(
                f"Saved global_step={self.global_step} exceeds configured "
                f"unio4.bppo_steps={self.cfg.unio4.bppo_steps}."
            )

        saved_ema_enabled = bool(shared_state.get("ema_enabled", False))
        current_ema_enabled = self.ema is not None
        if saved_ema_enabled != current_ema_enabled:
            raise RuntimeError(
                "Full PPO resume requires training.use_ema to match the saved run."
            )
        if self.ema is not None:
            self.ema_model.load_state_dict(shared_state["ema_model"], strict=True)
            self.ema.optimization_step = int(shared_state["ema_optimization_step"])
            self.ema.decay = float(shared_state["ema_decay"])

        if restore_data_cursor:
            rank_path = os.path.join(
                self._resume_checkpoint_dir,
                f"rank_{self.rank:04d}_state.pt",
            )
            if not os.path.isfile(rank_path):
                raise FileNotFoundError(
                    f"Missing rank-local PPO resume state: {rank_path}"
                )
            rank_state = self._torch_load(rank_path, map_location="cpu")
            self._restore_rank_runtime_state(rank_state)

        self._resume_loaded = True
        if self.is_ddp:
            dist.barrier()
        if self.rank == 0:
            print(
                f"Resumed PPO state from {self._resume_checkpoint_dir} "
                f"at global_step={self.global_step}"
            )

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
        if self._resume_loaded:
            if self._best_mean_q is None or not self._ope_history:
                raise RuntimeError(
                    "Full PPO resume checkpoint is missing OPE gating state."
                )
            best_mean_q = float(self._best_mean_q)
            opes = list(self._ope_history)
            if self.rank == 0:
                print(
                    f"Continuing Dynamics OPE gate: best_q={best_mean_q:.6f}, "
                    f"global_step={self.global_step}"
                )
        else:
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
            self._best_mean_q = best_mean_q
            self._ope_history = list(opes)
            if self.rank == 0:
                print(
                    f"Initial Dynamics OPE: mean_q={best_mean_q:.6f}, "
                    f"mean_reward={initial_reward:.6f}"
                )

        start_step = int(self.global_step)
        iterator_range = range(start_step, steps)
        iterator = (
            tqdm.tqdm(
                iterator_range,
                desc="BPPO updating",
                mininterval=cfg.training.tqdm_interval_sec,
            )
            if self.rank == 0
            else iterator_range
        )
        decay_stop_step = int(cfg.unio4.decay_stop_step)
        checkpoint_every = int(cfg.unio4.checkpoint_every_steps)

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
            self.global_step = step + 1

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
                self._best_mean_q = best_mean_q
                self._ope_history = list(opes)
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

            self._best_mean_q = best_mean_q
            self._ope_history = list(opes)
            if checkpoint_every > 0 and self.global_step % checkpoint_every == 0:
                self._save_resume_checkpoint(
                    f"step_{self.global_step:08d}"
                )

        self._best_mean_q = best_mean_q
        self._ope_history = list(opes)
        self._save_resume_checkpoint("final")

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
        self._apply_resume_config(cfg)
        self._apply_debug_overrides(cfg)
        self._validate_scheme_c_contract(cfg)
        self.cfg = cfg

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(self.output_dir, "config.yaml"))
        if self.is_ddp:
            dist.barrier()

        self.buffer = self._load_buffer()
        self._validate_resume_dataset()
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
            from .workspace import init_wandb_run

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
            if self._resume_checkpoint_dir is not None:
                self._load_resume_checkpoint(restore_data_cursor=False)
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
            if self._resume_checkpoint_dir is not None:
                self._load_resume_checkpoint(restore_data_cursor=True)
            self._train_offline_ppo()
        return None
