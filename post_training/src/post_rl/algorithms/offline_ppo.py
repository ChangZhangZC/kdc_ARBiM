import csv
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from .ppo import ProximalPolicyOptimization
from ..critic.iql_critic import IQLCritic
from ..dynamics.utils.act_obs_adapter import ACTObservationAdapter
from ..policy.stochastic_act_policy import StochasticACTPolicyWrapper
from ..utils.common import CONST_EPS


class BehaviorProximalPolicyOptimization(ProximalPolicyOptimization):
    def __init__(
        self,
        policy: StochasticACTPolicyWrapper,
        device: torch.device,
        obs_adapter: ACTObservationAdapter,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        fix_encoder: bool,
        cfg,
        dynamics=None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            device=device,
            policy_lr=policy_lr,
            clip_ratio=clip_ratio,
            entropy_weight=entropy_weight,
            decay=decay,
            optimizer_cfg=cfg.unio4.optimizer,
            lr_scheduler_cfg=cfg.unio4.lr_scheduler,
            fix_encoder=fix_encoder,
        )
        if not cfg.critic.is_iql:
            raise ValueError("Offline PPO V1 requires critic.is_iql=true.")
        if cfg.unio4.use_gae:
            raise ValueError("Offline PPO V1 does not use online-style GAE.")

        self.obs_adapter = obs_adapter
        self.dynamics = dynamics
        self.temperature = temperature
        self.cfg = cfg
        self.iteration = 0
        self._enable_ratio_logging = bool(cfg.ppo.enable_ratio_logging)
        self._ratio_log_every_updates = max(
            1,
            int(cfg.ppo.ratio_log_every_updates),
        )
        self._ratio_plot_on_final_flush = bool(
            cfg.ppo.ratio_plot_on_final_flush
        )
        self._ratio_records = []
        self._delta_cache = []
        self._ratio_log_counter = 0
        self._ratio_log_dir = None
        self._ratio_log_flush_interval = 50
        self._ratio_log_written_until = 0

        self._enable_monitoring_csv = bool(
            cfg.ppo.get("enable_monitoring_csv", True)
        )
        self._monitor_every_updates = max(
            1,
            int(cfg.ppo.get("monitor_every_updates", 10)),
        )
        self._monitor_records = []
        self._monitor_log_dir = None
        self._monitor_flush_interval = 10
        self._pending_advantage_stats = None

    def _resume_hparams(self) -> dict:
        dataset_path = os.path.realpath(
            os.path.abspath(os.path.expanduser(str(self.cfg.input.dataset_path)))
        )
        return {
            "training_seed": int(self.cfg.training.seed),
            "training_use_ema": bool(self.cfg.training.use_ema),
            "dataset_path": dataset_path,
            "dataset_reward_scaling": str(self.cfg.dataset.reward_scaling),
            "dataset_fixed_reward_scale": float(self.cfg.dataset.fixed_reward_scale),
            "dataset_pad_before": int(self.cfg.dataset.pad_before),
            "dataset_pad_after": int(self.cfg.dataset.pad_after),
            "max_train_episodes": self.cfg.dataset.max_train_episodes,
            "act_chunk_size": int(self.cfg.act_chunk_size),
            "rl_chunk_size": int(self.cfg.rl_chunk_size),
            "n_action_steps": int(self.cfg.n_action_steps),
            "chunk_as_single_action": bool(self.cfg.chunk_as_single_action),
            "offline_chunk_ratio_mode": str(self.cfg.offline_chunk_ratio_mode),
            "offline_chunk_adv_mode": str(self.cfg.offline_chunk_adv_mode),
            "chunk_adv_clip": self.cfg.get("chunk_adv_clip", None),
            "bppo_steps": int(self.cfg.unio4.bppo_steps),
            "bppo_lr": float(self.cfg.unio4.bppo_lr),
            "clip_ratio": float(self.cfg.unio4.clip_ratio),
            "entropy_weight": float(self.cfg.unio4.entropy_weight),
            "max_grad_norm": float(self.cfg.unio4.max_grad_norm),
            "decay": float(self.cfg.unio4.decay),
            "decay_stop_step": int(self.cfg.unio4.decay_stop_step),
            "is_clip_decay": bool(self.cfg.unio4.is_clip_decay),
            "is_bppo_lr_decay": bool(self.cfg.unio4.is_bppo_lr_decay),
            "is_update_old_policy": bool(self.cfg.unio4.is_update_old_policy),
            "is_linear_decay": bool(self.cfg.unio4.is_linear_decay),
            "temperature": self.cfg.unio4.temperature,
            "eval_step": int(self.cfg.unio4.eval_step),
            "finetune_batch_size": int(self.cfg.unio4.finetune_batch_size),
            "finetune_sequence_stride": int(
                self.cfg.dataset.finetune_sequence_stride
            ),
            "ope_rollout_length": int(self.cfg.dynamics.ope_rollout_length),
            "ema_update_after_step": int(self.cfg.ema.update_after_step),
            "ema_inv_gamma": float(self.cfg.ema.inv_gamma),
            "ema_power": float(self.cfg.ema.power),
            "ema_min_value": float(self.cfg.ema.min_value),
            "ema_max_value": float(self.cfg.ema.max_value),
            "optimizer": OmegaConf.to_container(
                self.cfg.unio4.optimizer,
                resolve=True,
            ),
            "lr_scheduler": OmegaConf.to_container(
                self.cfg.unio4.lr_scheduler,
                resolve=True,
            ),
        }

    def training_state_dict(self) -> dict:
        state = super().training_state_dict()
        state["offline_hparams"] = self._resume_hparams()
        return state

    def load_training_state_dict(self, state: dict) -> None:
        if "offline_hparams" not in state:
            raise KeyError("PPO resume state is missing offline_hparams.")
        saved = state["offline_hparams"]
        current = self._resume_hparams()
        mismatch = {
            key: (saved.get(key), current.get(key))
            for key in current
            if saved.get(key) != current.get(key)
        }
        if mismatch:
            raise RuntimeError(
                "Full PPO resume requires the original data/PPO/OPE/EMA contract. "
                f"Mismatch: {mismatch}. Use input.policy_checkpoint_type=rl "
                "for a new warm-start RL run with changed settings."
            )
        super().load_training_state_dict(state)

    def _get_offline_chunk_modes(self) -> tuple[str, str]:
        ratio_mode = str(self.cfg.offline_chunk_ratio_mode)
        adv_mode = str(self.cfg.offline_chunk_adv_mode)
        valid_ratio_modes = {"scalar", "per_step"}
        valid_adv_modes = {
            "scalar_iql",
            "per_step_vdelta",
            "chunk_vdelta_scalar",
            "chunk_vdelta_gae",
        }
        if ratio_mode not in valid_ratio_modes:
            raise ValueError(f"Unsupported offline_chunk_ratio_mode={ratio_mode}")
        if adv_mode not in valid_adv_modes:
            raise ValueError(f"Unsupported offline_chunk_adv_mode={adv_mode}")
        if ratio_mode == "scalar" and adv_mode == "per_step_vdelta":
            raise ValueError(
                "offline_chunk_ratio_mode=scalar is incompatible with "
                "offline_chunk_adv_mode=per_step_vdelta"
            )
        if adv_mode == "chunk_vdelta_gae" and ratio_mode != "scalar":
            raise ValueError(
                "offline_chunk_adv_mode=chunk_vdelta_gae requires "
                "offline_chunk_ratio_mode=scalar"
            )
        return ratio_mode, adv_mode

    def _normalize_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        flat = advantage.detach().reshape(-1).to(dtype=torch.float64)
        local = torch.stack(
            [
                flat.sum(),
                flat.square().sum(),
                torch.tensor(
                    float(flat.numel()),
                    device=flat.device,
                    dtype=torch.float64,
                ),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

        count = float(local[2].item())
        if count <= 0:
            raise RuntimeError("Cannot normalize an empty advantage tensor.")
        mean = local[0] / count
        if count > 1:
            centered_ss = local[1] - local[0].square() / count
            variance = torch.clamp(centered_ss / (count - 1.0), min=0.0)
            std = torch.sqrt(variance)
        else:
            std = torch.zeros((), device=flat.device, dtype=torch.float64)

        mean = mean.to(dtype=advantage.dtype)
        std = std.to(dtype=advantage.dtype)
        return (advantage - mean) / (std + CONST_EPS)

    def _normalize_advantage_by_step(self, advantage: torch.Tensor) -> torch.Tensor:
        if advantage.ndim != 2:
            raise ValueError(
                f"Expected per-step advantage [B,H], got {tuple(advantage.shape)}"
            )
        values = advantage.detach().to(dtype=torch.float64)
        local_sum = values.sum(dim=0)
        local_sq_sum = values.square().sum(dim=0)
        count = torch.tensor(
            float(values.shape[0]),
            device=values.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if float(count.item()) <= 0:
            raise RuntimeError("Cannot normalize an empty per-step advantage tensor.")
        mean = local_sum / count
        if float(count.item()) > 1:
            centered_ss = local_sq_sum - local_sum.square() / count
            variance = torch.clamp(centered_ss / (count - 1.0), min=0.0)
            std = torch.sqrt(variance)
        else:
            std = torch.zeros_like(mean)
        return (
            advantage
            - mean.to(dtype=advantage.dtype).unsqueeze(0)
        ) / (
            std.to(dtype=advantage.dtype).unsqueeze(0) + CONST_EPS
        )

    def _monitor_this_update(self) -> bool:
        if not self._enable_monitoring_csv:
            return False
        if self.iteration % self._monitor_every_updates != 0:
            return False
        return not (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() != 0
        )

    def _capture_advantage_stats(self, advantage: torch.Tensor) -> None:
        if not self._monitor_this_update():
            return
        flat = advantage.detach().float().reshape(-1)
        self._pending_advantage_stats = {
            "adv_pre_norm_mean": float(flat.mean().item()),
            "adv_pre_norm_std": float(flat.std(unbiased=False).item()),
        }

    @torch.no_grad()
    def advantage_computation(
        self,
        obs: dict | torch.Tensor,
        action: torch.Tensor,
        critic: IQLCritic,
    ) -> torch.Tensor:
        advantage = critic.get_advantage(obs, action)
        if self.temperature is not None:
            advantage = torch.minimum(
                torch.exp(advantage * self.temperature),
                torch.ones_like(advantage) * 100.0,
            )
        self._capture_advantage_stats(advantage)
        advantage = self._normalize_advantage(advantage)
        clip_value = self.cfg.get("chunk_adv_clip", None)
        if clip_value is not None:
            advantage = torch.clamp(
                advantage,
                -float(clip_value),
                float(clip_value),
            )
        return advantage

    @torch.no_grad()
    def _compute_per_step_iql_advantages(
        self,
        policy_obs: dict,
        action_chunk: torch.Tensor,
        critic: IQLCritic,
    ) -> torch.Tensor:
        if self.dynamics is None:
            raise RuntimeError(
                "chunk_as_single_action=false requires the trained transition model "
                "to estimate sequential per-step IQL advantages."
            )
        state_tokens, _ = self._old_policy.encode_observation(policy_obs)
        state_tokens = self.dynamics._as_tokens(state_tokens)
        advantages = []

        for step_idx in range(action_chunk.shape[1]):
            action_step = action_chunk[:, step_idx]
            critic_state = self.dynamics._critic_readout(state_tokens)
            advantage = critic.get_advantage(critic_state, action_step).reshape(
                action_chunk.shape[0]
            )
            advantages.append(advantage)
            if step_idx + 1 < action_chunk.shape[1]:
                next_state, _, _, _ = self.dynamics.step(
                    state_tokens,
                    action_step,
                )
                state_tokens = torch.as_tensor(
                    next_state,
                    device=self._device,
                    dtype=state_tokens.dtype,
                )

        advantages = torch.stack(advantages, dim=1)
        if self.temperature is not None:
            advantages = torch.minimum(
                torch.exp(advantages * self.temperature),
                torch.ones_like(advantages) * 100.0,
            )
        self._capture_advantage_stats(advantages)
        advantages = self._normalize_advantage_by_step(advantages)
        clip_value = self.cfg.get("chunk_adv_clip", None)
        if clip_value is not None:
            advantages = torch.clamp(
                advantages,
                -float(clip_value),
                float(clip_value),
            )
        return advantages

    def _compute_chunk_advantage(
        self,
        batch: dict,
        action_chunk: torch.Tensor,
        critic: IQLCritic,
        adv_mode: str,
    ) -> torch.Tensor:
        if adv_mode == "scalar_iql":
            advantage = self.advantage_computation(
                batch["obs"],
                action_chunk,
                critic,
            ).detach()
            advantage = advantage.reshape(advantage.shape[0], -1)
            if advantage.shape[1] != 1:
                raise ValueError(
                    "scalar_iql requires one scalar advantage per chunk, got "
                    f"{tuple(advantage.shape)}"
                )
            return advantage[:, 0]

        raise NotImplementedError(
            f"offline_chunk_adv_mode={adv_mode} is a preserved RL-100 experiment "
            "interface, but ARBiM transformer-token dynamics V1 keeps predict_r=false. "
            "The vdelta modes require reward-predicting dynamics and are not wired yet."
        )

    @staticmethod
    def _sum_step_event_dims(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,H,D], got {tuple(x.shape)}")
        return x.sum(dim=-1)

    def set_ratio_log_dir(self, log_dir: str | None) -> None:
        self._ratio_log_dir = log_dir if self._enable_ratio_logging else None
        if self._ratio_log_dir is not None:
            os.makedirs(self._ratio_log_dir, exist_ok=True)

        self._monitor_log_dir = None
        if self._enable_monitoring_csv and log_dir is not None:
            self._monitor_log_dir = os.path.join(
                os.path.dirname(log_dir),
                "monitoring",
            )
            os.makedirs(self._monitor_log_dir, exist_ok=True)

    def _record_ratio_stats(
        self,
        phase: str,
        ratio: torch.Tensor,
        old_logprob: torch.Tensor,
        new_logprob: torch.Tensor,
    ) -> None:
        if not self._enable_ratio_logging:
            return
        if (
            self._ratio_log_every_updates > 1
            and self.iteration % self._ratio_log_every_updates != 0
        ):
            return

        ratio_flat = ratio.detach().float().reshape(-1).cpu()
        old_flat = old_logprob.detach().float().reshape(-1).cpu()
        new_flat = new_logprob.detach().float().reshape(-1).cpu()
        delta_flat = new_flat - old_flat
        record = {
            "idx": self._ratio_log_counter,
            "iteration": self.iteration,
            "phase": phase,
            "ratio_mean": float(ratio_flat.mean()),
            "ratio_q05": float(torch.quantile(ratio_flat, 0.05)),
            "ratio_q25": float(torch.quantile(ratio_flat, 0.25)),
            "ratio_q50": float(torch.quantile(ratio_flat, 0.50)),
            "ratio_q75": float(torch.quantile(ratio_flat, 0.75)),
            "ratio_q95": float(torch.quantile(ratio_flat, 0.95)),
            "old_logprob_mean": float(old_flat.mean()),
            "new_logprob_mean": float(new_flat.mean()),
            "delta_mean": float(delta_flat.mean()),
            "delta_std": float(delta_flat.std(unbiased=False)),
            "delta_q05": float(torch.quantile(delta_flat, 0.05)),
            "delta_q25": float(torch.quantile(delta_flat, 0.25)),
            "delta_q50": float(torch.quantile(delta_flat, 0.50)),
            "delta_q75": float(torch.quantile(delta_flat, 0.75)),
            "delta_q95": float(torch.quantile(delta_flat, 0.95)),
        }
        self._ratio_records.append(record)
        self._delta_cache.append(delta_flat.numpy())
        if len(self._delta_cache) > 200:
            self._delta_cache = self._delta_cache[-200:]

        self._ratio_log_counter += 1
        if (
            self._ratio_log_dir is not None
            and self._ratio_log_counter % self._ratio_log_flush_interval == 0
        ):
            self.flush_ratio_logs(force=False)

    def flush_ratio_logs(self, force: bool = True) -> None:
        if not self._enable_ratio_logging or self._ratio_log_dir is None:
            return
        if not self._ratio_records:
            return
        if not force and len(self._ratio_records) < self._ratio_log_flush_interval:
            return

        csv_path = os.path.join(self._ratio_log_dir, "ratio_stats.csv")
        pending_records = self._ratio_records[self._ratio_log_written_until:]
        if pending_records:
            headers = list(pending_records[0].keys())
            file_exists = os.path.exists(csv_path)
            mode = "a" if file_exists and self._ratio_log_written_until > 0 else "w"
            with open(csv_path, mode, newline="") as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                if mode == "w":
                    writer.writeheader()
                writer.writerows(pending_records)
            self._ratio_log_written_until = len(self._ratio_records)

        if not force:
            self._ratio_records = self._ratio_records[self._ratio_log_written_until:]
            self._ratio_log_written_until = 0
            return
        if not self._ratio_plot_on_final_flush:
            self._ratio_records = self._ratio_records[self._ratio_log_written_until:]
            self._ratio_log_written_until = 0
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [record["idx"] for record in self._ratio_records]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, [r["ratio_mean"] for r in self._ratio_records], label="ratio_mean")
            plt.plot(xs, [r["ratio_q05"] for r in self._ratio_records], label="ratio_q05", alpha=0.7)
            plt.plot(xs, [r["ratio_q50"] for r in self._ratio_records], label="ratio_q50", alpha=0.9)
            plt.plot(xs, [r["ratio_q95"] for r in self._ratio_records], label="ratio_q95", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self._ratio_log_dir, "ratio_quantiles.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.plot(xs, [r["delta_mean"] for r in self._ratio_records], label="delta_mean")
            plt.plot(xs, [r["delta_q05"] for r in self._ratio_records], label="delta_q05", alpha=0.7)
            plt.plot(xs, [r["delta_q95"] for r in self._ratio_records], label="delta_q95", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self._ratio_log_dir, "delta_quantiles.png"), dpi=150)
            plt.close()
        except Exception:
            pass

        self._ratio_records.clear()
        self._ratio_log_written_until = 0

    def _record_monitor_stats(
        self,
        ratio: torch.Tensor,
        old_logprob: torch.Tensor,
        new_logprob: torch.Tensor,
        policy_loss: torch.Tensor,
        entropy: torch.Tensor,
        loss: torch.Tensor,
        grad_norm: float,
        lr_used: float,
        log_std_mean: float,
    ) -> None:
        if not self._monitor_this_update():
            return

        ratio_flat = ratio.detach().float().reshape(-1)
        log_ratio = (
            new_logprob.detach().float() - old_logprob.detach().float()
        ).reshape(-1)
        approx_kl = ((ratio_flat - 1.0) - log_ratio).mean()
        clip_fraction = (
            (ratio_flat - 1.0).abs() > float(self._clip_ratio)
        ).float().mean()
        advantage_stats = self._pending_advantage_stats or {
            "adv_pre_norm_mean": float("nan"),
            "adv_pre_norm_std": float("nan"),
        }
        record = {
            "iteration": int(self.iteration),
            "loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "ratio_mean": float(ratio_flat.mean().item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "adv_pre_norm_mean": float(advantage_stats["adv_pre_norm_mean"]),
            "adv_pre_norm_std": float(advantage_stats["adv_pre_norm_std"]),
            "grad_norm": float(grad_norm),
            "log_std_mean": float(log_std_mean),
            "lr": float(lr_used),
            "clip_ratio": float(self._clip_ratio),
        }
        self._monitor_records.append(record)
        if len(self._monitor_records) >= self._monitor_flush_interval:
            self.flush_monitor_logs(force=False)

    def flush_monitor_logs(self, force: bool = True) -> None:
        if not self._enable_monitoring_csv or self._monitor_log_dir is None:
            return
        if not self._monitor_records:
            return
        if not force and len(self._monitor_records) < self._monitor_flush_interval:
            return

        csv_path = os.path.join(self._monitor_log_dir, "ppo_metrics.csv")
        file_exists = os.path.isfile(csv_path)
        headers = list(self._monitor_records[0].keys())
        with open(csv_path, "a" if file_exists else "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._monitor_records)
        self._monitor_records.clear()

    def update_distribution(
        self,
        batch: dict,
        critic: IQLCritic,
        is_clip_decay: bool,
        is_lr_decay: bool,
        is_linear_decay: bool = False,
        bppo_lr_now: float | None = None,
        clip_ratio_now: float | None = None,
    ) -> float:
        self.iteration += 1
        self._pending_advantage_stats = None
        monitor_this_update = self._monitor_this_update()
        self._sync_old_policy()
        normalized_obs = self.obs_adapter.normalize_obs(batch["obs"])
        policy_obs = {key: value[:, 0] for key, value in normalized_obs.items()}

        with torch.no_grad():
            action_chunk, old_log_prob_raw, _ = self._old_policy.sample_action_chunk(
                policy_obs
            )

        if action_chunk.ndim != 3:
            raise ValueError(
                f"Expected ACT action chunk [B,H,D], got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] != int(self.cfg.n_action_steps):
            raise ValueError(
                f"ACT sampled {action_chunk.shape[1]} actions but "
                f"n_action_steps={self.cfg.n_action_steps}."
            )

        chunk_as_single_action = bool(self.cfg.chunk_as_single_action)
        if chunk_as_single_action:
            ratio_mode, adv_mode = self._get_offline_chunk_modes()
            advantages = self._compute_chunk_advantage(
                batch,
                action_chunk,
                critic,
                adv_mode,
            )
        else:
            ratio_mode = "per_step"
            advantages = self._compute_per_step_iql_advantages(
                policy_obs,
                action_chunk,
                critic,
            ).detach()

        new_log_prob_raw, entropy_raw = self._policy.evaluate_action_chunk(
            policy_obs,
            action_chunk,
        )

        if is_clip_decay:
            if bool(self.cfg.unio4.is_linear_decay):
                if clip_ratio_now is None:
                    raise ValueError("clip_ratio_now is required for linear clip decay.")
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio *= self._decay

        if chunk_as_single_action and ratio_mode == "scalar":
            old_logprob = self._sum_chunk_event_dims(old_log_prob_raw)
            new_logprob = self._sum_chunk_event_dims(new_log_prob_raw)
            ratio = torch.exp(new_logprob - old_logprob)
            self._record_ratio_stats(
                "offline_chunk_scalar",
                ratio,
                old_logprob,
                new_logprob,
            )
            loss1 = ratio * advantages
            loss2 = torch.clamp(
                ratio,
                1.0 - self._clip_ratio,
                1.0 + self._clip_ratio,
            ) * advantages
            policy_loss = -torch.min(loss1, loss2).mean()
            entropy = self._sum_chunk_event_dims(entropy_raw).mean()
        else:
            old_logprob = self._sum_step_event_dims(old_log_prob_raw)
            new_logprob = self._sum_step_event_dims(new_log_prob_raw)
            ratio = torch.exp(new_logprob - old_logprob)
            self._record_ratio_stats(
                "offline_chunk_per_step" if chunk_as_single_action else "offline_per_step",
                ratio,
                old_logprob,
                new_logprob,
            )
            if chunk_as_single_action:
                advantages = advantages.unsqueeze(-1).expand_as(ratio)
            if advantages.shape != ratio.shape:
                raise ValueError(
                    f"Per-step PPO advantage shape {tuple(advantages.shape)} must match "
                    f"ratio shape {tuple(ratio.shape)}"
                )
            loss1 = ratio * advantages
            loss2 = torch.clamp(
                ratio,
                1.0 - self._clip_ratio,
                1.0 + self._clip_ratio,
            ) * advantages
            policy_loss = -torch.min(loss1, loss2).mean()
            entropy = self._sum_step_event_dims(entropy_raw).mean()

        loss = policy_loss - self._entropy_weight * entropy
        lr_used = float(self._optimizer.param_groups[0]["lr"])
        log_std_mean = (
            float(self._policy._get_log_std().detach().float().mean().item())
            if monitor_this_update
            else float("nan")
        )

        self._optimizer.zero_grad()
        loss.backward()
        trainable_params = [
            param
            for param in self._policy.parameters()
            if param.requires_grad and param.grad is not None
        ]
        max_grad_norm = float(self.cfg.unio4.max_grad_norm)
        if max_grad_norm > 0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_grad_norm,
            )
            grad_norm = (
                float(grad_norm_tensor.item())
                if monitor_this_update
                else float("nan")
            )
        elif monitor_this_update and trainable_params:
            grad_norm = float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [param.grad.detach().norm(2) for param in trainable_params]
                    )
                ).item()
            )
        else:
            grad_norm = float("nan")
        self._optimizer.step()

        linear_decay = bool(self.cfg.unio4.is_linear_decay)
        if linear_decay:
            progress = (self.iteration - 1) / max(int(self.cfg.unio4.bppo_steps), 1)
            lr_now = float(self.cfg.unio4.bppo_lr) * (1.0 - progress)
            for group in self._optimizer.param_groups:
                group["lr"] = lr_now
        elif is_lr_decay:
            self._scheduler.step()

        self._record_monitor_stats(
            ratio=ratio,
            old_logprob=old_logprob,
            new_logprob=new_logprob,
            policy_loss=policy_loss,
            entropy=entropy,
            loss=loss,
            grad_norm=grad_norm,
            lr_used=lr_used,
            log_std_mean=log_std_mean,
        )
        if self.iteration >= int(self.cfg.unio4.bppo_steps):
            self.flush_monitor_logs(force=True)
        return float(loss.item())