import csv
import os

import torch

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
        if not cfg.chunk_as_single_action:
            raise ValueError("Offline PPO V1 requires chunk_as_single_action=true.")
        if not cfg.critic.is_iql:
            raise ValueError("Offline PPO V1 requires critic.is_iql=true.")
        if cfg.unio4.use_gae:
            raise ValueError("Offline PPO V1 does not use GAE.")

        self.obs_adapter = obs_adapter
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
        advantage = (advantage - advantage.mean()) / (
            advantage.std() + CONST_EPS
        )
        clip_value = self.cfg.get("chunk_adv_clip", None)
        if clip_value is not None:
            advantage = torch.clamp(
                advantage,
                -float(clip_value),
                float(clip_value),
            )
        return advantage

    def set_ratio_log_dir(self, log_dir: str | None) -> None:
        if not self._enable_ratio_logging:
            self._ratio_log_dir = None
            return
        self._ratio_log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

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

        advantages = self.advantage_computation(
            batch["obs"],
            action_chunk,
            critic,
        ).detach()
        advantages = advantages.reshape(advantages.shape[0], -1)
        if advantages.shape[1] != 1:
            raise ValueError(
                f"Scheme C chunk PPO requires scalar IQL advantage, got {tuple(advantages.shape)}"
            )
        advantages = advantages[:, 0]

        new_log_prob_raw, entropy_raw = self._policy.evaluate_action_chunk(
            policy_obs,
            action_chunk,
        )
        old_logprob = self._sum_chunk_event_dims(old_log_prob_raw)
        new_logprob = self._sum_chunk_event_dims(new_log_prob_raw)
        ratio = torch.exp(new_logprob - old_logprob)
        self._record_ratio_stats(
            "offline_chunk",
            ratio,
            old_logprob,
            new_logprob,
        )

        if is_clip_decay:
            if bool(self.cfg.unio4.is_linear_decay):
                if clip_ratio_now is None:
                    raise ValueError("clip_ratio_now is required for linear clip decay.")
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio *= self._decay

        loss1 = ratio * advantages
        loss2 = torch.clamp(
            ratio,
            1.0 - self._clip_ratio,
            1.0 + self._clip_ratio,
        ) * advantages
        policy_loss = -torch.min(loss1, loss2).mean()

        entropy = self._sum_chunk_event_dims(entropy_raw).mean()
        loss = policy_loss - self._entropy_weight * entropy

        self._optimizer.zero_grad()
        loss.backward()
        trainable_params = [
            param
            for param in self._policy.parameters()
            if param.requires_grad and param.grad is not None
        ]
        max_grad_norm = float(self.cfg.unio4.max_grad_norm)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        self._optimizer.step()

        linear_decay = bool(self.cfg.unio4.is_linear_decay)
        if linear_decay:
            progress = (self.iteration - 1) / max(int(self.cfg.unio4.bppo_steps), 1)
            lr_now = float(self.cfg.unio4.bppo_lr) * (1.0 - progress)
            for group in self._optimizer.param_groups:
                group["lr"] = lr_now
        elif is_lr_decay:
            self._scheduler.step()

        return float(loss.item())
