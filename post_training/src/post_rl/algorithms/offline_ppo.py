import torch
import csv
import os


from ppo import ProximalPolicyOptimization
from OfflineCritic import IQLCritic
from StochasticACTPolicyWrapper import StochasticACTPolicyWrapper
from utils import CONST_EPS
from transition_model.dynamics.ensemble_dynamics_for_batch import EnsembleDynamics_batch
from transition_model.utils.act_obs_adapter import ACTObservationAdapter


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
        omega: float,
        batch_size: int,
        is_iql: bool,
        temperature: float | None = None,
        ratio_strategy: str = "scalar",
        fix_encoder: bool = False,
        cfg=None,
    ) -> None:
        super().__init__(
            policy=policy,
            device=device,
            policy_lr=policy_lr,
            clip_ratio=clip_ratio,
            entropy_weight=entropy_weight,
            decay=decay,
            omega=omega,
            batch_size=batch_size,
            is_iql=is_iql,
            ratio_strategy=ratio_strategy,
            fix_encoder=fix_encoder,
        )
        self.obs_adapter = obs_adapter
        self.temperature = temperature
        self.cfg = cfg
        self.iteration = 0
        ppo_cfg = getattr(cfg, "ppo", None)
        self._enable_ratio_logging = bool(
            getattr(ppo_cfg, "enable_ratio_logging", False)
        )
        self._ratio_log_every_updates = max(
            1, int(getattr(ppo_cfg, "ratio_log_every_updates", 10))
        )
        self._ratio_plot_on_final_flush = bool(
            getattr(ppo_cfg, "ratio_plot_on_final_flush", True)
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
        if self._is_iql:
            advantage = critic.get_advantage(obs, action)
            if self.temperature:
                advantage = torch.minimum(
                    torch.exp(advantage * self.temperature),
                    torch.ones_like(advantage) * 100.0,
                )
            return (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)

        advantage = critic.minQ(obs, action) - critic.value(obs)
        advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
        return self.weighted_advantage(advantage)

    def _compute_advantage_actor_only(
        self,
        obs: dict | torch.Tensor,
        action: torch.Tensor,
        critic: IQLCritic,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.advantage_computation(obs, action, critic)

    @torch.no_grad()
    def _evaluate_value_function(
        self,
        state: dict | torch.Tensor,
        critic: IQLCritic,
    ) -> torch.Tensor:
        return critic.value(state)


    @torch.no_grad()
    def NStepValueEstimation(
        self,
        nobs_features: torch.Tensor,
        nactions: torch.Tensor,
        dynamics: EnsembleDynamics_batch,
        critic: IQLCritic,
        opt_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        policy_features = nobs_features.reshape(
            batch_size,
            self.cfg.n_obs_steps,
            feature_dim,
        )
        advantages, rewards = [], []

        for i in range(opt_steps):
            state_features = policy_features.reshape(batch_size, -1)
            advantage = self.advantage_computation(
                state_features,
                nactions[:, i],
                critic,
            )
            advantages.append(advantage)

            single_features = policy_features[:, -1, :]
            next_obs, reward, _, _ = dynamics.step(
                single_features,
                nactions[:, i],
                policy_features,
            )
            rewards.append(
                torch.as_tensor(
                    reward,
                    device=self._device,
                    dtype=policy_features.dtype,
                )
            )

            if dynamics.prediction_mode == "full":
                policy_features = torch.as_tensor(
                    next_obs,
                    device=self._device,
                    dtype=policy_features.dtype,
                )
            else:
                next_single = torch.as_tensor(
                    next_obs,
                    device=self._device,
                    dtype=policy_features.dtype,
                )
                policy_features = torch.cat(
                    (
                        policy_features[:, 1:, :],
                        next_single.unsqueeze(1),
                    ),
                    dim=1,
                )

        return torch.stack(advantages), torch.stack(rewards)


    @torch.no_grad()
    def GAE_withQ(
        self,
        advantages: torch.Tensor,
        gamma: float | torch.Tensor,
        lamda: float,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gae_advantages, gae = [], 0
        if dones is None:
            dones = torch.zeros_like(advantages)
        for delta, done in zip(
            reversed(advantages),
            reversed(dones),
        ):
            gae = delta + gamma * lamda * gae * (1.0 - done)
            gae_advantages.insert(0, gae)
        return torch.stack(gae_advantages)
    
    def _get_offline_chunk_modes(self) -> tuple[str, str]:
        ratio_mode = getattr(self.cfg, "offline_chunk_ratio_mode", "scalar")
        adv_mode = getattr(self.cfg, "offline_chunk_adv_mode", "scalar_iql")
        valid_ratio_modes = {"per_step", "scalar"}
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
        if ratio_mode == "scalar" and adv_mode not in {
            "scalar_iql",
            "chunk_vdelta_scalar",
            "chunk_vdelta_gae",
        }:
            raise ValueError(
                f"offline_chunk_ratio_mode={ratio_mode} requires scalar advantage, "
                f"got offline_chunk_adv_mode={adv_mode}"
            )
        if adv_mode == "chunk_vdelta_gae" and ratio_mode != "scalar":
            raise ValueError(
                "offline_chunk_adv_mode=chunk_vdelta_gae requires "
                "offline_chunk_ratio_mode=scalar"
            )
        if adv_mode == "per_step_vdelta":
            raise ValueError(
                "offline_chunk_adv_mode=per_step_vdelta is invalid with "
                "chunk_as_single_action=True"
            )
        return ratio_mode, adv_mode

    def _apply_chunk_adv_clip(
        self,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        chunk_adv_clip = getattr(self.cfg, "chunk_adv_clip", None)
        if chunk_adv_clip is None:
            return advantages
        return torch.clamp(
            advantages,
            -chunk_adv_clip,
            chunk_adv_clip,
        )
    
    @torch.no_grad()
    def _compute_chunk_step_advantages_vdelta(
        self,
        nobs_features: torch.Tensor,
        chunk_actions: torch.Tensor,
        dynamics: EnsembleDynamics_batch,
        critic: IQLCritic,
        gamma: float,
        lamda: float,
        use_gae: bool,
    ) -> torch.Tensor:
        if dynamics is None:
            raise ValueError(
                "dynamics is required for offline_chunk_adv_mode=per_step_vdelta"
            )
        if not getattr(dynamics, "predict_r", False):
            raise ValueError(
                "per_step_vdelta requires predict_r=True in dynamics config"
            )

        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        policy_features = nobs_features.reshape(
            batch_size,
            self.cfg.n_obs_steps,
            feature_dim,
        )
        single_features = policy_features[:, -1, :]
        advantages, terminals = [], []

        for step_idx in range(chunk_actions.shape[1]):
            state_features = policy_features.reshape(batch_size, -1)
            value_now = critic.value(state_features).reshape(
                batch_size, -1
            )[:, 0]

            next_obs, reward, terminal, _ = dynamics.step(
                single_features,
                chunk_actions[:, step_idx],
                policy_features,
            )

            reward_t = torch.as_tensor(
                reward,
                device=self._device,
                dtype=value_now.dtype,
            ).reshape(batch_size, -1)[:, 0]
            terminal_t = torch.as_tensor(
                terminal,
                device=self._device,
                dtype=value_now.dtype,
            ).reshape(batch_size, -1)[:, 0]

            if dynamics.prediction_mode == "full":
                next_policy_features = torch.as_tensor(
                    next_obs,
                    device=self._device,
                    dtype=policy_features.dtype,
                )
                next_single = next_policy_features[:, -1, :]
            else:
                next_single = torch.as_tensor(
                    next_obs,
                    device=self._device,
                    dtype=policy_features.dtype,
                )
                next_policy_features = torch.cat(
                    (
                        policy_features[:, 1:, :],
                        next_single.unsqueeze(1),
                    ),
                    dim=1,
                )

            next_state = next_policy_features.reshape(batch_size, -1)
            value_next = critic.value(next_state).reshape(
                batch_size, -1
            )[:, 0]

            delta = (
                reward_t
                + gamma * (1.0 - terminal_t) * value_next
                - value_now
            )
            advantages.append(delta)
            terminals.append(terminal_t)
            policy_features = next_policy_features
            single_features = next_single

        advantages = torch.stack(advantages)
        if use_gae:
            advantages = self.GAE_withQ(
                advantages,
                gamma,
                lamda,
                dones=torch.stack(terminals),
            )

        for j in range(advantages.shape[0]):
            adv = advantages[j]
            advantages[j] = (
                adv - adv.mean()
            ) / (adv.std() + CONST_EPS)

        return advantages 
    
    @torch.no_grad()
    def _compute_single_chunk_boundary_delta(
        self,
        policy_features: torch.Tensor,
        chunk_action: torch.Tensor,
        dynamics: EnsembleDynamics_batch,
        critic: IQLCritic,
        gamma: float,
    ) -> dict:
        if not getattr(dynamics, "predict_r", False):
            raise ValueError(
                "chunk boundary delta requires predict_r=True in dynamics config"
            )
        batch_size = policy_features.shape[0]
        single_features = policy_features[:, -1, :]
        state_features = policy_features.reshape(batch_size, -1)
        value_now = critic.value(state_features).reshape(batch_size, -1)[:, 0]

        next_obs, reward, terminal, _ = dynamics.step(
            single_features,
            chunk_action,
            policy_features,
        )
        reward_t = torch.as_tensor(
            reward,
            device=self._device,
            dtype=value_now.dtype,
        ).reshape(batch_size, -1)[:, 0]
        terminal_t = torch.as_tensor(
            terminal,
            device=self._device,
            dtype=value_now.dtype,
        ).reshape(batch_size, -1)[:, 0]

        if dynamics.prediction_mode == "full":
            next_policy_features = torch.as_tensor(
                next_obs,
                device=self._device,
                dtype=policy_features.dtype,
            )
        else:
            next_single = torch.as_tensor(
                next_obs,
                device=self._device,
                dtype=policy_features.dtype,
            )
            next_policy_features = torch.cat(
                (policy_features[:, 1:, :], next_single.unsqueeze(1)),
                dim=1,
            )

        next_state = next_policy_features.reshape(batch_size, -1)
        value_next = critic.value(next_state).reshape(batch_size, -1)[:, 0]
        K = chunk_action.shape[1] if chunk_action.ndim > 1 else 1
        gamma_K = torch.as_tensor(
            gamma,
            device=self._device,
            dtype=value_now.dtype,
        ) ** K
        delta = reward_t + gamma_K * (1.0 - terminal_t) * value_next - value_now

        return {
            "delta": delta,
            "reward": reward_t,
            "terminal": terminal_t,
            "value_now": value_now,
            "value_next": value_next,
            "next_policy_features": next_policy_features,
        }

    @torch.no_grad()
    def _compute_chunk_scalar_advantage_vdelta(
        self,
        nobs_features: torch.Tensor,
        chunk_actions: torch.Tensor,
        dynamics: EnsembleDynamics_batch,
        critic: IQLCritic,
        gamma: float,
    ) -> torch.Tensor:
        if dynamics is None:
            raise ValueError(
                "dynamics is required for offline_chunk_adv_mode=chunk_vdelta_scalar"
            )
        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        policy_features = nobs_features.reshape(
            batch_size,
            self.cfg.n_obs_steps,
            feature_dim,
        )
        result = self._compute_single_chunk_boundary_delta(
            policy_features,
            chunk_actions,
            dynamics,
            critic,
            gamma,
        )
        advantages = result["delta"]
        return (advantages - advantages.mean()) / (
            advantages.std() + CONST_EPS
        )
        
    @torch.no_grad()
    def _compute_chunk_gae_advantage_vdelta(
        self,
        nobs_features: torch.Tensor,
        chunk_actions: torch.Tensor,
        dynamics: EnsembleDynamics_batch,
        critic: IQLCritic,
        gamma: float,
        gae_lambda: float,
        n_rollout: int,
        chunk_source: str,
    ) -> torch.Tensor:
        if dynamics is None:
            raise ValueError(
                "dynamics is required for offline_chunk_adv_mode=chunk_vdelta_gae"
            )
        if not getattr(dynamics, "predict_r", False):
            raise ValueError(
                "chunk_vdelta_gae requires predict_r=True in dynamics config"
            )
        if chunk_source != "repeat_first":
            raise ValueError(
                f"chunk_vdelta_gae only supports chunk_source='repeat_first', "
                f"got '{chunk_source}'"
            )

        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        current_features = nobs_features.reshape(
            batch_size,
            self.cfg.n_obs_steps,
            feature_dim,
        )
        K = chunk_actions.shape[1] if chunk_actions.ndim > 1 else 1
        gamma_K = torch.as_tensor(
            gamma,
            device=self._device,
            dtype=current_features.dtype,
        ) ** K

        deltas, terminals = [], []
        alive_mask = torch.ones(
            batch_size,
            device=self._device,
            dtype=current_features.dtype,
        )

        for _ in range(n_rollout):
            if alive_mask.sum() == 0:
                remaining = n_rollout - len(deltas)
                for _ in range(remaining):
                    deltas.append(
                        torch.zeros(
                            batch_size,
                            device=self._device,
                            dtype=current_features.dtype,
                        )
                    )
                    terminals.append(
                        torch.ones(
                            batch_size,
                            device=self._device,
                            dtype=current_features.dtype,
                        )
                    )
                break

            result = self._compute_single_chunk_boundary_delta(
                current_features,
                chunk_actions,
                dynamics,
                critic,
                gamma,
            )
            delta = result["delta"]
            terminal = result["terminal"]
            next_features = result["next_policy_features"]

            deltas.append(delta * alive_mask)
            terminals.append(terminal)

            alive_mask = alive_mask * (1.0 - terminal)
            mask = alive_mask[:, None, None]
            current_features = (
                mask * next_features
                + (1.0 - mask) * current_features
            )

        deltas = torch.stack(deltas)
        terminals = torch.stack(terminals)
        gae_adv = self.GAE_withQ(
            deltas,
            gamma_K,
            gae_lambda,
            dones=terminals,
        )
        advantages = gae_adv[0]
        return (advantages - advantages.mean()) / (
            advantages.std() + CONST_EPS
        )
        
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
        if self._ratio_log_every_updates > 1 and self.iteration % self._ratio_log_every_updates != 0:
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
            with open(csv_path, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
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

            xs = [r["idx"] for r in self._ratio_records]
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
        dynamics: EnsembleDynamics_batch = None,
        use_gae: bool = True,
        gamma: float = 0.99,
        lamda: float = 0.95,
    ) -> float:
        self.iteration += 1

        normalized_obs = self.obs_adapter.normalize_obs(batch["obs"])
        obs_idx = int(getattr(self.cfg, "n_obs_steps", 1)) - 1
        policy_obs = {
            key: value[:, obs_idx]
            for key, value in normalized_obs.items()
        }

        with torch.no_grad():
            action_chunk, old_log_prob_raw, _ = (
                self._old_policy.sample_action_chunk(policy_obs)
            )

        if action_chunk.ndim != 3:
            raise ValueError(
                f"Expected ACT action chunk [B,H,D], got {tuple(action_chunk.shape)}"
            )

        if getattr(self.cfg, "ft_all_actions", False):
            opt_steps = action_chunk.shape[1]
        else:
            opt_steps = int(
                getattr(
                    self.cfg,
                    "n_action_steps",
                    action_chunk.shape[1],
                )
            )

        if opt_steps < 1 or opt_steps > action_chunk.shape[1]:
            raise ValueError(
                f"Invalid opt_steps={opt_steps} for ACT chunk "
                f"shape={tuple(action_chunk.shape)}"
            )

        actions = action_chunk[:, :opt_steps]
        chunk_as_single_action = bool(
            getattr(self.cfg, "chunk_as_single_action", False)
        )
        chunk_ratio_mode = "per_step"
        chunk_adv_mode = "scalar_iql"

        if chunk_as_single_action and opt_steps > 1:
            chunk_ratio_mode, chunk_adv_mode = (
                self._get_offline_chunk_modes()
            )

        if chunk_as_single_action:
            if opt_steps == 1:
                advantages = self._compute_advantage_actor_only(
                    batch["obs"],
                    actions[:, 0],
                    critic,
                )
            elif chunk_adv_mode == "scalar_iql":
                advantages = self._compute_advantage_actor_only(
                    batch["obs"],
                    actions,
                    critic,
                )
            else:
                with torch.no_grad():
                    nobs_features = self.obs_adapter.encode(
                        batch["obs"]
                    ).reshape(action_chunk.shape[0], -1)

                if chunk_adv_mode == "per_step_vdelta":
                    advantages = (
                        self._compute_chunk_step_advantages_vdelta(
                            nobs_features,
                            actions,
                            dynamics,
                            critic,
                            gamma,
                            lamda,
                            use_gae,
                        )
                    )
                elif chunk_adv_mode == "chunk_vdelta_scalar":
                    advantages = (
                        self._compute_chunk_scalar_advantage_vdelta(
                            nobs_features,
                            actions,
                            dynamics,
                            critic,
                            gamma,
                        )
                    )
                elif chunk_adv_mode == "chunk_vdelta_gae":
                    n_rollout = int(
                        getattr(
                            self.cfg,
                            "chunk_vdelta_gae_n_rollout",
                            3,
                        )
                    )
                    gae_lambda = float(
                        getattr(
                            self.cfg,
                            "chunk_vdelta_gae_lambda",
                            0.95,
                        )
                    )
                    chunk_source = str(
                        getattr(
                            self.cfg,
                            "chunk_vdelta_gae_chunk_source",
                            "repeat_first",
                        )
                    )
                    advantages = (
                        self._compute_chunk_gae_advantage_vdelta(
                            nobs_features,
                            actions,
                            dynamics,
                            critic,
                            gamma,
                            gae_lambda,
                            n_rollout,
                            chunk_source,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported offline_chunk_adv_mode="
                        f"{chunk_adv_mode}"
                    )
            if opt_steps > 1 :
                advantages = self._apply_chunk_adv_clip(
                    advantages
                )

        elif opt_steps > 1:
            with torch.no_grad():
                nobs_features = self.obs_adapter.encode(
                    batch["obs"]
                ).reshape(action_chunk.shape[0], -1)

            advantages, _ = self.NStepValueEstimation(
                nobs_features,
                actions,
                dynamics,
                critic,
                opt_steps,
            )

            if use_gae:
                advantages = self.GAE_withQ(
                    advantages,
                    gamma,
                    lamda,
                )

        else:
            advantages = self._compute_advantage_actor_only(
                batch["obs"],
                actions[:, 0],
                critic,
            )

        new_log_prob_raw, _ = (
            self._policy.evaluate_action_chunk(
                policy_obs,
                action_chunk,
            )
        )

        old_log_prob_raw = old_log_prob_raw[:, :opt_steps]
        new_log_prob_raw = new_log_prob_raw[:, :opt_steps]

        if is_clip_decay:
            if is_linear_decay:
                if clip_ratio_now is None:
                    raise ValueError(
                        "clip_ratio_now is required for "
                        "linear clip decay"
                    )
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio *= self._decay

        use_chunk_level_ratio = getattr(
            self.cfg,
            "bppo_chunk_level_ratio",
            True,
        )

        if (
            chunk_as_single_action
            and opt_steps > 1
            and use_chunk_level_ratio
        ):
            if chunk_ratio_mode == "scalar":
                old_logprob = self._sum_chunk_event_dims(
                    old_log_prob_raw
                )
                new_logprob = self._sum_chunk_event_dims(
                    new_log_prob_raw
                )
                ratio = torch.exp(
                    new_logprob - old_logprob
                )

                self._record_ratio_stats(
                    "offline_multi",
                    ratio,
                    old_logprob,
                    new_logprob,
                )

                adv = advantages.detach().reshape(
                    advantages.shape[0],
                    -1,
                )
                if adv.shape[1] != 1:
                    raise ValueError(
                        "Scalar chunk ratio requires scalar "
                        f"advantage, got {tuple(adv.shape)}"
                    )

                adv = adv[:, 0]
                loss1 = ratio * adv
                loss2 = torch.clamp(
                    ratio,
                    1.0 - self._clip_ratio,
                    1.0 + self._clip_ratio,
                ) * adv
                loss = -torch.min(
                    loss1,
                    loss2,
                ).mean()

            else:
                old_logprob = self._sum_step_event_dims(
                    old_log_prob_raw
                )
                new_logprob = self._sum_step_event_dims(
                    new_log_prob_raw
                )
                ratio = torch.exp(
                    new_logprob - old_logprob
                )

                self._record_ratio_stats(
                    "offline_multi",
                    ratio,
                    old_logprob,
                    new_logprob,
                )

                if chunk_adv_mode == "per_step_vdelta":
                    loss = 0.0

                    for j in range(opt_steps):
                        step_adv = advantages[j].detach()

                        if step_adv.ndim > 1:
                            step_adv = step_adv.squeeze(-1)

                        ratio_j = ratio[:, j]
                        loss1 = ratio_j * step_adv
                        loss2 = torch.clamp(
                            ratio_j,
                            1.0 - self._clip_ratio,
                            1.0 + self._clip_ratio,
                        ) * step_adv

                        loss += -torch.min(
                            loss1,
                            loss2,
                        ).mean()

                    loss = loss / opt_steps

                else:
                    adv = advantages.detach().reshape(
                        advantages.shape[0],
                        -1,
                    )

                    if adv.shape[1] != 1:
                        raise ValueError(
                            "Per-step chunk ratio requires "
                            "scalar advantage, got "
                            f"{tuple(adv.shape)}"
                        )

                    adv = adv[:, 0].unsqueeze(-1)

                    loss1 = ratio * adv
                    loss2 = torch.clamp(
                        ratio,
                        1.0 - self._clip_ratio,
                        1.0 + self._clip_ratio,
                    ) * adv
                    loss = -torch.min(
                        loss1,
                        loss2,
                    ).mean()

        elif opt_steps > 1:
            old_logprob = self._sum_step_event_dims(
                old_log_prob_raw
            )
            new_logprob = self._sum_step_event_dims(
                new_log_prob_raw
            )
            ratio = torch.exp(
                new_logprob - old_logprob
            )

            self._record_ratio_stats(
                "offline_multi",
                ratio,
                old_logprob,
                new_logprob,
            )

            loss = 0.0

            for j in range(opt_steps):
                if chunk_as_single_action:
                    step_adv = advantages.detach()
                else:
                    step_adv = advantages[j].detach()

                step_adv = step_adv.reshape(
                    step_adv.shape[0],
                    -1,
                )

                if step_adv.shape[1] != 1:
                    raise ValueError(
                        "Offline PPO expects scalar step "
                        f"advantage, got {tuple(step_adv.shape)}"
                    )

                step_adv = step_adv[:, 0]
                ratio_j = ratio[:, j]

                loss1 = ratio_j * step_adv
                loss2 = torch.clamp(
                    ratio_j,
                    1.0 - self._clip_ratio,
                    1.0 + self._clip_ratio,
                ) * step_adv

                loss += -torch.min(
                    loss1,
                    loss2,
                ).mean()

            loss = loss / opt_steps

        else:
            old_logprob = self._sum_step_event_dims(
                old_log_prob_raw
            )[:, 0]
            new_logprob = self._sum_step_event_dims(
                new_log_prob_raw
            )[:, 0]

            ratio = torch.exp(
                new_logprob - old_logprob
            )

            self._record_ratio_stats(
                "offline_single",
                ratio,
                old_logprob,
                new_logprob,
            )

            adv = advantages.detach().reshape(
                advantages.shape[0],
                -1,
            )

            if adv.shape[1] != 1:
                raise ValueError(
                    "Offline PPO expects scalar advantage, "
                    f"got {tuple(adv.shape)}"
                )

            adv = adv[:, 0]

            loss1 = ratio * adv
            loss2 = torch.clamp(
                ratio,
                1.0 - self._clip_ratio,
                1.0 + self._clip_ratio,
            ) * adv

            loss = -torch.min(
                loss1,
                loss2,
            ).mean()

        self._optimizer.zero_grad()
        loss.backward()

        trainable_params = [
            p
            for p in self._policy.parameters()
            if p.requires_grad and p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            0.5,
        )
        self._optimizer.step()

        if is_lr_decay:
            self._scheduler.step()

        if is_linear_decay:
            if bppo_lr_now is None:
                raise ValueError(
                    "bppo_lr_now is required for "
                    "linear LR decay"
                )
            for group in self._optimizer.param_groups:
                group["lr"] = bppo_lr_now

        return loss.item()
