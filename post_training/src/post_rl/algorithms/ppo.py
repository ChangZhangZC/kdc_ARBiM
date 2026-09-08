from copy import deepcopy

import torch
import torch.distributed as dist

from ..policy.stochastic_act_policy import StochasticACTPolicyWrapper


class ProximalPolicyOptimization:
    def __init__(
        self,
        policy: StochasticACTPolicyWrapper,
        device: torch.device,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        is_iql: bool,
        ratio_strategy: str,
        fix_encoder: bool,
    ) -> None:
        self._device = device
        self._policy = deepcopy(policy).to(device)
        self._fix_encoder = fix_encoder
        self._policy_lr = policy_lr
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega
        self._batch_size = batch_size
        self._is_iql = is_iql
        self._ratio_strategy = ratio_strategy

        if ratio_strategy not in {"per_step", "scalar"}:
            raise ValueError(
                f"Unsupported ratio_strategy={ratio_strategy}. "
                "Expected 'per_step' or 'scalar'."
            )

        self._policy.eval()
        self.set_old_policy()
        self._build_optimizer()
        self._build_scheduler()

    def _configure_trainable_params(self) -> None:
        for param in self._policy.parameters():
            param.requires_grad = True
        if not self._fix_encoder:
            return

        model = self._policy.model
        frontend_names = [
            "backbone",
            "encoder_img_feat_input_proj",
            "encoder_robot_state_input_proj",
            "encoder_env_state_input_proj",
            "depth_backbone",
            "encoder_depth_feat_input_proj",
            "cross_modal_fusion",
            "cross_modal_fusion_proj",
        ]
        for name in frontend_names:
            module = getattr(model, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def _build_optimizer(self) -> None:
        self._configure_trainable_params()
        params = [p for p in self._policy.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("PPO policy has no trainable parameters.")
        self._optimizer = torch.optim.Adam(params, lr=self._policy_lr)

    @staticmethod
    def _sync_gradients(params: list[torch.nn.Parameter]) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for param in params:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

    def weighted_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        weight = torch.where(
            advantage > 0,
            torch.as_tensor(self._omega, device=advantage.device, dtype=advantage.dtype),
            torch.as_tensor(1.0 - self._omega, device=advantage.device, dtype=advantage.dtype),
        )
        return weight * advantage

    @staticmethod
    def _sum_step_event_dims(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,H,D], got {tuple(x.shape)}")
        return x.sum(dim=-1)

    @staticmethod
    def _sum_chunk_event_dims(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,H,D], got {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1).sum(dim=-1)

    def _reduce_event_dims(self, x: torch.Tensor) -> torch.Tensor:
        if self._ratio_strategy == "per_step":
            return self._sum_step_event_dims(x)
        if self._ratio_strategy == "scalar":
            return self._sum_chunk_event_dims(x)
        raise ValueError(
            f"Unsupported ratio_strategy={self._ratio_strategy}. "
            "Expected 'per_step' or 'scalar'."
        )

    def _align_advantage(
        self,
        advantage: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        if self._ratio_strategy == "scalar":
            advantage = advantage.reshape(advantage.shape[0], -1)
            if advantage.shape[1] != 1:
                raise ValueError(
                    f"Scalar ratio requires scalar advantage, got {tuple(advantage.shape)}"
                )
            return advantage[:, 0]

        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)
        elif advantage.ndim == 3 and advantage.shape[-1] == 1:
            advantage = advantage.squeeze(-1)
        if advantage.ndim != 2:
            raise ValueError(
                f"Per-step ratio requires advantage [B,1] or [B,H], got {tuple(advantage.shape)}"
            )
        if advantage.shape[1] not in {1, ratio.shape[1]}:
            raise ValueError(
                f"Advantage shape {tuple(advantage.shape)} cannot match ratio "
                f"shape {tuple(ratio.shape)}"
            )
        return advantage

    def loss(
        self,
        obs: dict,
        action: torch.Tensor,
        advantage: torch.Tensor,
        is_clip_decay: bool = False,
        is_linear_decay: bool = False,
        clip_ratio_now: float = None,
    ) -> torch.Tensor:
        action = action.detach()
        advantage = advantage.detach()

        with torch.no_grad():
            old_log_prob_raw, _ = self._old_policy.evaluate_action_chunk(obs, action)

        new_log_prob_raw, entropy_raw = self._policy.evaluate_action_chunk(obs, action)
        old_log_prob = self._reduce_event_dims(old_log_prob_raw)
        new_log_prob = self._reduce_event_dims(new_log_prob_raw)
        entropy = self._reduce_event_dims(entropy_raw)

        ratio = torch.exp(new_log_prob - old_log_prob)
        advantage = self.weighted_advantage(advantage.detach())
        advantage = self._align_advantage(advantage, ratio)

        if is_clip_decay:
            if is_linear_decay:
                if clip_ratio_now is None:
                    raise ValueError("clip_ratio_now is required for linear clip decay.")
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio *= self._decay

        loss1 = ratio * advantage
        loss2 = torch.clamp(
            ratio,
            1.0 - self._clip_ratio,
            1.0 + self._clip_ratio,
        ) * advantage
        entropy_bonus = entropy * self._entropy_weight
        return -(torch.min(loss1, loss2) + entropy_bonus).mean()

    def update(
        self,
        obs: dict,
        action: torch.Tensor,
        advantage: torch.Tensor,
        is_clip_decay: bool = False,
        is_lr_decay: bool = False,
        is_linear_decay: bool = False,
        bppo_lr_now: float = None,
        clip_ratio_now: float = None,
    ) -> float:
        policy_loss = self.loss(
            obs=obs,
            action=action,
            advantage=advantage,
            is_clip_decay=is_clip_decay,
            is_linear_decay=is_linear_decay,
            clip_ratio_now=clip_ratio_now,
        )

        self._optimizer.zero_grad()
        policy_loss.backward()
        trainable_params = [
            p for p in self._policy.parameters()
            if p.requires_grad and p.grad is not None
        ]
        self._sync_gradients(trainable_params)
        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
        self._optimizer.step()

        if is_lr_decay:
            self._scheduler.step()
        if is_linear_decay:
            if bppo_lr_now is None:
                raise ValueError("bppo_lr_now is required for linear LR decay.")
            for group in self._optimizer.param_groups:
                group["lr"] = bppo_lr_now

        return policy_loss.item()

    @torch.no_grad()
    def select_action(
        self,
        obs: dict,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return self._policy.get_action_mean(obs)
        action, _, _ = self._policy.sample_action_chunk(obs)
        return action

    def _build_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=2,
            gamma=0.98,
        )

    def save(self, path: str) -> None:
        self._policy.save_pretrained(path)

    def load(self, path: str) -> None:
        self._policy = StochasticACTPolicyWrapper.from_pretrained(path).to(self._device)
        self._policy.eval()
        self.set_old_policy()
        self._build_optimizer()
        self._build_scheduler()

    def set_policy(
        self,
        policy: StochasticACTPolicyWrapper,
        device: torch.device | None = None,
    ) -> None:
        device = self._device if device is None else device
        self._device = device
        self._policy = deepcopy(policy).to(device)
        self._old_policy = self._old_policy.to(device)
        self._policy.eval()
        self._old_policy.eval()
        self._build_optimizer()
        self._build_scheduler()

    def set_old_policy(self) -> None:
        self._old_policy = deepcopy(self._policy).to(self._device)
        self._old_policy.eval()
        for param in self._old_policy.parameters():
            param.requires_grad = False
