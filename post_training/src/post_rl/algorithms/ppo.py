from copy import deepcopy

import hydra
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
        optimizer_cfg,
        lr_scheduler_cfg,
        fix_encoder: bool,
    ) -> None:
        if not fix_encoder:
            raise ValueError(
                "ACT Scheme C requires unio4.fix_encoder=true so the OPE latent space stays fixed."
            )

        self._device = device
        self._policy = deepcopy(policy).to(device)
        self._policy_lr = float(policy_lr)
        self._clip_ratio = float(clip_ratio)
        self._entropy_weight = float(entropy_weight)
        self._decay = float(decay)
        self._optimizer_cfg = optimizer_cfg
        self._lr_scheduler_cfg = lr_scheduler_cfg
        self._grad_hook_handles = []
        self._old_policy_version = 0

        self._policy.eval()
        self.set_old_policy()
        self._build_optimizer()
        self._build_scheduler()

    def _configure_trainable_params(self) -> None:
        for param in self._policy.parameters():
            param.requires_grad = True

        model = self._policy.model
        model.requires_grad_(False)
        for name in ("decoder", "decoder_pos_embed", "action_head"):
            module = getattr(model, name, None)
            if module is not None:
                module.requires_grad_(True)
        self._policy.raw_log_std.requires_grad_(True)

    def _register_gradient_sync_hooks(self, params) -> None:
        for handle in self._grad_hook_handles:
            handle.remove()
        self._grad_hook_handles = []

        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()

        def sync_grad(grad):
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad / world_size

        for param in params:
            self._grad_hook_handles.append(param.register_hook(sync_grad))

    def _build_optimizer(self) -> None:
        self._configure_trainable_params()
        params = [param for param in self._policy.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("PPO policy has no trainable parameters.")
        self._register_gradient_sync_hooks(params)
        self._optimizer = hydra.utils.instantiate(
            self._optimizer_cfg,
            params=params,
            lr=self._policy_lr,
        )

    def _build_scheduler(self) -> None:
        self._scheduler = hydra.utils.instantiate(
            self._lr_scheduler_cfg,
            optimizer=self._optimizer,
        )

    @staticmethod
    def _sum_chunk_event_dims(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,H,D], got {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1).sum(dim=-1)

    def _sync_old_policy(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        version = torch.tensor(
            [self._old_policy_version],
            device=self._device,
            dtype=torch.long,
        )
        dist.broadcast(version, src=0)
        target_version = int(version.item())
        needs_sync = torch.tensor(
            [int(self._old_policy_version != target_version)],
            device=self._device,
            dtype=torch.long,
        )
        dist.all_reduce(needs_sync, op=dist.ReduceOp.MAX)
        if int(needs_sync.item()) == 0:
            return

        for param in self._old_policy.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self._old_policy.buffers():
            dist.broadcast(buffer.data, src=0)
        self._old_policy_version = target_version

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

    def save(self, path: str) -> None:
        self._policy.save_pretrained(path)

    def load(self, path: str) -> None:
        self._policy = StochasticACTPolicyWrapper.from_pretrained(path).to(
            self._device
        )
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
        self._old_policy_version += 1
