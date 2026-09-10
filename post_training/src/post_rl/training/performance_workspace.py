from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from post_rl.critic.cached_iql_critic import CachedIQLCritic
from post_rl.data.latent_cache import (
    FrozenLatentCache,
    build_frozen_latent_cache,
    default_latent_cache_dir,
)
from post_rl.data.offline_dataset import OfflineDataset

from .contract_workspace import TrainACTWorkspace as _ContractTrainACTWorkspace


class TrainACTWorkspace(_ContractTrainACTWorkspace):
    """Performance layer: endpoint sampling plus persistent frozen ACT latents."""

    def _performance_enabled(self) -> bool:
        return bool(self.cfg.dataset.get("use_latent_cache", False))

    def _endpoint_sampling_enabled(self) -> bool:
        return bool(self.cfg.dataset.get("endpoint_obs_only", False))

    def _build_main_dataloaders(self) -> None:
        cfg = self.cfg
        use_cache = self._performance_enabled()
        endpoint_only = self._endpoint_sampling_enabled() and use_cache
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
            endpoint_obs_only=endpoint_only,
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

    def _latent_cache_metadata(self) -> dict:
        return {
            "version": 1,
            "source_dataset": os.path.realpath(
                os.path.abspath(os.path.expanduser(str(self.cfg.input.dataset_path)))
            ),
            "encoder_sha256": str(self._encoder_sha256),
            "normalizer_sha256": str(self._normalizer_sha256),
            "use_depth": bool(self.cfg.dataset.use_depth),
            "feature_dim": int(self.obs_feature_dim),
        }

    def _resolve_latent_cache_dir(self) -> str:
        configured = self.cfg.dataset.get("latent_cache_dir", None)
        if configured:
            return os.path.abspath(os.path.expanduser(str(configured)))
        return default_latent_cache_dir(
            str(self.cfg.input.dataset_path),
            str(self._encoder_sha256),
            str(self._normalizer_sha256),
        )

    def _build_act_observation_frontends(self) -> None:
        super()._build_act_observation_frontends()
        self.latent_cache = None
        if not self._performance_enabled():
            return
        if not self._endpoint_sampling_enabled():
            raise ValueError(
                "dataset.use_latent_cache=true requires dataset.endpoint_obs_only=true."
            )

        cache_dir = self._resolve_latent_cache_dir()
        metadata = self._latent_cache_metadata()
        cache_batch_size = int(self.cfg.dataset.get("latent_cache_batch_size", 64))

        if self.rank == 0:
            self.latent_cache = build_frozen_latent_cache(
                buffer=self.buffer,
                obs_adapter=self.obs_adapter,
                cache_dir=cache_dir,
                metadata=metadata,
                batch_size=cache_batch_size,
                use_depth=bool(self.cfg.dataset.use_depth),
                progress=True,
            )
        if self.is_ddp:
            import torch.distributed as dist

            dist.barrier()
        if self.rank != 0:
            self.latent_cache = FrozenLatentCache(
                cache_dir,
                expected_metadata=metadata,
            )

        self.dataset.set_latent_cache(self.latent_cache)
        self.val_dataset.set_latent_cache(self.latent_cache)
        if self.rank == 0:
            print(
                "Frozen ACT latent cache ready: "
                f"{cache_dir} shape={self.latent_cache.obs.shape}"
            )

    def _build_critic_dataset(self):
        if not self._performance_enabled():
            return super()._build_critic_dataset()

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
            endpoint_obs_only=True,
            latent_cache=self.latent_cache,
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
            print(
                f"Critic dataset: {len(dataset)} samples (stride={sequence_stride}, latent_cache=true)"
            )
        return dataset, dataloader

    def _build_critic(self) -> None:
        if not self._performance_enabled():
            return super()._build_critic()

        cfg = self.cfg
        self.critic = CachedIQLCritic(
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

    def _build_finetune_dataloader(self) -> None:
        if not self._endpoint_sampling_enabled():
            return super()._build_finetune_dataloader()

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
            endpoint_obs_only=True,
            latent_cache=None,
            include_next_obs=False,
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
                f"(stride={stride}, batch_size={cfg.unio4.finetune_batch_size}, endpoint_obs_only=true)"
            )
