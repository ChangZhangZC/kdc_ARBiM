import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
from lerobot.configs.policies import PolicyFeature
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import os
import socket
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed
from lerobot.policies.factory import make_pre_post_processors
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from kuavo_train.utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext
from lerobot.processor import ProcessorStep, NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
# import ipdb


def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    # If `max_training_step` is specified, it takes precedence; 
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        updates_per_epoch = (total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler

def build_policy(name, policy_cfg):
    if name == "diffusion":
        from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper

        return CustomDiffusionPolicyWrapper(policy_cfg)
    if name == "act":
        from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper

        return CustomACTPolicyWrapper(policy_cfg)
    raise KeyError(f"Unsupported policy name: {name}")

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

class AugmentationProcessorStep(ProcessorStep):
    def __init__(self, transform, cam_keys):
        super().__init__()
        self.transform = transform
        self.cam_keys = [k for k in cam_keys if "depth" not in k]  # list of keys in the transition dict to augment

    def __call__(self, transition):
        # Store the current transition (required by ProcessorStep)
        new_transition = transition.copy()

        # Apply transform to each camera key
        data_dict = new_transition.get(TransitionKey.OBSERVATION)
        if data_dict is not None:
            # new_data_dict = {
            #     k: self.transform(v) if k in self.cam_keys else v
            #     for k, v in data_dict.items()
            # }
            new_data_dict = {}
            for k, v in data_dict.items():
                
                if k in self.cam_keys:
                    # print(k)
                    new_data_dict[k] = self.transform(v)
                else:
                    new_data_dict[k] = v
            # print(new_data_dict['observation.images.head_cam_h'].device)
            new_transition[TransitionKey.OBSERVATION] = new_data_dict
            return new_transition
        else:
            return new_transition
        

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features
    

def insert_before_normalizer(pipeline, new_step):
    """
    Insert a processor step before the first NormalizerProcessorStep.
    If no NormalizerProcessorStep is found, append at the end.
    """
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, NormalizerProcessorStep):
            pipeline.steps.insert(i, new_step)
            print(f"Inserted {new_step.__class__.__name__} before NormalizerProcessorStep", {i})
            return new_step
    pipeline.steps.append(new_step)
    print(f"No NormalizerProcessorStep found, appended {new_step.__class__.__name__} at the end")
    return new_step

def remove_aug_step(pipeline, step_to_remove):
    """
    Remove the given step from the pipeline if it exists.
    """
    if step_to_remove in pipeline.steps:
        pipeline.steps.remove(step_to_remove)
        print(f"Removed {step_to_remove.__class__.__name__}")
    else:
        print(f"Step {step_to_remove.__class__.__name__} not found in pipeline")


def _infer_visible_gpu_count() -> int:
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        return max(1, len([x for x in cuda_visible_devices.split(",") if x.strip()]))

    try:
        raw = os.popen("nvidia-smi -L 2>/dev/null | wc -l").read().strip()
        count = int(raw)
        return max(1, count)
    except Exception:
        return 1


def _normalize_gpu_ids(gpu_ids) -> list[int]:
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, ListConfig):
        gpu_ids = list(gpu_ids)
    return [int(x) for x in gpu_ids]


def _pick_available_master_port(preferred_port: int) -> int:
    def _can_listen(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                return False
            return True

    if _can_listen(preferred_port):
        return preferred_port

    for port in range(preferred_port + 1, preferred_port + 200):
        if _can_listen(port):
            print(f"[WARN] LingBot master_port {preferred_port} is busy, fallback to {port}")
            return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
    print(f"[WARN] LingBot master_port {preferred_port} is busy, fallback to ephemeral port {port}")
    return port


def _launch_lingbot_from_policy_name(cfg: DictConfig) -> int:
    from kuavo_train.wrapper.policy.lingbot import (
        CustomLingbotConfigWrapper,
        CustomLingbotPolicyWrapper,
    )

    repo_root = Path(__file__).resolve().parent.parent
    policy_cfg = cfg.get("policy", {})
    if isinstance(policy_cfg, DictConfig):
        policy_cfg = OmegaConf.to_container(policy_cfg, resolve=True)
    if policy_cfg is None:
        policy_cfg = {}
    if isinstance(policy_cfg, dict):
        policy_cfg.pop("_target_", None)

    legacy_cfg = cfg.get("lingbot", {})
    if isinstance(legacy_cfg, DictConfig):
        legacy_cfg = OmegaConf.to_container(legacy_cfg, resolve=True)
    if legacy_cfg is None:
        legacy_cfg = {}

    lingbot_cfg = {}
    lingbot_cfg.update(policy_cfg)
    lingbot_cfg.update(legacy_cfg)

    configured_gpu_ids = _normalize_gpu_ids(getattr(cfg.training, "gpu_ids", []))
    env_overrides = dict(lingbot_cfg.get("env", {}) or {})
    if configured_gpu_ids and "CUDA_VISIBLE_DEVICES" not in env_overrides and not os.getenv("CUDA_VISIBLE_DEVICES"):
        env_overrides["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in configured_gpu_ids)
    if env_overrides.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = env_overrides["CUDA_VISIBLE_DEVICES"]

    nproc = len(env_overrides["CUDA_VISIBLE_DEVICES"].split(",")) if env_overrides.get("CUDA_VISIBLE_DEVICES") else _infer_visible_gpu_count()
    nnodes = int(lingbot_cfg.get("nnodes", int(os.getenv("NNODES", "1"))))
    preferred_master_port = int(lingbot_cfg.get("master_port", int(os.getenv("MASTER_PORT", "62500"))))
    master_port = _pick_available_master_port(preferred_master_port)

    wrapper_cfg = CustomLingbotConfigWrapper(
        lingbot_root=lingbot_cfg.get("lingbot_root", ""),
        lerobot_root=lingbot_cfg.get("lerobot_root", ""),
        train_entry=lingbot_cfg.get("train_entry", "kuavo_train/lingbot/tasks/vla/train_lingbotvla.py"),
        config_path=lingbot_cfg.get("config_path", "configs/policy/lingbot/robotwin_load20000h.yaml"),
        nnodes=nnodes,
        node_rank=int(lingbot_cfg.get("node_rank", int(os.getenv("NODE_RANK", "0")))),
        master_addr=lingbot_cfg.get("master_addr", os.getenv("MASTER_ADDR", "0.0.0.0")),
        master_port=master_port,
        dry_run=bool(lingbot_cfg.get("dry_run", False)),
        env=env_overrides,
    )

    extra_args = list(lingbot_cfg.get("extra_args", []))

    if getattr(cfg, "root", None):
        dataset_root = Path(str(cfg.root))
        extra_args.extend(["--data.train_path", str(dataset_root)])
        try:
            dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=dataset_root)
            state_shape = dataset_metadata.features["observation.state"]["shape"]
            action_shape = dataset_metadata.features["action"]["shape"]
            action_dim = int(action_shape[0]) if action_shape else 0
            state_dim = int(state_shape[0]) if state_shape else 0
            if action_dim > 0:
                extra_args.extend(["--train.action_dim", str(action_dim)])
            print(
                f"[INFO] LingBot dataset dims from LeRobot metadata: "
                f"state_dim={state_dim}, action_dim={action_dim}, root={dataset_root}"
            )
        except Exception as exc:
            print(f"[WARN] Failed to infer LingBot dataset dims from {dataset_root}: {exc!r}")

    if getattr(cfg.training, "batch_size", None):
        micro_bs = int(cfg.training.batch_size)
        extra_args.extend(["--train.micro_batch_size", str(micro_bs)])
        extra_args.extend(["--train.global_batch_size", str(micro_bs * nproc * nnodes)])

    resume_enabled = bool(getattr(cfg.training, "resume", False))
    resume_timestamp = str(getattr(cfg.training, "resume_timestamp", "") or "").strip()
    if resume_enabled and resume_timestamp:
        resume_run_dir = resume_timestamp if resume_timestamp.startswith("run_") else f"run_{resume_timestamp}"
        output_dir = Path(cfg.training.output_directory) / resume_run_dir
        print(f"[INFO] LingBot resume enabled, reusing output_dir: {output_dir}")
    else:
        output_dir = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
    extra_args.extend(["--train.output_dir", str(output_dir)])

    if lingbot_cfg.get("model_path"):
        extra_args.extend(["--model.model_path", str(lingbot_cfg["model_path"])])
    if lingbot_cfg.get("tokenizer_path"):
        extra_args.extend(["--model.tokenizer_path", str(lingbot_cfg["tokenizer_path"])])
    if lingbot_cfg.get("moge_path"):
        extra_args.extend(["--model.moge_path", str(lingbot_cfg["moge_path"])])
    if lingbot_cfg.get("morgbd_path"):
        extra_args.extend(["--model.morgbd_path", str(lingbot_cfg["morgbd_path"])])

    print(f"[INFO] policy_name=lingbot detected, dispatching via wrapper. extra_args={extra_args}")
    runner = CustomLingbotPolicyWrapper(wrapper_cfg)
    return runner.launch(repo_root=repo_root, extra_args=extra_args)

@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)
def main(cfg: DictConfig):
    if cfg.policy_name == "lingbot":
        code = _launch_lingbot_from_policy_name(cfg)
        if code != 0:
            raise RuntimeError(f"LingBot training failed with exit code {code}")
        return

    set_seed(cfg.training.seed)

    # Setup output directory
    output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # Dataset metadata and features
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    print("Camera_keys:", dataset_metadata.camera_keys)
    print("Original dataset features:", dataset_metadata.features)

    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print("policy_cfg", policy_cfg)

    # Build policy
    policy = build_policy(cfg.policy_name, policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"])
    
    # Initialize AMP GradScaler if use_amp is True
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    # autocast context (cuda, or no-op when disabled/non-cuda)
    has_torch_autocast = hasattr(torch, "autocast")
    def make_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        if device.type == "cuda":
            if has_torch_autocast:
                return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)  # noqa
            else:
                from torch.cuda.amp import autocast as cuda_autocast  # noqa
                return cuda_autocast()
        # Fallback: disable on non-cuda to avoid dtype surprises
        return nullcontext()

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(device=device.type, enabled=amp_enabled)
    # print("scaler", device.type, make_autocast(amp_enabled))
    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # ===== Resume logic (perfect resume for AMP & RNG) =====
    
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        print("Resuming from:", resume_path)
        try:
            # Load RNG state
            load_rng_state(resume_path / "rng_state.pth")
            
            # Load policy
            policy = policy.from_pretrained(resume_path, strict=True)
            preprocessor = preprocessor.from_pretrained(resume_path,config_filename="policy_preprocessor.json")

            """ Warning: using `from_pretrained` creates a new policy instance, 
            so the optimizer must be reinitialized here! """
            # print("load policy done ! ")
            optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"])
            
            # Load optimizer, scheduler, scaler and training state
            checkpoint = torch.load(resume_path / "learning_state.pth", map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer"])

            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
            if "scaler" in checkpoint and amp_enabled:
                scaler.load_state_dict(checkpoint["scaler"])
            
            if "steps" in checkpoint:
                steps = checkpoint["steps"]
            
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            
            if "best_loss" in checkpoint:
                best_loss = checkpoint["best_loss"]
            
            # Copy and load log_event
            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)
                
            print(f"Resumed training from epoch {start_epoch}, step {steps}")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            return
    else:
        print("Training from scratch!")

    policy.train().to(device)
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Using AMP: {amp_enabled}")
    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        image_transforms=None,
    )
    # Training loop
    aug_step = insert_before_normalizer(preprocessor, AugmentationProcessorStep(image_transforms, dataset.meta.camera_keys))  # just for training
    
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    for epoch in range(start_epoch, cfg.training.max_epoch):
        dataloader = DataLoader(
            dataset,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=(device.type != "cpu"),
            drop_last=cfg.training.drop_last,
            prefetch_factor=2 if cfg.training.num_workers > 0 else None,
        )

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}")

        
        total_loss = 0.0
        for batch in epoch_bar:
            batch = preprocessor(batch)  # will normalize and put batch to device
            with make_autocast(amp_enabled):
                loss, _ = policy.forward(batch)
            # Scale loss and backward with AMP if enabled
            scaled_loss = loss / cfg.training.accumulation_steps
            
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if steps % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    # Optionally unscale and clip gradients here if you use clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)
                epoch_bar.set_postfix(loss=f"{scaled_loss.item():.3f}", step=steps, lr=lr_scheduler.get_last_lr()[0])

            steps += 1
            total_loss += scaled_loss.item()
        
        # Update best loss
        if total_loss < best_loss:
            best_loss = total_loss
            # Save best model
            policy.save_pretrained(output_directory / "epochbest")
        # Save checkpoint every N epochs
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            policy.save_pretrained(output_directory / f"epoch{epoch+1}")
            # preprocessor.save_pretrained(output_directory)

        # Save last checkpoint (includes AMP scaler & progress for perfect resume)
        # Save last checkpoint
        policy.save_pretrained(output_directory)

        # Save training state including optimizer, scheduler, scaler, and step/epoch info
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "steps": steps,
            "epoch": epoch + 1,
            "best_loss": best_loss
        }
        torch.save(checkpoint, output_directory / "learning_state.pth")
        save_rng_state(output_directory / "rng_state.pth")

    writer.close()


if __name__ == "__main__":
    main()
