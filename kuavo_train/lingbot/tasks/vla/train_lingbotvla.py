import json
from copy import deepcopy
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuavo_train.lingbot.compat import (
    patch_lingbot_model_loader,
    patch_pi0_config_for_lingbot,
    patch_transformers_for_lingbot,
)

patch_transformers_for_lingbot()
patch_pi0_config_for_lingbot()

from lingbotvla.checkpoint import build_checkpointer, ckpt_to_state_dict
from lingbotvla.data import (
    VLADataCollatorWithPacking,
    build_dataloader,
)
from lingbotvla.data.vla_data import liberoDataset, RobotwinDataset, CustomizedRobotwinDataset
from lingbotvla.data.vla_data.transform import Normalizer, prepare_action, prepare_images, prepare_language, prepare_state
from lingbotvla.distributed.offloading import build_activation_offloading_context
from lingbotvla.distributed.parallel_state import get_parallel_state, init_parallel_state
from lingbotvla.distributed.torch_parallelize import build_parallelize_model
from lingbotvla.models import build_foundation_model, build_processor, save_model_assets, save_model_weights, build_tokenizer
from lingbotvla.optim import build_lr_scheduler, build_optimizer
from lingbotvla.utils import helper
from lingbotvla.utils.ema import ema_update
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from lingbotvla.utils.dist_utils import all_reduce

from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, get_depth_target, log_depth

patch_lingbot_model_loader()

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from lingbotvla.data.chat_template import ChatTemplate


logger = helper.create_logger(__name__)
# try:
#     from aistudio_tracking import training_tracking as wandb
# except Exception as e:
#     logger.info_rank0(f"Failed to import aistudio_tracking: {repr(e)}.")


def _normalize_dataset_tasks(tasks: Any) -> Any:
    if hasattr(tasks, "iloc"):
        columns = getattr(tasks, "columns", None)
        if columns is not None:
            if "task_index" in columns:
                return {
                    int(task_index): str(task_name)
                    for task_name, task_index in zip(tasks.index.tolist(), tasks["task_index"].tolist())
                }
            if "task" in columns:
                return tasks["task"].tolist()
            if len(columns) == 1:
                return tasks.iloc[:, 0].tolist()
        return tasks.squeeze().tolist()

    if isinstance(tasks, dict):
        if "task" in tasks:
            value = tasks["task"]
            return value.tolist() if hasattr(value, "tolist") else list(value)
        try:
            ordered_keys = sorted(tasks)
            return [tasks[key] for key in ordered_keys]
        except Exception:
            return list(tasks.values())

    if hasattr(tasks, "tolist"):
        return tasks.tolist()

    return tasks


def _lookup_task_text(tasks: Any, task_index: Any, fallback: str = "") -> str:
    try:
        task_index = int(task_index)
    except Exception:
        return fallback

    if isinstance(tasks, dict):
        return str(tasks.get(task_index, fallback))
    if isinstance(tasks, list):
        if 0 <= task_index < len(tasks):
            return str(tasks[task_index])
        return fallback
    return fallback


def _get_first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _to_uint8_chw_tensor(image: Any) -> torch.Tensor:
    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape={tuple(tensor.shape)}")
    if tensor.dtype.is_floating_point:
        tensor = (tensor * 255).clamp(0, 255)
    return tensor.to(torch.uint8)


def _build_dataset_normalizer(dataset_obj: Any) -> Normalizer:
    dataset_stats = getattr(getattr(dataset_obj, "dataset", None), "meta", None)
    dataset_stats = getattr(dataset_stats, "stats", None) or dataset_obj.dataset_meta.stats
    norm_type: dict[str, str] = {}

    for image_key in [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.images.head_cam_h",
        "observation.images.wrist_cam_l",
        "observation.images.wrist_cam_r",
    ]:
        if image_key in dataset_stats:
            norm_type[image_key] = "identity"

    for state_key in ["observation.state", "state"]:
        if state_key in dataset_stats:
            norm_type[state_key] = getattr(dataset_obj, "config", None) and "bounds_99_woclip" or "bounds_99_woclip"

    for action_key in ["action", "actions"]:
        if action_key in dataset_stats:
            norm_type[action_key] = getattr(dataset_obj, "config", None) and "bounds_99_woclip" or "bounds_99_woclip"

    return Normalizer(
        norm_stats=dataset_stats,
        from_file=False,
        norm_type=norm_type,
    )


def _generic_robot_dataset_getdata(dataset_obj: Any, idx: int) -> dict[str, Any]:
    item = dataset_obj.dataset[idx]
    task_text = item.get(
        "task",
        _lookup_task_text(dataset_obj.dataset_meta.tasks, item.get("task_index", -1), ""),
    )

    normalized_item = dataset_obj.normalizer.normalize(item)
    head = _get_first_present(
        normalized_item,
        ["observation.images.cam_high", "observation.images.head_cam_h"],
    )
    left = _get_first_present(
        normalized_item,
        ["observation.images.cam_left_wrist", "observation.images.wrist_cam_l"],
    )
    right = _get_first_present(
        normalized_item,
        ["observation.images.cam_right_wrist", "observation.images.wrist_cam_r"],
    )
    state = _get_first_present(normalized_item, ["observation.state", "state"])
    action = _get_first_present(normalized_item, ["action", "actions"])
    action_is_pad = _get_first_present(normalized_item, ["action_is_pad", "actions_is_pad"])

    if state is None or action is None:
        raise KeyError(
            f"Missing state/action keys in sample. Available keys: {sorted(normalized_item.keys())}"
        )

    if action_is_pad is None:
        action_is_pad = torch.zeros(action.shape[0], dtype=torch.bool)

    if left is None:
        left = right
    if right is None:
        right = left

    batch_images: dict[str, torch.Tensor] = {}
    if head is not None:
        batch_images["base_0_rgb"] = _to_uint8_chw_tensor(head)
    if left is not None:
        batch_images["left_wrist_0_rgb"] = _to_uint8_chw_tensor(left)
    if right is not None:
        batch_images["right_wrist_0_rgb"] = _to_uint8_chw_tensor(right)

    batch_dict = {
        "image": batch_images,
        "state": torch.as_tensor(state).to(torch.float32),
        "action": torch.as_tensor(action).to(torch.float32),
        "action_is_pad": torch.as_tensor(action_is_pad, dtype=torch.bool),
        "prompt": [task_text],
    }
    state = prepare_state(dataset_obj.config, batch_dict)
    lang_tokens, lang_masks = prepare_language(dataset_obj.config, dataset_obj.tokenizer, batch_dict)
    actions = prepare_action(dataset_obj.config, batch_dict)
    images, img_masks, pil_images = prepare_images(
        dataset_obj.config,
        dataset_obj.image_processor,
        batch_dict,
        use_depth_align=dataset_obj.use_depth_align,
    )

    result = {
        "images": images,
        "img_masks": img_masks,
        "state": state,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "actions": actions,
        "action_is_pad": batch_dict["action_is_pad"],
    }
    if dataset_obj.use_depth_align:
        result["pil_images"] = pil_images
    return result


def _patch_lerobot_task_lookup() -> None:
    def _wrap_init(dataset_cls: type) -> None:
        original_init = dataset_cls.__init__
        if getattr(original_init, "_kuavo_task_patch_applied", False):
            return

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.dataset_meta.tasks = _normalize_dataset_tasks(self.dataset_meta.tasks)
            if hasattr(self, "dataset") and hasattr(self.dataset, "video_backend"):
                # Real-robot LeRobot datasets often store camera streams as videos.
                # Force a stable backend here because torchcodec is not usable in the
                # current training environment, while pyav works without extra system fixes.
                self.dataset.video_backend = "pyav"
            if hasattr(self, "dataset") and hasattr(self.dataset, "meta") and hasattr(self, "normalizer"):
                dataset_stats = getattr(self.dataset.meta, "stats", {})
                state_stats = dataset_stats.get("observation.state") or dataset_stats.get("state")
                if state_stats is not None and "q01" in state_stats:
                    current_stats = self.normalizer.norm_stats.get("observation.state", {})
                    current_q01 = current_stats.get("q01")
                    if current_q01 is None or len(np.asarray(current_q01)) != len(np.asarray(state_stats["q01"])):
                        self.normalizer = _build_dataset_normalizer(self)

        patched_init._kuavo_task_patch_applied = True
        dataset_cls.__init__ = patched_init

    for dataset_cls in (liberoDataset, RobotwinDataset, CustomizedRobotwinDataset):
        _wrap_init(dataset_cls)

    RobotwinDataset.getdata = _generic_robot_dataset_getdata
    CustomizedRobotwinDataset.getdata = _generic_robot_dataset_getdata


_patch_lerobot_task_lookup()

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]

@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    freeze_vision_encoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vision encoder in VLA model."},
    )
    tokenizer_max_length: int = field(
        default=48,
        metadata={"help": "Maximum length of the tokenizer."},
    )
    enable_expert_vision: bool = field(
        default=False,
        metadata={"help": "Whether to enable expert vision."},
    )
    expert_vision_type: str | None = field(
        default=None,
        metadata={"help": "Type of expert vision. Currently only support vit."},
    )
    expert_vision_path: str | None = field(
        default=None,
        metadata={"help": "Path to expert vision model."},
    )
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension."},
    )
    max_action_dim: int = field(
        default=32,
        metadata={"help": "Action dimension after padding."},
    )
    max_state_dim: int = field(
        default=32,
        metadata={"help": "State dimension after padding."},
    )
    chunk_size: int = field(
        default=50,
        metadata={"help": "Chunk size of action."},
    )
    vlm_causal: bool = field(
        default=False,
        metadata={"help": "Whether to use causal atten for img anb lang tokens in vlm."},
    )
    use_ema: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA."},
    )
    qwenvl_bos: bool = field(
        default=False,
        metadata={"help": "Whether to use qwenvl bos."},
    )
    ema_rate:  float = field(
        default=0.9999,
        metadata={"help": "Rate of EMA."},
    )
    pre_train: bool = field(
        default=False,
        metadata={"help": "Whether to apply pretraining."},
    )
    loss_type: str = field(
        default='fm',
        metadata={"help": "Which loss to use."},
    )
    align_params: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "The config of vaco"},
    )
    use_ki: bool = field(
        default=False,
        metadata={"help": "Whether to apply knowledge insulating."},
    )
    ignore_depth: bool = field(
        default=False,
        metadata={"help": "Whether to ignore depth model in FSDP2."},
    )
    my_tokenizer_max_length: int = field(
        default=72,
        metadata={"help": ""},
    )
    use_subtask: bool = field(
        default=False,
        metadata={"help": "Whether to predict subtask from vlm."},
    )
    use_state: bool = field(
        default=False,
        metadata={"help": "Whether to use stringfy state in prefix."},
    )
    use_fast_action: bool = field(
        default=False,
        metadata={"help": "Whether to use fast action prediction."},
    )
    skip_max_norm: bool = field(
        default=False,
        metadata={"help": "Whether to skip batch with too large grad norm."},
    )
    decayed_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum norm for the decayed gradients."},
    )
    stable_train_steps: int = field(
        default=100000,
        metadata={"help": "Training steps for stable training, after this step, the decayed_max_grad_norm will be applied."},
    )
    resume_dataloader_state: bool = field(
        default=True,
        metadata={"help": "Whether to resume dataloader."},
    )
    norm_qkv: bool = field(
        default=False,
        metadata={"help": "Whether to apply RMSNorm for qkv."},
    )
    use_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt condition."},
    )
    embodiment_name: str = field(
        default=None,
        metadata={"help": "Name of the embodiment type."},
    )

@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    robot_config_root: str = field(
        default=None,
        metadata={"help": "Path to get all robot configs."},
    )
    joints: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of joints and their dim"},
    )
    cameras:Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of used images"},
    )
    norm_type:Literal["meanstd", "bounds_99", "bounds_98", "bounds_98_woclip", "bounds_99_woclip"] = field(
        default="bounds_99",
        metadata={"help": "Type of the normalization."},
    )
    img_size: int = field(
        default=224,
        metadata={"help": "Size of the image."},
    )
    norm_stats_file: str = field(
        default=None,
        metadata={"help": "Path to the normalization stats file."},
    )


@dataclass
class Arguments:

    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    config_kwargs = {'vlm_repo_id': getattr(args.model, "vlm_repo_id", None)}
    config_kwargs['action_dim'] = getattr(args.train, "action_dim", 7)
    config_kwargs['max_action_dim'] = getattr(args.train, "max_action_dim", 32)
    config_kwargs['max_state_dim'] = getattr(args.train, "max_state_dim", 32)
    config_kwargs['chunk_size'] = getattr(args.train, "chunk_size", 50)
    config_kwargs['tokenizer_path'] = getattr(args.model, "tokenizer_path", None)
    config_kwargs['post_training'] = getattr(args.model, "post_training", False)
    config_kwargs['incremental_training'] = getattr(args.model, "incremental_training", False)
    config_kwargs['depth_incremental_training'] = getattr(args.model, "depth_incremental_training", False)
    config_kwargs['norm_qkv'] = getattr(args.train, "norm_qkv", False)
    config_kwargs['enable_expert_vision'] = args.train.enable_expert_vision
    config_kwargs['expert_vision_type'] = getattr(args.train, "expert_vision_type", None)
    config_kwargs['expert_vision_path'] = getattr(args.train, "expert_vision_path", None)
    config_kwargs['adanorm_time'] = getattr(args.model, "adanorm_time", False)
    if not getattr(args.model, "adanorm_time", False):
        assert not getattr(args.model, "separate_time_proj", False), 'separate_time_proj should be dropped when we do not apply adanorm_time!!'
    config_kwargs['split_gate_liner'] = getattr(args.model, "split_gate_liner", False)
    config_kwargs['nosplit_gate_liner'] = getattr(args.model, "nosplit_gate_liner", False)
    config_kwargs['separate_time_proj'] = getattr(args.model, "separate_time_proj", False)
    config_kwargs['old_adanorm'] = getattr(args.model, "old_adanorm", False)
    if getattr(args.model, "old_adanorm", False):
        assert getattr(args.model, "adanorm_time", False), 'Apply old_adanorm should apply adanorm_time!!'
    config_kwargs['final_norm_adanorm'] = getattr(args.model, "final_norm_adanorm", False)
    config_kwargs['loss_type'] = getattr(args.train, "loss_type", 'fm')
    config_kwargs['align_params'] = getattr(args.train, "align_params", None)
    if args.train.enable_expert_vision and not args.model.post_training:
        assert args.train.expert_vision_path is not None, "expert_vision_path is required when enable_expert_vision is True!!!"
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        init_device=args.train.init_device,
        freeze_vision_encoder=args.train.freeze_vision_encoder,
        tokenizer_max_length=args.train.tokenizer_max_length,
        vocab_size=args.model.vocab_size,
        use_lm_head=args.model.use_lm_head,
        force_use_huggingface=args.model.force_use_huggingface,
        config_kwargs=config_kwargs,
    )
    use_depth_align = True if args.train.align_params != {} else False
    depth_model_type = None
    if use_depth_align:
        assert args.model.moge_path is not None and args.model.morgbd_path is not None, 'Depth models need to be loaded when uing LingBot-VLA-Depth!!!'
        args.train.align_params['visual_dir'] = os.path.join(args.train.output_dir, 'images')
        args.train.align_params['depth']['moge_path'] = args.model.moge_path
        args.train.align_params['depth']['morgbd_path'] = args.model.morgbd_path
        depth_model_type = args.train.align_params['depth']['model_type']
        moge_model, morgbd_model = build_depth_model(args.train.align_params)
        if args.train.use_compile:
            moge_model = torch.compile(moge_model)
            morgbd_model = torch.compile(morgbd_model)
        os.makedirs(args.train.align_params['visual_dir'], exist_ok=True)
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path) # if use build_processor,  tokenizer is processor.tokenizer

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.data.datasets_type == 'vla':
        data_collate_fn.append(VLADataCollatorWithPacking())
    else:
        if args.train.rmpad_with_pos_ids:
            data_collate_fn.append(OmniDataCollatorWithPacking()) # TODO 8.21
        else:
            data_collate_fn.append(OmniDataCollatorWithPadding())

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == 'vla':
            logger.info_rank0("Start building VLA dataset")
            args.data.chunk_size = args.train.chunk_size
            if args.data.data_name == 'libero':
                train_dataset = liberoDataset(repo_id=args.data.train_path, config=model.config, tokenizer=processor.tokenizer, data_config=args.data, image_processor=processor.image_processor if 'qwen' in args.model.tokenizer_path.lower() else None,use_depth_align=use_depth_align)
            elif 'custom' in (args.data.data_name or "").lower():
                train_dataset = CustomizedRobotwinDataset(repo_id=args.data.train_path, config=model.config, tokenizer=processor.tokenizer, data_config=args.data, image_processor=processor.image_processor if 'qwen' in args.model.tokenizer_path.lower() else None, use_depth_align=use_depth_align)
            elif 'robotwin' in args.data.data_name.lower():
                train_dataset = RobotwinDataset(repo_id=args.data.train_path, config=model.config, tokenizer=processor.tokenizer, data_config=args.data, image_processor=processor.image_processor if 'qwen' in args.model.tokenizer_path.lower() else None, use_depth_align=use_depth_align)
            else:
                train_dataset = CustomizedRobotwinDataset(repo_id=args.data.train_path, config=model.config, tokenizer=processor.tokenizer, data_config=args.data, image_processor=processor.image_processor if 'qwen' in args.model.tokenizer_path.lower() else None, use_depth_align=use_depth_align)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))
        
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor if args.data.num_workers > 0 else None,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    if args.train.use_ema:
        model_ema = deepcopy(model).eval()
    else:
        model_ema = None

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_fp32=args.train.enable_fp32,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules if args.train.module_fsdp_enable else None,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        fsdp_llm_blocks=False,
        ignore_norm=False,
        use_depth_align=use_depth_align,
        ignore_depth=args.train.ignore_depth,
    )
    if model_ema is not None:
        model_ema = build_parallelize_model(
            model_ema,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_fp32=args.train.enable_fp32,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            init_device=args.train.init_device,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            fsdp_kwargs=fsdp_kwargs,
            basic_modules=model_ema._no_split_modules if args.train.module_fsdp_enable else None,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            fsdp_llm_blocks=False,
            ignore_norm=False,
            use_depth_align=use_depth_align,
            ignore_depth=args.train.ignore_depth,
        )
    if args.train.use_compile:
        model = torch.compile(model)
        if model_ema is not None: model_ema = torch.compile(model_ema)

    if args.train.use_ema:
        ema_update(model_ema, model, 0)

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer,
        post_training=args.model.post_training,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        log_dir=f"{args.train.output_dir}/runs/"
        writer = SummaryWriter(log_dir=log_dir)
        if args.train.use_wandb:
            wandb.init(
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    load_checkpoint_path = None
    candidates = []
    if args.train.load_checkpoint_path or args.train.enable_resume:
        if args.train.load_checkpoint_path:
            load_checkpoint_path = args.train.load_checkpoint_path
            candidates = [load_checkpoint_path]
        elif args.train.enable_resume:
            checkpoint_dir = f'{args.train.output_dir}/checkpoints'
            if os.path.exists(checkpoint_dir):
                pattern = re.compile(r"global_step_(\d+)")
                tmp = []
                for dirname in os.listdir(checkpoint_dir):
                    match = pattern.fullmatch(dirname)
                    if match:
                        step = int(match.group(1))
                        tmp.append((step, os.path.join(checkpoint_dir, dirname)))
                tmp.sort(key=lambda x: x[0], reverse=True)
                candidates = [p for _, p in tmp]
            if candidates:
                load_checkpoint_path = candidates[0]
            else:
                logger.info_rank0(f"No checkpoints in {args.train.output_dir} now!")
    if candidates:
        last_err = None
        loaded = False
        for cp in candidates:
            state = {"model": model, "ema": model_ema, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            try:
                Checkpointer.load(cp, state)
                global_step = state["extra_state"]["global_step"]
                start_epoch = global_step // args.train.train_steps
                start_step = global_step % args.train.train_steps
                lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                if start_step > 0 and args.train.resume_dataloader_state:
                    train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
                environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])
                if start_step == 0:  # resume at the end of epoch
                    iter(train_dataloader)  # clear resume state and prefetch data
                dist.barrier()
                logger.info_rank0(f"Load distributed checkpoint from {cp} successfully!")
                loaded = True
                break
            except Exception as e:
                last_err = e
                logger.info_rank0(f"Failed to load checkpoint {cp}: {repr(e)}. Trying older one...")
                continue
        if not loaded:
            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")
    else:
        logger.info_rank0("Starting training from scratch.")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    if model_ema is not None:
        model_ema.eval()
    # create the path in advance to save loss log
    if args.train.global_rank == 0:
        os.makedirs(args.train.save_checkpoint_path, exist_ok=True)
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            total_vla_loss = 0
            total_depth_loss = 0
            depth_targets = None
            depth_preds = None
            torch.cuda.synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                dataset_names = micro_batch.pop('rep_id', None)
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                depth_forward_time = 0
                if use_depth_align:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pil_images = micro_batch.pop('pil_images', None)
                            depth_targets, cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), pil_images)

                with model_fwd_context:
                    # torch.cuda.synchronize()
                    loss, vla_loss, depth_loss, loss_log, depth_preds = model(**micro_batch, vlm_causal = args.train.vlm_causal, use_ki = args.train.use_ki, depth_targets=depth_targets)
                    # torch.cuda.synchronize()

                    loss = loss / len(micro_batches)
                    vla_loss = vla_loss / len(micro_batches)
                    depth_loss = depth_loss / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                total_vla_loss += vla_loss.item()
                if not (isinstance(depth_loss, int) or isinstance(depth_loss, float)):
                    total_depth_loss += depth_loss.item()
                del micro_batch
            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
            else:
                max_grad_norm = args.train.max_grad_norm
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, total_vla_loss, total_depth_loss, grad_norm = all_reduce((total_loss, total_vla_loss, total_depth_loss, grad_norm), group=get_parallel_state().fsdp_group)
            if model_ema is not None:
                ema_update(model_ema, model, args.train.ema_rate)
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            data_loader_tqdm.update()
            logger.info_rank0(
                f"Step {global_step}/{args.train.train_steps}, "
                f"Epoch {epoch+1}, "
                f"Loss {total_loss:.4f}, "
                f"VLA_Loss {total_vla_loss:.4f}, "
                f"Depth_Loss {total_depth_loss:.4f}, "
                f"GradNorm {grad_norm:.4f}, "
                f"LR {lr:.2e}, "
                f"StepTime {delta_time:.3f}s, "
            )


            if args.train.global_rank == 0:
                writer.add_scalar("training/loss", total_loss, global_step)
                writer.add_scalar("training/vla_loss", total_vla_loss, global_step)
                writer.add_scalar("training/depth_loss", total_depth_loss, global_step)
                writer.add_scalar("training/grad_norm", grad_norm, global_step)
                writer.add_scalar("training/lr", lr, global_step)
                writer.add_scalar("steptime", delta_time, global_step)
                # we only log the last mini batch if grad acc is activated
                if dataset_names is not None and 'batch_mean_losses' in loss_log:
                    batch_mean_losses = loss_log['batch_mean_losses']  # shape (B,)
                    if hasattr(batch_mean_losses, "detach"):
                        batch_mean_losses = batch_mean_losses.detach().cpu()

                    group_losses = defaultdict(list)
                    for name, loss_value in zip(dataset_names, batch_mean_losses):
                        group_losses[name].append(loss_value.item() if hasattr(loss_value, "item") else float(loss_value))

                    for name, values in group_losses.items():
                        mean_loss = sum(values) / len(values)
                        writer.add_scalar(f"detailed_loss/{name}", mean_loss, global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
                        helper.upload_trace(
                            args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir
                        )

                loss_record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": total_loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    "step_time": delta_time
                }
                loss_file_path = os.path.join(args.train.save_checkpoint_path, "loss.jsonl")
                try:
                    with open(loss_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(loss_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.info_rank0(f"⚠️ Failed to write loss.jsonl: {e}")

                # if use_depth_align:
                #     if global_step % args.train.align_params['visual_steps'] == 0:
                #         with torch.no_grad():
                #             with torch.autocast(device_type="cuda", dtype=torch.bfloat16):                
                #                 log_depth(morgbd_model, depth_preds, depth_targets, steps=global_step, config=args.train.align_params, cls_token=cls_token)

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")

                state = {
                    "model": model,
                    "ema": model_ema,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                if args.train.global_rank == 0:
                    if args.train.save_hf_weights and save_checkpoint_path is not None:
                        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        if args.train.enable_fp32:
                            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                        else:
                            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                        if "ema" in state and state["ema"] is not None:
                            ema_hf_weights_path = os.path.join(save_checkpoint_path, "ema_hf_ckpt")
                            ema_model_state_dict = ckpt_to_state_dict(
                                save_checkpoint_path=save_checkpoint_path,
                                output_dir=args.train.output_dir,
                                ckpt_manager=args.train.ckpt_manager,
                                ema=True
                            )
                            if args.train.enable_fp32:
                                save_model_weights(ema_hf_weights_path, ema_model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                            else:
                                save_model_weights(ema_hf_weights_path, ema_model_state_dict, model_assets=model_assets)
                            logger.info_rank0(f"Huggingface EMA checkpoint saved at {ema_hf_weights_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "ema": model_ema,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            if args.train.global_rank == 0:
                if args.train.save_hf_weights and save_checkpoint_path is not None:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                    model_state_dict = ckpt_to_state_dict(
                        save_checkpoint_path=save_checkpoint_path,
                        output_dir=args.train.output_dir,
                        ckpt_manager=args.train.ckpt_manager,
                    )
                    if args.train.enable_fp32:
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                    else:
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                    if "ema" in state and state["ema"] is not None:
                        ema_hf_weights_path = os.path.join(save_checkpoint_path, "ema_hf_ckpt")
                        ema_model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                            ema=True
                        )
                        if args.train.enable_fp32:
                            save_model_weights(ema_hf_weights_path, ema_model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                        else:
                            save_model_weights(ema_hf_weights_path, ema_model_state_dict, model_assets=model_assets)
                        logger.info_rank0(f"Huggingface EMA checkpoint saved at {ema_hf_weights_path} successfully!")

    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0:
        if args.train.save_hf_weights and save_checkpoint_path is not None:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager,
            )
            if args.train.enable_fp32:
                save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
            else:
                save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
            if "ema" in state and state["ema"] is not None:
                ema_hf_weights_path = os.path.join(save_checkpoint_path, "ema_hf_ckpt")
                ema_model_state_dict = ckpt_to_state_dict(
                    save_checkpoint_path=save_checkpoint_path,
                    output_dir=args.train.output_dir,
                    ckpt_manager=args.train.ckpt_manager,
                    ema=True
                )
                if args.train.enable_fp32:
                    save_model_weights(ema_hf_weights_path, ema_model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                else:
                    save_model_weights(ema_hf_weights_path, ema_model_state_dict, model_assets=model_assets)
                logger.info_rank0(f"Huggingface EMA checkpoint saved at {ema_hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
