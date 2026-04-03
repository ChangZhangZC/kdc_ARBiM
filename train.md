# LingBot 训练说明

本文说明如何在当前仓库中启动 LingBot 训练，以及两层配置文件的实际关系。

## 1. 训练入口

当前训练入口命令：

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

这条命令会走 `policy_name=lingbot` 分支，由 `kuavo_train/train_policy.py` 调起 `torchrun`，最终执行：

```text
kuavo_train/lingbot/tasks/vla/train_lingbotvla.py
```

## 2. 两层配置文件

训练实际由两层配置组成：

### 外层启动配置

文件：

```text
configs/policy/lingbot_config.yaml
```

作用：

- 指定训练入口使用 `lingbot`
- 指定数据根目录 `root`
- 指定模型路径 `policy.model_path`
- 指定 tokenizer 路径 `policy.tokenizer_path`
- 指定显卡 `training.gpu_ids` / `policy.env.CUDA_VISIBLE_DEVICES`
- 指定外层输出目录模板 `training.output_directory`
- 指定是否续训 `training.resume`

### 内层 LingBot 训练配置

文件：

```text
configs/policy/lingbot/robotwin_load20000h.yaml
```

作用：

- 控制 LingBot 训练细节
- 包括 `train.*`、`data.*`、`model.*` 的默认值

## 3. 配置优先级

优先级不是简单二选一，而是：

1. 先读取 `lingbot_config.yaml`
2. 再读取 `robotwin_load20000h.yaml`
3. 外层启动器把部分参数转成命令行参数
4. 命令行参数覆盖 `robotwin_load20000h.yaml` 中对应字段

也就是说：

- **没有被外层传递的参数**，最终使用 `robotwin_load20000h.yaml`
- **被外层显式传递的参数**，以外层为准

当前会被外层覆盖进去的关键参数包括：

- `data.train_path`
- `train.action_dim`
- `train.micro_batch_size`
- `train.global_batch_size`
- `train.output_dir`
- `model.model_path`
- `model.tokenizer_path`
- 可选的 `model.moge_path`
- 可选的 `model.morgbd_path`

所以如果你改了 `lingbot_config.yaml` 里的很多 `training.*` 字段，但启动器没有把它透传给 LingBot，那么这些字段不会自动覆盖内层训练配置。

## 4. 当前推荐训练方式

### 4.1 先检查关键路径

请先确认这些路径存在：

```text
/home/yunxi/lmy/VLA/lingbot-vla
/home/yunxi/lmy/VLA/lingbot-vla-4b
/home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct
/home/yunxi/lmy/VLA/huggingface/lerobot/pick_and_place0_4_2
```

以及当前仓库自带：

```text
third_party/lerobot/src
```

当前训练链路已经优先使用本仓库内的：

```text
third_party/lerobot/src
```

而不是旧仓库的历史目录。

### 4.2 直接启动训练

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

### 4.3 只看最终会启动什么命令

如果只想检查最终生成的 `torchrun` 命令，不真正训练：

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml policy.dry_run=true
```

## 5. 当前配置建议

### 显卡

如果你希望多卡训练，建议直接在 `configs/policy/lingbot_config.yaml` 中设置：

```yaml
policy:
  env:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

如果不设置 `policy.env.CUDA_VISIBLE_DEVICES`，启动器会尝试参考 `training.gpu_ids`。

### batch size

当前外层配置中的：

```yaml
training:
  batch_size: 1
```

会被转换为：

- `--train.micro_batch_size 1`
- `--train.global_batch_size = micro_batch_size * GPU数 * 节点数`

例如 8 卡时，会变成：

```text
micro_batch_size=1
global_batch_size=8
```

如果单卡训练出现 OOM，优先保持 `batch_size: 1`，并改用多卡分片。

## 6. 续训

外层配置里：

```yaml
training:
  resume: true
  resume_timestamp: "20260320_063225"
```

当 `resume=true` 且 `resume_timestamp` 非空时，启动器会把输出目录切回旧 run 目录。

如果不续训：

```yaml
training:
  resume: false
```

则会生成新的：

```text
outputs/train/${task}/${method}/run_${timestamp}
```

## 7. 如何修改训练逻辑

### 想改训练细节

请优先修改：

```text
configs/policy/lingbot/robotwin_load20000h.yaml
```

例如：

- 学习率
- epoch
- save_steps
- FSDP / offload
- tokenizer 长度
- 其它 `train.*` / `data.*` 默认行为

### 想改入口层行为

请修改：

```text
configs/policy/lingbot_config.yaml
```

例如：

- 数据根目录
- 模型路径
- tokenizer 路径
- 显卡
- 是否 dry run
- 是否 resume
- 输出目录模板

## 8. 常见问题

### 训练启动时报 `No module named kuavo_train.wrapper`

说明入口脚本的模块路径没处理好。当前仓库已经修过这个问题，按本文命令直接运行即可。

### 训练日志里出现 pydantic 的 `UnsupportedFieldAttributeWarning`

这通常只是警告噪声，不是训练失败原因。只要训练继续进入：

- `torchrun`
- `Prepare model`
- `Prepare data`
- `Start training`

就可以先忽略。

### 单卡显存不够

LingBot-VLA 体量较大，单卡容易在首个 optimizer step OOM。更稳的做法是：

- `batch_size: 1`
- 使用多卡
- 优先让 `CUDA_VISIBLE_DEVICES` 和实际 `nproc-per-node` 一致

## 9. 推荐使用习惯

- 把真正的 LingBot 训练参数放在 `robotwin_load20000h.yaml`
- 把外层路径、显卡、resume、输出目录放在 `lingbot_config.yaml`
- 改完后先跑一次 `policy.dry_run=true`
- 确认生成的 `torchrun` 命令正确，再开始正式训练
