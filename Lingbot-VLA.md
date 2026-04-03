# Lingbot 环境配置
1. 安装好kdc环境
2. 参考https://github.com/Robbyant/lingbot-vla的README.md，下载lingbot-vla-4b，Qwen2.5-VL-3B-Instruct模型
3. 在kdc环境中，使用提供的lingbot-vla仓库，运行pip install -e .和pip install -r requirements.txt
4. 在kdc环境中安装flash-attention，pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
5. 进入到kdc的third_party/lerobot文件夹，pip install -e .安装lerobot
6. 边侧机使用1.3.3版本的kdc，pip install kuavo-humannoid-sdk==1.3.3，下位机使用了1.3.3版本的kuavo-ros-control，可以通过git reset --hard 1.3.3然后使用catkin build humanoid_cotrollers编译，需要修改src/kuavo_assets/config/kuavo_v49/kuavo.json把qiangnao替换成lejuclaw，否则无法监听到/leju_claw_state

# LingBot 训练说明

本文说明如何在当前仓库中启动 LingBot 训练，以及两层配置文件的实际关系。

## 1. 数据处理

将rosbag转换为Lerobot数据集：

```bash
python kuavo_data/CvtRosbag2Lerobot.py --config-path=../configs/data/ --config-name=KuavoRosbag2Lerobot.yaml rosbag.rosbag_dir=/home/lmy/lmy_ws/Leju/kuavo_data_challenge/raw_data rosbag.lerobot_dir=/home/lmy/lmy_ws/Leju/pick_and_place
```

根据数据集转换生成norm_stats配置文件用于训练和推理：

`  python kuavo_train/convert_stats_to_norm_stats.py \
    /home/lmy/lmy_ws/pick_and_place/pick_and_place0_4_2/meta/stats.json`


## 1. 训练入口

当前训练入口命令：

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

lingbot_config.yaml需要修改：

1. `root: /data/limingyang/VLA/huggingface/pick_and_place/pick_and_place0_4_2`

2. `model_path: /home/yunxi/lmy/VLA/lingbot-vla-4b`

3. `tokenizer_path: /home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct`

robotwin_load20000h.yaml需要修改：

1. `norm_stats_file: assets/norm_stats/pick_and_place0_4_2.json`   需要和使用的数据集匹配


Tensorboard可视化：

```bash
tensorboard --logdir outputs/train/lingbot_task/lingbot_post_train --port 6007
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
- `--train.global_batch_size = micro_batch_size * GPU数`

例如 8 卡时，会变成：

```text
micro_batch_size=1
global_batch_size=8
```

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

## 8. 推荐使用习惯

- 把真正的 LingBot 训练参数放在 `robotwin_load20000h.yaml`
- 把外层路径、显卡、resume、输出目录放在 `lingbot_config.yaml`
- 改完后先跑一次 `policy.dry_run=true`
- 确认生成的 `torchrun` 命令正确，再开始正式训练

## 9. 真机部署与测试

### 9.1 前提假设

默认你已经完成：

- LingBot 模型训练
- `configs/deploy/kuavo_env.yaml` 已改成当前数据集对应的部署配置
- 当前使用环境为 `kdc_dev`
- 真机侧已具备 `kuavo-ros-opensource` 运行条件

当前部署特征对应：

- `observation.images.head_cam_h`
- `observation.images.wrist_cam_r`
- `observation.state`
- `action`

也就是“头部相机 + 右腕相机 + 单右臂状态/动作”的配置。

### 9.2 机器人与网络准备

连接 `Kuavo-manipulation` 网络后，可通过 `http://192.168.5.1/` 查看当前设备 IP。确保上下位机和边侧设备处于同一网段并且可以互相通信。

连接机器人下位机：

```bash
ssh lab@192.168.5.117
```

进入控制仓库后启动运动控制节点：

```bash
cd kuavo-ros-control
sudo su
roslaunch humanoid_controllers load_kuavo_real.launch
```

连接机器人上位机：

```bash
ssh leju_kuavo@192.168.5.111
```

启动相机：

```bash
sudo systemctl start start_camera.service
sudo systemctl restart start_camera.service
```

如果相机起不来，优先检查上位机的 `/etc/kuavo.conf` 里的 `ROS_MASTER_URI` 是否指向正确的远程 ROS Master，注意不要误改 `ROS_IP`。

如果边侧机运行部署脚本时一直提示话题为空，但三台机器之间网络互通，需要检查边侧机 `/etc/hosts` 中是否补了 `kuavo_master` 映射，例如：

```text
127.0.0.1 localhost
127.0.1.1 myl
192.168.5.165 kuavo_master
```
注意如果出现消息不通的情况，要检查对应设备上的~/.bashrc和/etc/hosts

如果边侧机消息定义缺失，也可以直接从下位机拷贝 `kuavo-ros-control`。

### 9.3 上机前检查

建议先确认：

- 当前环境是 `kdc_dev`
- `configs/deploy/kuavo_env.yaml` 中 `policy_type: lingbot`
- `configs/deploy/kuavo_env.yaml` 中 `go_bag_path` 已改为真实绝对路径
- `configs/deploy/kuavo_env.yaml` 中 `pretrained_path` 指向真实存在的 LingBot `hf_ckpt`
- 真机 ROS 话题和部署配置一致

可以先检查模型目录：

```bash
ls /home/yunxi/lmy/VLA/kuavo_data_challenge/outputs/train/lingbot_task/lingbot_post_train/run_20260314_061741/checkpoints/global_step_15300/hf_ckpt
```

### 9.4 检查 ROS 观测链路

在新终端执行：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
rostopic hz /cam_h/color/image_raw/compressed
rostopic hz /cam_r/color/image_raw/compressed
rostopic echo /sensors_data_raw
rostopic echo /leju_claw_state
```

至少确认以下四类观测存在：

- 头部 RGB
- 右腕 RGB
- 关节状态
- 夹爪状态

如果这些话题和 `configs/deploy/kuavo_env.yaml` 不一致，先改配置，再继续部署。

### 9.5 真机测试顺序

建议严格按下面顺序进行：

1. 启动真机底层
2. 检查 4 个关键话题
3. 测 `go`
4. 测 `run`
5. 最后测 `go_run`

不要一开始直接跑 `go_run`。

#### 第一阶段：只测试 `go.bag`

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task go --config configs/deploy/kuavo_env.lmy_go_run.yaml
```
kuavo_env.lmy_go_run.yaml需要修改的地方：
1. rosbag路径：

`go_bag_path: /home/lmy/lmy_ws/Leju/kuavo_data_challenge_dev/raw_data/A01-A02-A02-A02-A02-A03-P4_000-leju_claw-20260309165457-v002.bag`

2. 预训练模型路径：

`pretrained_path: "/home/lmy/lmy_ws/Leju/kuavo_data_challenge_master/models/run_20260323_060215_7005/run_20260323_060215/checkpoints/global_step_7005/hf_ckpt"`

3. lingbot-vla路径：

`lingbot_root: "/home/lmy/lmy_ws/Leju/lingbot-vla"`

4. 推理的chunk:

`lingbot_use_length: 5`

5. norm_stats文件：

`lingbot_norm_stats_file: "/home/lmy/lmy_ws/Leju/kuavo_data_challenge_master/assets/norm_stats/pick_and_place0_4_2.json"`

6. 如果lingbot_use_length大于1，lingbot_chunk_ret需要设置为false

7. Qwen2.5路径：

`qwen25_path: "/home/lmy/lmy_ws/Leju/Qwen2.5-VL-3B-Instruct"`

通过这一步说明：

- `go_bag_path` 可读
- 轨迹回放链路正常
- 当前右臂和夹爪控制映射没有明显错误
- 起始位至少可达

如果这一步动作不安全，不要继续跑模型。

#### 第二阶段：从当前位置直接跑模型

在终端给QWEN2.5的路径：

`export QWEN25_PATH=/home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct`

将机器人手动放到安全起始位，然后执行：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task run --config configs/deploy/kuavo_env.lmy_go_run.yaml
```

这一步主要验证：

- LingBot checkpoint 能否加载
- 头部 + 右腕 + 8 维 state 能否正常进入模型
- 模型输出动作能否成功下发给真机
- 动作方向是否基本合理

真机第一次测试建议：

- `eval_episodes=1`
- 旁边有人值守
- 随时准备物理急停
- 只观察前几步动作是否合理

#### 第三阶段：完整 `go_run`

前两步都通过后，再执行：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task go_run --config configs/deploy/kuavo_env.lmy_go_run.yaml
```

该流程会先按 `go.bag` 到起始姿态，再启动 LingBot 推理。

### 9.6 暂停、停止与回零

头部控制可以通过发布：

```bash
rostopic pub /robot_head_motion_data kuavo_msgs/robotHeadMotionData "joint_data: [0.0, 27.0]" --once
```

`script.py` 启动后会打印 PID，可用下面方式控制：

暂停或恢复：

```bash
kill -USR1 <pid>
```

停止：

```bash
kill -USR2 <pid>
```

如果停止后需要回安全位：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task back_to_zero --config configs/deploy/kuavo_env.lmy_go_run.yaml
```

建议在第一次真机测试前，先单独验证一次 `back_to_zero`。

### 9.7 常见问题

- `机械臂初始化失败`
  - 优先检查 `kuavo-humanoid-sdk` 版本是否和机器人侧一致。

- `LingBot import 失败`
  - 确认当前环境为 `kdc_dev`，并且该环境里能正常导入 LingBot 相关模块。

- `缺少 wrist_cam_l`
  - 当前仓库已兼容只有右腕图的部署场景。

- 模型能启动但动作异常
  - 优先检查：
  - `which_arm: right`
  - `obs_key_map` 是否对应真机实际话题
  - `pretrained_path` 是否是正确的单右臂 LingBot checkpoint
  - `go.bag` 是否适配当前机器人姿态

- `go.bag` 回放正常，但 `run` 异常
  - 说明轨迹链路没问题，重点排查模型输入、checkpoint 和观测维度。

### 9.8 相关文件

- 部署配置：`configs/deploy/kuavo_env.yaml`
- 真机脚本：`kuavo_deploy/src/scripts/script.py`
- 真机推理入口：`kuavo_deploy/src/eval/real_single_test.py`
- LingBot 部署适配器：`kuavo_deploy/utils/lingbot_adapter.py`

