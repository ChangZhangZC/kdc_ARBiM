# LingBot 真机测试流程

## 机器人启动流程

连接Kuavo-manipulation网络，密码：manipulation
通过http://192.168.5.1/可以登陆后台界面查看连接设备的IP

确保上下位机和边侧设备之间在同一个网段可以互相通信

连接机器人下位机： ssh lab@192.168.5.117   密码：三个空格

`cd kuavo-ros-control`

`sudo su`

`roslaunch humanoid_controllers load_kuavo_real.launch`   启动运动控制节点

连接机器人上位机： ssh leju_kuavo@192.168.5.111   密码：leju_kuavo

本地4090电脑，需要和上位机配置好主从

上位机启动相机

`sudo systemctl start start_camera.service`

`sudo systemctl restart start_camera.service`

如果发现相机启动不起来，可能是主机的ROS_MASTER_URI的问题，需要修改/etc/kuavo.conf，将 ROS_MASTER_URI 从指向本机改为指向远程 ROS Master 所在的机器。不要修改到ROS_IP

如果在边侧机启动python kuavo_deploy/src/scripts/script.py推理不了，一直报错话题为空等待，而上位机，下位机和边侧机之间能相互ping通，这时候需要修改边侧机的/etc/hosts，增加kuavo_master
```
127.0.0.1	localhost
127.0.1.1	myl
192.168.5.165   kuavo_master
```
边侧机消息的却缺失可以直接从下位机上拷贝kuavo-ros-control

头部控制：

下位机随便一个仓库（我自己是kuavo-ros-control）source后：

`rostopic pub /robot_head_motion_data kuavo_msgs/robotHeadMotionData "joint_data: [0.0, 27.0]" --once`

- 恢复零位

`python kuavo_deploy/src/scripts/script.py --task back_to_zero --config configs/deploy/kuavo_env.lmy_go_run.yaml`

- 重播bag包测试

`python kuavo_deploy/src/scripts/script.py --task go --config configs/deploy/kuavo_env.lmy_go_run.yaml`

- 模型推理测试
`python kuavo_deploy/src/scripts/script.py --task run --config configs/deploy/kuavo_env.lmy_go_run.yaml`

---

本文档针对当前仓库中基于 LingBot 的真机部署测试，假设你已经完成：

- LingBot 模型训练；
- `configs/deploy/kuavo_env.yaml` 已按当前数据集改成单右臂部署配置；
- 使用环境为 `kdc_dev`；
- 真机侧已具备 `kuavo-ros-opensource` 运行条件。

当前配置对应的数据特征为：

- `observation.images.head_cam_h`
- `observation.images.wrist_cam_r`
- `observation.state`
- `action`

也就是“头部相机 + 右腕相机 + 右臂状态/动作”的单臂任务。

## 1. 上机前检查

先确认下面几项已经就绪：

- 当前环境为 `kdc_dev`
- `configs/deploy/kuavo_env.yaml` 中 `policy_type` 为 `lingbot`
- `configs/deploy/kuavo_env.yaml` 中 `go_bag_path` 已改成真实绝对路径
- `configs/deploy/kuavo_env.yaml` 中 `pretrained_path` 指向真实存在的 LingBot `hf_ckpt`
- 真机 ROS 话题与配置一致

建议先检查模型路径：

```bash
ls /home/yunxi/lmy/VLA/kuavo_data_challenge/outputs/train/lingbot_task/lingbot_post_train/run_20260314_061741/checkpoints/global_step_15300/hf_ckpt
```

## 2. 启动真机底层

在部署终端中执行：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
roslaunch humanoid_controllers load_kuavo_real.launch
```

如果是首次开机并需要校准：

```bash
roslaunch humanoid_controllers load_kuavo_real.launch cali:=true
```

## 3. 检查 ROS 观测链路

在新终端中执行：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
rostopic hz /cam_h/color/image_raw/compressed
rostopic hz /cam_r/color/image_raw/compressed
rostopic echo /sensors_data_raw
rostopic echo /leju_claw_state
```

至少要确认：

- 头部 RGB 有数据；
- 右腕 RGB 有数据；
- 关节状态有数据；
- 夹爪状态有数据。

如果这些话题和 `configs/deploy/kuavo_env.yaml` 不一致，先改配置再继续。

## 4. 第一阶段测试：只测试 `go.bag`

先不要跑模型，只测到达起始位：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task go \
  --config configs/deploy/kuavo_env.yaml
```

这一步通过说明：

- `go_bag_path` 可读；
- 轨迹回放链路正常；
- 当前右臂和夹爪控制映射没有明显错误；
- 起始位至少是可达的。

如果这一步动作不安全，不要继续跑模型。

## 5. 第二阶段测试：从当前位置直接跑模型

将机器人手动放到安全起始位，先从当前位置直接推理，不走 `go.bag`：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_env.yaml
```

这一步主要验证：

- LingBot checkpoint 能否加载；
- 头部 + 右腕 + 8 维 state 能否正常进入模型；
- 模型输出动作能否下发给真机；
- 动作方向是否基本合理。

真机第一次测试建议：

- `eval_episodes=1`
- 旁边有人值守
- 随时准备物理急停
- 只看前几步动作是否合理

## 6. 第三阶段测试：完整 `go_run`

当前两步都通过后，再执行完整测试：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task go_run \
  --config configs/deploy/kuavo_env.yaml
```

它会执行：

1. 先按 `go.bag` 到任务起始姿态
2. 然后启动 LingBot 推理

这是最接近正式部署的流程。

## 7. 暂停、停止与回零

`script.py` 启动后会打印当前进程 PID。

暂停 / 恢复：

```bash
kill -USR1 <pid>
```

停止：

```bash
kill -USR2 <pid>
```

如果停止后需要回到安全位：

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task back_to_zero \
  --config configs/deploy/kuavo_env.yaml
```

建议在第一次真机测试前，单独验证一次 `back_to_zero`。

## 8. 推荐测试顺序

建议严格按下面顺序执行：

1. 启动真机底层
2. 检查 4 个关键话题
3. 测 `go`
4. 测 `run`
5. 最后测 `go_run`

不要一开始就直接上 `go_run`。

## 9. 常见问题

- `机械臂初始化失败`
  - 优先检查 `kuavo-humanoid-sdk` 版本是否和机器人侧一致。

- `LingBot import 失败`
  - 确认当前环境为 `kdc_dev`，并且该环境里能正常导入 LingBot 相关模块。

- `缺少 wrist_cam_l`
  - 当前仓库已修改 `lingbot_adapter.py`，允许在只有右腕图的情况下部署。

- 模型能启动但动作异常
  - 优先检查：
    - `which_arm: right`
    - `obs_key_map` 是否对应真机实际话题
    - `pretrained_path` 是否是正确的单右臂 LingBot checkpoint
    - `go.bag` 是否适配当前机器人姿态

- `go.bag` 回放正常，但 `run` 异常
  - 说明轨迹链路没问题，重点排查模型输入、checkpoint 和观测维度。

## 10. 关键文件

- 部署配置：[configs/deploy/kuavo_env.yaml](/home/yunxi/lmy/VLA/kuavo_data_challenge/configs/deploy/kuavo_env.yaml)
- 真机脚本：[kuavo_deploy/src/scripts/script.py](/home/yunxi/lmy/VLA/kuavo_data_challenge/kuavo_deploy/src/scripts/script.py)
- 真机推理入口：[kuavo_deploy/src/eval/real_single_test.py](/home/yunxi/lmy/VLA/kuavo_data_challenge/kuavo_deploy/src/eval/real_single_test.py)
- LingBot 部署适配器：[kuavo_deploy/utils/lingbot_adapter.py](/home/yunxi/lmy/VLA/kuavo_data_challenge/kuavo_deploy/utils/lingbot_adapter.py)
