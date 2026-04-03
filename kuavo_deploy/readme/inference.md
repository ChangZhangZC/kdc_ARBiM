# 🤖 Kuavo机器人控制示例

> 基于ROS的Kuavo机器人控制示例程序，支持机械臂运动控制、轨迹回放和模型推理等功能。

## 📁 文件结构

```
kuavo_deploy/examples/
├── eval/               # 评估脚本
│   ├── eval_kuavo.py   # Kuavo环境评估脚本
│   └── auto_test/      # 自动化测试
│       ├── eval_kuavo.py           # Kuavo环境自动化评估脚本
│       └── eval_kuavo_autotest.py  # 自动测试脚本
└── scripts/            # 控制脚本
    ├── script.py       # 主要控制脚本
    ├── controller.py   # 远程控制指令发送器
    └── script_auto_test.py  # 自动化控制脚本
```

## 🎯 控制系统概览

Kuavo机器人控制系统包含以下核心组件：

1. **`script.py`** - 主要控制脚本，执行具体的机器人任务
2. **`controller.py`** - 远程控制器，用于向运行中的任务发送控制指令
3. **`eval_kuavo.py`** - 评估脚本，用于模型推理和性能评估
4. **`script_auto_test.py`** - 自动化控制脚本，用于批量测试

## 🚀 快速开始

### 方法一：使用交互式脚本 eval_kuavo.sh

启动交互式控制界面
```bash
bash kuavo_deploy/eval_kuavo.sh
```
命令行弹出提示：
```bash
=== Kuavo机器人控制示例 ===
此脚本展示如何使用命令行参数控制不同的任务
-e 支持暂停、继续、停止功能

📋 控制功能说明:
  🔄 暂停/恢复: 发送 SIGUSR1 信号 (kill -USR1 <PID>)
  ⏹️  停止任务: 发送 SIGUSR2 信号 (kill -USR2 <PID>)
  📊 查看日志: tail -f log/kuavo_deploy/kuavo_deploy.log

kuavo_deploy/eval_kuavo.sh: 16: Bad substitution
1. 显示帮助信息:
python kuavo_deploy/examples/scripts/script.py --help

2. 干运行模式 - 查看将要执行的操作:
python kuavo_deploy/examples/scripts/script.py --task go --dry_run --config /path/to/custom_config.yaml

3. 到达工作位置:
python kuavo_deploy/examples/scripts/script.py --task go --config /path/to/custom_config.yaml

4. 从当前位置直接运行模型:
python kuavo_deploy/examples/scripts/script.py --task run --config /path/to/custom_config.yaml

5. 插值至bag的最后一帧状态开始运行:
python kuavo_deploy/examples/scripts/script.py --task go_run --config /path/to/custom_config.yaml

6. 从go_bag的最后一帧状态开始运行:
python kuavo_deploy/examples/scripts/script.py --task here_run --config /path/to/custom_config.yaml

7. 回到零位:
python kuavo_deploy/examples/scripts/script.py --task back_to_zero --config /path/to/custom_config.yaml

8. 仿真中自动测试模型，执行eval_episodes次:
python kuavo_deploy/examples/scripts/script_auto_test.py --task auto_test --config /path/to/custom_config.yaml

9. 启用详细输出:
python kuavo_deploy/examples/scripts/script.py --task go --verbose --config /path/to/custom_config.yaml

=== 任务说明 ===
go          - 先插值到bag第一帧的位置，再回放bag包前往工作位置
run         - 从当前位置直接运行模型
go_run      - 到达工作位置直接运行模型
here_run    - 插值至bag的最后一帧状态开始运行
back_to_zero - 中断模型推理后，倒放bag包回到0位
auto_test   - 仿真中自动测试模型，执行eval_episodes次

请选择要执行的示例: 1. 显示普通测试帮助信息 2. 显示自动测试帮助信息 3. 进一步选择示例
1. 执行: python kuavo_deploy/examples/scripts/script.py --help
2. 执行: python kuavo_deploy/examples/scripts/script_auto_test.py --help
3. 进一步选择示例
请选择要执行的示例 (1-3) 或按 Enter 退出:
```

在命令行输入3，按 Enter ，弹出提示
```bash
请输入自定义配置文件路径:
```

输入自定义配置文件路径，默认配置文件参考`configs/deploy/kuavo_sim_env.yaml`，弹出提示
```bash
📁 配置文件路径: configs/deploy/kuavo_sim_env.yaml
🔍 正在解析配置文件...
📋 模型配置信息:
   Task: your_task
   Method: your_methof
   Timestamp: your_timestamp
   Epoch: 300
📂 完整模型路径: your_path
✅ 模型路径存在
可选择要执行的示例如下:
1. 先插值到bag第一帧的位置，再回放bag包前往工作位置(干运行模式)
执行: python kuavo_deploy/examples/scripts/script.py --task go --dry_run --config /path/to/config.yaml
2. 先插值到bag第一帧的位置，再回放bag包前往工作位置
执行: python kuavo_deploy/examples/scripts/script.py --task go --config /path/to/config.yaml
3. 从当前位置直接运行模型
执行: python kuavo_deploy/examples/scripts/script.py --task run --config /path/to/config.yaml
4. 到达工作位置并直接运行模型
执行: python kuavo_deploy/examples/scripts/script.py --task go_run --config /path/to/config.yaml
5. 插值至bag的最后一帧状态开始运行
执行: python kuavo_deploy/examples/scripts/script.py --task here_run --config /path/to/config.yaml
6. 回到零位
执行: python kuavo_deploy/examples/scripts/script.py --task back_to_zero --config /path/to/config.yaml
7. 先插值到bag第一帧的位置，再回放bag包前往工作位置(启用详细输出)
执行: python kuavo_deploy/examples/scripts/script.py --task go --verbose --config /path/to/config.yaml
8. 仿真中自动测试模型，执行eval_episodes次
执行: python kuavo_deploy/examples/scripts/script_auto_test.py --task auto_test --config /path/to/config.yaml
9. 退出
请选择要执行的示例 (1-9)
```

选择需要的功能，一般选择8在仿真中进行自动化测试

交互式脚本提供以下功能：
- 📋 显示所有可用命令示例
- 🎮 交互式任务选择
- 🔄 实时任务控制（暂停/恢复/停止）
- 📊 实时日志查看

⚠️ 注意：如需使用仿真环境中的自动化测试，先在本机roscore，再启动仿真环境kuavo-ros-opensource的自动化测试脚本，最后启动本脚本


#### 📋 支持的任务类型

| 任务 | 描述 | 使用场景 |
|------|------|----------|
| `go` | 先插值到bag第一帧位置，再回放bag包前往工作位置 | 准备阶段 |
| `run` | 从当前位置直接运行模型 | 快速测试 |
| `go_run` | 到达工作位置直接运行模型 | 完整流程 |
| `here_run` | 插值至bag的最后一帧状态开始运行 | 连续推理 |
| `back_to_zero` | 中断模型推理后，倒放bag包回到0位 | 安全回退 |
| `auto_test` | 仿真环境中自动执行多次测试，评估模型性能 | 批量测试 | 

### 方法二：直接运行python脚本

#### 1. 查看帮助信息
```bash
python kuavo_deploy/examples/scripts/script.py --help
```

#### 2. 基本任务执行
```bash
# 先插值到bag第一帧位置，再回放bag包前往工作位置
python kuavo_deploy/examples/scripts/script.py --task go --config /path/to/config.yaml

# 从当前位置直接运行模型
python kuavo_deploy/examples/scripts/script.py --task run --config /path/to/config.yaml

# 到达工作位置并直接运行模型
python kuavo_deploy/examples/scripts/script.py --task go_run --config /path/to/config.yaml

# 插值至bag的最后一帧状态开始运行
python kuavo_deploy/examples/scripts/script.py --task here_run --config /path/to/config.yaml

# 回到零位
python kuavo_deploy/examples/scripts/script.py --task back_to_zero --config /path/to/config.yaml

# 执行自动化测试（仿真环境）
python kuavo_deploy/examples/scripts/script_auto_test.py --task auto_test --config /path/to/config.yaml
```

#### 3. 当任务运行时，您可以使用 controller.py 进行远程控制：

`controller.py` 提供了更友好的远程控制接口：

```bash
# 基本用法
python kuavo_deploy/examples/scripts/controller.py <command>

# 可用命令
python kuavo_deploy/examples/scripts/controller.py pause    # 暂停任务
python kuavo_deploy/examples/scripts/controller.py resume   # 恢复任务  
python kuavo_deploy/examples/scripts/controller.py stop     # 停止任务
python kuavo_deploy/examples/scripts/controller.py status   # 查看任务状态

# 指定特定进程
python kuavo_deploy/examples/scripts/controller.py pause --pid 12345
```

##### controller.py 功能特点：

- 🔍 **自动进程发现**：自动查找运行中的 script.py 进程
- 🎯 **精确控制**：支持指定特定进程ID进行控制
- 📊 **状态监控**：显示进程详细信息（CPU、内存、运行时间等）
- 🛡️ **安全验证**：验证目标进程是否为有效的 script.py 进程

#### 4. 命令行参数

###### script.py 参数

###### 必需参数
- `--task` : 任务类型 (`go`, `run`, `go_run`, `here_run`, `back_to_zero`)
- `--config` : 配置文件路径

###### 可选参数
- `--verbose, -v` : 启用详细输出
- `--dry_run` : 干运行模式（仅显示操作，不实际执行）

##### script_auto_test.py 参数

###### 必需参数
- `--task` : 任务类型 (`auto_test`)
- `--config` : 配置文件路径

###### 可选参数
- `--verbose, -v` : 启用详细输出
- `--dry_run` : 干运行模式（仅显示操作，不实际执行）

##### controller.py 参数

###### 必需参数
- `command` : 控制指令 (`pause`, `resume`, `stop`, `status`)

###### 可选参数
- `--pid` : 指定进程PID（如果不指定，将自动查找）

## ⚙️ 配置文件

默认配置文件：`configs/deploy/kuavo_sim_env.yaml`

### 关键配置项

```yaml
# 1. 环境配置（与 configs/deploy/kuavo_sim_env.yaml 对齐）
real: false                   # 是否使用真实机器人
only_arm: true                # 是否只使用手臂数据
eef_type: rq2f85              # 末端执行器类型: qiangnao, leju_claw, rq2f85
control_mode: joint           # 关节控制或笛卡尔控制: joint / eef
which_arm: both               # 使用的手臂: left, right, both
head_init: [0, 0.209]         # 头初始角度
input_images: ["head_cam_h", "wrist_cam_r", "wrist_cam_l", "depth_h", "depth_r", "depth_l"]
image_size: [480, 640]        # 图像大小
ros_rate: 10                  # 推理频率(Hz)

# 高级配置（不建议修改）
qiangnao_dof_needed: 1        # 强脑手自由度：1=简单开合
leju_claw_dof_needed: 1       # 夹爪自由度
rq2f85_dof_needed: 1          # rq2f85 自由度
arm_init: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
arm_min: [-180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180]
arm_max: [ 180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180]
eef_min: [0]
eef_max: [1]
is_binary: false

# 2. 推理配置
go_bag_path: /path/to/your/bag/file.bag  # rosbag 路径

policy_type: "diffusion"
use_delta: false
eval_episodes: 1
seed: 42
start_seed: 42
device: "cuda"  # or "cpu"

# 模型路径: outputs/train/{task}/{method}/{timestamp}/epoch{epoch}
task: "ruichen"                  # ← 按你的训练任务替换
method: "test_git_model"         # ← 按你的训练方法替换
timestamp: "run_20250819_115313" # ← 按你的时间戳替换
epoch: 29                         # ← 按你的 epoch 替换

max_episode_steps: 500
env_name: Kuavo-Real
```

### 末端执行器配置

| 类型 | 说明 | 自由度 | 控制模式 |
|------|------|--------|----------|
| `qiangnao` | 强脑灵巧手 | 1个自由度 | 简单开合控制 |
| `leju_claw` | 夹爪 | 1个自由度 | 夹紧/张开控制 |

## 🔧 环境要求

- ✅ ROS环境已配置
- ✅ 机器人硬件连接正常
- ✅ 配置文件路径正确
- ✅ 模型文件完整
- ✅ Python依赖包已安装

## 🐛 故障排除

## LingBot 部署补充

当 `configs/deploy/kuavo_env.yaml` 中设置 `inference.policy_type=lingbot` 时：

- 可用 `inference.pretrained_path` 直接指定 LingBot 导出的 `hf_ckpt` 路径
- 可用 `inference.task_prompt` 传递文本任务描述
- 可用 `inference.lingbot_norm_stats_file` 指定自定义归一化统计文件
- 适配代码入口在 `kuavo_deploy/utils/lingbot_adapter.py`

真实机器人部署示例说明见仓库根目录 `README_LINGBOT_REAL_TEST.md`。

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 配置文件不存在 | 检查配置文件路径是否正确 |
| 机械臂初始化失败 | 检查ROS环境和硬件连接 |
| 模型路径不存在 | 确认配置文件中的模型路径 |
| controller.py找不到进程 | 确保script.py正在运行，或使用--pid指定 |
| 权限不足 | 使用sudo或检查进程权限 |
| 自动化测试失败率高 | 检查模型训练质量，调整 `eval_episodes` 参数 |

### 调试技巧

1. **测试优先**：首次使用建议先使用 `--dry_run` 模式
2. **硬件检查**：确保机器人硬件状态正常
3. **进程监控**：使用 `python kuavo_deploy/examples/scripts/controller.py status` 查看任务状态
4. **日志分析**：查看 `log/kuavo_deploy/kuavo_deploy.log` 获取详细信息

## 📝 日志系统

- `log_model` : 网络/模型相关日志
- `log_robot` : 机器人控制相关日志

日志文件位置：`log/kuavo_deploy/kuavo_deploy.log`

## ⚠️ 安全注意事项

1. **测试优先**：首次使用建议先使用 `--dry_run` 模式
2. **硬件检查**：确保机器人硬件状态正常
3. **紧急停止**：支持 `Ctrl+C` 中断操作和 `kill -USR2` 信号停止
4. **配置验证**：确认配置文件中的路径和参数正确
5. **权限管理**：确保有足够的权限控制目标进程
6. **进程监控**：定期检查任务状态，确保正常运行

## 🔄 扩展开发

如需添加新任务类型：

1. 在 `ArmMove` 类中添加新方法
2. 在 `parse_args()` 中添加新选项
3. 在 `task_map` 中添加新映射
4. 更新文档和示例

如需扩展控制功能：

1. 在 `controller.py` 中添加新的控制指令
2. 在 `script.py` 中添加对应的信号处理
3. 更新帮助文档和示例

## 📚 最佳实践

### 推荐工作流程

1. **配置验证** → 使用 `--dry_run` 测试配置
2. **任务启动** → 运行 `run_example.sh`
3. **日志分析** → 查看日志文件进行问题诊断
4. **安全退出** → 执行 `back_to_zero` 任务安全回退

### 性能优化建议

- 使用 `--verbose` 模式进行调试，生产环境可关闭
- 合理设置 `ros_rate` 参数平衡性能和稳定性
- 定期清理日志文件避免磁盘空间不足
- 使用 `kuavo_deploy/examples/scripts/controller.py` 进行精确控制，避免直接kill进程
- 自动化测试建议设置合理的 `eval_episodes` 数量，使用仿真环境进行验证

---
