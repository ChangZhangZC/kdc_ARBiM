## 2026-08-30 22:44 CEST：冻结前五项 `train_ddp.py` 修正

- 任务目标：按已确认方案补齐 Critic shared encoder checkpoint、runner 参数兼容、PPO 的 DDP 辅助行为，并明确 ARBiM V1 Dynamics 范围。
- 涉及文件：`RL Post Training/train/train_ddp.py`；`AI logs.md`。
- 改动结论：`is_share_encoder=True` 时 Critic checkpoint 额外保存/加载 `encoder.pt`，加载后切换为 eval；`eval()` 与 `unio4_eval()` 按 runner 签名过滤 `eval_env_num` 和 IDQL 参数；PPO 仅 rank 0 创建 tqdm，周期 OPE 记录 `current_mean_qs` 到 W&B，且仅将输出限制在 rank 0，不改变所有 rank 的 old-policy 更新；Diffusion Dynamics 改为明确的 V1 out-of-scope 错误信息。
- 确认状态：用户本轮明确回复“按此执行”。
- 静态检查结果：GitNexus MCP 不可用，已只读检查 `.gitnexus/meta.json` 并降级使用 `rg`、源码及 RL-100 对照。确认 `IQLCritic.save/load` 均支持 `encoder_path`；普通 eval 不会提前访问 `idql_run`，仅在 `idql_eval=True` 时检查其签名；检查函数锚点、参数名和缩进。未运行训练、推理、测试或 Python 语法命令。
- 遗留风险：实际 env runner 的可选参数和 `idql_run` 接口仅能在获准运行环境验证；当前仓库未检索到 runner 实现，故静态检查无法确认具体签名。

## 2026-08-30 22:38 CEST：对齐 Offline PPO 记录与 Critic/Dynamics DDP 行为

- 任务目标：依据已确认方案，修复 `train_ddp.py` 的重复初始评估、未定义 OPE 变量与 OPE/scores 写入时机，并恢复 RL-100 对 `fix_encoder=False` 的 Critic/Dynamics DDP 行为。
- 涉及文件：`RL Post Training/train/train_ddp.py`；本日志文件。
- 改动结论：移除了 Critic/Dynamics 中额外的 DDP `fix_encoder=True` 限制；Dynamics 在非 rank 0 构造和 DDP 包装后均会在 `fix_encoder=False` 时将 ACT observation encoder 参数加入 optimizer；初始 normal eval 保留一次，删除未定义的 `current_mean_q` 追加与初始 OPE CSV 写入，恢复周期 OPE 和每步 scores 的 CSV 写入位置。既定的 Critic/Dynamics epoch 数及 artifact/checkpoint 路径未改动。
- 确认状态：用户在本轮明确要求依据引用对话最后一次修改意见直接实施。
- 静态检查结果：因 GitNexus MCP 不可用，已只读检查 `.gitnexus/meta.json` 并降级使用 `rg`/源码对照；逐项核对 RL-100 `train_ddp.py::finetune_dp3()` 的 OPE/scores 写入缩进及 Dynamics optimizer 参数组合，确认 `ACTObservationAdapter.encoder` 存在且在 `fix_encoder=False` 时允许梯度。未运行训练、推理、测试或 Python 语法命令。`git diff --check --no-index` 报告的是该未纳入 Git 的脚本既有多处尾随空白，未在本轮清理以避免超出范围。
- 遗留风险：共享 encoder 按 RL-100 原逻辑不单独 DDP 包装；多卡实际同步行为尚未运行验证，需在后续获授权的运行环境中验证。
