# AI 工作日志

## 2026-08-21｜对齐 LeRobot NPY 与 RL Transition 拼装

- 日期时间：2026-08-21 20:09 CEST。
- 任务目标：将 `process_raw_teleop_to_npy` 的多模态 `.npy` 输出与 `append_processed_transitions` 的 RL buffer 输入契约对齐，保留 RL-100 单步 transition 主逻辑并移除点云依赖。
- 涉及文件：`RL Post Training/data/data_prepare.py`、`AI logs.md`。
- 代码改动：新增 RGB/depth feature 到 buffer 的共享映射和 processed-data 契约校验；补充 `process_raw_teleop_to_npy` 的三路 RGB、可选三路 depth、字段等长及 `done == timeout` 校验；迁移并适配 `make_buffers(use_depth=False)`、`load_processed_npy`、`_push_episode_end`、`append_processed_transitions`；终止帧的 state、action、RGB 和可选 depth 的 `next_*` 均使用当前帧自环，避免跨 episode。
- 方案确认状态：用户已确认方案；GitNexus 对 RL-100 原实现分析显示 `make_buffers` 与 `_push_episode_end` 为高风险共享函数，用户在获知 build/extend/rollout 下游影响后再次明确“确认执行”，并授权使用 LunaWorks 子 Agent 并行实施与审查。
- 静态检查结果：已核对 Kuavo converter 的三路 RGB 与三路 depth feature 名称、producer/consumer 字段一致性、深度开关、episode 边界、buffer 字段、函数签名和目标文件的 point-cloud 残留；`git diff --no-index --check` 未发现空白错误；双重只读复审未发现阻塞性缺陷。遵守项目规定，未运行 Python、测试、训练、推理或数据转换。
- 待办/风险：目标脚本与日志当前仍为 Git 未跟踪文件；`write_zarr`、`run_build_zarr`、`run_extend_zarr` 和 ACT action chunk 尚未迁移；默认配置路径仍指向不存在的 `RL Post Training/data/configs/data_prepare.yaml`，应在后续配置/入口批次修正；当前固定三相机契约仅适用于双臂数据集。


## 2026-08-21｜重构协作规范与日志

- 任务目标：根据审阅建议完善 `AGENTS.md`，并将日志重构为按任务记录。
- 已完成：明确日志以 `##` 为单位且最新置顶；区分代码/配置任务和纯讨论/文档任务的静态检查要求；定义代码改动的用户确认口径；限制本地 GitNexus 仅可静态读取；补充配置目录与模块归属。
- 涉及文件：`AGENTS.md`、`AI logs.md`。
- 方案确认状态：用户已明确要求按建议修改。
- 静态检查：已检查文档结构、文件名、相互引用和措辞一致性；代码静态检查不适用，未运行任何代码。
- 待办/风险：无。


## 2026-08-21｜建立协作规范与代码地图

- 任务目标：建立面向 Kuavo ACT 离线强化学习后训练的协作规范与代码地图。
- 已完成：静态检查仓库目录、README、迁移设计文档、原有 `AGENTS.md` 和本地 `.gitnexus/` 索引；建立新的协作规范、代码地图、日志规范和 Done When。
- 涉及文件：`AGENTS.md`、`AI logs.md`。
- 方案确认状态：用户已明确要求先产出首版供审阅；未改动任何业务代码。
- 静态检查：确认主链路为 `kuavo_data → kuavo_train → kuavo_deploy`，离线 RL 迁移材料和初始脚本位于 `RL Post Training/`；代码静态检查不适用，未运行代码、测试或安装命令。
- 待办/风险：`RL Post Training/` 当前为未跟踪目录，后续需由用户决定版本管理安排。
