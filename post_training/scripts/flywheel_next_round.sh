#!/usr/bin/env bash
set -euo pipefail

# ARBiM Data Wheel：每轮只修改下面这一区域，然后运行：
# bash post_training/scripts/flywheel_next_round.sh
# 相对路径均以 kdc_ARBiM 仓库根目录为基准。

# ====================== 1. 每轮必改：输入/输出路径 ======================
STAGING=""          # 本轮 *_staging.npy 文件，或存放它们的 staging 目录
PREV_LEROBOT=""     # 上一轮已合并的 LeRobot（保留 RGB + 三路 Depth）
PREV_ZARR=""        # 上一轮已合并的 Offline PPO Zarr
OUTPUT_ROOT=""      # 本轮新的输出目录，例如 outputs/flywheel_round_02

# ====================== 2. 可选参数 ======================
USE_DEPTH=false     # 新 PPO Zarr 是否保存 Depth：false / true；需与上一轮 Zarr 一致
TASK="ARBiM Data Wheel"
ROLLOUT_REPO_ID="arbim/flywheel_rollout"
MERGED_REPO_ID="arbim/flywheel_merged"
STREAM_BATCH_SIZE=50
JPEG_QUALITY=95
RETURN_GAMMA=0.99
PYTHON_BIN=python

# 其他备选项（按需在下方 Python 命令处修改）：
# staging_to_lerobot.py：默认保留全部 3 路 Depth；
#   --require-valid-reward 仅导出 Reward V2 有效的成功 Rollout；
#   --rgb-only-legacy 用于历史 RGB-only Staging，不建议新 Data Wheel 使用。
# lerobot_merge.py：默认要求完整 RGB + 3 路 Depth Schema；
#   --allow-rgb-only 仅供旧数据兼容使用。
# data_prepare.py：reward_source=recorded 表示保留本轮 Staging 的 Reward V2；
#   reward_source=auto 可以按来源自动选择。当前流程建议使用 recorded。

# ====================== 3. 运行准备 ======================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for value in "$STAGING" "$PREV_LEROBOT" "$PREV_ZARR" "$OUTPUT_ROOT"; do
  [[ -n "$value" ]] || { echo "[ERROR] 请先填写顶部四个输入/输出路径。" >&2; exit 1; }
done
[[ -e "$STAGING" ]] || { echo "[ERROR] Staging 不存在: $STAGING" >&2; exit 1; }
[[ -f "$PREV_LEROBOT/meta/info.json" ]] || { echo "[ERROR] 上一轮 LeRobot 无效: $PREV_LEROBOT" >&2; exit 1; }
[[ -d "$PREV_ZARR" ]] || { echo "[ERROR] 上一轮 Zarr 不存在: $PREV_ZARR" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "[ERROR] 输出目录已存在，拒绝覆盖: $OUTPUT_ROOT" >&2; exit 1; }

ROLLOUT_LEROBOT="$OUTPUT_ROOT/rollout_lerobot"
ROLLOUT_ZARR="$OUTPUT_ROOT/rollout_reward_v2.zarr"
MERGED_LEROBOT="$OUTPUT_ROOT/merged_il_lerobot"
MERGED_ZARR="$OUTPUT_ROOT/merged_offline_rl.zarr"

mkdir -p "$OUTPUT_ROOT"

# ====================== 4. 按顺序转换 ======================
echo "[1/4] Staging NPY -> Rollout LeRobot"
"$PYTHON_BIN" post_training/src/post_rl/data/staging_to_lerobot.py \
  "$STAGING" \
  --template-lerobot-root "$PREV_LEROBOT" \
  --output-root "$ROLLOUT_LEROBOT" \
  --repo-id "$ROLLOUT_REPO_ID" \
  --task "$TASK" \
  --require-valid-reward

echo "[2/4] Previous LeRobot + Rollout LeRobot -> Merged LeRobot"
"$PYTHON_BIN" post_training/src/post_rl/data/lerobot_merge.py \
  "$PREV_LEROBOT" "$ROLLOUT_LEROBOT" \
  --output-root "$MERGED_LEROBOT" \
  --repo-id "$MERGED_REPO_ID" \
  --fallback-task "$TASK"

echo "[3/4] Rollout LeRobot -> Rollout Zarr"
# data_prepare.py 通过 --config 指定模式和参数，因此这里生成一份本轮配置。
cat > "$OUTPUT_ROOT/rollout_to_zarr.yaml" <<EOF
mode: lerobot_to_zarr
lerobot_root: "$ROLLOUT_LEROBOT"
zarr_output_path: "$ROLLOUT_ZARR"
overwrite: false
use_depth: $USE_DEPTH
stream_batch_size: $STREAM_BATCH_SIZE
jpeg_quality: $JPEG_QUALITY
reward_mode: last_gripper_release_sparse
reward_source: recorded
EOF

"$PYTHON_BIN" post_training/src/post_rl/data/data_prepare.py \
  --config "$OUTPUT_ROOT/rollout_to_zarr.yaml"

echo "[4/4] Previous Zarr + Rollout Zarr -> Merged Zarr"
cat > "$OUTPUT_ROOT/merge_zarr.yaml" <<EOF
mode: merge_zarr
overwrite: false
zarr_output_path: "$MERGED_ZARR"
return_gamma: $RETURN_GAMMA
zarr_sources:
  - name: previous_merged
    path: "$PREV_ZARR"
  - name: new_rollout
    path: "$ROLLOUT_ZARR"
EOF

"$PYTHON_BIN" post_training/src/post_rl/data/data_prepare.py \
  --config "$OUTPUT_ROOT/merge_zarr.yaml"

echo ""
echo "DATA WHEEL DATA PREPARATION FINISHED"
echo "新增 LeRobot: $ROLLOUT_LEROBOT"
echo "新增 Zarr:    $ROLLOUT_ZARR"
echo "合并 LeRobot: $MERGED_LEROBOT"
echo "合并 Zarr:    $MERGED_ZARR"
