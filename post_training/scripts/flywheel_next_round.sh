#!/usr/bin/env bash
set -euo pipefail

# ARBiM Data Wheel：每轮只修改下面这一区域，然后运行：
# bash post_training/scripts/flywheel_next_round.sh
# 相对路径均以 kdc_ARBiM 仓库根目录为基准。

# ====================== 1. 每轮必改：输入/输出路径 ======================
# 本轮支持多个 Staging 目录或单个 *_staging.npy 文件；至少填写一个
STAGING_DIRS=(
  # "outputs/eval/collection_01/data_wheel/staging"
  # "outputs/eval/collection_02/data_wheel/staging"
)
PREV_LEROBOT=""     # 上一轮已合并的 LeRobot（保留 RGB + 三路 Depth）
PREV_ZARR=""        # 上一轮已合并的 Offline PPO Zarr
OUTPUT_ROOT=""      # 本轮新的输出目录，例如 outputs/flywheel_round_02

# ====================== 2. 可选参数 ======================
# 四个输出名称仅控制 OUTPUT_ROOT 下的子目录，可以按轮次调整
ROLLOUT_LEROBOT_NAME="rollout_lerobot"
ROLLOUT_ZARR_NAME="rollout.zarr"
MERGED_LEROBOT_NAME="merged_il_lerobot"
MERGED_ZARR_NAME="merged_offline_rl.zarr"

USE_DEPTH=false     # 新 PPO Zarr 是否保存 Depth：false / true；需与上一轮 Zarr 一致
TASK="ARBiM Data Wheel"
ROLLOUT_REPO_ID="arbim/flywheel_rollout"
MERGED_REPO_ID="arbim/flywheel_merged"
STREAM_BATCH_SIZE=50
JPEG_QUALITY=95
RETURN_GAMMA=0.99
PYTHON_BIN=python

# ====================== 3. 运行准备 ======================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if (("${#STAGING_DIRS[@]}" == 0)); then
  echo "[ERROR] 请在 STAGING_DIRS 中至少填写一个 Staging 路径。" >&2
  exit 1
fi
for staging in "${STAGING_DIRS[@]}"; do
  [[ -n "$staging" && -e "$staging" ]] || { echo "[ERROR] Staging 不存在或为空: $staging" >&2; exit 1; }
done
for value in "$PREV_LEROBOT" "$PREV_ZARR" "$OUTPUT_ROOT"; do
  [[ -n "$value" ]] || { echo "[ERROR] 请先填写上一轮 LeRobot、Zarr 和本轮输出路径。" >&2; exit 1; }
done
[[ -f "$PREV_LEROBOT/meta/info.json" ]] || { echo "[ERROR] 上一轮 LeRobot 无效: $PREV_LEROBOT" >&2; exit 1; }
[[ -d "$PREV_ZARR" ]] || { echo "[ERROR] 上一轮 Zarr 不存在: $PREV_ZARR" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "[ERROR] 输出目录已存在，拒绝覆盖: $OUTPUT_ROOT" >&2; exit 1; }

ROLLOUT_LEROBOT="$OUTPUT_ROOT/$ROLLOUT_LEROBOT_NAME"
ROLLOUT_ZARR="$OUTPUT_ROOT/$ROLLOUT_ZARR_NAME"
MERGED_LEROBOT="$OUTPUT_ROOT/$MERGED_LEROBOT_NAME"
MERGED_ZARR="$OUTPUT_ROOT/$MERGED_ZARR_NAME"

mkdir -p "$OUTPUT_ROOT"

# ====================== 4. 按顺序转换 ======================
echo "[1/5] Staging NPY -> Rollout LeRobot"
"$PYTHON_BIN" post_training/src/post_rl/data/staging_to_lerobot.py \
  "${STAGING_DIRS[@]}" \
  --template-lerobot-root "$PREV_LEROBOT" \
  --output-root "$ROLLOUT_LEROBOT" \
  --repo-id "$ROLLOUT_REPO_ID" \
  --task "$TASK" \
  --require-valid-reward

echo "[2/5] Previous LeRobot + Rollout LeRobot -> Merged LeRobot"
"$PYTHON_BIN" post_training/src/post_rl/data/lerobot_merge.py \
  "$PREV_LEROBOT" "$ROLLOUT_LEROBOT" \
  --output-root "$MERGED_LEROBOT" \
  --repo-id "$MERGED_REPO_ID" \
  --fallback-task "$TASK"

echo "[3/5] Rollout LeRobot -> Rollout Zarr"
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

echo "[4/5] Previous Zarr + Rollout Zarr -> Merged Zarr"
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

echo "[5/5] Write Data Wheel round summary"
"$PYTHON_BIN" post_training/src/post_rl/data/flywheel_report.py \
  --staging "${STAGING_DIRS[@]}" \
  --prev-lerobot "$PREV_LEROBOT" \
  --prev-zarr "$PREV_ZARR" \
  --rollout-lerobot "$ROLLOUT_LEROBOT" \
  --rollout-zarr "$ROLLOUT_ZARR" \
  --merged-lerobot "$MERGED_LEROBOT" \
  --merged-zarr "$MERGED_ZARR" \
  --output "$OUTPUT_ROOT/flywheel_summary.txt"

echo ""
echo "DATA WHEEL DATA PREPARATION FINISHED"
echo "新增 LeRobot: $ROLLOUT_LEROBOT"
echo "新增 Zarr:    $ROLLOUT_ZARR"
echo "合并 LeRobot: $MERGED_LEROBOT"
echo "合并 Zarr:    $MERGED_ZARR"
echo "本轮汇总:     $OUTPUT_ROOT/flywheel_summary.txt"
