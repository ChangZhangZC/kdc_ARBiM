#!/usr/bin/env bash
# Batch 04 LeRobot merge synthetic tests; run in final data contract test stage.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m py_compile \
  post_training/src/post_rl/data/lerobot_merge.py \
  post_training/tests/smoke/smoke_17_merge_lerobot.py
python post_training/tests/smoke/smoke_17_merge_lerobot.py
echo "BATCH 04 SYNTHETIC TESTS PASSED"
