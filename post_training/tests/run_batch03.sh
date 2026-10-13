#!/usr/bin/env bash
# Batch 03: Staging -> LeRobot synthetic verification (deferred until full integration).
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m py_compile \
  post_training/src/post_rl/data/staging_to_lerobot.py \
  post_training/tests/smoke/smoke_16_staging_to_lerobot.py

python post_training/tests/smoke/smoke_16_staging_to_lerobot.py
echo "BATCH 03 SYNTHETIC TESTS PASSED"
