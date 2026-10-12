#!/usr/bin/env bash
# Batch 01: run the direct LeRobot -> Zarr synthetic and regression checks.
# Usage: bash post_training/tests/run_batch01.sh
# For real-data validation, run tests/integration/check_lerobot_to_zarr_real.py separately.
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m py_compile \
  post_training/src/post_rl/data/data_prepare.py \
  post_training/tests/smoke/smoke_12_lerobot_to_zarr.py \
  post_training/tests/smoke/smoke_13_lerobot_to_zarr_edge.py \
  post_training/tests/integration/check_lerobot_to_zarr_real.py

python post_training/tests/smoke/smoke_12_lerobot_to_zarr.py
python post_training/tests/smoke/smoke_13_lerobot_to_zarr_edge.py
python post_training/tests/smoke/smoke_reward_last_release.py
python post_training/tests/smoke/smoke_reward_v2_zarr_migration.py

echo "BATCH 01 SYNTHETIC TESTS PASSED"
