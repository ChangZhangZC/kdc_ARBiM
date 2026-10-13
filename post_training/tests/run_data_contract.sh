#!/usr/bin/env bash
# Consolidated ARBiM Data Contract test suite.
# Run only AFTER all contract integration batches have been merged.
# The real 600-episode/4-rollout test is a separate explicit command.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== Batch 01: direct LeRobot -> Zarr ==="
bash post_training/tests/run_batch01.sh

echo "=== Batch 02: merge Zarr ==="
bash post_training/tests/run_batch02.sh

echo "=== Batch 03: Staging -> LeRobot ==="
bash post_training/tests/run_batch03.sh

echo "=== Batch 04: merge LeRobot ==="
bash post_training/tests/run_batch04.sh

echo "=== Regression: LeRobot v0.4.2 fast merge shard mappings ==="
python -m py_compile \
  post_training/src/post_rl/data/lerobot_v042_fast_merge.py \
  post_training/tests/smoke/smoke_19_fast_lerobot_shards.py
python post_training/tests/smoke/smoke_19_fast_lerobot_shards.py

echo "=== Contract Reward provenance ==="
python -m py_compile \
  post_training/tests/smoke/smoke_18_contract_provenance.py \
  post_training/tests/integration/check_data_contract_real.py \
  post_training/tests/integration/resume_data_contract_merge_lerobot.py \
  kuavo_deploy/src/eval/sim_auto_test.py \
  kuavo_deploy/src/eval/rollout_recorder.py
python post_training/tests/smoke/smoke_18_contract_provenance.py

echo "=== Existing Data Wheel regression suite ==="
python post_training/tests/smoke/smoke_07_flywheel_recorder.py -v
python post_training/tests/smoke/smoke_08_flywheel_npy.py -v
python post_training/tests/smoke/smoke_09_flywheel_dataset.py -v
python post_training/tests/smoke/smoke_10_flywheel_eval.py -v
python post_training/tests/smoke/smoke_11_flywheel_target.py -v

echo "=== ALL DATA CONTRACT SYNTHETIC TESTS PASSED ==="
