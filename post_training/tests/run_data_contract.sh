#!/usr/bin/env bash
# Final ARBiM data-contract synthetic regression suite.
# Run AFTER implementing all batches, from repository root or any directory:
#   bash post_training/tests/run_data_contract.sh
# Optional real data is checked separately with the integration scripts.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "========== ARBiM DATA CONTRACT: BATCH 01 =========="
bash post_training/tests/run_batch01.sh

echo "========== ARBiM DATA CONTRACT: BATCH 02 =========="
bash post_training/tests/run_batch02.sh

echo "========== ARBiM DATA CONTRACT: BATCH 03 =========="
bash post_training/tests/run_batch03.sh

echo "========== ARBiM DATA CONTRACT: BATCH 04 =========="
bash post_training/tests/run_batch04.sh

echo "========== ARBiM DATA CONTRACT: PROVENANCE =========="
python -m py_compile \
  post_training/tests/smoke/smoke_18_contract_provenance.py \
  post_training/tests/integration/check_data_contract_real.py
python post_training/tests/smoke/smoke_18_contract_provenance.py

echo "========== ARBiM DATA CONTRACT: LEGACY FLYWHEEL =========="
python post_training/tests/smoke/smoke_07_flywheel_recorder.py
python post_training/tests/smoke/smoke_08_flywheel_npy.py
python post_training/tests/smoke/smoke_09_flywheel_dataset.py
python post_training/tests/smoke/smoke_10_flywheel_eval.py
python post_training/tests/smoke/smoke_11_flywheel_target.py

echo "ARBiM DATA CONTRACT SYNTHETIC SUITE PASSED"
