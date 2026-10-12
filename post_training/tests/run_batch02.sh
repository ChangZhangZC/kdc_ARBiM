#!/usr/bin/env bash
# Batch 02: synthetic merge tests; execute during consolidated contract testing.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m py_compile \
  post_training/src/post_rl/data/zarr_merge.py \
  post_training/tests/smoke/smoke_14_merge_zarr.py \
  post_training/tests/smoke/smoke_15_merge_zarr_edges.py

python post_training/tests/smoke/smoke_14_merge_zarr.py
python post_training/tests/smoke/smoke_15_merge_zarr_edges.py
echo "BATCH 02 SYNTHETIC TESTS PASSED"
