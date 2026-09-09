#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/act_checkpoint /path/to/offline_dataset.zarr [device]"
  exit 2
fi

CHECKPOINT="$1"
DATASET="$2"
DEVICE="${3:-cuda:0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/smoke_01_contract.py" \
  --checkpoint "$CHECKPOINT" --device "$DEVICE"

python "$SCRIPT_DIR/smoke_02_policy_data.py" \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --device "$DEVICE"

python "$SCRIPT_DIR/smoke_03_critic.py" \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --device "$DEVICE"

python "$SCRIPT_DIR/smoke_04_dynamics.py" \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --device "$DEVICE"

for CASE in chunk_scalar chunk_per_step single_step unsupported_vdelta; do
  python "$SCRIPT_DIR/smoke_05_ppo.py" \
    --checkpoint "$CHECKPOINT" --dataset "$DATASET" --device "$DEVICE" --case "$CASE"
done

python "$SCRIPT_DIR/smoke_06_e2e.py" \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --device "$DEVICE"

echo "ALL ARBiM POST-RL SMOKE TESTS PASSED"
