#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 /path/to/act_checkpoint /path/to/offline_dataset.zarr [device] [chunk_size]"
  exit 2
fi

CHECKPOINT="$1"
DATASET="$2"
DEVICE="${3:-cuda:0}"
CHUNK_SIZE="${4:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(--checkpoint "$CHECKPOINT" --device "$DEVICE")
DATA_COMMON=(--checkpoint "$CHECKPOINT" --dataset "$DATASET" --device "$DEVICE")
if [[ -n "$CHUNK_SIZE" ]]; then
  COMMON+=(--chunk-size "$CHUNK_SIZE")
  DATA_COMMON+=(--chunk-size "$CHUNK_SIZE")
fi

python "$SCRIPT_DIR/smoke_01_contract.py" "${COMMON[@]}"
python "$SCRIPT_DIR/smoke_02_policy_data.py" "${DATA_COMMON[@]}"
python "$SCRIPT_DIR/smoke_03_critic.py" "${DATA_COMMON[@]}"
python "$SCRIPT_DIR/smoke_04_dynamics.py" "${DATA_COMMON[@]}"

for CASE in chunk_scalar chunk_per_step single_step unsupported_vdelta; do
  python "$SCRIPT_DIR/smoke_05_ppo.py" \
    "${DATA_COMMON[@]}" --case "$CASE"
done

python "$SCRIPT_DIR/smoke_06_latent_cache.py" "${DATA_COMMON[@]}"
python "$SCRIPT_DIR/smoke_07_stage_gates.py"
python "$SCRIPT_DIR/smoke_08_e2e.py" "${DATA_COMMON[@]}"

echo "ALL CORE ARBiM POST-RL SMOKE TESTS PASSED"
echo "Manual tests not run here: smoke_00_data_prepare.py, smoke_09_checkpoint_export.py"
