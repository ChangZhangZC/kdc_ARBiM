#!/usr/bin/env bash
set -euo pipefail
REPO="/home/kuavo/changzhang/kdc_ARBiM"
RUN="$REPO/post_training/outputs/sim_task1_flywheel01_single_step_stage12_ppo8k_bs256_20261017_073145"
OUT="$REPO/post_training/outputs/dynamics_diagnostics/flywheel01_offline"
cd "$REPO"
python post_training/tests/analysis/test_flywheel01_dynamics/evaluate.py \
  --run-dir "$RUN" --source zarr --split val \
  --windows "${WINDOWS:-64}" --horizon 32 --stride 32 \
  --device "${DEVICE:-cuda:0}" --output-dir "$OUT"
echo "Open plots:"
echo "  $OUT/latent_error_vs_horizon.png"
echo "  $OUT/advantage_disagreement.png"
