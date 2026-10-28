#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/home/kuavo/changzhang/kdc_ARBiM}"
RUN="${RUN:-$REPO/post_training/outputs/sim_task1_flywheel01_single_step_stage12_ppo8k_bs256_20261017_073145}"
OUT="${OUT:-$REPO/post_training/outputs/dynamics_diagnostics/flywheel01_components}"
cd "$REPO"
python post_training/tests/analysis/test_flywheel01_dynamics/evaluate_components.py \
  --run-dir "$RUN" --source zarr --split val \
  --windows "${WINDOWS:-64}" --repeats "${REPEATS:-8}" \
  --horizon 32 --stride 32 --seed "${SEED:-20261028}" \
  --device "${DEVICE:-cuda:0}" --output-dir "$OUT"
echo "Results: $OUT"
echo "  three_mode_rollout_mse.png"
echo "  teacher_one_step_components.png"
echo "  variance_calibration.png"
echo "  component_advantage_direction.png"
