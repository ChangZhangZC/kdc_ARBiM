#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/home/kuavo/changzhang/kdc_ARBiM}"
RUN="${RUN:-$REPO/post_training/outputs/sim_task1_flywheel01_single_step_stage12_ppo8k_bs256_20261017_073145}"
OUT="${OUT:-$REPO/post_training/outputs/dynamics_diagnostics/flywheel01_stochastic}"
cd "$REPO"
python post_training/tests/analysis/test_flywheel01_dynamics/evaluate_stochastic.py \
  --run-dir "$RUN" --source zarr --split val \
  --windows "${WINDOWS:-64}" --repeats "${REPEATS:-8}" \
  --horizon 32 --stride 32 --seed "${SEED:-20261023}" \
  --device "${DEVICE:-cuda:0}" --output-dir "$OUT"
echo "Stochastic diagnostics are under $OUT"
echo "  $OUT/stochastic_vs_elite_mean.png"
echo "  $OUT/stochastic_advantage_sign.png"
echo "  $OUT/advantage_margin_sensitivity.png"
