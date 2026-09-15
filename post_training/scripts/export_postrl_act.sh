#!/usr/bin/env bash
set -euo pipefail

# Post-RL stochastic checkpoint produced by one experiment.
# Prefer the Dynamics-OPE best policy for deployment/export.
POSTRL_CHECKPOINT="/path/to/post_training/outputs/experiment_YYYYMMDD_HHMMSS/stage2/offline_ppo/best_ope"

# Kuavo ACT output hierarchy: outputs/train/<task>/<method>/run_*/epochbest
OUTPUT_ROOT="outputs/train"
TASK="your_task"
METHOD="your_method"

python post_training/scripts/export_postrl_act.py \
  --checkpoint "${POSTRL_CHECKPOINT}" \
  --task "${TASK}" \
  --method "${METHOD}" \
  --output-root "${OUTPUT_ROOT}"
