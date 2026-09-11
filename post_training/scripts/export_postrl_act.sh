#!/usr/bin/env bash
set -euo pipefail

# Fill these three values before running.
POSTRL_CHECKPOINT="/path/to/post_training/outputs/experiment_YYYYMMDD_HHMMSS/offline_ppo/last"
TASK="your_task"
METHOD="your_method"

python post_training/scripts/export_postrl_act.py \
  --checkpoint "${POSTRL_CHECKPOINT}" \
  --task "${TASK}" \
  --method "${METHOD}"
