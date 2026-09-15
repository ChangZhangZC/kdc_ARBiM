#!/usr/bin/env bash
set -euo pipefail

# Input must already be a complete stochastic Post-RL ACT pretrained bundle:
# config.json + model.safetensors + policy pre/post-processors.
# Prefer the Dynamics-OPE best policy. This exporter does NOT read training_state.pt;
# it converts stochastic ACT -> deterministic Kuavo ACT while preserving the mean action.
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
