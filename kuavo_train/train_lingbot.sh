#!/usr/bin/env bash
set -euo pipefail

# LingBot-VLA launcher integrated into kuavo_train.
# Usage (internal default entry):
#   bash kuavo_train/train_lingbot.sh configs/policy/lingbot/robotwin_load20000h.yaml --model.model_path ...
# Usage (explicit entry):
#   bash kuavo_train/train_lingbot.sh kuavo_train/lingbot/tasks/vla/train_lingbotvla.py configs/policy/lingbot/robotwin_load20000h.yaml --model.model_path ...

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash kuavo_train/train_lingbot.sh <config.yaml> [extra args ...]"
  echo "  bash kuavo_train/train_lingbot.sh <train_entry.py> <config.yaml> [extra args ...]"
  exit 1
fi

if [[ "$1" == *.yaml ]]; then
  TRAIN_ENTRY="kuavo_train/lingbot/tasks/vla/train_lingbotvla.py"
  CONFIG_PATH="$1"
  shift
else
  TRAIN_ENTRY="$1"
  shift
  CONFIG_PATH="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export REPO_ROOT
export TRAIN_ENTRY
export CONFIG_PATH
export EXTRA_ARGS_STR="$(printf '%s\n' "$@")"
export DRY_RUN="${DRY_RUN:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python - <<'PY'
import os
from pathlib import Path

from kuavo_train.wrapper.policy.lingbot import (
    CustomLingbotConfigWrapper,
    CustomLingbotPolicyWrapper,
)

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
train_entry = os.environ["TRAIN_ENTRY"]
config_path = os.environ["CONFIG_PATH"]
dry_run = os.environ.get("DRY_RUN", "0") == "1"
extra_args_raw = os.environ.get("EXTRA_ARGS_STR", "")
extra_args = [x for x in extra_args_raw.splitlines() if x]

cfg = CustomLingbotConfigWrapper(
    lingbot_root=os.environ.get("LINGBOT_ROOT", ""),
    train_entry=train_entry,
    config_path=config_path,
    nnodes=int(os.environ.get("NNODES", "1")),
    node_rank=int(os.environ.get("NODE_RANK", "0")),
    master_addr=os.environ.get("MASTER_ADDR", "0.0.0.0"),
    master_port=int(os.environ.get("MASTER_PORT", "62500")),
    dry_run=dry_run,
)

runner = CustomLingbotPolicyWrapper(cfg)
code = runner.launch(repo_root=repo_root, extra_args=extra_args)
raise SystemExit(code)
PY
