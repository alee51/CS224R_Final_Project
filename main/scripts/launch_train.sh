#!/usr/bin/env bash
# GRPO train on Modal — always main/configs/train_real.yaml.
# Only difference between smoke and full: step count (env override).
#
# Usage (from repo root):
#   bash main/scripts/launch_train.sh --mode smoke
#   bash main/scripts/launch_train.sh --mode full

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

CFG="main/configs/train_real.yaml"
MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_train.sh --mode smoke|full" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash main/scripts/launch_train.sh --mode smoke|full" >&2
  exit 1
fi

export CS224R_TRAIN_MODE="$MODE"
if [[ "$MODE" == "smoke" ]]; then
  export CS224R_TOTAL_STEPS=10
else
  unset CS224R_TOTAL_STEPS
fi

TRAIN_OP="$(main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('operator','unknown'))")"
TRAIN_ARM="$(main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('arm','grpo'))")"
TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-train-${TRAIN_ARM}-${MODE}-${TRAIN_OP}-${TS}"
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi

echo "Launching train mode=$MODE config=$CFG total_steps=${CS224R_TOTAL_STEPS:-from yaml} app=$CS224R_APP_NAME"
exec main/.venv/bin/modal run --detach main/train/trainer.py::train_remote --config-path "$CFG"
