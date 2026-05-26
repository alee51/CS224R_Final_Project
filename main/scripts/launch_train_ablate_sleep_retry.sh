#!/usr/bin/env bash
# Retry sleep ablations C and D after empty_cache-before-wake fix.
#
# Usage (from repo root):
#   bash main/scripts/launch_train_ablate_sleep_retry.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

CFG="main/configs/train_real.yaml"
export CS224R_TRAIN_MODE=smoke
export CS224R_TOTAL_STEPS=10

TRAIN_OP="$(main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('operator','unknown'))")"
TRAIN_ARM="$(main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('arm','grpo'))")"
TS=$(date +%m-%d-%H%M)
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi

MODAL=main/.venv/bin/modal
LOG_DIR="${REPO_ROOT}/main/docs/probes/artifacts/ablate_sleepfix_${TS}"
mkdir -p "$LOG_DIR"

launch_one() {
  local label=$1
  local sleep_flag=$2
  local chunk=$3
  export CS224R_APP_NAME="cs224r-train-${TRAIN_ARM}-smoke-ablate-${label}-wakefix-${TRAIN_OP}-${TS}"
  local log="${LOG_DIR}/ablate_${label}.log"
  echo "Launching ${label} wakefix (vllm_sleep=${sleep_flag} logprob_chunk=${chunk}) app=${CS224R_APP_NAME}"
  $MODAL run --detach main/train/trainer.py::train_remote \
    --config-path "$CFG" \
    --ablation "${label}-wakefix" \
    --vllm-sleep "$sleep_flag" \
    --logprob-chunk "$chunk" \
    >"$log" 2>&1 &
  echo $! >> "${LOG_DIR}/pids.txt"
  echo "  log → ${log}"
}

: > "${LOG_DIR}/pids.txt"
launch_one C 1 0
launch_one D 1 64

echo ""
echo "Sleep retry smokes C + D launched. Logs: ${LOG_DIR}/"
wait || true
