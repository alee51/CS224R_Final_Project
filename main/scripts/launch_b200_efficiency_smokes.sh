#!/usr/bin/env bash
# Minimal B200 minority_answer efficiency matrix (10 steps each, detached).
#
# Baseline reference: wandb wdl3fczm (B200, budget=105k, gc on, seq_batch=1).
#
# Usage (from repo root):
#   bash main/scripts/launch_b200_efficiency_smokes.sh          # all 4
#   bash main/scripts/launch_b200_efficiency_smokes.sh sleep_only sleep_gc75
#
# Variants:
#   sleep_only   vllm_sleep=1, else baseline (gc on, 105k) — measure sleep tax + VRAM
#   sleep_gc75   sleep + gc off + token_budget=75000 — OOM / fit probe
#   sleep        legacy alias (same as sleep_only)
#   budget       token_budget=130000, gc on
#   gc_off       gc off only (no sleep; expected OOM)
#   seqbatch     logprob_seq_batch=8

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

MODAL="${REPO_ROOT}/main/.venv/bin/modal"
ARM="minority_answer"
TS=$(date +%m-%d-%H%M)
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi

LOG_DIR="${REPO_ROOT}/main/docs/probes/artifacts/b200_eff_${TS}"
mkdir -p "$LOG_DIR"
: > "${LOG_DIR}/launched.txt"

launch_one() {
  local label=$1
  local cfg=$2
  local sleep_flag=${3:-0}
  local seq_batch=${4:-1}
  export CS224R_APP_NAME="cs224r-train-${ARM}-smoke-b200eff-${label}-${TS}"
  local log="${LOG_DIR}/${label}.log"
  echo "Launching ${label} cfg=${cfg} sleep=${sleep_flag} seq_batch=${seq_batch} app=${CS224R_APP_NAME}"
  echo "${label} ${CS224R_APP_NAME} ${cfg}" >> "${LOG_DIR}/launched.txt"
  "$MODAL" run --detach "main/train/trainer.py::train_remote_b200" \
    --config-path "$cfg" \
    --ablation "b200eff-${label}" \
    --vllm-sleep "$sleep_flag" \
    --logprob-chunk 0 \
    --logprob-seq-batch "$seq_batch" \
    --launch-mode smoke \
    --total-steps-override 10 \
    --no-resume \
    --arm-override "$ARM" \
    >"$log" 2>&1 &
  echo "  log → ${log}"
}

CFG_BASE="main/configs/train_real_b200.yaml"
CFG_BUDGET="main/configs/train_real_b200_ablate_budget130k.yaml"
CFG_GC="main/configs/train_real_b200_ablate_gc_off.yaml"
CFG_SLEEP_GC75="main/configs/train_real_b200_ablate_sleep_gc_off_75k.yaml"

run_variant() {
  case "$1" in
    sleep_only|sleep) launch_one sleep_only "$CFG_BASE" 1 1 ;;
    sleep_gc75) launch_one sleep_gc75 "$CFG_SLEEP_GC75" 1 1 ;;
    budget) launch_one budget "$CFG_BUDGET" 0 1 ;;
    gc_off) launch_one gc_off "$CFG_GC" 0 1 ;;
    seqbatch) launch_one seqbatch "$CFG_BASE" 0 8 ;;
    *)
      echo "Unknown variant: $1" >&2
      exit 1
      ;;
  esac
}

if [ $# -eq 0 ]; then
  run_variant sleep
  run_variant budget
  run_variant gc_off
  run_variant seqbatch
  echo ""
  echo "Launched 4 B200 efficiency smokes (detached). Manifest: ${LOG_DIR}/launched.txt"
  wait || true
else
  for v in "$@"; do
    run_variant "$v"
  done
  wait || true
fi

echo "Done. Summarize when finished:"
echo "  main/.venv/bin/python main/scripts/summarize_efficiency_smokes.py ${LOG_DIR}/launched.txt"
