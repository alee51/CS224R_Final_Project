#!/usr/bin/env bash
# One-step REAL minority_cot training + judge; dumps /vol/ artifacts after step 1.
# Run from repo root. Requires JUDGE_BASE_URL (judge must be deployed).
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"

: "${CS224R_APP_NAME:=cs224r-verl-stage03b-train-trace}"
export CS224R_APP_NAME

if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
  echo "ERROR: JUDGE_BASE_URL is required." >&2
  exit 1
fi

echo "Launching REAL 1-step minority_cot training with judge artifacts:"
echo "  CS224R_APP_NAME=${CS224R_APP_NAME}"
echo "  MODAL_PROFILE=${MODAL_PROFILE}"
echo "  JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "  CS224R_JUDGE_TRACE_PROMPT_IDX=${CS224R_JUDGE_TRACE_PROMPT_IDX:-0}"
echo "  Artifacts: /vol/judge_train_step_log.jsonl + /vol/judge_trace_training_prompt0.json"
echo ""
echo "After run, grep driver log for:"
echo "  JUDGE_STEP_RECORD  JUDGE_TRACE_META  JUDGE_TRACE_PROBLEM_PREVIEW"

PYTHONPATH=main-verl modal run -d main-verl/probes/minority_cot_judge_trace_smoke.py
