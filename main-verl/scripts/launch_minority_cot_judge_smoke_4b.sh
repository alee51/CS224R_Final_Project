#!/usr/bin/env bash
# Stage 6: single 4B probe — minority_cot + judge + ladder knobs.
#
# Next run (ladder 1e — gpu_mem 0.75 + 2-container judge + trace on):
#   1. Redeploy judge (picks up max_containers=2): ./main-verl/scripts/launch_judge_service.sh
#   2. export MODAL_PROFILE=chicken602
#   3. export JUDGE_BASE_URL=https://chicken602--v1-chat-completions.modal.run
#   4. export CS224R_SMOKE_CONFIG=minority_cot_smoke_judge_4b_ladder1e
#   5. ./main-verl/scripts/launch_minority_cot_judge_smoke_4b.sh
#
# After it kicks off, pull the trace mid-run:
#   modal volume get cs224r-artifacts judge_trace_4b_ladder1e_step.json -
#
# Judge batch sizing is in the yaml (algorithm.minority_cot.judge_http_batch_size).
# Only deploy secrets are passed via env: JUDGE_BASE_URL, JUDGE_AUTH_TOKEN.
#
# Override step count without editing yaml:
#   CS224R_SMOKE_STEPS=3 ...
# Run from repository root.
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"

: "${CS224R_APP_NAME:=cs224r-verl-stage06-minority-judge-4b}"
export CS224R_APP_NAME

if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
  echo "ERROR: JUDGE_BASE_URL is required. Example:" >&2
  echo "  export JUDGE_BASE_URL=https://chicken602--v1-chat-completions.modal.run" >&2
  exit 1
fi

echo "Launching 4B minority_cot + judge step-time probe:"
echo "  CS224R_APP_NAME=${CS224R_APP_NAME}"
echo "  MODAL_PROFILE=${MODAL_PROFILE:-default}"
echo "  JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "  Config: ${CS224R_SMOKE_CONFIG:-minority_cot_smoke_judge_4b_ladder1e}.yaml"
echo "  Steps: ${CS224R_SMOKE_STEPS:-(yaml default)}"

PYTHONPATH=main-verl modal run -d main-verl/probes/minority_cot_judge_smoke_4b.py "$@"
