#!/usr/bin/env bash
# Fast judge cluster trace: 1 Polaris prompt, 8 vLLM rollouts, real judge — ~5–8 min.
# Same assign_judge_clusters() as training; no VeRL/Ray. Artifact on artifacts volume.
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"

: "${CS224R_APP_NAME:=cs224r-verl-judge-trace-fast}"
export CS224R_APP_NAME

if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
  echo "ERROR: JUDGE_BASE_URL required." >&2
  exit 1
fi

echo "Fast judge trace (B200:1, NOT full trainer smoke):"
echo "  CS224R_APP_NAME=${CS224R_APP_NAME}"
echo "  JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "  CS224R_JUDGE_TRACE_PROMPT_IDX=${CS224R_JUDGE_TRACE_PROMPT_IDX:-0}"
echo "  CS224R_TRACE_ACTOR_MODEL=${CS224R_TRACE_ACTOR_MODEL:-Qwen/Qwen3-1.7B-Base}"
echo "  Artifact: /vol/judge_trace_prompt\${CS224R_JUDGE_TRACE_PROMPT_IDX:-0}_<actor>.json"
echo ""
echo "After run, grep: JUDGE_TRACE_ARTIFACT=  or download artifact from Modal volume."

PYTHONPATH=main-verl modal run main-verl/probes/judge_cluster_trace_fast.py
