#!/usr/bin/env bash
# Stage 6: Qwen3-4B-Base GRPO fit check (10-step OOM discovery, then 50-step gate).
# Run from repository root.
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"

: "${CS224R_APP_NAME:=cs224r-verl-stage06-grpo}"
export CS224R_APP_NAME

echo "Launching GRPO 4B fit smoke:"
echo "  CS224R_APP_NAME=${CS224R_APP_NAME}"
echo "  MODAL_PROFILE=${MODAL_PROFILE:-default}"
echo "  Config: grpo_smoke_4b.yaml (ladder 1b: micro=4, gpu_mem=0.45)"

PYTHONPATH=main-verl modal run -d main-verl/probes/grpo_smoke_4b.py "$@"
