#!/usr/bin/env bash
# Launch Stage 3b smoke (minority_cot with real judge clusters).
#
# Prereqs (set in caller's shell):
#   CS224R_APP_NAME=cs224r-verl-stage03b
#   MODAL_PROFILE=chicken602
#   JUDGE_BASE_URL=https://chicken602--v1-chat-completions.modal.run
#   JUDGE_AUTH_TOKEN=...   (optional)
#   JUDGE_HEALTH_URL=https://chicken602--health.modal.run   (optional override)
#
# Image rebuild count 5 (adds maxrl_expose_data_to_adv_est.patch).
# Triggers automatically on first launch after modal_image.py changes.
set -euo pipefail

: "${CS224R_APP_NAME:=cs224r-verl-stage03b}"
export CS224R_APP_NAME

if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
  echo "ERROR: JUDGE_BASE_URL is required. Example:" >&2
  echo "  export JUDGE_BASE_URL=https://chicken602--v1-chat-completions.modal.run" >&2
  exit 1
fi

echo "Launching Stage 3b smoke with:"
echo "  CS224R_APP_NAME=${CS224R_APP_NAME}"
echo "  MODAL_PROFILE=${MODAL_PROFILE:-<default>}"
echo "  JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "  JUDGE_AUTH_TOKEN=${JUDGE_AUTH_TOKEN:+<set>}${JUDGE_AUTH_TOKEN:-<unset>}"

PYTHONPATH=main-verl modal run -d main-verl/probes/minority_cot_judge_smoke.py
