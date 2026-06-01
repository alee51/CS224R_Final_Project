#!/usr/bin/env bash
# One-shot Stage 3b smoke launcher. Bundles judge URL + chicken602 profile.
#
# Requires the maxrl fork at cs224r-patches HEAD to include commit 572a592
# (passes DataProto to registered adv_estimator hooks).
# Inner script runs probes/minority_cot_judge_smoke.py with health-probe pre-flight.
set -euo pipefail

export MODAL_PROFILE="${MODAL_PROFILE:-chicken602}"
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage03b}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://chicken602--v1-chat-completions.modal.run}"
export JUDGE_HEALTH_URL="${JUDGE_HEALTH_URL:-https://chicken602--health.modal.run}"

echo "[stage-3b] MODAL_PROFILE=${MODAL_PROFILE}"
echo "[stage-3b] CS224R_APP_NAME=${CS224R_APP_NAME}"
echo "[stage-3b] JUDGE_BASE_URL=${JUDGE_BASE_URL}"

./main-verl/scripts/launch_minority_cot_judge_smoke.sh
