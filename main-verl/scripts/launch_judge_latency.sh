#!/usr/bin/env bash
# Run from repository root (cs224r_finalproject/).
# Optional first arg: config stem (default judge_latency_smoke).
# Serial parse diagnostic: ./main-verl/scripts/launch_judge_latency.sh judge_latency_smoke_serial
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage04-judge}"
CONFIG="${1:-judge_latency_smoke}"
PYTHONPATH=main-verl python3 -m modal run main-verl/probes/judge_latency_smoke.py --config-name "${CONFIG}"
