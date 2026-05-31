#!/usr/bin/env bash
# Run from repository root (cs224r_finalproject/).
# Stage 3a smoke: minority_cot advantage estimator on B200:4.
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage03a}"
PYTHONPATH=main-verl python3 -m modal run main-verl/probes/minority_cot_smoke.py "$@"
