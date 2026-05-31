#!/usr/bin/env bash
# Run from repository root (cs224r_finalproject/).
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage01}"
PYTHONPATH=main-verl python3 -m modal run main-verl/probes/hello_verl.py "$@"
