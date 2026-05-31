#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=modal_bringup_env.sh
source "$(dirname "$0")/modal_bringup_env.sh"
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage05}"
PYTHONPATH=main-verl python3 -m modal run main-verl/probes/poly_epo_registry_check.py "$@"
