#!/usr/bin/env bash
# Run from repository root (cs224r_finalproject/).
set -euo pipefail
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage01}"
# Optional: MODAL_PROFILE=chicken602 if not default
python3 -m modal run main-verl/probes/hello_verl.py "$@"
