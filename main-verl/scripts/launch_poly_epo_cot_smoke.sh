#!/usr/bin/env bash
set -euo pipefail
export CS224R_APP_NAME="${CS224R_APP_NAME:-cs224r-verl-stage05}"
python3 -m modal run main-verl/probes/poly_epo_cot_smoke.py "$@"
