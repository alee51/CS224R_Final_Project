#!/usr/bin/env bash
# Worst-case n_kept=512 VRAM stress on Modal H200 (train_real.yaml rollout + microbatch=64).
#
# Usage (repo root):
#   bash main/scripts/launch_stress_n_kept.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"
export CS224R_VLLM_SLEEP=1

TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-stress-nkept512-${TS}"

echo "Launching stress n_kept=512 on H200 app=$CS224R_APP_NAME"
exec main/.venv/bin/modal run --detach main/probes/stress_n_kept_probe.py::stress_remote
