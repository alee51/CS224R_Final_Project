#!/usr/bin/env bash
# Launch Group B step probe (smoke or full) with a per-run Modal app name.
#
# Usage (from repo root):
#   bash main/scripts/launch_probe_step_b.sh
#   bash main/scripts/launch_probe_step_b.sh main/configs/probe_step_b_05-25.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"

CFG="${1:-main/configs/probe_step_b_05-25.yaml}"
{
  read -r PHASE
  read -r OP
  read -r RUNTIME_CFG
} <<EOF
$(PYTHONPATH=main main/.venv/bin/python -c "
import sys
sys.path.insert(0, 'main')
from probes.group_b_step_probe import load_merged_config, write_merged_runtime_config
cfg = load_merged_config(sys.argv[1])
runtime = write_merged_runtime_config(sys.argv[1])
print('smoke' if cfg.get('smoke') else 'full')
print(cfg['operator'])
print(runtime)
" "$CFG")
EOF
TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-probe-b-${PHASE}-${OP}-${TS}"
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi
if [ "$PHASE" = "smoke" ]; then
  # Foreground: local entrypoint passes --config and blocks on pipeline.
  exec main/.venv/bin/modal run main/probes/group_b_step_probe.py::run_full_sync --config "$RUNTIME_CFG"
else
  exec main/.venv/bin/modal run --detach main/probes/group_b_step_probe.py::run_full --config "$RUNTIME_CFG"
fi
