#!/usr/bin/env bash
# Launch Group A probe (smoke or full) with a per-run Modal app name.
#
# Usage (from repo root):
#   bash main/scripts/launch_probe_a.sh
#   bash main/scripts/launch_probe_a.sh main/configs/probe_a_05-24.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${1:-main/configs/probe_a_05-24.yaml}"
read -r PHASE OP <<EOF
$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
smoke = bool(cfg.get('smoke', False))
print('smoke' if smoke else 'full')
print(cfg['operator'])
" "$CFG")
EOF
TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-probe-a-${PHASE}-${OP}-${TS}"
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi
exec main/.venv/bin/modal run --detach main/probes/group_a_rollout_judge.py::run_full --config "$CFG"
