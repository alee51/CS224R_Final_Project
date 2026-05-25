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
PHASE=$(yq '.smoke' "$CFG" | grep -q true && echo smoke || echo full)
OP=$(yq '.operator' "$CFG")
TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-probe-a-${PHASE}-${OP}-${TS}"
exec modal run --detach main/probes/group_a_rollout_judge.py::run_full --config "$CFG"
