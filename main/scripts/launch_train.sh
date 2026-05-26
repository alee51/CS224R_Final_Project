#!/usr/bin/env bash
# Optional smoke/full GRPO train on Modal (detach).
#
# Usage (from repo root):
#   bash main/scripts/launch_train.sh
#   bash main/scripts/launch_train.sh main/configs/train_grpo_05-25.yaml smoke

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${1:-main/configs/train_grpo_05-25.yaml}"
MODE="${2:-smoke}"
read -r OP ARM <<EOF
$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get('operator', 'unknown'))
print(cfg.get('arm', 'grpo'))
" "$CFG")
EOF
TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-train-${ARM}-${MODE}-${OP}-${TS}"
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi

if [ "$MODE" = "smoke" ]; then
  exec main/.venv/bin/modal run --detach main/train/trainer.py::train_remote --config-path "$CFG"
fi
exec main/.venv/bin/modal run --detach main/train/trainer.py::train_remote --config-path "$CFG"
