#!/usr/bin/env bash
# HF->vLLM weight-sync smoke on Modal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-smoke-weight-sync-${TS}"
GPU_CLASS="b200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-class)
      GPU_CLASS="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_smoke_weight_sync.sh [--gpu-class h200|b200]" >&2
      exit 1
      ;;
  esac
done

if [[ "$GPU_CLASS" != "h200" && "$GPU_CLASS" != "b200" ]]; then
  echo "--gpu-class must be h200 or b200" >&2
  exit 1
fi

echo "Launching weight-sync smoke app=$CS224R_APP_NAME gpu=$GPU_CLASS"
exec main/.venv/bin/modal run main/probes/smoke_weight_sync.py --gpu-class "$GPU_CLASS"
