#!/usr/bin/env bash
# FA2 isolation smoke on Modal (same image as train). From repo root:
#   bash main/scripts/launch_smoke_flash_attn.sh
#   bash main/scripts/launch_smoke_flash_attn.sh --gpu-class b200
#   bash main/scripts/launch_smoke_flash_attn.sh --all   # run every stage even after failure

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-smoke-flash-attn-${TS}"
GPU_CLASS="h200"
ALL_STAGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-class)
      GPU_CLASS="${2:-}"
      shift 2
      ;;
    --all)
      ALL_STAGES=1
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_smoke_flash_attn.sh [--gpu-class h200|b200] [--all]" >&2
      exit 1
      ;;
  esac
done

if [[ "$GPU_CLASS" != "h200" && "$GPU_CLASS" != "b200" ]]; then
  echo "--gpu-class must be h200 or b200" >&2
  exit 1
fi

echo "Launching FA2 smoke app=$CS224R_APP_NAME gpu=$GPU_CLASS"
if [[ $ALL_STAGES -eq 1 ]]; then
  exec main/.venv/bin/modal run main/probes/smoke_flash_attn.py --gpu-class "$GPU_CLASS" --all-stages
fi
exec main/.venv/bin/modal run main/probes/smoke_flash_attn.py --gpu-class "$GPU_CLASS"
