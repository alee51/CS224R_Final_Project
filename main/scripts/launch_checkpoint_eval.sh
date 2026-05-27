#!/usr/bin/env bash
# Checkpoint rollout eval on Modal.
#
# Usage (from repo root):
#   bash main/scripts/launch_checkpoint_eval.sh
#   bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_2k.yaml
#   bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_2k.yaml --detach

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

CFG="main/configs/checkpoint_eval.yaml"
DETACH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH="--detach"
      shift
      ;;
    --config)
      CFG="${2:?--config requires a path}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_checkpoint_eval.sh [--config path] [--detach]" >&2
      exit 1
      ;;
  esac
done

OP="$(main/.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('operator','unknown'))")"
GPU_CLASS="$(main/.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('gpu_class','H200'))")"
SUFFIX="$(basename "$CFG" .yaml | sed 's/^checkpoint_eval//; s/^_//')"
TS=$(date +%m-%d-%H%M)
if [[ -n "$SUFFIX" ]]; then
  export CS224R_APP_NAME="cs224r-checkpoint-eval-${SUFFIX}-${OP}-${TS}"
else
  export CS224R_APP_NAME="cs224r-checkpoint-eval-${OP}-${TS}"
fi
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
export CS224R_GPU_CLASS="${GPU_CLASS}"

echo "Launching checkpoint eval config=$CFG app=${CS224R_APP_NAME} gpu=${CS224R_GPU_CLASS}"
exec main/.venv/bin/modal run $DETACH main/probes/checkpoint_rollout_eval.py::run_parallel_eval \
  --config-path "$CFG"
