#!/usr/bin/env bash
# Train on Modal — single config main/configs/train_real.yaml + arm_profiles.
# Difference between smoke and full: step count (env override).
#
# Usage (from repo root):
#   bash main/scripts/launch_train.sh --mode smoke
#   bash main/scripts/launch_train.sh --mode full
#   bash main/scripts/launch_train.sh --mode smoke --gpu-class b200
#   bash main/scripts/launch_train.sh --mode smoke --arm minority_answer
#   bash main/scripts/launch_train.sh --mode full  --arm poly_epo_answer
# Agent copy-paste guide: main/docs/launch_training.md
# Legacy: --config main/configs/train_real_minority_answer.yaml (arm-only shim)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export CS224R_MAIN_ROOT="${REPO_ROOT}/main"
export PYTHONPATH="${CS224R_MAIN_ROOT}"

CFG=""
MODE=""
ARM=""
FRESH_WANDB=""
NO_RESUME=""
CHECKPOINT_DIR=""
RESUME_RUN=""
GPU_CLASS="h200"
CFG_EXPLICIT=0
SMOKE_STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --config)
      CFG="${2:-}"
      CFG_EXPLICIT=1
      shift 2
      ;;
    --arm)
      ARM="${2:-}"
      shift 2
      ;;
    --fresh-wandb)
      FRESH_WANDB="--fresh-wandb"
      shift 1
      ;;
    --no-resume)
      NO_RESUME=1
      shift 1
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="${2:-}"
      shift 2
      ;;
    --resume-run)
      RESUME_RUN="${2:-}"
      shift 2
      ;;
    --gpu-class)
      GPU_CLASS="${2:-}"
      shift 2
      ;;
    --steps)
      SMOKE_STEPS="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_train.sh --mode smoke|full [--gpu-class h200|b200] [--arm <name>] [--config <path>] [--steps N] [--fresh-wandb] [--no-resume] [--checkpoint-dir PATH] [--resume-run RUN_ID]" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash main/scripts/launch_train.sh --mode smoke|full [--gpu-class h200|b200] [--arm <name>] [--config <path>] [--steps N] [--fresh-wandb]" >&2
  exit 1
fi
if [[ -n "$SMOKE_STEPS" && "$MODE" != "smoke" ]]; then
  echo "--steps is only valid with --mode smoke" >&2
  exit 1
fi
if [[ "$GPU_CLASS" != "h200" && "$GPU_CLASS" != "b200" ]]; then
  echo "gpu-class must be one of: h200, b200" >&2
  exit 1
fi
if [[ $CFG_EXPLICIT -eq 0 ]]; then
  if [[ "$GPU_CLASS" == "b200" ]]; then
    CFG="main/configs/train_real_b200.yaml"
  else
    CFG="main/configs/train_real.yaml"
  fi
fi
if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  if [[ "$GPU_CLASS" == "b200" && $CFG_EXPLICIT -eq 0 ]]; then
    echo "Tip: add main/configs/train_real_b200.yaml (or pass --config)." >&2
  fi
  exit 1
fi

if [[ -n "$ARM" ]]; then
  export CS224R_ARM="$ARM"
fi
if [[ -n "$CHECKPOINT_DIR" ]]; then
  export CS224R_CHECKPOINT_DIR="$CHECKPOINT_DIR"
fi

main/.venv/bin/python main/scripts/preflight_train_launch.py \
  --gpu-class "$GPU_CLASS" \
  --config "$CFG" \
  --arm "$ARM" \
  --mode "$MODE"

export CS224R_TRAIN_MODE="$MODE"
if [[ "$MODE" == "smoke" ]]; then
  export CS224R_TOTAL_STEPS="${SMOKE_STEPS:-10}"
else
  unset CS224R_TOTAL_STEPS
fi

TRAIN_OP="$(main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('operator','unknown'))")"
if [[ -n "$ARM" ]]; then
  TRAIN_ARM="$ARM"
elif main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); exit(0 if set(c.keys())<={'arm'} else 1)"; then
  TRAIN_ARM="$(main/.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG'))['arm'])")"
else
  TRAIN_ARM="$(main/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('arm','grpo'))")"
fi
if [[ -n "$RESUME_RUN" ]]; then
  RUN_ID="$RESUME_RUN"
  export CS224R_CHECKPOINT_RUN_ID="$RUN_ID"
  export CS224R_APP_NAME="cs224r-train-${TRAIN_ARM}-${MODE}-${TRAIN_OP}-resume"
else
  TS=$(date +%m-%d-%H%M)
  RUN_ID="cs224r-train-${TRAIN_ARM}-${MODE}-${TRAIN_OP}-${TS}"
  export CS224R_CHECKPOINT_RUN_ID="$RUN_ID"
  export CS224R_APP_NAME="$RUN_ID"
fi
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi

MODAL_LAUNCH_ARGS=()
if [[ "$MODE" == "smoke" ]]; then
  MODAL_LAUNCH_ARGS+=(--launch-mode smoke --total-steps-override 10 --no-resume)
elif [[ -n "$NO_RESUME" ]]; then
  MODAL_LAUNCH_ARGS+=(--no-resume)
fi
if [[ -n "$ARM" ]]; then
  MODAL_LAUNCH_ARGS+=(--arm-override "$ARM")
fi
MODAL_LAUNCH_ARGS+=(--checkpoint-run-id "$RUN_ID")

TRAIN_FN="train_remote_${GPU_CLASS}"
echo "Launching train mode=$MODE gpu_class=$GPU_CLASS fn=$TRAIN_FN arm=${TRAIN_ARM} config=$CFG total_steps=${CS224R_TOTAL_STEPS:-from yaml} app=$CS224R_APP_NAME run_id=$RUN_ID fresh_wandb=${FRESH_WANDB:-no} no_resume=${NO_RESUME:-no} checkpoint_family=${CS224R_CHECKPOINT_DIR:-from config}"
# -q: return after submit; do not stream container logs (allows chaining launches).
main/.venv/bin/modal run --detach -q "main/train/trainer.py::${TRAIN_FN}" \
  --config-path "$CFG" \
  $FRESH_WANDB \
  "${MODAL_LAUNCH_ARGS[@]}"
