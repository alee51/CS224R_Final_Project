#!/usr/bin/env bash
# Launch checkpoint+resume smoke in two phases:
#   phase 1: fresh 10-step smoke (no resume) to force ckpt at step 9
#   phase 2: resume smoke to at least step 10 from that checkpoint
#
# Usage:
#   bash main/scripts/launch_smoke_ckpt_resume.sh --arm grpo --gpu-class h200
#   bash main/scripts/launch_smoke_ckpt_resume.sh --arm minority_answer --gpu-class b200
#
# After phase 1 finishes, rerun with:
#   --phase resume
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARM=""
GPU_CLASS="h200"
PHASE="fresh"
CFG=""
CFG_EXPLICIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)
      ARM="${2:-}"
      shift 2
      ;;
    --gpu-class)
      GPU_CLASS="${2:-}"
      shift 2
      ;;
    --phase)
      PHASE="${2:-}"
      shift 2
      ;;
    --config)
      CFG="${2:-}"
      CFG_EXPLICIT=1
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_smoke_ckpt_resume.sh --arm <grpo|minority_answer> [--gpu-class h200|b200] [--phase fresh|resume] [--config <path>]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ARM" ]]; then
  echo "--arm is required (e.g. grpo or minority_answer)" >&2
  exit 1
fi
if [[ "$GPU_CLASS" != "h200" && "$GPU_CLASS" != "b200" ]]; then
  echo "--gpu-class must be h200 or b200" >&2
  exit 1
fi
if [[ "$PHASE" != "fresh" && "$PHASE" != "resume" ]]; then
  echo "--phase must be fresh or resume" >&2
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
  exit 1
fi

if [[ "$PHASE" == "fresh" ]]; then
  echo "Launching phase=fresh arm=$ARM gpu=$GPU_CLASS (10 steps, no resume; expect checkpoint at step 9)"
  bash main/scripts/launch_train.sh \
    --mode smoke \
    --gpu-class "$GPU_CLASS" \
    --arm "$ARM" \
    --config "$CFG" \
    --fresh-wandb

  echo
  echo "When phase=fresh finishes, run:"
  echo "  bash main/scripts/launch_smoke_ckpt_resume.sh --arm $ARM --gpu-class $GPU_CLASS --phase resume --config $CFG"
  exit 0
fi

echo "Launching phase=resume arm=$ARM gpu=$GPU_CLASS (resume from step_000009.pt to at least step 10)"
export CS224R_TRAIN_MODE="smoke"
export CS224R_TOTAL_STEPS=11
unset CS224R_NO_RESUME

main/.venv/bin/modal run --detach "main/train/trainer.py::train_remote_${GPU_CLASS}" \
  --config-path "$CFG" \
  --launch-mode smoke \
  --total-steps-override 11 \
  --arm-override "$ARM"
