#!/usr/bin/env bash
# Generic Phase 1 eval launcher: one arm × one dataset-shard per Modal app.
#
# Required env vars:
#   CS224R_ARM      one of: base | grpo | minority | polyepo
#   CS224R_SHARD    one of: math500 | smallood
#
# Optional env vars (all have Phase-1 defaults from eval.md):
#   CS224R_EVAL_N_ROLLOUTS    default 64
#   CS224R_EVAL_LOGPROBS      default 20
#   CS224R_EVAL_GPU_COUNT     default 1
#   CS224R_EVAL_OUTPUT_DIR    default /vol/probes/eval_4b
#
# Each (arm, shard) combo runs as its own Modal app on abao. We use --detach
# + .spawn() (see run_eval.py local_entrypoint) so the remote job survives
# any local-side disconnect.
#
# Phase 1 layout: 4 arms × 2 shards = 8 concurrent Modal apps. Stays under
# abao's 10-GPU workspace cap.

set -euo pipefail

cd "$(dirname "$0")/../../.."

: "${CS224R_ARM:?must set CS224R_ARM=base|grpo|minority|polyepo}"
: "${CS224R_SHARD:?must set CS224R_SHARD=math500|smallood}"

NOW=$(date -u '+%H%M')

# ---- Resolve per-arm ckpt + base-mode flag ----
CS224R_EVAL_BASE=0
case "$CS224R_ARM" in
  base)
    CS224R_EVAL_BASE=1
    CS224R_EVAL_CKPT_PATH=""
    ARM_LABEL="base"
    ;;
  grpo)
    CS224R_EVAL_CKPT_PATH="/vol/merged_hf/grpo_step400"
    ARM_LABEL="grpo"
    ;;
  minority)
    CS224R_EVAL_CKPT_PATH="/vol/merged_hf/minority_step400"
    ARM_LABEL="minority"
    ;;
  polyepo)
    CS224R_EVAL_CKPT_PATH="/vol/merged_hf/polyepo_step400"
    ARM_LABEL="polyepo"
    ;;
  *)
    echo "ERROR: unknown CS224R_ARM=$CS224R_ARM" >&2
    exit 2
    ;;
esac

# ---- Resolve per-shard dataset list ----
case "$CS224R_SHARD" in
  math500)
    CS224R_EVAL_DATASETS="math500"
    ;;
  smallood)
    CS224R_EVAL_DATASETS="aime25,aime26,hmmt_feb25,hmmt_nov25,beyondaime"
    ;;
  *)
    echo "ERROR: unknown CS224R_SHARD=$CS224R_SHARD" >&2
    exit 2
    ;;
esac

# Per-(arm, shard) label so output JSONs don't collide:
#   <arm>_step400_math500.json   vs   <arm>_step400_smallood.json
CS224R_EVAL_LABEL="${ARM_LABEL}_step400_${CS224R_SHARD}"

MODAL_PROFILE="${MODAL_PROFILE:-abao}" \
CS224R_APP_NAME="cs224r-eval-${ARM_LABEL}-step400-${CS224R_SHARD}-${NOW}" \
CS224R_EVAL_BASE="$CS224R_EVAL_BASE" \
CS224R_EVAL_CKPT_PATH="$CS224R_EVAL_CKPT_PATH" \
CS224R_EVAL_LABEL="$CS224R_EVAL_LABEL" \
CS224R_EVAL_DATASETS="$CS224R_EVAL_DATASETS" \
CS224R_EVAL_N_ROLLOUTS="${CS224R_EVAL_N_ROLLOUTS:-64}" \
CS224R_EVAL_LOGPROBS="${CS224R_EVAL_LOGPROBS:-20}" \
CS224R_EVAL_OUTPUT_DIR="${CS224R_EVAL_OUTPUT_DIR:-/vol/probes/eval_4b}" \
CS224R_EVAL_GPU_COUNT="${CS224R_EVAL_GPU_COUNT:-1}" \
PYTHONPATH=main-verl \
modal run --detach main-verl/eval/run_eval.py
