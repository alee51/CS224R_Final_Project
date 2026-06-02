#!/usr/bin/env bash
# Eval the minority_cot step-400 ckpt on the full panel.
# Run from repo root: bash main-verl/eval/launchers/minority.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."
NOW=$(date -u '+%H%M')

MODAL_PROFILE=emma \
CS224R_APP_NAME=cs224r-eval-minority-step400-${NOW} \
CS224R_EVAL_CKPT_PATH=/vol/checkpoints/main-verl/minority_cot_train_4b_1epoch_lr3e6/global_step_400/actor \
CS224R_EVAL_LABEL=minority_step400 \
CS224R_EVAL_DATASETS=aime25,polaris_val,math500,hmmt_feb25,hmmt_nov25,beyondaime \
CS224R_EVAL_N_ROLLOUTS=16 \
CS224R_EVAL_OUTPUT_DIR=/vol/probes/eval_4b \
CS224R_EVAL_GPU_COUNT=1 \
PYTHONPATH=main-verl \
modal run --detach main-verl/eval/run_eval.py
