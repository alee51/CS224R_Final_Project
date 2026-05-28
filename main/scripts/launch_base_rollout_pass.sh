#!/usr/bin/env bash
# Launch sharded Phase-1 base rollouts over polaris_train.jsonl on Modal B200.
#
# Usage (from repo root):
#   bash main/scripts/launch_base_rollout_pass.sh --num-shards 8 --detach
#   MODAL_PROFILE=anastasia bash main/scripts/launch_base_rollout_pass.sh --num-shards 4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG="main/configs/base_rollout_pass_polaris_51k_b200.yaml"
NUM_SHARDS=8
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
    --num-shards)
      NUM_SHARDS="${2:?--num-shards requires a number}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash main/scripts/launch_base_rollout_pass.sh [--config path] [--num-shards N] [--detach]" >&2
      exit 1
      ;;
  esac
done

OP="$(main/.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('operator','unknown'))")"
GPU_CLASS="$(main/.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('gpu_class','B200'))")"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TS=$(date +%m-%d-%H%M)

export CS224R_RUN_STAMP="$RUN_STAMP"
export CS224R_NUM_SHARDS="$NUM_SHARDS"
export CS224R_GPU_CLASS="$GPU_CLASS"
export CS224R_GIT_SHA="$(git rev-parse HEAD)"
export CS224R_GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  export CS224R_GIT_DIRTY="true"
else
  export CS224R_GIT_DIRTY="false"
fi

LOG_DIR="${REPO_ROOT}/main/docs/probes/artifacts/base_rollout_pass_polaris_51k"
mkdir -p "$LOG_DIR"
LAUNCHED="${LOG_DIR}/launched_${RUN_STAMP}.txt"
: >"$LAUNCHED"

echo "Run stamp: ${RUN_STAMP}"
echo "Shards: ${NUM_SHARDS} x ${GPU_CLASS} (operator=${OP})"
echo "Volume output: probes/base_rollout_pass_polaris_51k/${RUN_STAMP}/shard_XX_of_${NUM_SHARDS}/"
echo "Merge after completion:"
echo "  main/.venv/bin/python3 main/scripts/merge_base_rollout_shards.py --stamp ${RUN_STAMP}"

PIDS=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  (
    export CS224R_SHARD_INDEX="$i"
    export CS224R_APP_NAME="cs224r-base-rollout-51k-s${i}-${OP}-${TS}"
    echo "Launching shard ${i}/${NUM_SHARDS} app=${CS224R_APP_NAME}"
    main/.venv/bin/modal run $DETACH \
      main/probes/group_a_rollout_judge.py::run_phase1_only \
      --config "$CFG"
    echo "shard ${i} ${CS224R_APP_NAME}"
  ) >>"$LAUNCHED" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$((FAILED + 1))
  fi
done

if [[ "$FAILED" -gt 0 ]]; then
  echo "ERROR: ${FAILED}/${NUM_SHARDS} shard launchers failed. See ${LAUNCHED}" >&2
  exit 1
fi

echo "Launched ${NUM_SHARDS} shards. Manifest: ${LAUNCHED}"
