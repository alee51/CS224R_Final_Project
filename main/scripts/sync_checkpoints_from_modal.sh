#!/usr/bin/env bash
# Copy training checkpoints between Modal profiles/workspaces via local disk.
#
# Typical flow:
#   1. modal token set --token-id <id> --token-secret <secret> --profile friend --activate
#   2. bash main/scripts/sync_checkpoints_from_modal.sh --source-profile friend --latest 2
#   3. Edit main/configs/checkpoint_eval_2k_polaris_later_b200.yaml checkpoint_steps if needed
#   4. bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_2k_polaris_later_b200.yaml --detach
#
# Usage:
#   bash main/scripts/sync_checkpoints_from_modal.sh --source-profile friend --steps 199,249
#   bash main/scripts/sync_checkpoints_from_modal.sh --source-profile friend --latest 2
#   bash main/scripts/sync_checkpoints_from_modal.sh --steps 199,249   # uses active profile as source

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODAL="${REPO_ROOT}/main/.venv/bin/modal"
VOLUME="${VOLUME:-main-artifacts}"
REMOTE_DIR="${REMOTE_DIR:-checkpoints/train_real}"
DEST_PROFILE=""
SOURCE_PROFILE=""
STEPS=""
LATEST=""
TMPDIR="${TMPDIR:-/tmp/cs224r_ckpt_sync}"

usage() {
  echo "Usage: $0 [--source-profile NAME] [--dest-profile NAME] (--steps N,N | --latest K)" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-profile) SOURCE_PROFILE="${2:?}"; shift 2 ;;
    --dest-profile) DEST_PROFILE="${2:?}"; shift 2 ;;
    --steps) STEPS="${2:?}"; shift 2 ;;
    --latest) LATEST="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

activate_profile() {
  local name="$1"
  if [[ -n "$name" ]]; then
    echo "Activating Modal profile: $name"
    "$MODAL" profile activate "$name"
  fi
}

list_remote_steps() {
  "$MODAL" volume ls "$VOLUME" "$REMOTE_DIR/" 2>/dev/null \
    | rg -o 'step_([0-9]+)\.pt' -r '$1' \
    | sort -n
}

if [[ -z "$DEST_PROFILE" ]]; then
  DEST_PROFILE="$("$MODAL" profile current)"
  echo "Destination profile (current): $DEST_PROFILE"
fi

if [[ -n "$SOURCE_PROFILE" ]]; then
  activate_profile "$SOURCE_PROFILE"
fi

if [[ -n "$LATEST" ]]; then
  mapfile -t ALL_STEPS < <(list_remote_steps)
  if ((${#ALL_STEPS[@]} < LATEST)); then
    echo "Only ${#ALL_STEPS[@]} checkpoint(s) on volume; need --latest $LATEST" >&2
    exit 1
  fi
  STEPS=$(IFS=,; echo "${ALL_STEPS[@]: -LATEST}")
  echo "Latest $LATEST step(s) on source volume: $STEPS"
fi

if [[ -z "$STEPS" ]]; then
  echo "Provide --steps or --latest" >&2
  usage
fi

IFS=',' read -r -a STEP_ARR <<< "$STEPS"
mkdir -p "$TMPDIR"

for step in "${STEP_ARR[@]}"; do
  step="$(echo "$step" | tr -d ' ')"
  remote="${REMOTE_DIR}/step_$(printf '%06d' "$step").pt"
  local="${TMPDIR}/step_$(printf '%06d' "$step").pt"
  echo "Downloading $remote from source profile..."
  activate_profile "$SOURCE_PROFILE"
  "$MODAL" volume get "$VOLUME" "$remote" "$local" --force
  echo "Uploading to dest profile -> $remote"
  activate_profile "$DEST_PROFILE"
  "$MODAL" volume put "$VOLUME" "$local" "$remote" --force
done

echo "Done. Synced steps: ${STEP_ARR[*]}"
echo "Update checkpoint_steps in main/configs/checkpoint_eval_2k_polaris_later_b200.yaml if needed."
