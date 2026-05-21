#!/usr/bin/env bash
# Pilot Modal launcher — always detached unless --wait or PILOT_MODAL_ATTACH=1.
#
# Usage (from repo root):
#   ./pilot/scripts/modal_run_pilot.sh --run-id smoke
#   ./pilot/scripts/modal_run_pilot.sh --run-id run1_grpo --debug-max-prompts 5
#   ./pilot/scripts/modal_run_pilot.sh --run-id run1_grpo --wait   # interactive; laptop must stay on
#
# Set PILOT_MODAL_ATTACH=1 to run without --detach (debug only; do not use for overnight jobs).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

APP="pilot/infra/modal_app.py"
WAIT=0
FORWARD=()

for arg in "$@"; do
  if [[ "$arg" == "--wait" ]]; then
    WAIT=1
  else
    FORWARD+=("$arg")
  fi
done

if [[ "$WAIT" -eq 1 ]] || [[ "${PILOT_MODAL_ATTACH:-}" == "1" ]]; then
  exec modal run "$APP" "${FORWARD[@]}"
fi

exec modal run --detach "$APP" "${FORWARD[@]}"
