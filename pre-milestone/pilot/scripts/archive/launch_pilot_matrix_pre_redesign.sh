#!/usr/bin/env bash
# ARCHIVED 2026-05-19 — pre–Stage-1 redesign matrix launcher.
#
# Superseded by ../launch_pilot_matrix.sh (three-run GRPO matrix: run1/2/3; run0 waived).
# Kept for reference: old matrix was run1_grpo, run1b_grpo, run2, run3 (no run0;
# preflight caps 36; no modal_run_pilot.sh wrapper).
#
# Do not use for new launches.

set -euo pipefail

MATRIX_RUNS=(run1_grpo run1b_grpo run2_inverse_freq run3_f_grpo)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCK="${REPO_ROOT}/pilot/preflight_lock.json"
LOG_DIR="${REPO_ROOT}/pilot/artifacts/matrix_logs"
MODE="parallel"
DRY_RUN=0

usage() {
  echo "ARCHIVED — use ./pilot/scripts/launch_pilot_matrix.sh"
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequential) MODE="sequential" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
  shift
done

echo "ERROR: This script is archived. Use: ./pilot/scripts/launch_pilot_matrix.sh" >&2
exit 1
