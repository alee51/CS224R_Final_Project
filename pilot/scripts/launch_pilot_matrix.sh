#!/usr/bin/env bash
# Stage-1 pilot matrix — three detached Modal jobs (PILOT_REDESIGN.md §3).
#
# Runs: run1_grpo + run2_inverse_freq + run3_f_grpo (GRPO matrix only).
# Run 0 waived — use pre-redesign artifacts (see pilot/docs/decisions/20260519_skip_run0_stage1_redesign.md).
# All use seed 42 in configs; single seed, parallel by default.
#
# Usage (from repo root):
#   ./pilot/scripts/launch_pilot_matrix.sh
#   ./pilot/scripts/launch_pilot_matrix.sh --sequential
#   ./pilot/scripts/launch_pilot_matrix.sh --dry-run
#
# Launches via ./pilot/scripts/modal_run_pilot.sh (--detach by default).

set -euo pipefail

MATRIX_RUNS=(run1_grpo run2_inverse_freq run3_f_grpo)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${REPO_ROOT}/pilot/preflight_lock.json"
LOG_DIR="${REPO_ROOT}/pilot/artifacts/matrix_logs"
MODAL_RUN="${REPO_ROOT}/pilot/scripts/modal_run_pilot.sh"
MODE="parallel"
DRY_RUN=0

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Options:"
  echo "  --sequential   Spawn one run after another (still detached)"
  echo "  --dry-run      Print budget caps and commands; do not launch"
  echo "  -h, --help     Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequential) MODE="sequential" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$REPO_ROOT"

if [[ ! -x "$MODAL_RUN" ]]; then
  chmod +x "$MODAL_RUN"
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Activating ${REPO_ROOT}/.venv ..."
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
  else
    echo "ERROR: No active venv. Run: source .venv/bin/activate" >&2
    exit 1
  fi
fi

if ! command -v modal >/dev/null 2>&1; then
  echo "ERROR: modal CLI not found. pip install -r pilot/requirements.txt" >&2
  exit 1
fi

if [[ ! -f "$LOCK" ]]; then
  echo "ERROR: missing preflight lock: $LOCK" >&2
  exit 1
fi

echo "=== Pilot Stage-1 matrix (3 runs) ==="
echo "Repo: $REPO_ROOT"
echo "Mode: $MODE"
echo "Launcher: $MODAL_RUN (detach default; --wait opt-in on wrapper only)"
echo ""
echo "Budget caps (from preflight_lock.json):"
python3 - "$LOCK" "${MATRIX_RUNS[@]}" <<'PY'
import json
import sys

lock_path = sys.argv[1]
run_ids = sys.argv[2:]
lock = json.loads(open(lock_path).read())
caps = lock.get("budget_caps_usd", {})
pilot_total = caps.get("pilot_total")
matrix_sum = 0.0
for rid in run_ids:
    cap = float(caps[rid])
    hard = 1.5 * cap
    matrix_sum += cap
    print(f"  {rid}: cap ${cap:.0f}  |  hard abort (1.5×) ${hard:.0f}")
print(f"  matrix subtotal (these {len(run_ids)} runs): ${matrix_sum:.0f}")
if pilot_total is not None:
    print(f"  pilot_total ceiling: ${float(pilot_total):.0f}")
    if matrix_sum > float(pilot_total):
        print(f"  WARNING: matrix subtotal exceeds pilot_total by ${matrix_sum - float(pilot_total):.0f}")
PY
echo ""
echo "Runs are independent (objectives differ); parallel spawn is default."
echo "Monitor: modal app list  |  logs: modal app logs <app-id>"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN — commands that would run:"
  for rid in "${MATRIX_RUNS[@]}"; do
    echo "  $MODAL_RUN --run-id $rid"
  done
  if [[ "$MODE" == "parallel" ]]; then
    echo "  (parallel mode: each in background; logs under pilot/artifacts/matrix_logs/)"
  fi
  exit 0
fi

if [[ "$MODE" == "sequential" ]]; then
  echo "Launching sequential matrix (detached spawn per run)..."
  for rid in "${MATRIX_RUNS[@]}"; do
    echo "=== $rid ==="
    "$MODAL_RUN" --run-id "$rid"
  done
  echo "All matrix runs spawned. Monitor with: modal app list"
  exit 0
fi

mkdir -p "$LOG_DIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
echo "Launching ${#MATRIX_RUNS[@]} detached Modal spawns (logs: $LOG_DIR/${stamp}_*.log)"
pids=()
for rid in "${MATRIX_RUNS[@]}"; do
  log="${LOG_DIR}/${stamp}_${rid}.log"
  echo "  $rid -> $log"
  "$MODAL_RUN" --run-id "$rid" >"$log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!MATRIX_RUNS[@]}"; do
  rid="${MATRIX_RUNS[$i]}"
  pid="${pids[$i]}"
  if wait "$pid"; then
    echo "Spawned: $rid (see ${LOG_DIR}/${stamp}_${rid}.log for call id + local dir)"
  else
    echo "SPAWN FAILED: $rid (see ${LOG_DIR}/${stamp}_${rid}.log)" >&2
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "$failed / ${#MATRIX_RUNS[@]} matrix spawns failed." >&2
  exit 1
fi
echo "All matrix runs spawned on Modal (GPU work continues if laptop disconnects)."
echo "Monitor: modal app list"
echo "Pull when done: python pilot/scripts/pull_run_artifacts.py --run-id <id> --local-dir <dir>"
