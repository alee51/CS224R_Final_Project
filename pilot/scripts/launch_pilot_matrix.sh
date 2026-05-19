#!/usr/bin/env bash
# Launch Run1–Run3 overnight matrix on Modal (independent jobs, parallel by default).
#
# Run0 (run0_proxy) is separate — proxy rollouts only, not part of this matrix.
#
# Usage (from repo root):
#   ./pilot/scripts/launch_pilot_matrix.sh              # parallel (4 Modal jobs)
#   ./pilot/scripts/launch_pilot_matrix.sh --sequential   # one process, --run-ids list
#   ./pilot/scripts/launch_pilot_matrix.sh --dry-run      # print caps + commands only

set -euo pipefail

MATRIX_RUNS=(run1_grpo run1b_grpo run2_inverse_freq run3_f_grpo)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${REPO_ROOT}/pilot/preflight_lock.json"
LOG_DIR="${REPO_ROOT}/pilot/artifacts/matrix_logs"
MODE="parallel"
DRY_RUN=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Options:"
  echo "  --sequential   Single modal process: --run-ids run1_grpo,..."
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

echo "=== Pilot overnight matrix (Run1–Run3) ==="
echo "Repo: $REPO_ROOT"
echo "Mode: $MODE"
echo "Note: run0_proxy is NOT included (launch separately for proxy rollouts)."
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
PY
echo ""
echo "Runs are independent (different objectives/seeds) and may run in parallel."
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN — commands that would run:"
  if [[ "$MODE" == "parallel" ]]; then
    for rid in "${MATRIX_RUNS[@]}"; do
      echo "  modal run pilot/infra/modal_app.py --run-id $rid"
    done
    echo "  (each in background; logs under pilot/artifacts/matrix_logs/)"
  else
    ids_csv=$(IFS=,; echo "${MATRIX_RUNS[*]}")
    echo "  modal run pilot/infra/modal_app.py --run-ids $ids_csv"
  fi
  exit 0
fi

if [[ "$MODE" == "sequential" ]]; then
  ids_csv=$(IFS=,; echo "${MATRIX_RUNS[*]}")
  echo "Launching sequential matrix in one Modal local entrypoint..."
  exec modal run pilot/infra/modal_app.py --run-ids "$ids_csv"
fi

mkdir -p "$LOG_DIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
echo "Launching ${#MATRIX_RUNS[@]} parallel Modal jobs (logs: $LOG_DIR/${stamp}_*.log)"
pids=()
for rid in "${MATRIX_RUNS[@]}"; do
  log="${LOG_DIR}/${stamp}_${rid}.log"
  echo "  $rid -> $log"
  modal run pilot/infra/modal_app.py --run-id "$rid" >"$log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!MATRIX_RUNS[@]}"; do
  rid="${MATRIX_RUNS[$i]}"
  pid="${pids[$i]}"
  if wait "$pid"; then
    echo "OK: $rid"
  else
    echo "FAILED: $rid (see ${LOG_DIR}/${stamp}_${rid}.log)" >&2
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "$failed / ${#MATRIX_RUNS[@]} matrix runs failed." >&2
  exit 1
fi
echo "All matrix runs finished. Check pilot/artifacts/<run_id>/latest per run."
