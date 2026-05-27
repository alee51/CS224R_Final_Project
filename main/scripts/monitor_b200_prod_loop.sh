#!/usr/bin/env bash
# Background monitor: polls B200 prod runs, emits AGENT_LOOP_WAKE for agent continuation.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

PROMPT='Run bash main/scripts/monitor_b200_prod.sh. If exit 2 and simple crash, relaunch with --relaunch. Update user on milestones 10 and 20 and hourly after. Read main/docs/probes/artifacts/b200_prod_monitor/state.json.'

while true; do
  SECS=300
  if [[ -f main/docs/probes/artifacts/b200_prod_monitor/state.json ]]; then
    SECS="$(main/.venv/bin/python -c "
import json
from pathlib import Path
p = Path('main/docs/probes/artifacts/b200_prod_monitor/state.json')
d = json.loads(p.read_text())
print(d.get('last_report', {}).get('next_poll_seconds', 300))
")"
  fi
  sleep "$SECS"
  main/.venv/bin/python -c "import json; print('AGENT_LOOP_WAKE_B200_PROD', json.dumps({'prompt': '''$PROMPT'''}))"
done
