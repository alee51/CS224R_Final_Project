#!/usr/bin/env bash
# Monitor B200 fresh GRPO + minority prod runs (W&B + Modal).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/main"
main/.venv/bin/python main/scripts/monitor_b200_prod.py "$@"
