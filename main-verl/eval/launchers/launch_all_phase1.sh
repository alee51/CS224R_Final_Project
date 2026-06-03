#!/usr/bin/env bash
# Phase 1 launcher: fire all 4 arms × 2 shards = 8 Modal apps in parallel
# on abao. Each app uses `modal run --detach` + .spawn() so it survives any
# local-side disconnect.
#
# Layout per eval_build.md:
#   - shard "math500"  = MATH-500 only (500 prompts; the long pole)
#   - shard "smallood" = aime25, aime26, hmmt_feb25, hmmt_nov25, beyondaime
#                        (5 datasets, 220 prompts total)
#
# Output JSON labels are <arm>_step400_<shard>.json so the two shards never
# overwrite each other. 8 concurrent GPUs sits under abao's 10-GPU cap.
#
# Usage:
#   bash main-verl/eval/launchers/launch_all_phase1.sh
#
# Run from anywhere — the underlying run.sh cd's to repo root.
#
# Each `modal run` call only blocks until the .spawn() returns the call_id,
# then exits. So this script wraps up quickly even if all 8 jobs run for
# hours in the cloud.

set -euo pipefail

LAUNCHER_DIR="$(cd "$(dirname "$0")" && pwd)"

ARMS=(base grpo minority polyepo)
SHARDS=(math500 smallood)

echo "=== Phase 1 launch: ${#ARMS[@]} arms × ${#SHARDS[@]} shards = $((${#ARMS[@]} * ${#SHARDS[@]})) Modal apps ==="
for arm in "${ARMS[@]}"; do
  for shard in "${SHARDS[@]}"; do
    echo "--- spawning arm=$arm shard=$shard ---"
    CS224R_ARM="$arm" CS224R_SHARD="$shard" \
      bash "$LAUNCHER_DIR/run.sh"
  done
done
echo "=== all 8 Phase 1 Modal apps spawned ==="
echo "Watch with: modal app list --profile abao"
echo "Outputs land on abao's main-artifacts volume at /vol/probes/eval_4b/<arm>_step400_<shard>.json"
