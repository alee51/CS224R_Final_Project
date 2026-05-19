#!/usr/bin/env python3
"""Pull completed run artifacts from Modal volume into a local timestamped dir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot.infra.artifacts import REQUIRED_ARTIFACTS, link_latest_run  # noqa: E402
from pilot.infra.modal_volumes import pull_run_artifacts_from_volume  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull pilot run artifacts from Modal volume")
    ap.add_argument("--run-id", required=True, help="e.g. run0_proxy, run1_grpo")
    ap.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local timestamped dir printed at launch (pilot/artifacts/<run_id>/<UTC>)",
    )
    args = ap.parse_args()

    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pulling volume:pilot-artifacts/{args.run_id}/ -> {local_dir}")
    pull_run_artifacts_from_volume(args.run_id, local_dir)
    latest = link_latest_run(args.run_id, local_dir, artifacts_root=ROOT / "pilot" / "artifacts")
    print(f"Latest symlink: {latest} -> {local_dir.name}")

    missing = [name for name in REQUIRED_ARTIFACTS if not (local_dir / name).exists()]
    if missing:
        print(f"WARNING: missing after pull: {missing}", file=sys.stderr)
        sys.exit(1)

    metrics_path = local_dir / "metrics.json"
    if metrics_path.exists():
        print(metrics_path.read_text())
    print(f"OK: {local_dir}")


if __name__ == "__main__":
    main()
