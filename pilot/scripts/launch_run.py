#!/usr/bin/env python3
"""
Orchestrator entry: launch a pilot run with frozen config and budget caps.

Examples:
  python pilot/scripts/launch_run.py --run-id run1_grpo --dry-run
  modal run --detach pilot/infra/modal_app.py --run-id run0_proxy   # production
  modal run pilot/infra/modal_app.py --run-id run0_proxy --wait     # interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot.infra.modal_launch import launch_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch a pilot training run")
    ap.add_argument("--run-id", required=True, help="e.g. run1_grpo, run0_proxy")
    ap.add_argument("--dry-run", action="store_true", help="Print config and cap only")
    ap.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "pilot" / "preflight_lock.json",
        help="Path to preflight_lock.json",
    )
    ap.add_argument(
        "--no-modal",
        action="store_true",
        help="Skip Modal SDK check; invoke train_fn in-process only",
    )
    args = ap.parse_args()

    try:
        launch_run(
            args.run_id,
            dry_run=args.dry_run,
            lock_path=args.lock,
            repo_root=ROOT,
            use_modal=not args.no_modal,
        )
    except NotImplementedError as exc:
        print(f"LAUNCH BLOCKED: {exc}", file=sys.stderr)
        sys.exit(2)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"LAUNCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
