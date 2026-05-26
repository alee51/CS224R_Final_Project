#!/usr/bin/env python3
"""Rewind a wandb run's history past a given step.

Wandb's `wandb.log({...}, step=N)` silently drops writes when N <= the run's
current global step. After we stop a Modal training leg and resume from a
checkpoint at step K, the live wandb run may have logged through K+M; the
resumed trainer's logs at K+1..K+M will be lost unless we rewind first.

Usage:
  python main/scripts/wandb_rewind.py --run-id 8qesa78k --step 149
  python main/scripts/wandb_rewind.py --run-id 8qesa78k --step 149 \\
      --project cs224r-minority-voting --entity 224r-project

The rewind uses wandb's `resume_from` parameter, which truncates history to
steps strictly less than the target. Requires wandb >= 0.17 (and the wandb
plan that supports rewind).

This is a one-shot operation. Do not bake it into trainer.py — auto-resume
after a crash should preserve history, not delete it.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="wandb run id, e.g. 8qesa78k")
    ap.add_argument(
        "--step",
        type=int,
        required=True,
        help="Rewind so history < step is preserved; >= step is deleted.",
    )
    ap.add_argument("--entity", default="224r-project")
    ap.add_argument("--project", default="cs224r-minority-voting")
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    full = f"{args.entity}/{args.project}/{args.run_id}"
    try:
        existing = api.run(full)
    except Exception as exc:
        print(f"ERROR: could not fetch run {full}: {exc}", file=sys.stderr)
        return 2

    cur_step = existing.summary.get("_step")
    print(
        f"Run {full}: state={existing.state}, current _step={cur_step}, "
        f"target rewind step={args.step}"
    )
    if existing.state == "running":
        print(
            "WARNING: run is still 'running' in wandb. Stop the producer "
            "(modal app stop ...) and wait ~30s before rewinding.",
            file=sys.stderr,
        )

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume_from=f"{args.run_id}?_step={args.step}",
    )
    print(f"Rewound to step {args.step}. New run handle: {run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
