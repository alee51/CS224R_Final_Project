"""Backfill `production` / `smoke` tags on existing wandb runs.

Classification rules:
    - probe-* group: ensure `probe` tag; if lastHistoryStep < 50, also `smoke`.
    - train-real group + name contains "ablate": ensure `smoke` tag.
    - train-real group + name no "ablate" + lastHistoryStep < 50: ensure `smoke`.
    - train-real group + name no "ablate" + lastHistoryStep >= 50: ensure `production`.

Idempotent: only adds tags that are missing. Pass --apply to actually write.
"""

import argparse
import wandb

ENTITY = "224r-project"
PROJECT = "cs224r-minority-voting"
PROD_STEP_THRESHOLD = 50


def classify(run) -> set[str]:
    """Return tags this run *should* have (beyond existing)."""
    needed: set[str] = set()
    group = run.group or ""
    name = run.name or ""
    steps = run.lastHistoryStep or 0

    if group.startswith("probe-"):
        needed.add("probe")
        if steps < PROD_STEP_THRESHOLD:
            needed.add("smoke")
    elif group == "train-real":
        if "ablate" in name:
            needed.add("smoke")
        elif steps < PROD_STEP_THRESHOLD:
            needed.add("smoke")
        else:
            needed.add("production")
    return needed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually update tags")
    args = ap.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", per_page=200)

    changes: list[tuple[str, str, list[str]]] = []
    for r in runs:
        should = classify(r)
        existing = set(r.tags or [])
        missing = should - existing
        if missing:
            changes.append((r.id, r.name, sorted(missing)))

    print(f"{len(changes)} runs need tag updates:")
    for rid, name, miss in changes:
        print(f"  {rid:12} {name:45}  + {miss}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return

    for rid, name, miss in changes:
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        run.tags = sorted(set(run.tags or []) | set(miss))
        run.update()
        print(f"  ✓ {rid} {name} → +{miss}")
    print(f"\nApplied to {len(changes)} runs.")


if __name__ == "__main__":
    main()
