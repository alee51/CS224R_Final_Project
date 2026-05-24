#!/usr/bin/env python3
"""Verify §6 smoke artifacts after pull (local path to one run dir)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="e.g. pilot/artifacts/smoke/20260520T052234Z")
    args = ap.parse_args()
    d = args.run_dir
    failed = 0

    def check(cond: bool, msg: str) -> None:
        global failed
        if cond:
            print(f"  OK  {msg}")
        else:
            print(f"  FAIL  {msg}", file=sys.stderr)
            failed += 1

    print(f"=== Smoke verify: {d} ===\n")

    state = d / "training_state.json"
    ckpt1 = d / "checkpoint_step1"
    diag = d / "step_diagnostics.jsonl"
    preds = d / "raw_predictions.jsonl"

    check(state.is_file(), "training_state.json exists")
    check(ckpt1.is_dir(), "checkpoint_step1/ exists")
    check(diag.is_file(), "step_diagnostics.jsonl exists")

    if preds.is_file():
        n = sum(1 for _ in preds.open() if _.strip())
        check(n == 768, f"raw_predictions.jsonl lines={n} (expected 768 after 3 steps)")
    else:
        check(False, "raw_predictions.jsonl missing")

    if diag.is_file():
        rows = [json.loads(ln) for ln in diag.read_text().splitlines() if ln.strip()]
        step1 = [r for r in rows if r.get("step") == 1 and r.get("phase") == "step_complete"]
        if step1:
            r = step1[0]
            pcr = float(r.get("parser_clean_rate", 0))
            ms = float(r.get("mechanism_signal_per_variant", 0))
            check(pcr >= 0.9, f"parser_clean_rate={pcr:.3f} (need >=0.9)")
            check(ms >= 0.9, f"mechanism_signal_per_variant={ms:.3f} (need >=0.9)")
        else:
            check(False, "no step_complete row for step 1 in diagnostics")

    wandb_dir = d / "wandb"
    check(wandb_dir.is_dir(), "wandb/ offline run dir exists")

    print()
    if failed:
        print(f"Verify FAILED ({failed} check(s))")
        return 1
    print("Artifact checks passed (timing/B6/preempt are manual).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
