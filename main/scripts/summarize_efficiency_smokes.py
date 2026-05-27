#!/usr/bin/env python3
"""Summarize wandb efficiency smokes (step time + VRAM ROI)."""

from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path

METRICS = [
    "train/t_rollout_s",
    "train/t_train_fwd_bwd_s",
    "train/t_logprob_fwd_s",
    "train/t_backward_s",
    "train/vram_peak_gb_step",
    "train/num_chunks",
    "train/n_kept_sequences",
]


def _med_p95(series: list[float]) -> tuple[float, float]:
    if not series:
        return float("nan"), float("nan")
    s = sorted(series)
    med = st.median(s)
    p95 = s[int(0.95 * (len(s) - 1))]
    return med, p95


def summarize_run(run_id: str, entity: str, project: str) -> dict[str, tuple[float, float]]:
    import wandb

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    hist = run.history(samples=500)
    out: dict[str, tuple[float, float]] = {}
    for key in METRICS:
        if key not in hist.columns:
            continue
        vals = [float(x) for x in hist[key].dropna().tolist()]
        out[key] = _med_p95(vals)
    step_med = float("nan")
    if "train/t_rollout_s" in out and "train/t_train_fwd_bwd_s" in out:
        tr, _ = out["train/t_rollout_s"]
        tt, _ = out["train/t_train_fwd_bwd_s"]
        if tr == tr and tt == tt:
            step_med = tr + tt
    out["_step_time_est"] = (step_med, step_med)
    return {
        "name": run.name,
        "state": run.state,
        "metrics": out,
        "run_id": run_id,
        "url": run.url,
    }


def _parse_manifest_line(line: str) -> tuple[str, str] | None:
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    label = parts[0]
    app = parts[1]
    return label, app


def _find_run_by_app(app_slug: str, entity: str, project: str) -> str | None:
    import wandb

    api = wandb.Api()
    # App name embeds b200eff-{label}; match recent runs.
    needle = app_slug.split("/")[-1] if "/" in app_slug else app_slug
    for run in api.runs(f"{entity}/{project}", order="-created_at", per_page=40):
        if needle in (run.name or "") or needle in str(run.tags):
            return run.id
        notes = getattr(run, "notes", "") or ""
        if needle in notes:
            return run.id
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "manifest",
        nargs="?",
        help="launched.txt from launch_b200_efficiency_smokes.sh",
    )
    ap.add_argument("--baseline", default="wdl3fczm", help="wandb run id for baseline")
    ap.add_argument("--entity", default="224r-project")
    ap.add_argument("--project", default="cs224r-minority-voting")
    ap.add_argument("--run-id", action="append", dest="run_ids", help="explicit run id")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.run_ids:
        for rid in args.run_ids:
            rows.append(summarize_run(rid, args.entity, args.project))
    elif args.manifest:
        path = Path(args.manifest)
        for line in path.read_text().splitlines():
            parsed = _parse_manifest_line(line)
            if not parsed:
                continue
            label, app = parsed
            rid = _find_run_by_app(app, args.entity, args.project)
            if not rid:
                print(f"skip {label}: no wandb run yet for {app}", file=sys.stderr)
                continue
            row = summarize_run(rid, args.entity, args.project)
            row["label"] = label
            rows.append(row)
    else:
        rows.append(summarize_run(args.baseline, args.entity, args.project))
        rows[0]["label"] = "baseline"

    base = summarize_run(args.baseline, args.entity, args.project)
    b_step, _ = base["metrics"].get("_step_time_est", (float("nan"), float("nan")))
    b_vram, _ = base["metrics"].get("train/vram_peak_gb_step", (float("nan"), float("nan")))

    print(f"Baseline {args.baseline}: step≈{b_step:.0f}s vram≈{b_vram:.1f}GB\n")
    for row in rows:
        label = row.get("label", row["run_id"])
        m = row["metrics"]
        step, _ = m.get("_step_time_est", (float("nan"), float("nan")))
        vram, _ = m.get("train/vram_peak_gb_step", (float("nan"), float("nan")))
        dt = b_step - step if step == step else float("nan")
        dv = vram - b_vram if vram == vram else float("nan")
        roi = dt / dv if dv and abs(dv) > 0.01 else float("nan")
        print(f"## {label} ({row['run_id']}) {row['state']}")
        print(f"   {row['url']}")
        print(f"   step≈{step:.0f}s (Δ{b_step - step:+.0f}s)  vram≈{vram:.1f}GB (Δ{dv:+.1f}GB)  ROI≈{roi:.1f}s/GB")
        for key in METRICS:
            if key in m:
                med, p95 = m[key]
                print(f"   {key}: med={med:.1f} p95={p95:.1f}")
        print()


if __name__ == "__main__":
    main()
