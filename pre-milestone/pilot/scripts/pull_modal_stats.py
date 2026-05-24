#!/usr/bin/env python3
"""Pull Modal billing + container stats for pilot matrix apps.

Usage (repo root, venv active):
  python pilot/scripts/pull_modal_stats.py
  python pilot/scripts/pull_modal_stats.py --start 2026-05-19 --hours 6
  python pilot/scripts/pull_modal_stats.py --out pilot/artifacts/matrix_logs/modal_stats.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import modal.billing as modal_billing

# From Modal pricing (~$2.50/hr A100-80GB); used only to estimate GPU-hours from cost.
A100_80GB_USD_PER_HR = 2.50

MATRIX_APPS: dict[str, str] = {
    "run0_proxy": "ap-Zk6zAIs9tWpGerJHufSud1",
    "run1_grpo": "ap-CpcEIWjwiNMb8MvGCZFpAT",
    "run1b_grpo": "ap-EWhmIPbGpflmnM2IcrKp77",
    "run2_inverse_freq": "ap-aAYroxfDF3TuZY5NJ1pbOP",
    "run3_f_grpo": "ap-MO0JD72gMTybU9Sv7VCSrn",
}


def _json_default(obj: object) -> object:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _containers() -> list[dict]:
    raw = subprocess.check_output(
        ["modal", "container", "list", "--json"],
        text=True,
    )
    return json.loads(raw)


def _preemption_lines(app_id: str, max_lines: int = 5000) -> list[str]:
    try:
        logs = subprocess.check_output(
            ["modal", "app", "logs", app_id],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        return [f"(logs error: {exc.output[:200]})"]
    hits: list[str] = []
    for line in logs.splitlines()[-max_lines:]:
        low = line.lower()
        if "preempt" in low or "runner interrupted" in low:
            hits.append(line.strip())
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="UTC start date YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours after start to include (default: 24)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report here",
    )
    args = parser.parse_args()

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=args.hours)

    items = modal_billing.workspace_billing_report(
        start=start,
        end=end,
        resolution="h",
    )
    app_ids = set(MATRIX_APPS.values())
    billing_rows = [r for r in items if r["object_id"] in app_ids]

    inv = {v: k for k, v in MATRIX_APPS.items()}
    by_run: dict[str, dict] = {}
    for row in billing_rows:
        run_id = inv[row["object_id"]]
        entry = by_run.setdefault(
            run_id,
            {
                "app_id": row["object_id"],
                "hourly": [],
                "cost_usd_total": 0.0,
                "est_gpu_hours_total": 0.0,
            },
        )
        cost = float(row["cost"])
        est_h = cost / A100_80GB_USD_PER_HR
        entry["hourly"].append(
            {
                "interval_start_utc": row["interval_start"],
                "cost_usd": cost,
                "est_gpu_hours": round(est_h, 3),
                "est_avg_gpus_in_hour": round(est_h, 2),
            }
        )
        entry["cost_usd_total"] += cost
        entry["est_gpu_hours_total"] += est_h

    containers = [c for c in _containers() if c.get("App ID") in app_ids]
    for run_id, app_id in MATRIX_APPS.items():
        if run_id in by_run:
            by_run[run_id]["containers"] = [
                c for c in containers if c.get("App ID") == app_id
            ]
            by_run[run_id]["preemption_log_lines"] = _preemption_lines(app_id)

    report = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"start_utc": start.isoformat(), "end_utc": end.isoformat()},
        "note": (
            "Modal billing API returns cost per app per hour, not raw GPU count. "
            "est_gpu_hours = cost / A100_80GB_USD_PER_HR. Values >1.0 in an hour "
            "usually mean two containers billed briefly (e.g. preemption restart)."
        ),
        "runs": by_run,
    }

    text = json.dumps(report, indent=2, default=_json_default)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nWrote {args.out}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
