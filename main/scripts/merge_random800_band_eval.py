#!/usr/bin/env python3
"""Merge random800 base grading (local) with Modal checkpoint eval (3 arms)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BANDS = [f"{k}/8" for k in range(8)]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _base_from_grading(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary.get("overall", summary)
    out: dict[str, Any] = {
        "label": "base",
        "checkpoint": "base",
        "n_prompts": overall.get("n_prompts"),
        "pass_at_8_mean": overall.get("pass_at_8_mean"),
        "mean_reward": overall.get("pass_at_1"),  # grading summary uses pass@1 as rollout acc
        "by_band": {},
        "source": "local_phase1_rollouts",
        "n_rollouts_note": summary.get("n_rollouts_graded"),
    }
    for band, stats in summary.get("by_band", {}).items():
        out["by_band"][band] = {
            "n_prompts": stats.get("n_prompts"),
            "pass_at_8_mean": stats.get("pass_at_8_mean"),
        }
    return out


def _arm_from_modal(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row["label"],
        "checkpoint": row.get("checkpoint"),
        "n_prompts": row.get("n_prompts"),
        "pass_at_8_mean": row.get("pass_at_8_mean"),
        "mean_reward": row.get("mean_reward"),
        "by_band": row.get("by_band", {}),
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    base_pass = results[0].get("pass_at_8_mean") or 0.0
    print("\n=== Overall pass@8 ===")
    for r in results:
        p = r.get("pass_at_8_mean") or 0.0
        delta = p - base_pass if r["label"] != "base" else 0.0
        note = "" if r["label"] == "base" else f"  ({delta:+.4f} vs base)"
        n = r.get("n_prompts", "?")
        print(f"  {r['label']:<24} n={n}  pass@8={p:.4f}{note}")

    print("\n=== By difficulty_band (pass@8) ===")
    header = f"{'band':<6}" + "".join(f"{r['label'][:12]:>14}" for r in results)
    print(header)
    for band in BANDS:
        cells = [f"{band:<6}"]
        base_b = (results[0].get("by_band") or {}).get(band, {})
        base_p = base_b.get("pass_at_8_mean")
        for r in results:
            b = (r.get("by_band") or {}).get(band, {})
            p = b.get("pass_at_8_mean")
            n = b.get("n_prompts", 0)
            if p is None:
                cells.append(f"{'—':>14}")
            elif r["label"] == "base":
                cells.append(f"{p:.3f} (n={n})"[:14].rjust(14))
            else:
                d = (p - base_p) if base_p is not None else 0.0
                cells.append(f"{p:.3f} ({d:+.3f})"[:14].rjust(14))
        print("".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal-summary", type=Path, required=True)
    ap.add_argument(
        "--base-summary",
        type=Path,
        default=Path("main/data/probes/05-27/random_fullgold_n800/grading_summary.json"),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    modal = _load(args.modal_summary)
    rows = modal.get("results", modal)
    if isinstance(rows, dict):
        rows = rows.get("polaris_random800", rows)
        if isinstance(rows, dict) and "results" in rows:
            rows = rows["results"]

    base_summary = _load(args.base_summary)
    results = [_base_from_grading(base_summary)]
    results.extend(_arm_from_modal(r) for r in rows)

    payload = {
        "manifest": "random_fullgold_n800",
        "base_note": "480/800 prompts (phase1_rollouts partial); arms run full 800 on Modal",
        "results": results,
    }
    out = args.out or args.modal_summary.with_name("merged_band_summary.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    _print_table(results)


if __name__ == "__main__":
    main()
