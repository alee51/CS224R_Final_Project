#!/usr/bin/env python3
"""Export human review queue as CSV (Numbers / Excel).

Reads labels/rollout_labels.jsonl, keeps rows with needs_human=true, joins
problem + tail from chunks/chunk_KKK_in.tsv.

Usage:
  python export_human_review_csv.py
  python export_human_review_csv.py -o labels/human_review_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from label_paths import ANALYSIS_ROOT, LABELING_ROOT, ROLLOUT_LABELS

CHUNKS = LABELING_ROOT / "chunks"
DEFAULT_OUT = LABELING_ROOT / "labels_archive" / "human_review_queue.csv"

FIELDS = [
    "seq",
    "chunk",
    "id",
    "rollout_key",
    "gold",
    "problem",
    "tail",
    "result_A",
    "result_B",
    "agreed",
    "review_reason",
    "human_result",
    "notes",
]


def load_in_by_chunk() -> dict[str, dict[int, tuple[str, str]]]:
    cache: dict[str, dict[int, tuple[str, str]]] = {}
    for in_path in sorted(CHUNKS.glob("chunk_*_in.tsv")):
        tag = in_path.stem.replace("chunk_", "").replace("_in", "")
        meta: dict[int, tuple[str, str]] = {}
        with in_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                meta[int(row["id"])] = (row.get("problem") or "", row.get("tail") or "")
        cache[tag] = meta
    return cache


def review_reason(row: dict) -> str:
    ra, rb = row.get("result_A", ""), row.get("result_B", "")
    if not row.get("agreed"):
        return "dispute"
    if ra == "needs_review" and rb == "needs_review":
        return "needs_review_both"
    if ra == "needs_review":
        return "needs_review_A"
    if rb == "needs_review":
        return "needs_review_B"
    return "needs_human"


def export(out_path: Path) -> int:
    in_cache = load_in_by_chunk()
    rows_out: list[dict[str, str]] = []

    with ROLLOUT_LABELS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("needs_human"):
                continue

            tag = row["chunk"]
            rid = int(row["id"])
            prob, tail = in_cache.get(tag, {}).get(rid, ("", ""))

            rows_out.append(
                {
                    "seq": str(row.get("seq", "")),
                    "chunk": tag,
                    "id": str(rid),
                    "rollout_key": row.get("rollout_key", ""),
                    "gold": row.get("gold", ""),
                    "problem": prob,
                    "tail": tail,
                    "result_A": row.get("result_A", ""),
                    "result_B": row.get("result_B", ""),
                    "agreed": "yes" if row.get("agreed") else "no",
                    "review_reason": review_reason(row),
                    "human_result": row.get("human_result") or "",
                    "notes": "",
                }
            )

    rows_out.sort(key=lambda r: (r["chunk"], int(r["id"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    by_reason: dict[str, int] = {}
    for r in rows_out:
        by_reason[r["review_reason"]] = by_reason.get(r["review_reason"], 0) + 1

    print(f"wrote {len(rows_out)} rows -> {out_path}")
    for reason, n in sorted(by_reason.items()):
        print(f"  {reason}: {n}")
    return len(rows_out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
