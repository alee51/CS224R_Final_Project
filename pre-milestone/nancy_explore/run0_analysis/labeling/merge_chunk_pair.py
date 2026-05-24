#!/usr/bin/env python3
"""Merge chunk_KKK out_A/out_B into single sequential labels/rollout_labels.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from label_paths import ANALYSIS_ROOT, LABELING_ROOT, ROLLOUT_LABELS, load_manifest

CHUNKS = LABELING_ROOT / "chunks"


def load_tsv(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[int(row["id"])] = (row.get("result") or "").strip()
    return out


def load_keys(path: Path) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = int(row["id"])
            meta[rid] = {
                "rollout_key": row["rollout_key"],
                "gold": row.get("gold", ""),
            }
    return meta


def load_in_meta(path: Path) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = int(row["id"])
            out[rid] = (row.get("problem") or "", row.get("tail") or "")
    return out


def merge_chunk(chunk_k: int) -> tuple[list[dict], int, int]:
    manifest = load_manifest(chunk_k)
    tag = manifest["chunk"]
    a = load_tsv(LABELING_ROOT / manifest["output_a"])
    b = load_tsv(LABELING_ROOT / manifest["output_b"])
    keys = load_keys(CHUNKS / f"chunk_{tag}_keys.tsv")
    in_meta = load_in_meta(CHUNKS / f"chunk_{tag}_in.tsv")

    ids = sorted(set(a) | set(b) | set(keys))
    rows: list[dict] = []
    disputes: list[list[str]] = []

    for rid in ids:
        ra = a.get(rid, "")
        rb = b.get(rid, "")
        km = keys.get(rid, {})
        prob, tail = in_meta.get(rid, ("", ""))
        agreed = ra == rb
        needs_human = (not agreed) or ra == "needs_review" or rb == "needs_review"

        row = {
            "chunk": tag,
            "chunk_index": chunk_k,
            "id": rid,
            "seq": chunk_k * 1000 + rid,
            "rollout_key": km.get("rollout_key", ""),
            "gold": km.get("gold", ""),
            "result_A": ra,
            "result_B": rb,
            "agreed": agreed,
            "needs_human": needs_human,
            "result": ra if agreed else None,
            "human_result": None,
        }
        rows.append(row)
        if not agreed:
            disputes.append([str(rid), prob, km.get("gold", ""), tail, ra, rb])

    dispute_path = CHUNKS / f"chunk_{tag}_dispute.tsv"
    if disputes:
        with dispute_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "problem", "gold", "tail", "result_A", "result_B"])
            w.writerows(disputes)

    return rows, len(disputes), sum(1 for r in rows if r["agreed"])


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("chunk", type=int, help="0-based chunk index")
    args = p.parse_args()

    manifest = load_manifest(args.chunk)
    tag = manifest["chunk"]
    for key in ("output_a", "output_b", "keys", "input"):
        req = LABELING_ROOT / manifest[key]
        if not req.exists():
            raise SystemExit(f"missing {req}")

    rows, n_dispute, n_agree = merge_chunk(args.chunk)
    append_rows(ROLLOUT_LABELS, rows)

    print(f"chunk_{tag}: appended {len(rows)} rows -> {ROLLOUT_LABELS}")
    print(f"  agreed={n_agree}  dispute={n_dispute}  needs_human={sum(1 for r in rows if r['needs_human'])}")


if __name__ == "__main__":
    main()
