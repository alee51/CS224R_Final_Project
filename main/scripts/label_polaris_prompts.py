#!/usr/bin/env python3
"""Add prompt-quality heuristic flags to a Polaris train jsonl manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from data.prompt_heuristics import any_heuristic, label_prompt_heuristics

DEFAULT_IN = _MAIN_ROOT / "data/polaris_train.jsonl"
DEFAULT_OUT = _MAIN_ROOT / "data/polaris_train_labeled.jsonl"
DEFAULT_SUMMARY = _MAIN_ROOT / "data/polaris_train_heuristic_summary.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        problem = str(row.get("problem", ""))
        gold = str(row.get("gold", row.get("answer", "")))
        flags = label_prompt_heuristics(problem, gold)
        labeled = dict(row)
        labeled["prompt_heuristics"] = flags
        labeled["prompt_heuristic_any"] = any_heuristic(flags)
        out.append(labeled)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    keys = (
        "last_starts_prove",
        "last_contains_prove",
        "contains_show_that",
        "gold_in_prompt",
    )
    counts = {k: 0 for k in keys}
    combo: Counter[tuple[bool, ...]] = Counter()
    any_count = 0
    for row in rows:
        h = row["prompt_heuristics"]
        tup = tuple(h[k] for k in keys)
        combo[tup] += 1
        for k in keys:
            if h[k]:
                counts[k] += 1
        if row.get("prompt_heuristic_any"):
            any_count += 1

    remove_or = {
        "last_starts_prove OR gold_in_prompt": sum(
            1
            for r in rows
            if r["prompt_heuristics"]["last_starts_prove"]
            or r["prompt_heuristics"]["gold_in_prompt"]
        ),
        "any_of_four": any_count,
        "all_four": sum(1 for r in rows if all(r["prompt_heuristics"][k] for k in keys)),
    }

    return {
        "n_rows": n,
        "per_flag": {k: {"count": counts[k], "pct": counts[k] / n if n else 0} for k in keys},
        "prompt_heuristic_any": {"count": any_count, "pct": any_count / n if n else 0},
        "suggested_remove_or_last_prove_gold_leak": remove_or,
        "top_combo_patterns": [
            {"flags": dict(zip(keys, tup)), "count": c}
            for tup, c in combo.most_common(12)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = _read_jsonl(args.in_path)
    labeled = label_rows(rows)
    _write_jsonl(args.out, labeled)
    summary = summarize(labeled)
    summary["input"] = str(args.in_path)
    summary["output"] = str(args.out)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Labeled {len(labeled)} rows -> {args.out}")
    print(f"Summary -> {args.summary}")
    for k, v in summary["per_flag"].items():
        print(f"  {k}: {v['count']:,} ({100*v['pct']:.2f}%)")
    print(f"  any_of_four: {summary['prompt_heuristic_any']['count']:,}")
    ro = summary["suggested_remove_or_last_prove_gold_leak"]
    print(f"  last_starts_prove OR gold_in_prompt: {ro['last_starts_prove OR gold_in_prompt']:,}")


if __name__ == "__main__":
    main()
