#!/usr/bin/env python3
"""Materialize filtered Polaris train jsonl (decisions.md §2026-05-27)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from data.paths import (  # noqa: E402
    POLARIS_TRAIN_DROPPED_JSONL,
    POLARIS_TRAIN_FULL_JSONL,
    POLARIS_TRAIN_FULL_META,
    POLARIS_TRAIN_JSONL,
    POLARIS_TRAIN_META,
)
from data.prompt_heuristics import (  # noqa: E402
    label_prompt_heuristics,
    should_drop_train_prompt_filter,
)

DEFAULT_IN = POLARIS_TRAIN_FULL_JSONL
DEFAULT_OUT = POLARIS_TRAIN_JSONL
DEFAULT_META = POLARIS_TRAIN_META
DEFAULT_PARENT_META = POLARIS_TRAIN_FULL_META
DEFAULT_DROPPED = POLARIS_TRAIN_DROPPED_JSONL

_TRAIN_ROW_KEYS = ("problem_id", "problem", "gold", "difficulty_band", "hf_index")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _train_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in _TRAIN_ROW_KEYS if k in row}


def _drop_reason(problem: str, gold: str, flags: dict[str, bool]) -> str:
    if flags["last_starts_prove"]:
        return "last_starts_prove"
    if flags["gold_in_prompt"] and (
        "prove" in problem.lower() or flags["contains_show_that"]
    ):
        return "gold_in_prompt_and_prove_or_show_that"
    return "other"


def apply_filter(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()

    for row in rows:
        problem = str(row.get("problem", ""))
        gold = str(row.get("gold", row.get("answer", "")))
        flags = label_prompt_heuristics(problem, gold)
        if should_drop_train_prompt_filter(problem, gold):
            reason = _drop_reason(problem, gold, flags)
            drop_reasons[reason] += 1
            dropped.append({**_train_row(row), "drop_reason": reason, "prompt_heuristics": flags})
        else:
            kept.append(_train_row(row))

    stats = {
        "n_input": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_reasons": dict(drop_reasons),
        "per_band_kept": dict(
            Counter(str(r.get("difficulty_band", "")) for r in kept)
        ),
        "per_band_dropped": dict(
            Counter(str(r.get("difficulty_band", "")) for r in dropped)
        ),
    }
    return kept, dropped, stats


def build_meta(
    *,
    stats: dict[str, Any],
    in_path: Path,
    out_jsonl: Path,
    out_meta: Path,
    parent_meta_path: Path,
    dropped_audit_path: Path | None,
) -> dict[str, Any]:
    parent: dict[str, Any] = {}
    if parent_meta_path.is_file():
        parent = json.loads(parent_meta_path.read_text())

    return {
        "source": {
            "input_jsonl": str(in_path),
            "parent_meta": str(parent_meta_path) if parent else None,
            "parent_freeze": parent.get("freeze_status"),
            "parent_dataset_id": parent.get("dataset_id"),
            "parent_dataset_revision": parent.get("dataset_revision"),
        },
        "prompt_filter": {
            "rule_id": "2026-05-27_train_prompt_filter",
            "predicate": (
                "last_starts_prove OR "
                "(gold_in_prompt AND (prove anywhere OR contains_show_that))"
            ),
            "decision_doc": "main/docs/decisions.md §2026-05-27",
        },
        "counts": stats,
        "materialized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "freeze_status": "frozen",
        "freeze_note": (
            "Canonical train manifest (prompt-filtered). Source pool: source/polaris_train_full.jsonl. "
            "Re-materialize only with dated note in main/docs/context.md."
        ),
        "artifact_role": "train",
        "output_files": {
            "jsonl": str(out_jsonl),
            "meta": str(out_meta),
            "dropped_audit_jsonl": str(dropped_audit_path) if dropped_audit_path else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--parent-meta", type=Path, default=DEFAULT_PARENT_META)
    parser.add_argument(
        "--dropped-audit",
        type=Path,
        default=DEFAULT_DROPPED,
        help="jsonl of dropped rows with drop_reason (not for training)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.in_path)
    kept, dropped, stats = apply_filter(rows)

    expected_drop = sum(
        1
        for r in rows
        if should_drop_train_prompt_filter(
            str(r.get("problem", "")), str(r.get("gold", ""))
        )
    )
    if expected_drop != stats["n_dropped"]:
        raise RuntimeError(
            f"drop count mismatch: expected {expected_drop}, got {stats['n_dropped']}"
        )

    print(f"Input:  {stats['n_input']:,}")
    print(f"Kept:   {stats['n_kept']:,} ({100 * stats['n_kept'] / stats['n_input']:.2f}%)")
    print(f"Dropped:{stats['n_dropped']:,} ({100 * stats['n_dropped'] / stats['n_input']:.2f}%)")
    for reason, count in sorted(stats["drop_reasons"].items()):
        print(f"  {reason}: {count:,}")

    if args.dry_run:
        print("Dry run — no files written.")
        return

    _write_jsonl(args.out, kept)
    meta = build_meta(
        stats=stats,
        in_path=args.in_path,
        out_jsonl=args.out,
        out_meta=args.meta,
        parent_meta_path=args.parent_meta,
        dropped_audit_path=args.dropped_audit,
    )
    args.meta.write_text(json.dumps(meta, indent=2) + "\n")
    if args.dropped_audit:
        _write_jsonl(args.dropped_audit, dropped)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.meta}")
    if args.dropped_audit:
        print(f"Wrote dropped audit {args.dropped_audit}")


if __name__ == "__main__":
    main()
