#!/usr/bin/env python3
"""Apply human labels from CSV into labels/rollout_labels.jsonl.

Reads CSV with human_result filled. Optionally auto-labels needs_human rows as
runon when the raw completion hit the generation token cap (Qwen tokenizer).

Usage:
  python import_human_labels.py labels/partial_annotated.csv
  python import_human_labels.py labels/partial_annotated.csv --auto-runon-at-tokens 1024
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from label_paths import ANALYSIS_ROOT, ROLLOUT_LABELS

RAW = ANALYSIS_ROOT / "data" / "raw_predictions.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"


@lru_cache(maxsize=1)
def _tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def completion_token_count(text: str, model_id: str) -> int:
    if not text:
        return 0
    tok = _tokenizer(model_id)
    return len(tok.encode(text, add_special_tokens=False))


def load_completions_by_rollout_key() -> dict[str, str]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_prompt[r["prompt_id"]].append(r)

    out: dict[str, str] = {}
    for pid, rollouts in by_prompt.items():
        for idx, r in enumerate(rollouts):
            out[f"{pid}#{idx}"] = r.get("completion") or ""
    return out


def load_csv_labels(path: Path) -> dict[int, dict[str, str]]:
    """Map seq -> {human_result, notes}."""
    by_seq: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            hr = (row.get("human_result") or "").strip()
            if not hr:
                continue
            seq = int(row["seq"])
            by_seq[seq] = {
                "human_result": hr,
                "notes": (row.get("notes") or "").strip(),
            }
    return by_seq


def apply_labels(
    csv_path: Path,
    *,
    model_id: str,
    auto_runon_at_tokens: int | None,
) -> None:
    human_by_seq = load_csv_labels(csv_path)
    completions = load_completions_by_rollout_key()

    rows: list[dict] = []
    n_csv = n_auto = n_open = 0

    with ROLLOUT_LABELS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            if row["seq"] in human_by_seq:
                h = human_by_seq[row["seq"]]
                row["human_result"] = h["human_result"]
                row["result"] = h["human_result"]
                row["needs_human"] = False
                n_csv += 1
            elif (
                auto_runon_at_tokens is not None
                and row.get("needs_human")
                and not row.get("human_result")
            ):
                comp = completions.get(row.get("rollout_key", ""), "")
                n_tok = completion_token_count(comp, model_id)
                if n_tok >= auto_runon_at_tokens:
                    row["human_result"] = "runon"
                    row["result"] = "runon"
                    row["needs_human"] = False
                    n_auto += 1

            if row.get("needs_human"):
                n_open += 1

            rows.append(row)

    with ROLLOUT_LABELS.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"updated {ROLLOUT_LABELS}")
    print(f"  from CSV: {n_csv} rows")
    if auto_runon_at_tokens is not None:
        print(
            f"  auto runon (Qwen {model_id}, >= {auto_runon_at_tokens} tokens): {n_auto} rows"
        )
    print(f"  still needs_human: {n_open} rows")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path, help="CSV with human_result column")
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model for tokenizer (default {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--auto-runon-at-tokens",
        type=int,
        default=1024,
        help="Auto runon when completion token count >= N (Run 0 cap). Pass 0 to disable.",
    )
    args = p.parse_args()

    auto = args.auto_runon_at_tokens if args.auto_runon_at_tokens > 0 else None
    apply_labels(args.csv, model_id=args.model_id, auto_runon_at_tokens=auto)


if __name__ == "__main__":
    main()
