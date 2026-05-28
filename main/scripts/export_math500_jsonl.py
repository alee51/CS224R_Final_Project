#!/usr/bin/env python3
"""Export HuggingFaceH4/math-500 test split into eval JSONL format."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    out_path = Path("main/data/eval/math500.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceH4/math-500", split="test")
    with out_path.open("w") as f:
        for i, row in enumerate(ds):
            rec = {
                "problem_id": i,
                "source_problem_id": row.get("unique_id"),
                "problem": str(row.get("problem", "")).strip(),
                "gold": str(row.get("answer", row.get("solution", ""))).strip(),
                "subject": row.get("subject"),
                "level": row.get("level"),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(ds)} rows to {out_path}")


if __name__ == "__main__":
    main()
