"""Load predictions JSONL into PromptRollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from pilot.eval.metrics import PromptRollouts


def load_predictions(
    path: Path,
    *,
    eval_splits: Sequence[str] | None = None,
) -> list[PromptRollouts]:
    """
    Load rollouts grouped by prompt_id.
    If eval_splits is set, keep only rows whose eval_split is in that list.
    """
    allowed = set(eval_splits) if eval_splits is not None else None
    by_id: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if allowed is not None:
                split = row.get("eval_split", "")
                if split not in allowed:
                    continue
            pid = row["prompt_id"]
            if pid not in by_id:
                by_id[pid] = {"correct": [], "cluster_ids": []}
            by_id[pid]["correct"].append(bool(row["correct"]))
            by_id[pid]["cluster_ids"].append(int(row["cluster_id"]))
    return [
        PromptRollouts(prompt_id=pid, correct=v["correct"], cluster_ids=v["cluster_ids"])
        for pid, v in sorted(by_id.items())
    ]


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
