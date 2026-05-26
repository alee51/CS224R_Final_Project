"""Jsonl prompt dataset for GRPO training."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonlPromptDataset:
    """Deterministic shuffle over jsonl rows; yields (prompts, golds) batches."""

    def __init__(self, path: str, seed: int) -> None:
        self.path = Path(path)
        self.seed = seed
        self._rows = self._load()
        self._rng = random.Random(seed)
        self._shuffle()
        self._cursor = 0
        logger.info("Loaded %s rows from %s", len(self._rows), self.path)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"dataset not found: {self.path}")
        rows: list[dict[str, Any]] = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"empty dataset: {self.path}")
        return rows

    def _shuffle(self) -> None:
        self._rng.shuffle(self._rows)

    def _problem_text(self, row: dict[str, Any]) -> str:
        return str(row.get("problem", row.get("question", ""))).strip()

    def _gold_text(self, row: dict[str, Any]) -> str:
        return str(row.get("gold", row.get("answer", ""))).strip()

    def next_batch(self, n: int) -> tuple[list[str], list[str]]:
        if n <= 0:
            raise ValueError("batch size must be positive")
        if self._cursor + n > len(self._rows):
            self._cursor = 0
            self._shuffle()
        batch = self._rows[self._cursor : self._cursor + n]
        self._cursor += n
        problems = [self._problem_text(r) for r in batch]
        golds = [self._gold_text(r) for r in batch]
        return problems, golds

    def next_batch_with_ids(
        self, n: int
    ) -> tuple[list[str], list[str], list[int]]:
        """Batch with manifest problem_id when present, else row index."""
        if n <= 0:
            raise ValueError("batch size must be positive")
        if self._cursor + n > len(self._rows):
            self._cursor = 0
            self._shuffle()
        batch = self._rows[self._cursor : self._cursor + n]
        self._cursor += n
        problems = [self._problem_text(r) for r in batch]
        golds = [self._gold_text(r) for r in batch]
        ids = [int(r["problem_id"]) if "problem_id" in r else i for i, r in enumerate(batch)]
        return problems, golds, ids

    def _row_key(self, row: dict[str, Any], index: int) -> int:
        return int(row["problem_id"]) if "problem_id" in row else index

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": self._cursor,
            "rng_state": self._rng.getstate(),
            "row_keys": [self._row_key(r, i) for i, r in enumerate(self._rows)],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        file_rows = self._load()
        by_key = {self._row_key(r, i): r for i, r in enumerate(file_rows)}
        keys = state.get("row_keys")
        if keys is not None:
            self._rows = [by_key[int(k)] for k in keys]
        self._cursor = int(state["cursor"])
        self._rng.setstate(state["rng_state"])
