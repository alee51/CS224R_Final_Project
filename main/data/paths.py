"""Canonical paths for Polaris train data artifacts."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCE_DIR = DATA_DIR / "source"

# Unfiltered HF freeze (53,291 rows) — analysis / re-filter only; not for training.
POLARIS_TRAIN_FULL_JSONL = SOURCE_DIR / "polaris_train_full.jsonl"
POLARIS_TRAIN_FULL_META = SOURCE_DIR / "polaris_train_full.meta.json"

# Train freeze (51,139 rows after prompt filter) — use for GRPO / Modal upload.
POLARIS_TRAIN_JSONL = DATA_DIR / "polaris_train.jsonl"
POLARIS_TRAIN_META = DATA_DIR / "polaris_train.meta.json"
POLARIS_TRAIN_DROPPED_JSONL = DATA_DIR / "polaris_train_dropped.jsonl"

# Optional analysis outputs (full-pool labels).
POLARIS_TRAIN_LABELED_JSONL = DATA_DIR / "polaris_train_labeled.jsonl"
POLARIS_TRAIN_HEURISTIC_SUMMARY = DATA_DIR / "polaris_train_heuristic_summary.json"
