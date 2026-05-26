"""Polaris preprocess unit tests (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.dataset import JsonlPromptDataset
from data.gold_utils import is_integer_gold, is_nonempty_gold, normalize_train_gold
from data.preprocess_polaris import (
    BANDS,
    _hamilton_quotas,
    clean_rows,
    materialize,
    stratified_sample,
    validate_output,
)


def _hf_row(
    hf_index: int,
    problem: str,
    answer: str,
    difficulty: str,
) -> dict:
    return {
        "problem": problem,
        "answer": answer,
        "difficulty": difficulty,
        "hf_index": hf_index,
    }


def test_train_gold_helpers():
    assert normalize_train_gold("  \\frac{1}{2}  ") == "\\frac{1}{2}"
    assert is_nonempty_gold("0")
    assert not is_nonempty_gold("")
    assert not is_nonempty_gold("   ")
    assert is_integer_gold("42")
    assert not is_integer_gold("\\frac{1}{2}")


def test_clean_rows_filters():
    rows = [
        _hf_row(0, "ok", "10", "3/8"),
        _hf_row(1, "   ", "1", "3/8"),
        _hf_row(2, "ok2", "", "3/8"),
        _hf_row(3, 123, "4", "3/8"),
        _hf_row(4, "ok3", r"\frac{2}{3}", "3/8"),
        _hf_row(5, "ok4", "  32.5 ", "7/8"),
    ]
    clean, stats = clean_rows(rows)
    assert len(clean) == 3
    assert stats["dropped_empty_problem"] == 1
    assert stats["dropped_empty_gold"] == 1
    assert stats["dropped_invalid_problem_type"] == 1
    assert clean[0]["gold"] == "10"
    assert clean[1]["gold"] == r"\frac{2}{3}"
    assert clean[2]["gold"] == "32.5"


def test_hamilton_quotas_proportional():
    counts = {"0/8": 50, "1/8": 50}
    for b in BANDS[2:]:
        counts[b] = 0
    quotas = _hamilton_quotas(counts, n=100, n_clean=100)
    assert quotas["0/8"] == 50
    assert quotas["1/8"] == 50
    assert sum(quotas.values()) == 100


def test_hamilton_tie_break_prefers_lower_band_index():
    counts = {b: 0 for b in BANDS}
    counts["0/8"] = 10
    counts["1/8"] = 10
    quotas = _hamilton_quotas(counts, n=3, n_clean=20)
    assert quotas["0/8"] == 2
    assert quotas["1/8"] == 1
    assert sum(quotas.values()) == 3


def test_stratified_sample_deterministic_and_ordered():
    rows = []
    idx = 0
    for band in BANDS:
        for _ in range(20):
            rows.append(_hf_row(idx, f"p{idx}", str(idx % 9) if idx % 3 else r"\frac{1}{2}", band))
            idx += 1
    clean, _ = clean_rows(rows)
    a, _ = stratified_sample(clean, n=40, seed=42)
    b, _ = stratified_sample(clean, n=40, seed=42)
    assert [r["hf_index"] for r in a] == [r["hf_index"] for r in b]
    assert [r["problem_id"] for r in a] == list(range(40))
    bands_in_file = [r["difficulty_band"] for r in a]
    assert bands_in_file == sorted(bands_in_file, key=BANDS.index)
    assert all(bands_in_file.count(b) == 5 for b in BANDS)


def test_stratified_drops_non_band_difficulty():
    rows = [
        _hf_row(0, "a", "1", "0/8"),
        _hf_row(1, "b", "2", "8/8"),
        _hf_row(2, "c", "3", "1/8"),
    ]
    clean, _ = clean_rows(rows)
    sampled, stats = stratified_sample(clean, n=2, seed=0)
    assert stats["dropped_bad_band"] == 1
    assert len(sampled) == 2
    assert all(r["difficulty_band"] in BANDS for r in sampled)


def test_sample_fails_when_n_exceeds_pool():
    rows = [_hf_row(0, "a", "1", "0/8")]
    clean, _ = clean_rows(rows)
    with pytest.raises(ValueError, match="exceeds clean pool"):
        stratified_sample(clean, n=5, seed=0)


def test_materialize_dry_run_writes_nothing(tmp_path: Path):
    rows = []
    for i, band in enumerate(BANDS * 3):
        rows.append(_hf_row(i, f"problem {i}", str(i + 1), band))
    out = tmp_path / "out"
    meta = materialize(
        rows,
        out_dir=out,
        target_n=16,
        seed=42,
        dataset_id="synthetic",
        dataset_revision="test",
        dry_run=True,
    )
    assert meta["counts"]["written"] == 16
    assert meta["cleaning"]["gold_policy"] == "verbatim_hf_strip_only"
    assert not (out / "polaris_train_full.jsonl").exists()
    assert not (out / "polaris_train_full.meta.json").exists()


def test_materialize_writes_jsonl_loader_smoke(tmp_path: Path):
    rows = []
    for i in range(32):
        gold = str(i + 1) if i % 2 == 0 else r"\frac{1}{2}"
        rows.append(_hf_row(i, f"problem {i}", gold, BANDS[i % len(BANDS)]))
    out = tmp_path / "out"
    materialize(
        rows,
        out_dir=out,
        target_n=16,
        seed=7,
        dataset_id="synthetic",
        dataset_revision="test",
        dry_run=False,
    )
    path = out / "polaris_train_full.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 16
    first = json.loads(lines[0])
    assert r"\frac{1}{2}" in {json.loads(ln)["gold"] for ln in lines}
    meta = json.loads((out / "polaris_train_full.meta.json").read_text())
    assert meta["sampling"]["seed"] == 7
    ds = JsonlPromptDataset(str(path), seed=0)
    probs, golds = ds.next_batch(8)
    assert len(probs) == 8
    assert all(p.strip() for p in probs)
    assert all(g.strip() for g in golds)


def test_validate_output_rejects_duplicate_hf_index():
    rows = [
        {
            "problem_id": 0,
            "problem": "a",
            "gold": "1",
            "difficulty_band": "0/8",
            "hf_index": 1,
        },
        {
            "problem_id": 1,
            "problem": "b",
            "gold": r"\frac{1}{2}",
            "difficulty_band": "0/8",
            "hf_index": 1,
        },
    ]
    per_band = {b: (1 if b == "0/8" else 0) for b in BANDS}
    with pytest.raises(ValueError, match="duplicate hf_index"):
        validate_output(rows, target_n=2, per_band_after_clean=per_band)
