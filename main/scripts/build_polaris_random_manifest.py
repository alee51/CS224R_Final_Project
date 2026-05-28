#!/usr/bin/env python3
"""Build a uniform random Polaris manifest (relaxed cleaning, full gold types)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from data.gold_utils import is_nonempty_gold, normalize_train_gold

POLARIS_DATASET_ID = "POLARIS-Project/Polaris-Dataset-53K"
DEFAULT_OUT = _MAIN_ROOT / "data/probes/05-27/random_fullgold_n800/manifest.jsonl"
DEFAULT_STRATIFIED_OUT = (
    _MAIN_ROOT / "data/probes/05-27/polaris_stratified800/manifest.jsonl"
)
DEFAULT_TRAIN_JSONL = _MAIN_ROOT / "data/polaris_train.jsonl"
BANDS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8"]


def _load_hf_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(POLARIS_DATASET_ID, split="train")
    rows: list[dict[str, Any]] = []
    for hf_index, row in enumerate(ds):
        rows.append(
            {
                "problem": row["problem"],
                "answer": row["answer"],
                "difficulty": row["difficulty"],
                "hf_index": hf_index,
            }
        )
    return rows


def _clean_relaxed(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop empty/non-string problems and empty gold only (no integer-gold filter)."""
    stats = Counter(
        dropped_invalid_problem_type=0,
        dropped_empty_problem=0,
        dropped_empty_gold=0,
    )
    clean: list[dict[str, Any]] = []
    for row in rows:
        problem_raw = row.get("problem")
        if not isinstance(problem_raw, str):
            stats["dropped_invalid_problem_type"] += 1
            continue
        problem = problem_raw.strip()
        if not problem:
            stats["dropped_empty_problem"] += 1
            continue

        gold_raw = row.get("answer", row.get("gold", ""))
        if not is_nonempty_gold(gold_raw):
            stats["dropped_empty_gold"] += 1
            continue

        clean.append(
            {
                "problem": problem_raw,
                "gold": normalize_train_gold(gold_raw),
                "difficulty": row["difficulty"],
                "hf_index": int(row["hf_index"]),
            }
        )
    return clean, dict(stats)


def _band_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(r["difficulty"] for r in rows)
    return {b: counts.get(b, 0) for b in BANDS}


def _load_train_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows.append(
                {
                    "problem": row["problem"],
                    "gold": row["gold"],
                    "difficulty": row.get("difficulty_band", row.get("difficulty")),
                    "hf_index": row.get("hf_index"),
                }
            )
    return rows


def _sample_manifest_stratified(
    clean: list[dict[str, Any]],
    *,
    per_band: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Uniform count per difficulty band (Group A style)."""
    from collections import defaultdict

    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_band[str(row["difficulty"])].append(row)

    rng = random.Random(seed)
    manifest: list[dict[str, Any]] = []
    problem_id = 0
    for band in BANDS:
        pool = by_band.get(band, [])
        if len(pool) < per_band:
            raise ValueError(
                f"Band {band} has only {len(pool)} rows; need {per_band}"
            )
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        for i in indices[:per_band]:
            row = pool[i]
            manifest.append(
                {
                    "problem_id": problem_id,
                    "problem": row["problem"],
                    "gold": row["gold"],
                    "difficulty_band": band,
                    "hf_index": row.get("hf_index"),
                }
            )
            problem_id += 1
    return manifest


def _sample_manifest(
    clean: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    if n > len(clean):
        raise ValueError(f"requested n={n} exceeds clean pool size {len(clean)}")
    rng = random.Random(seed)
    chosen = rng.sample(clean, n)
    manifest: list[dict[str, Any]] = []
    for problem_id, row in enumerate(chosen):
        manifest.append(
            {
                "problem_id": problem_id,
                "problem": row["problem"],
                "gold": row["gold"],
                "difficulty_band": row["difficulty"],
                "hf_index": row["hf_index"],
            }
        )
    return manifest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print clean pool size and difficulty histogram; do not write manifest",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Sample per_band rows from each difficulty band (uniform n per band)",
    )
    parser.add_argument(
        "--per-band",
        type=int,
        default=100,
        help="Prompts per band when --stratified (total = 8 * per_band)",
    )
    parser.add_argument(
        "--from-train-jsonl",
        type=Path,
        default=DEFAULT_TRAIN_JSONL,
        help="Source pool for --stratified (prompt-filtered train manifest)",
    )
    args = parser.parse_args()
    if args.stratified and args.out == DEFAULT_OUT:
        args.out = DEFAULT_STRATIFIED_OUT

    if args.stratified:
        print(f"Loading train pool {args.from_train_jsonl} ...", flush=True)
        clean = _load_train_jsonl(args.from_train_jsonl)
        drop_stats: dict[str, int] = {}
    else:
        print(f"Loading {POLARIS_DATASET_ID} ...", flush=True)
        hf_rows = _load_hf_rows()
        clean, drop_stats = _clean_relaxed(hf_rows)
    hist = _band_histogram(clean)

    print(f"Pool rows: {len(clean)}")
    if drop_stats:
        print(f"Drops: {drop_stats}")
    print("Per-band (pool):")
    for band in BANDS:
        print(f"  {band}: {hist[band]}")

    if args.dry_run:
        print("(dry-run: no manifest written)")
        return

    if args.stratified:
        manifest = _sample_manifest_stratified(
            clean, per_band=args.per_band, seed=args.seed
        )
    else:
        manifest = _sample_manifest(clean, args.n, args.seed)
    _write_jsonl(args.out, manifest)
    sample_hist = _band_histogram(
        [{"difficulty": m["difficulty_band"]} for m in manifest]
    )
    mode = f"stratified {args.per_band}/band" if args.stratified else f"random n={args.n}"
    print(f"Wrote {len(manifest)} prompts ({mode}) to {args.out}")
    print("Per-band (sample):")
    for band in BANDS:
        print(f"  {band}: {sample_hist[band]}")


if __name__ == "__main__":
    main()
