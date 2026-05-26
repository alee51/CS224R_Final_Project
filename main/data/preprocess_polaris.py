"""
One-shot Polaris full-pool freeze → source/polaris_train_full.jsonl (PLAN §2).

Not invoked by the trainer. Run filter_polaris_train.py to produce polaris_train.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.gold_utils import is_nonempty_gold, normalize_train_gold
from data.paths import POLARIS_TRAIN_FULL_JSONL, POLARIS_TRAIN_FULL_META

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = "POLARIS-Project/Polaris-Dataset-53K"
BANDS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8"]
REQUIRED_OUTPUT_KEYS = ("problem_id", "problem", "gold", "difficulty_band", "hf_index")
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def load_hf_rows(dataset_id: str) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset, load_dataset_builder

    try:
        builder = load_dataset_builder(dataset_id)
        revision = getattr(builder, "hash", None) or "unknown"
    except Exception:
        revision = "unknown"

    ds = load_dataset(dataset_id, split="train")
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
    return rows, str(revision)


def clean_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply §4 cleaning filters; return cleaned rows and drop statistics."""
    stats: dict[str, Any] = {
        "hf_rows": len(rows),
        "dropped_empty_problem": 0,
        "dropped_empty_gold": 0,
        "dropped_invalid_problem_type": 0,
    }
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

        answer = row.get("answer", row.get("gold", ""))
        if not is_nonempty_gold(answer):
            stats["dropped_empty_gold"] += 1
            continue

        clean.append(
            {
                "problem": problem_raw,
                "gold": normalize_train_gold(answer),
                "difficulty": row["difficulty"],
                "hf_index": int(row["hf_index"]),
            }
        )

    stats["after_clean"] = len(clean)
    stats["dropped_cleaning"] = stats["hf_rows"] - len(clean)
    stats["per_band_after_clean"] = {
        b: sum(1 for r in clean if r["difficulty"] == b) for b in BANDS
    }
    return clean, stats


def _hamilton_quotas(counts: dict[str, int], n: int, n_clean: int) -> dict[str, int]:
    """§3.1 steps 2–4: proportional floors + Hamilton remainder apportionment."""
    if n_clean <= 0:
        raise ValueError(f"clean pool is empty; cannot sample n={n}")
    if n > n_clean:
        raise ValueError(
            f"target n={n} exceeds clean pool size {n_clean}; will not pad"
        )

    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for b in BANDS:
        c_b = counts.get(b, 0)
        if c_b == 0:
            quotas[b] = 0
            remainders[b] = 0.0
            continue
        exact = n * c_b / n_clean
        quota_b = math.floor(exact)
        if quota_b > c_b:
            raise ValueError(
                f"band {b}: initial quota {quota_b} exceeds clean count {c_b}"
            )
        quotas[b] = quota_b
        remainders[b] = exact - quota_b

    r = n - sum(quotas.values())
    while r > 0:
        candidates = [b for b in BANDS if remainders[b] > 0]
        if not candidates:
            raise ValueError("Hamilton apportionment stalled with R > 0")
        best_remainder = max(remainders[b] for b in candidates)
        for b in BANDS:
            if remainders[b] == best_remainder:
                quotas[b] += 1
                remainders[b] = 0.0
                r -= 1
                break

    for b in BANDS:
        c_b = counts.get(b, 0)
        if quotas[b] > c_b:
            raise ValueError(
                f"band {b}: final quota {quotas[b]} exceeds clean count {c_b}"
            )
    if sum(quotas.values()) != n:
        raise ValueError(f"quotas sum to {sum(quotas.values())}, expected {n}")
    return quotas


def stratified_sample(
    rows: list[dict[str, Any]], n: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """§3.1 stratified proportional (Hamilton) sample; output rows in BANDS order."""
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped_bad_band = 0
    for row in rows:
        band = row["difficulty"]
        if band not in BANDS:
            dropped_bad_band += 1
            continue
        by_band[band].append(row)

    unexpected = set(by_band.keys()) - set(BANDS)
    if unexpected:
        raise ValueError(f"unexpected difficulty bands after partition: {sorted(unexpected)}")

    pool = [r for b in BANDS for r in by_band.get(b, [])]
    n_clean = len(pool)
    if dropped_bad_band:
        logger.info("Dropped %s rows with difficulty not in BANDS", dropped_bad_band)

    counts = {b: len(by_band.get(b, [])) for b in BANDS}
    quotas = _hamilton_quotas(counts, n, n_clean)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    problem_id = 0
    for band in BANDS:
        pool_b = list(by_band.get(band, []))
        rng.shuffle(pool_b)
        take = quotas[band]
        if take > len(pool_b):
            raise ValueError(
                f"band {band}: draw {take} exceeds pool size {len(pool_b)}"
            )
        for row in pool_b[:take]:
            sampled.append(
                {
                    "problem_id": problem_id,
                    "problem": row["problem"],
                    "gold": row["gold"],
                    "difficulty_band": band,
                    "hf_index": row["hf_index"],
                }
            )
            problem_id += 1

    sample_stats = {
        "dropped_bad_band": dropped_bad_band,
        "n_clean": n_clean,
        "per_band_in_output": {b: quotas[b] for b in BANDS},
    }
    return sampled, sample_stats


def validate_output(
    rows: list[dict[str, Any]],
    *,
    target_n: int,
    per_band_after_clean: dict[str, int],
) -> None:
    """§8 validation checks before declaring freeze."""
    if len(rows) != target_n:
        raise ValueError(f"expected {target_n} rows, got {len(rows)}")

    hf_indices = [r["hf_index"] for r in rows]
    if len(hf_indices) != len(set(hf_indices)):
        raise ValueError("duplicate hf_index in output")

    n_clean = sum(per_band_after_clean.values())
    band_out: Counter[str] = Counter()
    for i, row in enumerate(rows):
        for key in REQUIRED_OUTPUT_KEYS:
            if key not in row:
                raise ValueError(f"row {i} missing key {key!r}")
        if row["problem_id"] != i:
            raise ValueError(f"row {i} problem_id {row['problem_id']} != index {i}")
        if not is_nonempty_gold(row["gold"]):
            raise ValueError(f"row {i} gold is empty: {row['gold']!r}")
        band_out[row["difficulty_band"]] += 1

    if sum(band_out.values()) != target_n:
        raise ValueError("per-band output counts do not sum to target_n")

    for b in BANDS:
        if per_band_after_clean.get(b, 0) > 0 and band_out[b] == 0:
            raise ValueError(f"band {b} present after clean but missing from output")

    if n_clean > 0:
        for b in BANDS:
            clean_share = 100.0 * per_band_after_clean.get(b, 0) / n_clean
            out_share = 100.0 * band_out.get(b, 0) / target_n
            if abs(out_share - clean_share) > 1.0:
                raise ValueError(
                    f"band {b}: output share {out_share:.2f}% differs from clean "
                    f"share {clean_share:.2f}% by more than 1.0 pp"
                )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_meta(
    *,
    dataset_id: str,
    dataset_revision: str,
    target_n: int,
    seed: int,
    clean_stats: dict[str, Any],
    sample_stats: dict[str, Any],
    out_jsonl: Path,
    out_meta: Path,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": "train",
        "sampling": {
            "method": "stratified_proportional",
            "target_n": target_n,
            "seed": seed,
            "bands": list(BANDS),
        },
        "cleaning": {
            "drop_empty_gold": True,
            "drop_empty_problem": True,
            "gold_policy": "verbatim_hf_strip_only",
            "note": "No integer-gold filter; grade at train time via grade_parsed_answer (mathd OR sympy).",
        },
        "counts": {
            "hf_rows": clean_stats["hf_rows"],
            "after_clean": clean_stats["after_clean"],
            "written": target_n,
            "dropped_cleaning": clean_stats["dropped_cleaning"],
            "dropped_bad_band": sample_stats.get("dropped_bad_band", 0),
            "per_band_after_clean": clean_stats["per_band_after_clean"],
            "per_band_in_output": sample_stats["per_band_in_output"],
        },
        "materialized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "freeze_status": "frozen",
        "artifact_role": "full_pool",
        "freeze_note": (
            "Unfiltered clean pool; not for training. Train on data/polaris_train.jsonl "
            "(from filter_polaris_train.py)."
        ),
        "output_files": {
            "jsonl": str(out_jsonl),
            "meta": str(out_meta),
        },
    }


def materialize(
    raw_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    target_n: int,
    seed: int,
    dataset_id: str,
    dataset_revision: str,
    dry_run: bool,
) -> dict[str, Any]:
    clean, clean_stats = clean_rows(raw_rows)
    sampled, sample_stats = stratified_sample(clean, target_n, seed)
    validate_output(
        sampled,
        target_n=target_n,
        per_band_after_clean=clean_stats["per_band_after_clean"],
    )

    if out_dir.resolve() == _DEFAULT_OUT_DIR.resolve():
        out_jsonl = POLARIS_TRAIN_FULL_JSONL
        out_meta = POLARIS_TRAIN_FULL_META
    else:
        out_jsonl = out_dir / "polaris_train_full.jsonl"
        out_meta = out_dir / "polaris_train_full.meta.json"
    meta = build_meta(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        target_n=target_n,
        seed=seed,
        clean_stats=clean_stats,
        sample_stats=sample_stats,
        out_jsonl=out_jsonl,
        out_meta=out_meta,
    )

    if dry_run:
        logger.info("Dry run: would write %s rows to %s", len(sampled), out_jsonl)
    else:
        write_jsonl(out_jsonl, sampled)
        write_meta(out_meta, meta)
        logger.info("Wrote %s (%s rows)", out_jsonl, len(sampled))
        logger.info("Wrote %s", out_meta)

    logger.info(
        "Clean pool=%s target_n=%s per_band_output=%s",
        sample_stats["n_clean"],
        target_n,
        sample_stats["per_band_in_output"],
    )
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Polaris train freeze jsonl.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--n", type=int, default=16000, help="Target sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats and validate only; do not write files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger.info("Loading %s", args.dataset)
    raw_rows, revision = load_hf_rows(args.dataset)
    materialize(
        raw_rows,
        out_dir=args.out_dir,
        target_n=args.n,
        seed=args.seed,
        dataset_id=args.dataset,
        dataset_revision=revision,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
