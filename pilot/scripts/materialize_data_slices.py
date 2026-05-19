#!/usr/bin/env python3
"""
Pull frozen pilot + paper eval slices from HuggingFace and write JSONL under pilot/data/.

Train: open-r1/DAPO-Math-17k-Processed (config en) — stable sort prompt_id, shuffle seed=42, first 3000.
Beyond-AIME: ByteDance-Seed/BeyondAIME (test, 100).
MATH-500 sanity: 100 prompts, proportional across level × subject, seed=42.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "pilot" / "data"
LOCK_PATH = ROOT / "pilot" / "preflight_lock.json"

DAPO_HF = "open-r1/DAPO-Math-17k-Processed"
DAPO_CONFIG = "en"
BEYOND_AIME_HF = "ByteDance-Seed/BeyondAIME"
MATH500_HF = "HuggingFaceH4/MATH-500"
TRAIN_SEED = 42
TRAIN_N = 3000
MATH500_SANITY_SEED = 42
MATH500_SANITY_N = 100


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _first_str(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _dapo_ground_truth(row: dict) -> str:
    sol = _first_str(row, ("solution", "answer"))
    if sol:
        return sol
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        return _first_str(rm, ("ground_truth",))
    if isinstance(rm, str):
        try:
            parsed = json.loads(rm.replace("'", '"'))
            if isinstance(parsed, dict):
                return _first_str(parsed, ("ground_truth",))
        except json.JSONDecodeError:
            pass
    return ""


def _normalize_dapo_row(row: dict, idx: int) -> dict[str, Any]:
    problem = _first_str(row, ("prompt", "problem", "question", "input", "instruction"))
    answer = _dapo_ground_truth(row)
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    uid = _first_str(extra, ("index", "id")) or _first_str(row, ("uid", "id")) or f"dapo_en_{idx:06d}"
    return {
        "prompt_id": str(uid),
        "problem": problem,
        "answer": answer,
        "split": "train",
        "source_hf": f"{DAPO_HF}:{DAPO_CONFIG}",
    }


def _normalize_matharena_row(row: dict, idx: int, split: str, source_hf: str) -> dict[str, Any]:
    problem = _first_str(row, ("problem", "question", "prompt", "text"))
    answer = _first_str(row, ("answer", "final_answer", "label", "solution"))
    uid = _first_str(row, ("id", "problem_id", "idx", "number")) or f"{split}_{idx:05d}"
    return {
        "prompt_id": f"{split}_{uid}",
        "problem": problem,
        "answer": answer,
        "split": split,
        "source_hf": source_hf,
    }


def _load_hf_dataset(repo_id: str, config: str | None = None, split: str | None = None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split
    try:
        return load_dataset(repo_id, **kwargs)
    except Exception:
        return load_dataset(repo_id)


def materialize_dapo_slice(out_path: Path) -> int:
    ds = _load_hf_dataset(DAPO_HF, config=DAPO_CONFIG, split="train")
    rows = [dict(ds[i]) for i in range(len(ds))]

    normalized = [_normalize_dapo_row(r, i) for i, r in enumerate(rows)]
    normalized.sort(key=lambda r: r["prompt_id"])
    rng = random.Random(TRAIN_SEED)
    rng.shuffle(normalized)
    selected = normalized[:TRAIN_N]

    return _write_jsonl(out_path, iter(selected))


def _take_first_n(ds_rows: list[dict], n: int, split: str, source_hf: str) -> list[dict]:
    out = []
    for i, r in enumerate(ds_rows[:n]):
        row = _normalize_matharena_row(dict(r), i, split, source_hf)
        if row["problem"]:
            out.append(row)
    if len(out) < n:
        raise RuntimeError(f"{source_hf}: only {len(out)}/{n} rows with non-empty problem")
    return out[:n]


def materialize_beyondaime(out_path: Path, n: int = 100) -> int:
    """Beyond-AIME: ByteDance-Seed/BeyondAIME test split (100 problems)."""
    ds = _load_hf_dataset(BEYOND_AIME_HF, split="test")
    if len(ds) < n:
        raise RuntimeError(f"{BEYOND_AIME_HF}: only {len(ds)} rows, need {n}")

    def rows() -> Iterator[dict[str, Any]]:
        for i in range(n):
            row = dict(ds[i])
            yield {
                "prompt_id": f"beyond_aime_{i:03d}",
                "problem": _first_str(row, ("problem",)),
                "answer": str(row.get("answer", "")),
                "split": "beyond_aime_eval",
                "source_hf": BEYOND_AIME_HF,
            }

    return _write_jsonl(out_path, rows())


def materialize_matharena(repo_id: str, out_path: Path, split: str, n: int) -> int:
    ds = _load_hf_dataset(repo_id)
    if hasattr(ds, "keys"):
        key = next(iter(ds.keys()))
        rows = [dict(ds[key][i]) for i in range(len(ds[key]))]
    else:
        rows = [dict(ds[i]) for i in range(len(ds))]
    selected = _take_first_n(rows, n, split, repo_id)
    return _write_jsonl(out_path, iter(selected))


def _math_level(row: dict) -> str | None:
    for k in ("level", "difficulty", "type"):
        v = row.get(k)
        if v is None:
            continue
        s = str(v).upper()
        m = re.search(r"L?(\d)", s)
        if m:
            return f"L{m.group(1)}"
    return None


def _proportional_sample(
    pools: dict[tuple, list[dict]], n: int, seed: int
) -> list[dict]:
    """Sample n rows with counts proportional to cell sizes (largest remainder)."""
    keys = list(pools.keys())
    total = sum(len(pools[k]) for k in keys)
    if total < n:
        raise RuntimeError(f"cannot sample {n} from {total} rows")

    raw = {k: n * len(pools[k]) / total for k in keys}
    counts = {k: int(raw[k]) for k in keys}
    remainder = n - sum(counts.values())
    order = sorted(keys, key=lambda k: raw[k] - counts[k], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1

    rng = random.Random(seed)
    selected: list[dict] = []
    for k in keys:
        pool = list(pools[k])
        rng.shuffle(pool)
        take = min(counts[k], len(pool))
        selected.extend(pool[:take])
    if len(selected) < n:
        # top up from largest remaining pools
        rng.shuffle(selected)
        flat = [r for k in keys for r in pools[k] if r not in selected]
        rng.shuffle(flat)
        selected.extend(flat[: n - len(selected)])
    rng.shuffle(selected)
    return selected[:n]


def materialize_math500_sanity(out_path: Path) -> int:
    ds = _load_hf_dataset(MATH500_HF, split="test")
    pools: dict[tuple, list[dict]] = {}
    for i in range(len(ds)):
        row = dict(ds[i])
        key = (row.get("level"), row.get("subject"))
        pools.setdefault(key, []).append(row)

    selected = _proportional_sample(pools, MATH500_SANITY_N, MATH500_SANITY_SEED)

    def rows() -> Iterator[dict[str, Any]]:
        for row in selected:
            pid = _first_str(row, ("unique_id", "id")) or "unknown"
            yield {
                "prompt_id": f"math500_{pid}",
                "problem": _first_str(row, ("problem", "question")),
                "answer": _first_str(row, ("answer", "solution")),
                "split": "math500_sanity",
                "level": row.get("level"),
                "subject": row.get("subject"),
                "source_hf": MATH500_HF,
            }

    return _write_jsonl(out_path, rows())


def materialize_math500_full(out_path: Path) -> int:
    table = _load_hf_dataset(MATH500_HF, split="test")

    def rows() -> Iterator[dict]:
        for i in range(len(table)):
            row = dict(table[i])
            pid = _first_str(row, ("unique_id", "id", "problem_id")) or f"{i:05d}"
            yield {
                "prompt_id": f"math500_{pid}",
                "problem": _first_str(row, ("problem", "question")),
                "answer": _first_str(row, ("answer", "solution")),
                "split": "math500_full",
                "level": row.get("level"),
                "subject": row.get("subject"),
                "source_hf": MATH500_HF,
            }

    return _write_jsonl(out_path, rows())


def _update_lock_hashes(lock: dict, updates: dict[str, str]) -> None:
    lock["train"]["sha256"] = updates["train"]
    lock["pilot_eval"]["primary_sha256"] = updates["primary"]
    lock["pilot_eval"]["secondary_sha256"] = updates["secondary"]
    lock["pilot_eval"]["sanity_sha256"] = updates["sanity"]
    lock["paper_eval"]["primary_hard_sha256"] = updates["primary_hard"]
    lock["paper_eval"]["secondary_hard_sha256"] = updates["secondary_hard"]
    lock["paper_eval"]["math500_full_sha256"] = updates["math500_full"]
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n")


def _hash_existing(paths: dict[str, Path]) -> dict[str, str]:
    return {key: _sha256(path) for key, path in paths.items() if path.exists()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output JSONL already exists; refresh all SHA256 in lock.",
    )
    args = ap.parse_args()

    lock = json.loads(LOCK_PATH.read_text())

    tasks: list[tuple[str, Path, str, Callable[[], None]]] = [
        ("DaPO train 3k", DATA / "dapo_slice_3k.jsonl", "train", lambda: materialize_dapo_slice(DATA / "dapo_slice_3k.jsonl")),
        ("AIME 2025 eval 30", DATA / "aime25_eval_30.jsonl", "primary", lambda: materialize_matharena("MathArena/aime_2025", DATA / "aime25_eval_30.jsonl", "aime25_eval_30", 30)),
        ("HMMT Nov 2025 eval 30", DATA / "hmmt_nov25_eval_30.jsonl", "secondary", lambda: materialize_matharena("MathArena/hmmt_nov_2025", DATA / "hmmt_nov25_eval_30.jsonl", "hmmt_nov25_eval_30", 30)),
        ("MATH-500 sanity 100", DATA / "math500_sanity_100.jsonl", "sanity", lambda: materialize_math500_sanity(DATA / "math500_sanity_100.jsonl")),
        ("Beyond-AIME paper 100", DATA / "beyond_aime_eval_100.jsonl", "primary_hard", lambda: materialize_beyondaime(DATA / "beyond_aime_eval_100.jsonl", 100)),
        ("HMMT Feb 2025 paper 30", DATA / "hmmt_feb25_eval_30.jsonl", "secondary_hard", lambda: materialize_matharena("MathArena/hmmt_feb_2025", DATA / "hmmt_feb25_eval_30.jsonl", "hmmt_feb25_eval_30", 30)),
        ("MATH-500 full 500", DATA / "math500_eval_500.jsonl", "math500_full", lambda: materialize_math500_full(DATA / "math500_eval_500.jsonl")),
    ]

    for label, path, key, run in tasks:
        print(f"=== {label} ===")
        if args.resume and path.exists() and path.stat().st_size > 0:
            h = _sha256(path)
            print(f"  skip (exists) {path} sha256={h[:16]}...")
            continue
        run()
        h = _sha256(path)
        print(f"  wrote {path} ({path.stat().st_size} bytes) sha256={h[:16]}...")

    all_paths = {
        "train": DATA / "dapo_slice_3k.jsonl",
        "primary": DATA / "aime25_eval_30.jsonl",
        "secondary": DATA / "hmmt_nov25_eval_30.jsonl",
        "sanity": DATA / "math500_sanity_100.jsonl",
        "primary_hard": DATA / "beyond_aime_eval_100.jsonl",
        "secondary_hard": DATA / "hmmt_feb25_eval_30.jsonl",
        "math500_full": DATA / "math500_eval_500.jsonl",
    }
    _update_lock_hashes(lock, _hash_existing(all_paths))

    # Remove deprecated placeholder files
    deprecated = [
        "beyond_aime_hard_200.jsonl",
        "hmtt_hard_100.jsonl",
        "aime25_sanity_50.jsonl",
    ]
    for name in deprecated:
        p = DATA / name
        if p.exists():
            p.unlink()
            print(f"  removed deprecated {p}")

    print(f"\nUpdated {LOCK_PATH}")


if __name__ == "__main__":
    main()
