#!/usr/bin/env python3
"""Build static Run 0 review dashboard from jsonl artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR.parent
RAW_PREDICTIONS_PATH = ARTIFACT_DIR / "raw_predictions.jsonl"
CLEANED_PREDICTIONS_PATH = ARTIFACT_DIR / "cleaned" / "predictions.jsonl"
PROMPTS_PATH = ARTIFACT_DIR / "prompt_inputs.jsonl"
DATA_JS_PATH = SCRIPT_DIR / "data.js"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def rollout_key(row: dict) -> tuple[str, str]:
    return (row["prompt_id"], row.get("completion", "") or "")


def source_triplet(row: dict, prefix: str) -> dict:
    """Extract parsed_answer, correct, cluster_id for raw or clean."""
    if prefix == "clean":
        parsed = str(row.get("parsed_answer_clean", row.get("parsed_answer", "")))
        correct = bool(row.get("correct_clean", row.get("correct", False)))
        cluster_id = row.get("cluster_id_clean", row.get("cluster_id"))
        out: dict = {
            "parsed_answer": parsed,
            "correct": correct,
            "cluster_id": cluster_id,
        }
        if "extract_path_clean" in row:
            out["extract_path_clean"] = row["extract_path_clean"]
        if "is_runon_fallback" in row:
            out["is_runon_fallback"] = bool(row.get("is_runon_fallback", False))
        return out
    parsed = str(row.get("parsed_answer", ""))
    correct = bool(row.get("correct", False))
    cluster_id = row.get("cluster_id")
    return {
        "parsed_answer": parsed,
        "correct": correct,
        "cluster_id": cluster_id,
    }


def compute_delta(raw: dict, clean: dict) -> dict:
    """Meaningful label deltas for UI highlighting (not cluster_id or run-on rejection)."""
    parsed_diff = raw["parsed_answer"] != clean["parsed_answer"]
    if clean.get("is_runon_fallback"):
        # Clean empty/rejected a prose tail — not a substantive relabel.
        parsed_diff = False
    return {
        "parsed": parsed_diff,
        "correct": raw["correct"] != clean["correct"],
        "cluster": False,
    }


def index_predictions(rows: list[dict]) -> dict[tuple[str, str], dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = rollout_key(row)
        if key in by_key:
            print(
                f"warning: duplicate rollout key prompt_id={key[0]!r}",
                file=sys.stderr,
            )
        by_key[key] = row
    return by_key


def merge_rollout(
    raw_row: dict | None,
    clean_row: dict | None,
    *,
    single_source: str | None,
) -> dict | None:
    if raw_row is None and clean_row is None:
        return None

    if single_source == "raw":
        row = raw_row
        assert row is not None
        completion = row.get("completion", "") or ""
        raw = source_triplet(row, "raw")
        clean = dict(raw)
        delta = {"parsed": False, "correct": False, "cluster": False}
    elif single_source == "cleaned":
        row = clean_row
        assert row is not None
        completion = row.get("completion", "") or ""
        raw = source_triplet(row, "raw")
        clean = source_triplet(row, "clean")
        delta = compute_delta(raw, clean)
    else:
        row = raw_row or clean_row
        assert row is not None
        completion = row.get("completion", "") or ""
        if raw_row is not None and clean_row is not None:
            raw = source_triplet(raw_row, "raw")
            clean = source_triplet(clean_row, "clean")
        elif raw_row is not None:
            raw = source_triplet(raw_row, "raw")
            clean = source_triplet(raw_row, "clean")
        else:
            raw = source_triplet(clean_row, "raw")
            clean = source_triplet(clean_row, "clean")
        delta = compute_delta(raw, clean)

    return {
        "completion": completion,
        "char_count": len(completion),
        "raw": raw,
        "clean": clean,
        "delta": delta,
    }


def build_dataset(source: str) -> dict:
    prompts_by_id: dict[str, dict] = {}
    prompt_order: list[str] = []

    for row in load_jsonl(PROMPTS_PATH):
        pid = row["prompt_id"]
        prompt_order.append(pid)
        prompts_by_id[pid] = {
            "prompt_id": pid,
            "problem": row.get("problem", ""),
            "gold_answer": str(row.get("gold_answer", "")),
        }

    raw_by_key: dict[tuple[str, str], dict] = {}
    clean_by_key: dict[tuple[str, str], dict] = {}
    single_source: str | None = None

    if source in ("raw", "both"):
        raw_by_key = index_predictions(load_jsonl(RAW_PREDICTIONS_PATH))
    if source in ("cleaned", "both"):
        clean_by_key = index_predictions(load_jsonl(CLEANED_PREDICTIONS_PATH))
    if source == "raw":
        single_source = "raw"
    elif source == "cleaned":
        single_source = "cleaned"

    if source == "both":
        all_keys = set(raw_by_key) | set(clean_by_key)
        only_raw = set(raw_by_key) - set(clean_by_key)
        only_clean = set(clean_by_key) - set(raw_by_key)
        if only_raw:
            print(
                f"warning: {len(only_raw)} rollouts only in raw_predictions.jsonl",
                file=sys.stderr,
            )
        if only_clean:
            print(
                f"warning: {len(only_clean)} rollouts only in cleaned/predictions.jsonl",
                file=sys.stderr,
            )
    elif source == "raw":
        all_keys = set(raw_by_key)
    else:
        all_keys = set(clean_by_key)

    rollouts_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for key in all_keys:
        pid, _completion = key
        merged = merge_rollout(
            raw_by_key.get(key),
            clean_by_key.get(key),
            single_source=single_source,
        )
        if merged is not None:
            rollouts_by_prompt[pid].append(merged)

    prompts_out: list[dict] = []
    for index, pid in enumerate(prompt_order):
        base = prompts_by_id[pid]
        rollouts = rollouts_by_prompt.get(pid, [])
        rollouts.sort(key=lambda r: r["completion"])
        if len(rollouts) != 8:
            print(
                f"warning: prompt {pid} has {len(rollouts)} rollouts (expected 8)",
                file=sys.stderr,
            )

        n_correct_raw = sum(1 for r in rollouts if r["raw"]["correct"])
        n_correct_clean = sum(1 for r in rollouts if r["clean"]["correct"])
        n_clusters_raw = len({r["raw"]["cluster_id"] for r in rollouts})
        n_clusters_clean = len({r["clean"]["cluster_id"] for r in rollouts})
        has_delta = any(
            r["delta"]["parsed"] or r["delta"]["correct"] for r in rollouts
        )

        prompts_out.append(
            {
                "index": index,
                "prompt_id": pid,
                "problem": base["problem"],
                "gold_answer": base["gold_answer"],
                "n_correct_raw": n_correct_raw,
                "n_correct_clean": n_correct_clean,
                "n_clusters_raw": n_clusters_raw,
                "n_clusters_clean": n_clusters_clean,
                "has_delta": has_delta,
                "n_rollouts": len(rollouts),
                "rollouts": rollouts,
            }
        )

    return {
        "source_mode": source,
        "dual_source": source == "both",
        "prompt_count": len(prompts_out),
        "prompts": prompts_out,
    }


def write_data_js(dataset: dict) -> None:
    payload = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    DATA_JS_PATH.write_text(
        f"/* Generated by build_review_dashboard.py — do not edit */\n"
        f"window.REVIEW_DATA = {payload};\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Run 0 review dashboard data.js")
    parser.add_argument(
        "--source",
        choices=("raw", "cleaned", "both"),
        default="both",
        help="Label source(s) to embed (default: both)",
    )
    args = parser.parse_args()

    if args.source in ("raw", "both") and not RAW_PREDICTIONS_PATH.is_file():
        print(f"error: missing {RAW_PREDICTIONS_PATH}", file=sys.stderr)
        return 1
    if args.source in ("cleaned", "both") and not CLEANED_PREDICTIONS_PATH.is_file():
        print(f"error: missing {CLEANED_PREDICTIONS_PATH}", file=sys.stderr)
        return 1
    if not PROMPTS_PATH.is_file():
        print(f"error: missing {PROMPTS_PATH}", file=sys.stderr)
        return 1

    dataset = build_dataset(args.source)
    write_data_js(dataset)
    size_mb = DATA_JS_PATH.stat().st_size / (1024 * 1024)
    delta_prompts = sum(1 for p in dataset["prompts"] if p["has_delta"])
    print(
        f"wrote {DATA_JS_PATH} ({dataset['prompt_count']} prompts, "
        f"{delta_prompts} with deltas, {size_mb:.2f} MB, source={args.source})"
    )
    print("open index.html in a browser (see README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
