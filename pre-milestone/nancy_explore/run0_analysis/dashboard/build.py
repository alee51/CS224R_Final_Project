#!/usr/bin/env python3
"""Build per-prompt cluster-review dashboard.

Reads `data/cleaned_answers.parquet` (canonical human-verified answers),
`data/predictions_reparsed.jsonl` (for completion text only),
`data/prompt_inputs.jsonl`, and `analysis_a/llm_clusters_summary.parquet`.
Writes data.js for the static index.html.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREDS = ROOT / "data" / "predictions_reparsed.jsonl"
PROMPTS = ROOT / "data" / "prompt_inputs.jsonl"
CLEANED_PARQUET = ROOT / "data" / "cleaned_answers.parquet"
LLM_PARQUET = ROOT / "analysis_a" / "llm_clusters_summary.parquet"
OUT = HERE / "data.js"
POLY_EPO_DEGENERATE = 100


def normalize_llm_cluster_id(cid: int) -> int:
    if cid == POLY_EPO_DEGENERATE:
        return -1
    return int(cid)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def largest_correct_cluster_size(cluster_ids: list, correct: list[bool]) -> int:
    """Size of the largest cluster that contains at least one correct rollout."""
    if not any(correct):
        return 0
    correct_clusters = {cid for cid, ok in zip(cluster_ids, correct) if ok}
    counts = Counter(cluster_ids)
    return max(counts[c] for c in correct_clusters)


def has_minority_correct(cluster_ids: list, correct: list[bool]) -> bool:
    """True if correct rollouts span >=2 clusters with at least one minority."""
    correct_clusters = [c for c, ok in zip(cluster_ids, correct) if ok]
    if len(correct_clusters) < 2:
        return False
    freq = Counter(correct_clusters)
    majority = max(freq.values())
    return any(v < majority for v in freq.values())


def main() -> None:
    preds = load_jsonl(PREDS)
    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS)}
    cleaned = pd.read_parquet(CLEANED_PARQUET)
    cleaned_by_key: dict[tuple[str, int], dict] = {}
    for row in cleaned.itertuples(index=False):
        cleaned_by_key[(row.prompt_id, int(row.rollout_idx))] = {
            "cleaned_answer": row.cleaned_answer,
            "cleaned_state": row.cleaned_state,
            "cleaned_correct": bool(row.cleaned_correct),
            "cleaned_cluster_id": int(row.cleaned_cluster_id),
        }

    llm = pd.read_parquet(LLM_PARQUET)
    llm_by_pid: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in llm.itertuples(index=False):
        llm_by_pid[row.prompt_id][int(row.rollout_idx)] = {
            "llm_cluster_id": normalize_llm_cluster_id(int(row.llm_cluster_id)),
            "parse_ok": bool(row.parse_ok),
        }

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in preds:
        by_prompt[r["prompt_id"]].append(r)

    out_prompts = []
    for pid, rollouts in by_prompt.items():
        # rollout_idx not in jsonl — assign by order
        for i, r in enumerate(rollouts):
            r["_idx"] = i

        cleaned_per = [cleaned_by_key[(pid, i)] for i in range(len(rollouts))]
        cluster_cleaned = [c["cleaned_cluster_id"] for c in cleaned_per]
        llm_ids = [llm_by_pid.get(pid, {}).get(i, {}).get("llm_cluster_id", -999) for i in range(len(rollouts))]
        correct = [c["cleaned_correct"] for c in cleaned_per]

        cleaned_sizes = Counter(cluster_cleaned)
        llm_sizes = Counter(llm_ids)

        n_clusters_cleaned = len(cleaned_sizes)
        n_clusters_llm = len(llm_sizes)
        n_correct = sum(correct)

        largest_correct_cleaned = largest_correct_cluster_size(cluster_cleaned, correct)
        largest_correct_llm = largest_correct_cluster_size(llm_ids, correct)
        minority_cleaned = has_minority_correct(cluster_cleaned, correct)
        minority_llm = has_minority_correct(llm_ids, correct)

        prompt_meta = prompts.get(pid, {})

        out_rollouts = []
        for i, r in enumerate(rollouts):
            c = cleaned_per[i]
            cid = c["cleaned_cluster_id"]
            lid = llm_ids[i]
            out_rollouts.append({
                "idx": i,
                "completion": r.get("completion", ""),
                "cleaned_answer": c["cleaned_answer"],
                "cleaned_state": c["cleaned_state"],
                "cleaned_correct": c["cleaned_correct"],
                "cleaned_cluster_id": cid,
                "cleaned_cluster_size": cleaned_sizes[cid],
                "llm_cluster_id": lid,
                "llm_cluster_size": llm_sizes[lid],
                "llm_degenerate": lid == -1,
            })

        out_prompts.append({
            "prompt_id": pid,
            "problem": prompt_meta.get("problem", ""),
            "gold_answer": prompt_meta.get("gold_answer", ""),
            "n_rollouts": len(rollouts),
            "n_correct": n_correct,
            "n_clusters_cleaned": n_clusters_cleaned,
            "n_clusters_llm": n_clusters_llm,
            "largest_correct_cleaned": largest_correct_cleaned,
            "largest_correct_llm": largest_correct_llm,
            "minority_correct_cleaned": minority_cleaned,
            "minority_correct_llm": minority_llm,
            "rollouts": out_rollouts,
        })

    # Sort by prompt_id for stable ordering
    out_prompts.sort(key=lambda p: p["prompt_id"])

    payload = {
        "prompts": out_prompts,
        "n_prompts": len(out_prompts),
    }
    OUT.write_text("window.DASHBOARD = " + json.dumps(payload) + ";\n")
    print(f"wrote {OUT} — {len(out_prompts)} prompts")


if __name__ == "__main__":
    main()
