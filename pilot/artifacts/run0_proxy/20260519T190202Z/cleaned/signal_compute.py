#!/usr/bin/env python3
"""Run 0 cleaned-label signal stats for signal_investigation.md.

Reads cleaned/predictions.jsonl (immutable completions). No LLM calls.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from pilot.train.run_proxy import has_minority_correct_cluster  # noqa: E402

ARTIFACT = Path(__file__).resolve().parent.parent
CLEANED = Path(__file__).resolve().parent / "predictions.jsonl"
PROMPTS = ARTIFACT / "prompt_inputs.jsonl"
OUT_JSON = Path(__file__).resolve().parent / "signal_stats.json"


def dist(counter: Counter, max_k: int) -> dict[int, int]:
    return {k: counter.get(k, 0) for k in range(max_k + 1)}


def main() -> None:
    gold_by_pid: dict[str, str] = {}
    with PROMPTS.open() as f:
        for line in f:
            row = json.loads(line)
            gold_by_pid[row["prompt_id"]] = str(row["gold_answer"])

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with CLEANED.open() as f:
        for line in f:
            r = json.loads(line)
            by_prompt[r["prompt_id"]].append(r)

    assert len(by_prompt) == 500
    rollouts = [r for rs in by_prompt.values() for r in rs]
    assert len(rollouts) == 4000

    n_rollouts = len(rollouts)
    n_prompts = 500

    # --- rollout / prompt correctness ---
    n_correct_stored = sum(bool(r["correct"]) for r in rollouts)
    n_correct_clean = sum(bool(r["correct_clean"]) for r in rollouts)
    prompts_any_stored = sum(
        1 for rs in by_prompt.values() if any(bool(r["correct"]) for r in rs)
    )
    prompts_any_clean = sum(
        1 for rs in by_prompt.values() if any(bool(r["correct_clean"]) for r in rs)
    )

    dist_correct_stored = Counter()
    dist_correct_clean = Counter()
    for rs in by_prompt.values():
        dist_correct_stored[sum(bool(r["correct"]) for r in rs)] += 1
        dist_correct_clean[sum(bool(r["correct_clean"]) for r in rs)] += 1

    # --- minority correct cluster (stored vs clean) ---
    minority_stored = 0
    minority_clean = 0
    minority_clean_canon = 0  # group by canon_clean among correct

    prompt_rows: list[dict] = []
    for pid in sorted(by_prompt):
        rs = by_prompt[pid]
        correct_s = [bool(r["correct"]) for r in rs]
        cids_s = [r["cluster_id"] for r in rs]
        correct_c = [bool(r["correct_clean"]) for r in rs]
        cids_c = [r["cluster_id_clean"] for r in rs]
        canons_c = [r["canon_clean"] for r in rs]

        hm_s = has_minority_correct_cluster(correct_s, cids_s)
        hm_c = has_minority_correct_cluster(correct_c, cids_c)

        # Minority among correct using canon (stable semantic grouping)
        correct_canons = [c for ok, c in zip(correct_c, canons_c) if ok and c]
        hm_canon = False
        if correct_canons:
            freq = Counter(correct_canons)
            maj = max(freq.values())
            hm_canon = any(v < maj for v in freq.values())

        minority_stored += int(hm_s)
        minority_clean += int(hm_c)
        minority_clean_canon += int(hm_canon)

        buckets_all = {r["semantic_bucket"] for r in rs}
        buckets_nonempty = {b for b in buckets_all if b != "empty"}
        correct_buckets = {
            r["semantic_bucket"]
            for r in rs
            if r["correct_clean"] and r["semantic_bucket"] != "empty"
        }
        correct_canons_set = {r["canon_clean"] for r in rs if r["correct_clean"] and r["canon_clean"]}
        correct_clusters = {r["cluster_id_clean"] for r in rs if r["correct_clean"]}

        prompt_rows.append(
            {
                "prompt_id": pid,
                "gold_answer": gold_by_pid[pid],
                "n_correct_stored": sum(correct_s),
                "n_correct_clean": sum(correct_c),
                "has_minority_correct_stored": hm_s,
                "has_minority_correct_clean": hm_c,
                "has_minority_correct_canon": hm_canon,
                "n_semantic_buckets": len(buckets_nonempty),
                "n_semantic_buckets_correct": len(correct_buckets),
                "n_canon_correct": len(correct_canons_set),
                "n_cluster_correct": len(correct_clusters),
                "semantic_buckets": sorted(buckets_nonempty),
                "semantic_buckets_correct": sorted(correct_buckets),
                "canons_correct": sorted(correct_canons_set),
            }
        )

    # Actionable subsets
    actionable_all_modes = [
        p
        for p in prompt_rows
        if p["n_correct_clean"] >= 1 and p["n_semantic_buckets"] >= 2
    ]
    actionable_correct_modes = [
        p
        for p in prompt_rows
        if p["n_correct_clean"] >= 1 and p["n_semantic_buckets_correct"] >= 2
    ]
    actionable_canon = [
        p for p in prompt_rows if p["n_correct_clean"] >= 1 and p["n_canon_correct"] >= 2
    ]

    # Semantic diversity (clean buckets on all rollouts, exclude empty)
    dist_buckets = Counter(p["n_semantic_buckets"] for p in prompt_rows)
    mean_buckets = statistics.mean(p["n_semantic_buckets"] for p in prompt_rows)
    med_buckets = statistics.median(p["n_semantic_buckets"] for p in prompt_rows)

    # extract_path_clean
    path_counts = Counter(r["extract_path_clean"] for r in rollouts)

    # Delta categories
    parsed_changed = [r for r in rollouts if r["parsed_answer"] != r["parsed_answer_clean"]]
    correct_gained = [r for r in rollouts if not r["correct"] and r["correct_clean"]]
    correct_lost = [r for r in rollouts if r["correct"] and not r["correct_clean"]]
    parsed_only = [
        r
        for r in rollouts
        if r["parsed_answer"] != r["parsed_answer_clean"]
        and bool(r["correct"]) == bool(r["correct_clean"])
    ]

    def cat_flip(r: dict) -> str:
        path = r["extract_path_clean"]
        if path == "runon_rejected":
            return "runon_rejected"
        if r.get("parsed_answer_clean") and r["correct_clean"]:
            return "format_fix_correct"
        return "other_flip"

    flip_cats = Counter(cat_flip(r) for r in correct_gained)

    parsed_only_cats = Counter()
    for r in parsed_only:
        if r["extract_path_clean"] == "runon_rejected":
            parsed_only_cats["runon_rejected"] += 1
        elif r["extract_path_clean"] == "boxed_balanced":
            parsed_only_cats["boxed_balanced"] += 1
        elif r["extract_path_clean"] in ("answer_line", "last_line"):
            parsed_only_cats["fallback_line"] += 1
        else:
            parsed_only_cats["other"] += 1

    # Cluster merge/split on prompts with correct
    prompts_cluster_merge = 0
    for rs in by_prompt.values():
        if not any(r["correct_clean"] for r in rs):
            continue
        stored_c = len({r["cluster_id"] for r in rs if r["correct"]})
        clean_c = len({r["cluster_id_clean"] for r in rs if r["correct_clean"]})
        if clean_c < stored_c:
            prompts_cluster_merge += 1

    stats = {
        "n_rollouts": n_rollouts,
        "n_prompts": n_prompts,
        "rollout_correct_rate_stored": n_correct_stored / n_rollouts,
        "rollout_correct_rate_clean": n_correct_clean / n_rollouts,
        "prompt_any_correct_rate_stored": prompts_any_stored / n_prompts,
        "prompt_any_correct_rate_clean": prompts_any_clean / n_prompts,
        "n_correct_stored": n_correct_stored,
        "n_correct_clean": n_correct_clean,
        "dist_correct_per_prompt_stored": dist(dist_correct_stored, 8),
        "dist_correct_per_prompt_clean": dist(dist_correct_clean, 8),
        "minority_correct_prompt_rate_stored": minority_stored / n_prompts,
        "minority_correct_prompt_rate_clean": minority_clean / n_prompts,
        "minority_correct_prompt_rate_canon": minority_clean_canon / n_prompts,
        "n_minority_stored": minority_stored,
        "n_minority_clean": minority_clean,
        "n_minority_canon": minority_clean_canon,
        "mean_semantic_buckets_clean": mean_buckets,
        "median_semantic_buckets_clean": med_buckets,
        "dist_semantic_buckets_per_prompt": dist(dist_buckets, 8),
        "extract_path_clean": dict(path_counts),
        "n_parsed_changed": len(parsed_changed),
        "n_correct_gained": len(correct_gained),
        "n_correct_lost": len(correct_lost),
        "n_parsed_only": len(parsed_only),
        "flip_categories": dict(flip_cats),
        "parsed_only_categories": dict(parsed_only_cats),
        "actionable_all_modes_n": len(actionable_all_modes),
        "actionable_correct_modes_n": len(actionable_correct_modes),
        "actionable_canon_n": len(actionable_canon),
        "prompts_correct_cluster_merged": prompts_cluster_merge,
        "actionable_correct_modes_examples": [
            {
                "prompt_id": p["prompt_id"],
                "gold": p["gold_answer"],
                "n_correct_clean": p["n_correct_clean"],
                "buckets_correct": p["semantic_buckets_correct"],
                "canons_correct": p["canons_correct"],
            }
            for p in sorted(
                actionable_correct_modes,
                key=lambda x: (-x["n_semantic_buckets_correct"], -x["n_correct_clean"]),
            )[:12]
        ],
        "correct_gained_prompt_ids": sorted({r["prompt_id"] for r in correct_gained}),
    }

    OUT_JSON.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in stats if k != "actionable_correct_modes_examples"}, indent=2))


if __name__ == "__main__":
    main()
