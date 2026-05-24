#!/usr/bin/env python3
"""Re-label Run 0 rollouts with answer_clean (completion text unchanged).

Usage:
    python pilot/scripts/clean_run0_artifacts.py \\
        --artifact-dir pilot/artifacts/run0_proxy/20260519T190202Z
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pilot.train.answer_clean import (  # noqa: E402
    extract_answer_clean,
    is_correct_clean,
    is_runon,
    is_truncated_boxed,
    cluster_id_clean,
    nested_boxed_mismatch,
    normalize_answer_clean,
    semantic_bucket_clean,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dist_table(counter: Counter, max_key: int, label: str) -> str:
    lines = [f"| {label} | count | % |", "|---|---:|---:|"]
    total = sum(counter.values())
    for k in range(max_key + 1):
        c = counter.get(k, 0)
        pct = 100.0 * c / total if total else 0.0
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    return "\n".join(lines)


def process_rollout(r: dict, gold: str) -> dict:
    completion = r.get("completion", "")
    parsed, path = extract_answer_clean(completion)
    canon = normalize_answer_clean(parsed)
    correct = is_correct_clean(parsed, gold)
    cid = cluster_id_clean(parsed)
    bucket = semantic_bucket_clean(parsed)
    is_runon_fb = path == "runon_rejected" or (
        path in ("answer_line", "last_line") and is_runon(parsed)
    )

    out = dict(r)
    out.update(
        {
            "parsed_answer_clean": parsed,
            "correct_clean": correct,
            "cluster_id_clean": cid,
            "extract_path_clean": path,
            "semantic_bucket": bucket,
            "is_runon_fallback": is_runon_fb,
            "canon_clean": canon,
        }
    )
    return out


def prompt_stats(pid: str, gold: str, rollouts: list[dict]) -> dict:
    stored_correct = sum(bool(r["correct"]) for r in rollouts)
    clean_correct = sum(bool(r["correct_clean"]) for r in rollouts)
    return {
        "prompt_id": pid,
        "gold_answer": gold,
        "n_rollouts": len(rollouts),
        "n_correct_stored": stored_correct,
        "n_correct_clean": clean_correct,
        "n_distinct_parsed_stored": len({r["parsed_answer"] for r in rollouts}),
        "n_distinct_parsed_clean": len({r["parsed_answer_clean"] for r in rollouts}),
        "n_distinct_cluster_stored": len({r["cluster_id"] for r in rollouts}),
        "n_distinct_cluster_clean": len({r["cluster_id_clean"] for r in rollouts}),
        "n_semantic_buckets": len({r["semantic_bucket"] for r in rollouts}),
        "n_parsed_changed": sum(
            r["parsed_answer"] != r["parsed_answer_clean"] for r in rollouts
        ),
        "n_correct_flipped": sum(
            bool(r["correct"]) != bool(r["correct_clean"]) for r in rollouts
        ),
        "n_runon_rejected": sum(r["extract_path_clean"] == "runon_rejected" for r in rollouts),
        "extract_path_clean_counts": dict(
            Counter(r["extract_path_clean"] for r in rollouts)
        ),
    }


def build_delta_report(
    rollouts: list[dict],
    prompt_rows: list[dict],
    gold_by_pid: dict[str, str],
) -> str:
    n = len(rollouts)
    n_parsed_chg = sum(r["parsed_answer"] != r["parsed_answer_clean"] for r in rollouts)
    n_correct_chg = sum(bool(r["correct"]) != bool(r["correct_clean"]) for r in rollouts)

    # Cluster grouping change: same stored cluster partition vs clean canon groups
    def _grouping_changed(rs: list[dict], key_a: str, key_b: str) -> bool:
        pairs_a = {(r["prompt_id"], r[key_a]) for r in rs}
        pairs_b = {(r["prompt_id"], r[key_b]) for r in rs}
        for pid in {r["prompt_id"] for r in rs}:
            sub = [r for r in rs if r["prompt_id"] == pid]
            map_a: dict[int, set[str]] = defaultdict(set)
            map_b: dict[int, set[str]] = defaultdict(set)
            for r in sub:
                map_a[r[key_a]].add(r["parsed_answer"])
                map_b[r[key_b]].add(r["parsed_answer_clean"])
            if {frozenset(v) for v in map_a.values()} != {frozenset(v) for v in map_b.values()}:
                return True
        return False

    cluster_grouping_changed = 0
    for pid in {r["prompt_id"] for r in rollouts}:
        sub = [r for r in rollouts if r["prompt_id"] == pid]
        stored_parts = tuple(sorted({r["cluster_id"] for r in sub}))
        clean_parts = tuple(sorted({r["cluster_id_clean"] for r in sub}))
        stored_canon = tuple(sorted({normalize_answer_clean(r["parsed_answer"]) for r in sub}))
        clean_canon = tuple(sorted({r["canon_clean"] for r in sub}))
        if stored_canon != clean_canon or stored_parts != clean_parts:
            cluster_grouping_changed += 1

    # Prompt-level flip counts
    prompts_any_correct_flip = 0
    prompts_gained = 0
    prompts_lost = 0
    for pr in prompt_rows:
        pid = pr["prompt_id"]
        sub = [r for r in rollouts if r["prompt_id"] == pid]
        had_ft = any(bool(r["correct"]) != bool(r["correct_clean"]) for r in sub)
        if had_ft:
            prompts_any_correct_flip += 1
        if any(not r["correct"] and r["correct_clean"] for r in sub):
            prompts_gained += 1
        if any(r["correct"] and not r["correct_clean"] for r in sub):
            prompts_lost += 1

    dist_correct_stored = Counter(pr["n_correct_stored"] for pr in prompt_rows)
    dist_correct_clean = Counter(pr["n_correct_clean"] for pr in prompt_rows)
    mean_clusters_stored = statistics.mean(pr["n_distinct_cluster_stored"] for pr in prompt_rows)
    mean_clusters_clean = statistics.mean(pr["n_distinct_cluster_clean"] for pr in prompt_rows)

    def _collect(cat: str, pred) -> None:
        seen: set[str] = set()
        for r in rollouts:
            if len(seen) >= 10:
                break
            if not pred(r):
                continue
            pid = r["prompt_id"]
            if pid not in seen:
                seen.add(pid)
                cat_prompts[cat].append(pid)

    cat_prompts: dict[str, list[str]] = defaultdict(list)
    _collect(
        "correct_gained",
        lambda r: not r["correct"] and r["correct_clean"],
    )
    _collect(
        "correct_lost",
        lambda r: r["correct"] and not r["correct_clean"],
    )
    _collect(
        "parsed_changed_correct_unchanged",
        lambda r: r["parsed_answer"] != r["parsed_answer_clean"]
        and bool(r["correct"]) == bool(r["correct_clean"]),
    )
    _collect(
        "runon_rejected",
        lambda r: r["extract_path_clean"] == "runon_rejected",
    )
    _collect(
        "truncated_boxed",
        lambda r: is_truncated_boxed(r.get("completion", "")),
    )
    _collect(
        "nested_boxed_mismatch",
        lambda r: nested_boxed_mismatch(r.get("completion", "")),
    )

    path_counts = Counter(r["extract_path_clean"] for r in rollouts)
    runon_n = sum(r["extract_path_clean"] == "runon_rejected" for r in rollouts)
    trunc_n = sum(is_truncated_boxed(r.get("completion", "")) for r in rollouts)
    nested_n = sum(nested_boxed_mismatch(r.get("completion", "")) for r in rollouts)

    md: list[str] = []
    md.append("# Run 0 cleaned labels vs stored raw\n")
    md.append("**Source:** `raw_predictions.jsonl` (immutable completions)  \n")
    md.append("**Cleaner:** `pilot/train/answer_clean.py` (`extract_answer_clean`, ")
    md.append("`normalize_answer_clean`, brace-balanced `\\boxed`, run-on rejection)\n")

    md.append("\n## Headline counts (4000 rollouts)\n")
    md.append("| Metric | Count | Rate |\n|---|---:|---:|\n")
    md.append(
        f"| `parsed_answer` changed | {n_parsed_chg} | {100*n_parsed_chg/n:.2f}% |\n"
    )
    md.append(
        f"| `correct` flipped (either direction) | {n_correct_chg} | "
        f"{100*n_correct_chg/n:.2f}% |\n"
    )
    md.append(
        f"| Prompts with different cluster/canon grouping (8 rollouts) | "
        f"{cluster_grouping_changed} | {100*cluster_grouping_changed/500:.1f}% |\n"
    )
    md.append(
        f"| Correct gained (false→true) rollouts | "
        f"{sum(not r['correct'] and r['correct_clean'] for r in rollouts)} |\n"
    )
    md.append(
        f"| Correct lost (true→false) rollouts | "
        f"{sum(r['correct'] and not r['correct_clean'] for r in rollouts)} |\n"
    )

    md.append("\n## Prompt-level correctness\n")
    md.append(
        f"- Prompts with ≥1 rollout where `correct` flipped: **{prompts_any_correct_flip}**\n"
    )
    md.append(f"- Prompts with any correct **gained**: **{prompts_gained}**\n")
    md.append(f"- Prompts with any correct **lost**: **{prompts_lost}**\n")
    md.append(
        f"- Mean distinct stored clusters / prompt: **{mean_clusters_stored:.2f}**\n"
    )
    md.append(
        f"- Mean distinct clean clusters / prompt: **{mean_clusters_clean:.2f}**\n"
    )

    md.append("\n### Distribution: correct rollouts per prompt (stored)\n")
    md.append(_dist_table(dist_correct_stored, 8, "n_correct_stored"))
    md.append("\n\n### Distribution: correct rollouts per prompt (clean)\n")
    md.append(_dist_table(dist_correct_clean, 8, "n_correct_clean"))

    md.append("\n## Extract path (clean)\n")
    md.append("| Path | Count | % |\n|---|---:|---:|\n")
    for p, c in path_counts.most_common():
        md.append(f"| {p} | {c} | {100*c/n:.1f}% |\n")
    md.append(f"\n- Run-on rejected: **{runon_n}** ({100*runon_n/n:.1f}%)\n")
    md.append(
        f"- Truncated `\\boxed{{` (unclosed at end): **{trunc_n}** "
        f"({100*trunc_n/n:.1f}%)\n"
    )
    md.append(
        f"- Nested boxed: balanced ≠ shallow-regex: **{nested_n}** "
        f"({100*nested_n/n:.1f}%)\n"
    )

    md.append("\n## Flip categories (example prompt_ids)\n")
    labels = {
        "correct_gained": "Correct gained (format/LaTeX/boxed fix)",
        "correct_lost": "Correct lost",
        "parsed_changed_correct_unchanged": "Parsed changed, correct unchanged",
        "runon_rejected": "Run-on fallback rejected",
        "truncated_boxed": "Truncated `\\boxed{` / cut-off completion",
        "nested_boxed_mismatch": "Nested boxed: regex vs balanced inner mismatch",
    }
    for key, title in labels.items():
        pids = cat_prompts.get(key, [])[:10]
        md.append(f"\n### {title}\n")
        if pids:
            md.append(", ".join(f"`{p}`" for p in pids) + "\n")
        else:
            md.append("_None flagged at prompt level._\n")

    md.append("\n## Limitations\n")
    md.append(
        "- **Run-on heuristic** may reject valid long math tails or accept short prose.\n"
    )
    md.append(
        "- **Brace-balanced boxed** prefers the *last* boxed; multiple answers in one "
        "completion are not disambiguated.\n"
    )
    md.append(
        "- **`normalize_answer_clean`** merges some format variants but not all "
        "mathematically equivalent forms (e.g. unsimplified radicals).\n"
    )
    md.append(
        "- **Truncated boxed** detection is syntactic (unclosed opener), not semantic "
        "completeness of the math.\n"
    )
    md.append(
        "- Stored `cluster_id` uses Python `hash()` (process-dependent); "
        "`cluster_id_clean` uses SHA-256 — compare via `canon_clean`, not raw ints.\n"
    )

    return "".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Run 0 artifact labels")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing raw_predictions.jsonl and prompt_inputs.jsonl",
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    if not artifact_dir.is_absolute():
        artifact_dir = (REPO / artifact_dir).resolve()

    raw_path = artifact_dir / "raw_predictions.jsonl"
    prompts_path = artifact_dir / "prompt_inputs.jsonl"
    out_dir = artifact_dir / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_by_pid: dict[str, str] = {}
    for row in _load_jsonl(prompts_path):
        gold_by_pid[row["prompt_id"]] = str(row["gold_answer"])

    cleaned: list[dict] = []
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with raw_path.open() as f:
        for line in f:
            r = json.loads(line)
            gold = gold_by_pid[r["prompt_id"]]
            out = process_rollout(r, gold)
            cleaned.append(out)
            by_prompt[r["prompt_id"]].append(out)

    assert len(cleaned) == 4000, f"expected 4000 rollouts, got {len(cleaned)}"
    assert len(by_prompt) == 500, f"expected 500 prompts, got {len(by_prompt)}"

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_rows = [
        prompt_stats(pid, gold_by_pid[pid], by_prompt[pid])
        for pid in sorted(by_prompt)
    ]
    ps_path = out_dir / "prompt_stats.jsonl"
    with ps_path.open("w") as f:
        for row in prompt_rows:
            f.write(json.dumps(row) + "\n")

    n = len(cleaned)
    metrics = {
        "n_rollouts": n,
        "n_prompts": len(prompt_rows),
        "n_parsed_changed": sum(
            r["parsed_answer"] != r["parsed_answer_clean"] for r in cleaned
        ),
        "n_correct_flipped": sum(
            bool(r["correct"]) != bool(r["correct_clean"]) for r in cleaned
        ),
        "n_correct_stored": sum(bool(r["correct"]) for r in cleaned),
        "n_correct_clean": sum(bool(r["correct_clean"]) for r in cleaned),
        "n_correct_gained": sum(
            not r["correct"] and r["correct_clean"] for r in cleaned
        ),
        "n_correct_lost": sum(r["correct"] and not r["correct_clean"] for r in cleaned),
        "extract_path_clean": dict(Counter(r["extract_path_clean"] for r in cleaned)),
        "n_runon_rejected": sum(
            r["extract_path_clean"] == "runon_rejected" for r in cleaned
        ),
        "n_truncated_boxed": sum(
            is_truncated_boxed(r.get("completion", "")) for r in cleaned
        ),
        "n_nested_boxed_mismatch": sum(
            nested_boxed_mismatch(r.get("completion", "")) for r in cleaned
        ),
        "mean_distinct_cluster_stored": statistics.mean(
            pr["n_distinct_cluster_stored"] for pr in prompt_rows
        ),
        "mean_distinct_cluster_clean": statistics.mean(
            pr["n_distinct_cluster_clean"] for pr in prompt_rows
        ),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    delta_path = out_dir / "delta_vs_raw.md"
    delta_path.write_text(build_delta_report(cleaned, prompt_rows, gold_by_pid))

    print(f"Wrote {pred_path} ({n} lines)")
    print(f"Wrote {ps_path} ({len(prompt_rows)} lines)")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {delta_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "extract_path_clean"}, indent=2))


if __name__ == "__main__":
    main()
