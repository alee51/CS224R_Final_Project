#!/usr/bin/env python3
"""One-off analysis: Run 0 prompt-level rollout stats (as-recorded fields)."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
RAW = ARTIFACT_DIR / "raw_predictions.jsonl"
OUT_JSONL = ARTIFACT_DIR / "_prompt_level_stats.jsonl"
OUT_MD = ARTIFACT_DIR / "_prompt_level_summary.md"


def compute_prompt_metrics(rollouts: list[dict]) -> dict:
    parsed = [r["parsed_answer"] for r in rollouts]
    correct = [bool(r["correct"]) for r in rollouts]
    cluster_ids = [r["cluster_id"] for r in rollouts]

    wrong_cids = [cid for ok, cid in zip(correct, cluster_ids) if not ok]

    return {
        "prompt_id": rollouts[0]["prompt_id"],
        "n_rollouts": len(rollouts),
        "n_distinct_parsed": len(set(parsed)),
        "n_distinct_clusters": len(set(cluster_ids)),
        "n_correct_rollouts": sum(correct),
        "n_wrong_clusters": len(set(wrong_cids)) if wrong_cids else 0,
    }


def dist_table(counter: Counter, max_key: int, label: str) -> str:
    lines = [f"| {label} | count | % |", "|---|---:|---:|"]
    total = sum(counter.values())
    for k in range(max_key + 1):
        c = counter.get(k, 0)
        pct = 100.0 * c / total if total else 0
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    return "\n".join(lines)


def main() -> None:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with RAW.open() as f:
        for line in f:
            r = json.loads(line)
            by_prompt[r["prompt_id"]].append(r)

    assert len(by_prompt) == 500, f"expected 500 prompts, got {len(by_prompt)}"
    for pid, rs in by_prompt.items():
        assert len(rs) == 8, f"prompt {pid}: expected 8 rollouts, got {len(rs)}"

    stats = [compute_prompt_metrics(by_prompt[pid]) for pid in sorted(by_prompt)]
    n = len(stats)

    with OUT_JSONL.open("w") as f:
        for row in stats:
            f.write(json.dumps(row) + "\n")

    dist_clusters = Counter(s["n_distinct_clusters"] for s in stats)
    dist_correct_rollouts = Counter(s["n_correct_rollouts"] for s in stats)

    with_correct = [s for s in stats if s["n_correct_rollouts"] > 0]

    clusters_vals = [s["n_distinct_clusters"] for s in stats]
    mean_clusters = statistics.mean(clusters_vals)
    median_clusters = statistics.median(clusters_vals)
    pct_any_correct = 100.0 * len(with_correct) / n

    def fmt(s: dict) -> str:
        return (
            f"`{s['prompt_id'][:8]}…` — "
            f"clusters={s['n_distinct_clusters']}, "
            f"parsed={s['n_distinct_parsed']}, "
            f"correct={s['n_correct_rollouts']}/8, "
            f"wrong_clusters={s['n_wrong_clusters']}"
        )

    by_clusters = sorted(
        stats, key=lambda s: (-s["n_distinct_clusters"], -s["n_distinct_parsed"])
    )
    high_div = by_clusters[0]

    all_wrong = [s for s in stats if s["n_correct_rollouts"] == 0]
    all_wrong.sort(key=lambda s: -s["n_distinct_clusters"])
    aw = all_wrong[0]

    most_correct = max(stats, key=lambda s: s["n_correct_rollouts"])

    exemplars: list[tuple[str, dict]] = [
        ("Max answer diversity (clusters/parsed)", high_div),
        ("All 8 rollouts wrong, high wrong-cluster spread", aw),
        ("Most correct rollouts in one prompt", most_correct),
    ]

    one_correct = [s for s in stats if s["n_correct_rollouts"] == 1]
    if one_correct:
        one_correct.sort(key=lambda s: -s["n_distinct_clusters"])
        exemplars.append(("Single correct rollout among 8", one_correct[0]))

    low_div_many_correct = [
        s for s in with_correct if s["n_distinct_clusters"] <= 3 and s["n_correct_rollouts"] >= 5
    ]
    if low_div_many_correct:
        low_div_many_correct.sort(
            key=lambda s: (-s["n_correct_rollouts"], s["n_distinct_clusters"])
        )
        exemplars.append(("Low diversity, many correct", low_div_many_correct[0]))

    exemplars = exemplars[:8]

    md = []
    md.append("# Run 0 prompt-level rollout summary\n")
    md.append(f"**Artifact:** `{ARTIFACT_DIR.name}`  \n")
    md.append(f"**Prompts:** {n} × 8 rollouts = {n * 8} lines  \n")
    md.append(
        "**Fields:** as-recorded `correct`, `cluster_id`, `parsed_answer` "
        "(no re-canonicalization).\n"
    )
    md.append(
        "**Dropped metrics:** `n_correct_clusters`, `correct_cluster_sizes`, "
        "`largest_correct_cluster_size`, `smallest_correct_cluster_size`, "
        "`minority_correct_cluster`, and `has_any_correct` — with this pilot's "
        "`is_correct()` / `cluster_id` rules, every correct rollout on a prompt "
        "shares one cluster, so those fields are derivable from `n_correct_rollouts`.\n"
    )

    md.append("## Distribution: distinct clusters per prompt (0–8)\n")
    md.append(dist_table(dist_clusters, 8, "n_distinct_clusters"))
    md.append("")

    md.append("## Distribution: correct rollouts per prompt (0–8)\n")
    md.append(dist_table(dist_correct_rollouts, 8, "n_correct_rollouts"))
    md.append("")

    md.append("## Key aggregates\n")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Mean distinct clusters | {mean_clusters:.2f} |")
    md.append(f"| Median distinct clusters | {median_clusters:.1f} |")
    md.append(
        f"| Mean distinct parsed answers | "
        f"{statistics.mean(s['n_distinct_parsed'] for s in stats):.2f} |"
    )
    md.append(
        f"| % prompts with ≥1 correct | {pct_any_correct:.1f}% "
        f"({len(with_correct)}/{n}) |"
    )
    md.append(
        f"| Total correct rollouts | {sum(s['n_correct_rollouts'] for s in stats)} / "
        f"{n * 8} ({100 * sum(s['n_correct_rollouts'] for s in stats) / (n * 8):.1f}%) |"
    )
    md.append("")

    md.append("## Exemplar prompts\n")
    for title, s in exemplars:
        md.append(f"- **{title}:** {fmt(s)}")
    md.append("")

    md.append("## Canonicalization / parse caveats (interpretation skew)\n")
    md.append(
        "- Stored `cluster_id` is consistent with `hash(canonicalize(parsed))` in-process; "
        "**LaTeX/format splits** (e.g. `\\(50\\)` vs `50`) inflate `n_distinct_clusters` "
        "and can split semantically equivalent answers across clusters "
        "(`_audit_parse_cluster.md` §1–2).\n"
        "- **Format false negatives** (~0.15% rollouts) depress `n_correct_rollouts` "
        "without changing cluster structure much.\n"
        "- **Truncated `\\boxed{...}`** parses create spurious clusters and wrong "
        "`correct` flags; true-equivalent parses in different clusters can make "
        "`n_wrong_clusters` look larger than true answer modes.\n"
    )

    md.append("## Interpretation for next experiments\n")
    md.append(
        "The rollout substrate shows **high within-prompt diversity**: median ~"
        f"{median_clusters:.0f} distinct clusters per 8 samples, with only "
        f"**{pct_any_correct:.0f}%** of prompts seeing any correct answer at ~8% "
        "per-rollout accuracy. "
        "That pattern is viable for contrastive or cluster-based training signals—many "
        "wrong modes and occasional correct islands—but **semantic clustering bugs** mean "
        "cluster counts are a noisy superset of true answer modes. "
        "Before trusting cluster-level rewards, fix canonicalization (delimiters, `%`, "
        "brace handling) and boxed extraction; then re-run this table. "
        "Near-term experiments should treat **parsed string diversity** and "
        "**correct-hit rate** as primary viability metrics; use stored `cluster_id` only "
        "after canon fixes, or bucket by normalized parse in analysis.\n"
    )

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_JSONL} ({n} lines)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
