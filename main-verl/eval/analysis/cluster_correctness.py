"""Per-prompt cluster correctness distribution.

For each prompt with >=1 correct rollout, rank its non-degenerate clusters by
frequency (most-common = rank 1). Then for each rank, count P(rank's cluster
is the correct cluster).

If minority objective worked as designed (rarest cluster more likely to be the
correct cluster), we'd see P(correct | rarest rank) > P(correct | most-common rank).
If it's noise, all ranks are equiprobable at ~1/n_clusters.

Also reports:
- raw cluster size distribution per arm (how skewed are clusters?)
- per-arm "rarest = correct" hit rate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ARMS = {
    "minority": "main/data/probes/per_rollout_v2/minority",
    "grpo": "main/data/probes/per_rollout_v2/grpo",
    "polyepo": "main/data/probes/per_rollout_v2/polyepo",
}
DEGENERATE = -1


def iter_step_files(arm_dir: Path, step_set=None):
    files = {}
    for sub in arm_dir.iterdir():
        if not sub.is_dir():
            continue
        is_resume = sub.name != "unknown_run"
        for f in sub.glob("step_*.jsonl"):
            try:
                step = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if step_set is not None and step not in step_set:
                continue
            if step in files and not is_resume:
                continue
            files[step] = f
    for step in sorted(files):
        yield step, files[step]


def load_step(path: Path) -> dict[str, list[dict]]:
    by_prompt = defaultdict(list)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            by_prompt[row["prompt_id"]].append(row)
    return by_prompt


def analyze(arm_dir: Path, step_range=None, arm_label: str = ""):
    """Aggregate per-rank correctness probability across all sampled steps."""
    # rank_correct_count[rank] = #prompts where cluster ranked at this position is correct
    # rank_total_count[rank]   = #prompts where this rank position exists
    rank_correct_count = Counter()
    rank_total_count = Counter()
    rarest_correct = 0  # rarest cluster == correct cluster
    rarest_total = 0
    most_common_correct = 0  # most-common cluster == correct cluster
    cluster_size_histogram = Counter()  # cluster size in #rollouts per prompt
    n_prompts_eligible = 0
    n_prompts_with_correct = 0

    for step, path in iter_step_files(arm_dir, step_range):
        by_prompt = load_step(path)
        for prompt_id, rollouts in by_prompt.items():
            n_correct = sum(1 for r in rollouts if r["reward"] > 0.5)
            if n_correct == 0:
                continue
            n_prompts_with_correct += 1
            cluster_ids = [r["cluster_id"] for r in rollouts]
            nondeg = [c for c in cluster_ids if c != DEGENERATE]
            if len(set(nondeg)) < 2:
                continue
            n_prompts_eligible += 1
            freq = Counter(nondeg)
            # sort clusters by frequency (most common first)
            sorted_clusters = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
            cluster_correct_count = Counter()
            for r in rollouts:
                if r["reward"] > 0.5 and r["cluster_id"] != DEGENERATE:
                    cluster_correct_count[r["cluster_id"]] += 1
            if not cluster_correct_count:
                continue
            most_correct_cluster = cluster_correct_count.most_common(1)[0][0]
            # Record per-rank correctness
            for rank, (cluster_id, _) in enumerate(sorted_clusters, start=1):
                rank_total_count[rank] += 1
                if cluster_id == most_correct_cluster:
                    rank_correct_count[rank] += 1
            # Cluster size histogram
            for size in freq.values():
                cluster_size_histogram[size] += 1
            # Rarest = correct?
            rarest_size = sorted_clusters[-1][1]
            rarest_clusters = {c for c, v in freq.items() if v == rarest_size}
            rarest_total += 1
            if most_correct_cluster in rarest_clusters:
                rarest_correct += 1
            # Most common = correct?
            top_size = sorted_clusters[0][1]
            top_clusters = {c for c, v in freq.items() if v == top_size}
            if most_correct_cluster in top_clusters:
                most_common_correct += 1

    return {
        "arm": arm_label,
        "n_prompts_with_correct": n_prompts_with_correct,
        "n_prompts_eligible": n_prompts_eligible,
        "rank_correct_count": dict(rank_correct_count),
        "rank_total_count": dict(rank_total_count),
        "rarest_eq_correct_rate": rarest_correct / rarest_total if rarest_total else None,
        "most_common_eq_correct_rate": most_common_correct / rarest_total if rarest_total else None,
        "cluster_size_histogram": dict(cluster_size_histogram),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/nancybao/Desktop/dev/cs224r_finalproject")
    ap.add_argument("--step-min", type=int, default=100,
                    help="lower step bound (default 100, skip warmup)")
    ap.add_argument("--step-max", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=10)
    args = ap.parse_args()

    step_set = set(range(args.step_min, args.step_max + 1, args.sample_every))

    print(f"Sampling steps {sorted(step_set)[0]}..{sorted(step_set)[-1]} every {args.sample_every}")
    print()

    results = {}
    for arm, rel in ARMS.items():
        d = Path(args.root) / rel
        if not d.exists():
            continue
        results[arm] = analyze(d, step_set, arm_label=arm)

    # Print
    for arm, r in results.items():
        print(f"=== {arm} ===")
        print(f"  prompts with >=1 correct (across sampled steps): {r['n_prompts_with_correct']}")
        print(f"  eligible (>=2 non-deg clusters): {r['n_prompts_eligible']}")
        print()
        # P(rank == correct) for ranks 1, 2, 3, ...
        print(f"  P(cluster at rank R is the correct cluster):")
        print(f"    rank | count | total | P(correct)")
        for rank in sorted(r["rank_total_count"]):
            count = r["rank_correct_count"].get(rank, 0)
            total = r["rank_total_count"][rank]
            prob = count / total if total else 0
            print(f"      {rank}  |  {count:>4d} | {total:>4d} |  {prob:.3f}")
        print()
        rarest_val = r.get("rarest_eq_correct_rate")
        mc_val = r.get("most_common_eq_correct_rate")
        print(f"  rarest cluster == correct: {rarest_val:.3f}" if rarest_val is not None else "  rarest cluster == correct: n/a (no clusters; e.g. GRPO has no judge during training)")
        print(f"  most-common cluster == correct: {mc_val:.3f}" if mc_val is not None else "  most-common cluster == correct: n/a")
        print()
        print(f"  Cluster size histogram (size: #occurrences):")
        for size in sorted(r["cluster_size_histogram"]):
            print(f"    size={size}: {r['cluster_size_histogram'][size]}")
        print()


if __name__ == "__main__":
    main()
