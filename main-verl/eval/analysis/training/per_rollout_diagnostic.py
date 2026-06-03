"""Offline §3.3 diagnostic analyses on per-rollout JSONLs.

Comparing yfpxs7wo (minority_cot) vs rof8t8kf (GRPO) vs m29o33k1 (poly_epo_cot).

JSONL schema:
  {global_step, prompt_id, rollout_idx, parsed_answer, reward,
   cluster_id, finish_reason, response_length}

Where cluster_id = -1 indicates DEGENERATE (unparseable/parse fail).

Per `eval_2026-06-01.md` §3.3, this script computes:

  1. minority_vs_correctness_alignment:
     For each step, fraction of prompts where the MINORITY cluster
     (least-frequent non-degenerate cluster) is also the CORRECT cluster.
     If <= 1/N_clusters (chance), minority objective is upweighting wrong
     answers as often as right ones.

  2. all_wrong_rare_reward_signature:
     Fraction of prompts where all 8 rollouts have reward=0 (the
     `bin[0.0, 0.0]` bucket).

  3. response_length_by_reward:
     mean response length per step, split into (reward=0, reward=1).

  4. finish_reason_drift:
     fraction of {eos, stop, length} per step.

  5. distinct_parsed_answers_per_prompt:
     mean count of distinct parsed_answer strings across n rollouts.

  6. distinct_clusters_mean (re-derive what training logged):
     mean count of distinct non-degenerate cluster_ids per prompt.

  7. degenerate_fraction:
     mean fraction of cluster_id=-1 rollouts per prompt.

  8. parsed_answer_entropy:
     mean entropy of the parsed_answer distribution per prompt
     (higher = less mode-collapse).

Output: JSON with per-step metrics for each arm + summary table.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ARMS = {
    "minority": "main/data/probes/per_rollout_v2/minority",
    "grpo": "main/data/probes/per_rollout_v2/grpo",
    "polyepo": "main/data/probes/per_rollout_v2/polyepo",
}

DEGENERATE = -1


def iter_step_files(arm_dir: Path):
    """Yield (step, path) for all step_*.jsonl in arm_dir, sorted by step.

    Looks under both arm_dir/unknown_run/ (original run) and arm_dir/<run_id>/
    (resumed run with WANDB_RUN_ID set). When the same step appears in both,
    prefer the resumed-run file (later in time).
    """
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
            # resumed run takes priority for same step
            if step in files and not is_resume:
                continue
            files[step] = f
    for step in sorted(files):
        yield step, files[step]


def load_step(path: Path) -> dict[str, list[dict]]:
    """Load a step's JSONL, group by prompt_id."""
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            by_prompt[row["prompt_id"]].append(row)
    return by_prompt


def analyze_step(by_prompt: dict[str, list[dict]]) -> dict:
    """Compute all per-step metrics for one step's data."""
    n_prompts = len(by_prompt)
    if n_prompts == 0:
        return {"n_prompts": 0}

    # Accumulators
    minority_is_correct_count = 0
    minority_eligible_count = 0  # prompts with >=2 non-degenerate clusters
    all_wrong_count = 0
    all_right_count = 0
    response_lengths_r0 = []
    response_lengths_r1 = []
    finish_counter = Counter()
    distinct_answers_per_prompt = []
    distinct_clusters_per_prompt = []
    degenerate_fractions = []
    answer_entropies = []
    cluster_correct_alignment_count = 0
    cluster_correct_alignment_eligible = 0

    for prompt_id, rollouts in by_prompt.items():
        n = len(rollouts)
        rewards = [r["reward"] for r in rollouts]
        cluster_ids = [r["cluster_id"] for r in rollouts]
        parsed = [r["parsed_answer"] for r in rollouts]
        lengths = [r["response_length"] for r in rollouts]
        finishes = [r["finish_reason"] for r in rollouts]

        # Reward distribution
        n_correct = sum(1 for r in rewards if r > 0.5)
        if n_correct == 0:
            all_wrong_count += 1
        if n_correct == n:
            all_right_count += 1

        # Length by reward
        for r, l in zip(rewards, lengths):
            (response_lengths_r1 if r > 0.5 else response_lengths_r0).append(l)

        # Finish reasons
        for fr in finishes:
            finish_counter[fr] += 1

        # Distinct parsed answers
        distinct_answers_per_prompt.append(len(set(p for p in parsed if p)))

        # Distinct non-degenerate clusters
        nondeg = [c for c in cluster_ids if c != DEGENERATE]
        distinct_clusters_per_prompt.append(len(set(nondeg)) if nondeg else 0)
        degenerate_fractions.append(sum(1 for c in cluster_ids if c == DEGENERATE) / n)

        # Parsed-answer entropy (over non-empty answers)
        answer_counts = Counter(p for p in parsed if p)
        total = sum(answer_counts.values())
        if total > 0:
            probs = [c / total for c in answer_counts.values()]
            h = -sum(p * math.log2(p) for p in probs if p > 0)
            answer_entropies.append(h)

        # Cluster-correctness alignment: is the cluster_id where the MINORITY
        # of non-degenerate rollouts live ALSO the cluster where most CORRECT
        # rollouts live?
        if len(set(nondeg)) >= 2 and n_correct > 0:
            cluster_correct_alignment_eligible += 1
            # Frequency per non-degenerate cluster
            freq = Counter(nondeg)
            # Find rarest non-degenerate cluster
            min_count = min(freq.values())
            rarest_clusters = {c for c, v in freq.items() if v == min_count}
            # Find correct cluster: most common cluster_id among correct rollouts
            correct_clusters = Counter(
                rollouts[i]["cluster_id"]
                for i, r in enumerate(rewards)
                if r > 0.5 and rollouts[i]["cluster_id"] != DEGENERATE
            )
            if correct_clusters:
                most_correct = correct_clusters.most_common(1)[0][0]
                if most_correct in rarest_clusters:
                    cluster_correct_alignment_count += 1
                    minority_is_correct_count += 1
            minority_eligible_count += 1

    return {
        "n_prompts": n_prompts,
        "all_wrong_frac": all_wrong_count / n_prompts,
        "all_right_frac": all_right_count / n_prompts,
        "mean_response_len_r0": (sum(response_lengths_r0) / len(response_lengths_r0))
        if response_lengths_r0 else 0,
        "mean_response_len_r1": (sum(response_lengths_r1) / len(response_lengths_r1))
        if response_lengths_r1 else 0,
        "finish_eos_frac": finish_counter.get("eos", 0) / max(sum(finish_counter.values()), 1),
        "finish_stop_frac": finish_counter.get("stop", 0) / max(sum(finish_counter.values()), 1),
        "finish_length_frac": finish_counter.get("length", 0) / max(sum(finish_counter.values()), 1),
        "mean_distinct_parsed_answers": sum(distinct_answers_per_prompt) / n_prompts,
        "mean_distinct_clusters": sum(distinct_clusters_per_prompt) / n_prompts,
        "mean_degenerate_frac": sum(degenerate_fractions) / n_prompts,
        "mean_parsed_answer_entropy": (sum(answer_entropies) / len(answer_entropies))
        if answer_entropies else 0,
        "minority_eq_correct_frac": (
            cluster_correct_alignment_count / minority_eligible_count
        ) if minority_eligible_count else 0,
        "n_minority_eligible": minority_eligible_count,
    }


def analyze_arm(arm: str, arm_dir: Path, sample_every: int = 1) -> dict:
    """Run analysis over all steps for one arm."""
    print(f"[analyze] {arm}: scanning {arm_dir}")
    per_step = {}
    n_processed = 0
    for step, path in iter_step_files(arm_dir):
        if step % sample_every != 0:
            continue
        by_prompt = load_step(path)
        per_step[step] = analyze_step(by_prompt)
        n_processed += 1
        if n_processed % 50 == 0:
            print(f"[analyze] {arm}: processed {n_processed} steps (latest step={step})")
    print(f"[analyze] {arm}: total {n_processed} steps")
    return {"arm": arm, "per_step": per_step}


def summarize(results: dict[str, dict]) -> dict:
    """Cross-arm summary: compare metrics at step-matched points + final."""
    arms = list(results.keys())
    summary = {}

    # Step-matched comparison at step 200 (per eval_2026-06-01.md §3.1)
    for ref_step in [50, 100, 150, 200, 250, 300, 350]:
        summary[f"step_{ref_step}"] = {}
        for arm in arms:
            per_step = results[arm]["per_step"]
            if ref_step in per_step:
                summary[f"step_{ref_step}"][arm] = per_step[ref_step]

    # Final step per arm
    summary["final"] = {}
    for arm in arms:
        per_step = results[arm]["per_step"]
        if per_step:
            last_step = max(per_step)
            summary["final"][arm] = {"step": last_step, **per_step[last_step]}

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="main/data/probes/per_rollout_v2/diagnostic_summary.json")
    p.add_argument("--sample-every", type=int, default=10,
                   help="process every Nth step (default 10 for speed)")
    p.add_argument("--root", default="/Users/nancybao/Desktop/dev/cs224r_finalproject")
    args = p.parse_args()

    root = Path(args.root)
    results = {}
    for arm, rel in ARMS.items():
        arm_dir = root / rel
        if not arm_dir.exists():
            print(f"[analyze] {arm}: dir missing {arm_dir}")
            continue
        results[arm] = analyze_arm(arm, arm_dir, sample_every=args.sample_every)

    summary = summarize(results)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"per_arm": results, "summary": summary}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[analyze] wrote {out_path}")

    # Print headline cross-arm comparison
    print("\n=== STEP 200 COMPARISON ===")
    s200 = summary.get("step_200", {})
    metrics = [
        "all_wrong_frac", "mean_distinct_parsed_answers", "mean_distinct_clusters",
        "mean_parsed_answer_entropy", "mean_response_len_r1", "mean_response_len_r0",
        "finish_length_frac", "minority_eq_correct_frac",
    ]
    print(f"{'metric':40s} " + " ".join(f"{a:>10s}" for a in results))
    for m in metrics:
        vals = []
        for arm in results:
            v = s200.get(arm, {}).get(m, "—")
            vals.append(f"{v:10.3f}" if isinstance(v, (int, float)) else f"{str(v):>10s}")
        print(f"{m:40s} " + " ".join(vals))

    print("\n=== FINAL STEP COMPARISON ===")
    sf = summary.get("final", {})
    for m in metrics:
        vals = []
        for arm in results:
            v = sf.get(arm, {}).get(m, "—")
            vals.append(f"{v:10.3f}" if isinstance(v, (int, float)) else f"{str(v):>10s}")
        print(f"{m:40s} " + " ".join(vals))


if __name__ == "__main__":
    main()
