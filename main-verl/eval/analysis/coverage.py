"""Compute coverage@k from saved eval_4b JSON outputs.

Coverage@k = number of distinct *correct* answer clusters discovered in k
rollouts, averaged across problems. This is the actual minority hypothesis:
broader coverage of correct answers even if pass@1 is similar.

Also computes:
- Answer-cluster entropy@k: entropy of the answer-hash distribution over
  k rollouts, per problem.
- Distinct-answers@k: count of distinct parsed_answer strings over k.

Usage:
  python3 main/scripts/compute_coverage_at_k.py path/to/eval_4b_result.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def coverage_at_k(per_prompt, k):
    """Mean distinct correct answers per prompt over the first k rollouts.

    A "correct answer" is a unique parsed_answer string that has reward=1.
    """
    cov = []
    for p in per_prompt:
        first_k_rewards = p["rewards"][:k]
        first_k_preds = p["preds"][:k]
        distinct_correct = set()
        for r, pred in zip(first_k_rewards, first_k_preds):
            if r > 0.5 and pred and pred != "[INVALID]":
                distinct_correct.add(pred)
        cov.append(len(distinct_correct))
    return sum(cov) / len(cov)


def distinct_answers_at_k(per_prompt, k):
    da = []
    for p in per_prompt:
        first_k = p["preds"][:k]
        distinct = {pred for pred in first_k if pred and pred != "[INVALID]"}
        da.append(len(distinct))
    return sum(da) / len(da)


def answer_entropy_at_k(per_prompt, k):
    """Mean entropy (bits) of parsed_answer distribution over first k rollouts."""
    ents = []
    for p in per_prompt:
        first_k = [pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"]
        if not first_k:
            continue
        counts = Counter(first_k)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        h = -sum(prob * math.log2(prob) for prob in probs if prob > 0)
        ents.append(h)
    return (sum(ents) / len(ents)) if ents else 0


def majority_at_k(per_prompt, k):
    """Pick the most common non-empty pred over first k rollouts; correct if gt match."""
    mvotes = 0
    for p in per_prompt:
        first_k_preds = [pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"]
        first_k_rewards = p["rewards"][:k]
        if not first_k_preds:
            continue
        # Most common pred
        most_common, _ = Counter(first_k_preds).most_common(1)[0]
        # Is it correct? Check if any rollout with this pred has reward=1
        gt = p["ground_truth"]
        if most_common == gt:
            mvotes += 1
        else:
            # Check via rewards (string match might miss latex equiv)
            for pred, r in zip(p["preds"][:k], first_k_rewards):
                if pred == most_common and r > 0.5:
                    mvotes += 1
                    break
    return mvotes / len(per_prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", help="path to eval_4b JSON output")
    args = ap.parse_args()
    p = Path(args.eval_json)
    data = json.load(p.open())
    print(f"=== {data['label']} ===")
    print(f"ckpt: {data['ckpt_path']}")
    print(f"n_rollouts: {data['n_rollouts']}")

    for ds_name, ds in data["datasets"].items():
        print(f"\n--- {ds_name} (n={ds['n_prompts']} prompts) ---")
        print(f"  pass@k: {ds['pass_at_k']}")
        print(f"  mean_reward_at_1: {ds['mean_reward_at_1']:.4f}")
        pp = ds["per_prompt"]
        K_VALUES = [1, 4, 8, 16]
        print(f"\n  k    cov   dist  ent  maj")
        for k in K_VALUES:
            if k > data["n_rollouts"]:
                continue
            cov = coverage_at_k(pp, k)
            da = distinct_answers_at_k(pp, k)
            ent = answer_entropy_at_k(pp, k)
            maj = majority_at_k(pp, k)
            print(f"  {k:>2d}  {cov:.2f}  {da:.2f}  {ent:.2f}  {maj:.3f}")


if __name__ == "__main__":
    main()
