"""
Poly-EPO marginal set advantage analysis.

Compares standard GRPO advantage vs. Poly-EPO marginal set advantage
for various n values and synthetic cluster/reward distributions, all at N=8.
"""
from itertools import combinations
from math import comb
from collections import Counter


def f_poly(rewards, clusters):
    """Set score: (avg reward) * (unique clusters / n)."""
    n = len(rewards)
    avg_r = sum(rewards) / n
    diversity = len(set(clusters)) / n
    return avg_r * diversity


def marginal_set_advantages(rewards, clusters, n):
    """For each of the N generations, compute its Poly-EPO marginal advantage."""
    N = len(rewards)
    indices = list(range(N))
    subsets = list(combinations(indices, n))
    K = len(subsets)

    scores = [
        f_poly([rewards[i] for i in s], [clusters[i] for i in s]) for s in subsets
    ]
    baseline = sum(scores) / K
    set_adv = [s - baseline for s in scores]

    marg = []
    for i in range(N):
        adv_sum = sum(set_adv[j] for j, sub in enumerate(subsets) if i in sub)
        count = sum(1 for sub in subsets if i in sub)
        marg.append(adv_sum / count)

    return marg, baseline, scores


def grpo_adv(rewards):
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def show_advantages(advs, rewards, clusters):
    """Group generations by (reward, cluster) and report each type's advantage."""
    types = {}
    for i, (r, c) in enumerate(zip(rewards, clusters)):
        key = (r, c)
        if key not in types:
            types[key] = (advs[i], 1)
        else:
            types[key] = (types[key][0], types[key][1] + 1)
    for (r, c), (a, count) in sorted(
        types.items(), key=lambda x: (-x[0][0], x[0][1])
    ):
        tag = "CORRECT" if r == 1 else "wrong  "
        print(f"    {tag}  cluster={c} (x{count}):  A = {a:+.4f}")


def show_scenario(name, desc, rewards, clusters, n_values):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"  {desc}")
    print(f"  rewards:  {rewards}")
    print(f"  clusters: {clusters}")
    print(f"  cluster sizes: {dict(Counter(clusters))}, "
          f"#correct = {sum(rewards)}/{len(rewards)}")

    print(f"\n  >> standard GRPO  (== Poly-EPO at n=1):")
    show_advantages(grpo_adv(rewards), rewards, clusters)

    for n in n_values:
        marg, b, _ = marginal_set_advantages(rewards, clusters, n)
        K = comb(len(rewards), n)
        print(f"\n  >> Poly-EPO  n={n}  (K={K} subsets, baseline f_hat={b:.4f}):")
        show_advantages(marg, rewards, clusters)


print("=" * 70)
print("POLY-EPO MARGINAL SET ADVANTAGE,  N=8,  varying n in {2, 4, 6}")
print("=" * 70)

show_scenario(
    "S1: Diverse-correct, diverse-wrong",
    "2 correct strategies (2x A, 2x B) + 2 wrong strategies (2x C, 2x D)",
    [1, 1, 1, 1, 0, 0, 0, 0],
    ["A", "A", "B", "B", "C", "C", "D", "D"],
    [2, 4, 6],
)

show_scenario(
    "S2: Collapsed correct  (mid-training, model converged on one strategy)",
    "5x A correct, 3 unique wrongs (B, C, D)",
    [1, 1, 1, 1, 1, 0, 0, 0],
    ["A", "A", "A", "A", "A", "B", "C", "D"],
    [2, 4, 6],
)

show_scenario(
    "S3: Rare correct  (hard problem, early training)",
    "1x A correct, 4x B wrong (dominant wrong), 3 unique wrongs (C, D, E)",
    [1, 0, 0, 0, 0, 0, 0, 0],
    ["A", "B", "B", "B", "B", "C", "D", "E"],
    [2, 4, 6],
)

show_scenario(
    "S4: Common-correct + rare-correct  (your question.md RQ1)",
    "4x A correct (common), 1x B correct (rare), 3 unique wrongs (C, D, E)",
    [1, 1, 1, 1, 1, 0, 0, 0],
    ["A", "A", "A", "A", "B", "C", "D", "E"],
    [2, 4, 6],
)

show_scenario(
    "S5: Zero diversity  (all same cluster)",
    "8x A, mixed correctness (4 correct, 4 wrong, all same cluster)",
    [1, 1, 1, 1, 0, 0, 0, 0],
    ["A", "A", "A", "A", "A", "A", "A", "A"],
    [2, 4, 6],
)

show_scenario(
    "S6: Maximum diversity  (every generation in its own cluster)",
    "8 unique clusters, 4 correct, 4 wrong",
    [1, 1, 1, 1, 0, 0, 0, 0],
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    [2, 4, 6],
)
