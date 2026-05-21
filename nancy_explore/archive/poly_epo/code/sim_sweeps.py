"""
Sweeps over (N, n) for Poly-EPO marginal advantages.

Scenario (held fixed proportionally as N varies):
  - 50% common-correct (one cluster, all r=1)
  - ~12.5% rare-correct  (one cluster, all r=1, ideally a single rare strategy)
  - remainder = unique wrong clusters (each r=0, in its own cluster)

We report per-type marginal advantages and aggregate signals across:
  Sweep 1: fix n=4, vary N         (n/N changes; algorithmic regime shifts)
  Sweep 2: fix n/N=0.5, vary N     (algorithmic regime fixed; compute changes)
  Sweep 3: fix N=8, vary n         (compute fixed; algorithmic regime shifts)
"""
from itertools import combinations
from math import comb


def f_poly(rewards, clusters):
    n = len(rewards)
    return (sum(rewards) / n) * (len(set(clusters)) / n)


def marginal_set_advantages(rewards, clusters, n):
    N = len(rewards)
    if n > N:
        return None
    subsets = list(combinations(range(N), n))
    K = len(subsets)
    scores = [
        f_poly([rewards[i] for i in s], [clusters[i] for i in s]) for s in subsets
    ]
    baseline = sum(scores) / K
    set_adv = [s - baseline for s in scores]
    marg = []
    for i in range(N):
        s = sum(set_adv[j] for j, sub in enumerate(subsets) if i in sub)
        c = sum(1 for sub in subsets if i in sub)
        marg.append(s / c)
    return marg


def make_scenario(N):
    """Hold proportions roughly fixed: ~50% common-correct, ~12.5% rare-correct,
    rest = unique wrong clusters."""
    n_common = max(1, N // 2)
    n_rare = max(1, N // 8)
    n_wrong = N - n_common - n_rare
    if n_wrong < 1:
        n_wrong = 1
        n_common = N - n_rare - n_wrong
    rewards = [1] * n_common + [1] * n_rare + [0] * n_wrong
    clusters = (
        ["COMMON_OK"] * n_common
        + ["RARE_OK"] * n_rare
        + [f"WRONG{i}" for i in range(n_wrong)]
    )
    return rewards, clusters, n_common, n_rare, n_wrong


def summary(advs, rewards, clusters):
    by_type = {"COMMON_OK": [], "RARE_OK": [], "WRONG": []}
    for a, r, c in zip(advs, rewards, clusters):
        if c == "COMMON_OK":
            by_type["COMMON_OK"].append(a)
        elif c == "RARE_OK":
            by_type["RARE_OK"].append(a)
        else:
            by_type["WRONG"].append(a)
    avg = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_type.items()}
    total_mag = sum(abs(a) for a in advs)
    rare_minus_common = avg["RARE_OK"] - avg["COMMON_OK"]
    return avg, total_mag, rare_minus_common


def header():
    print(
        f"{'N':>4} {'n':>3} {'n/N':>5} {'K':>7} {'cnts(c/r/w)':>12} "
        f"{'A_common':>10} {'A_rare':>10} {'A_wrong':>10} "
        f"{'rare-common':>13} {'sum|A|':>9}"
    )


def row(N, n, rewards, clusters, n_c, n_r, n_w, advs):
    avg, mag, gap = summary(advs, rewards, clusters)
    K = comb(N, n)
    print(
        f"{N:>4} {n:>3} {n/N:>5.2f} {K:>7} "
        f"{f'{n_c}/{n_r}/{n_w}':>12} "
        f"{avg['COMMON_OK']:>+10.4f} "
        f"{avg['RARE_OK']:>+10.4f} "
        f"{avg['WRONG']:>+10.4f} "
        f"{gap:>+13.4f} "
        f"{mag:>9.4f}"
    )


print("=" * 100)
print("SWEEP 1:  fix n=4, vary N    (n/N changes; algorithmic regime SHIFTS)")
print("=" * 100)
header()
for N in [6, 8, 12, 16, 20]:
    n = 4
    rewards, clusters, n_c, n_r, n_w = make_scenario(N)
    advs = marginal_set_advantages(rewards, clusters, n)
    row(N, n, rewards, clusters, n_c, n_r, n_w, advs)

print()
print("=" * 100)
print("SWEEP 2:  fix n/N=0.5, vary N    (algorithmic regime ROUGHLY FIXED;")
print("                                   compute & granularity scale)")
print("=" * 100)
header()
for N in [4, 6, 8, 12, 16, 20]:
    n = N // 2
    rewards, clusters, n_c, n_r, n_w = make_scenario(N)
    advs = marginal_set_advantages(rewards, clusters, n)
    row(N, n, rewards, clusters, n_c, n_r, n_w, advs)

print()
print("=" * 100)
print("SWEEP 3:  fix N=8, vary n in {1..7}    (compute fixed; algorithmic regime SHIFTS)")
print("=" * 100)
header()
N = 8
rewards, clusters, n_c, n_r, n_w = make_scenario(N)
for n in [1, 2, 3, 4, 5, 6, 7]:
    advs = marginal_set_advantages(rewards, clusters, n)
    row(N, n, rewards, clusters, n_c, n_r, n_w, advs)


print()
print("=" * 100)
print("SWEEP 4:  fix N=16, vary n in {1..15}    (richer cluster pool, varying n)")
print("=" * 100)
header()
N = 16
rewards, clusters, n_c, n_r, n_w = make_scenario(N)
for n in [1, 2, 4, 6, 8, 10, 12, 14, 15]:
    advs = marginal_set_advantages(rewards, clusters, n)
    row(N, n, rewards, clusters, n_c, n_r, n_w, advs)
