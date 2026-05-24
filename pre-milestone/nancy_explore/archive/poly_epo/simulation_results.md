# Poly-EPO Simulation Results

Numerical analysis of the Poly-EPO marginal set advantage under varying batch
configurations. All numbers are computed exactly (full enumeration of subsets,
no Monte Carlo).

The two scripts that produced these results:
- `code/sim_poly_epo.py` — six synthetic batches at $N=8$, sweeping $n \in \{2, 4, 6\}$
- `code/sim_sweeps.py` — four sweeps over the joint $(N, n)$ space on a unified scenario

---

## Part 1: Per-scenario marginal advantages, $N=8$

Each scenario fixes a synthetic batch of 8 generations with specified rewards
and cluster assignments, then computes the **marginal set advantage** $A_i$ for
each generation under three Poly-EPO settings ($n \in \{2, 4, 6\}$) and the
GRPO baseline (= Poly-EPO at $n=1$). The marginal set advantage is the scalar
weight that gets multiplied with $\nabla_\theta \log \pi(y_i \mid x)$ in the
gradient update.

### S1: Diverse-correct, diverse-wrong (4 clusters of size 2 each)

```
rewards:  [1, 1, 1, 1, 0, 0, 0, 0]
clusters: [A, A, B, B, C, C, D, D]
```

| Setting | A-correct (×2) | B-correct (×2) | C-wrong (×2) | D-wrong (×2) |
|---|---|---|---|---|
| GRPO          | +0.5000 | +0.5000 | −0.5000 | −0.5000 |
| Poly-EPO n=2  | +0.1786 | +0.1786 | −0.1786 | −0.1786 |
| Poly-EPO n=4  | +0.0500 | +0.0500 | −0.0500 | −0.0500 |
| Poly-EPO n=6  | +0.0146 | +0.0146 | −0.0146 | −0.0146 |

**Read:** When clusters are uniformly distributed, Poly-EPO is just GRPO with
a smaller effective learning rate. The diversity term doesn't differentiate
because every cluster contributes equally.

### S2: Collapsed correct (mid-training scenario)

```
rewards:  [1, 1, 1, 1, 1, 0, 0, 0]
clusters: [A, A, A, A, A, B, C, D]   # 5x same correct cluster, 3 unique wrongs
```

| Setting | A-correct (×5) | B-wrong (×1) | C-wrong (×1) | D-wrong (×1) |
|---|---|---|---|---|
| GRPO          | +0.3750 | −0.6250 | −0.6250 | −0.6250 |
| Poly-EPO n=2  | +0.0536 | −0.0893 | −0.0893 | −0.0893 |
| Poly-EPO n=4  | −0.0000 | −0.0000 | −0.0000 | −0.0000 |
| Poly-EPO n=6  | −0.0020 | +0.0033 | +0.0033 | +0.0033 |

**Read:** This is the algorithm's **failure mode**. At $n=4$ all advantages
collapse to exactly zero (no learning signal at all). At $n=6$, the **signs
flip**: the dominant correct cluster gets pushed *down* and the unique wrongs
get pushed *up*. The diversity term has overpowered the reward term and the
algorithm is now actively unlearning correct behavior.

### S3: Rare correct (hard problem, sparse signal)

```
rewards:  [1, 0, 0, 0, 0, 0, 0, 0]
clusters: [A, B, B, B, B, C, D, E]   # 1x correct, 4x dominant wrong, 3 unique wrongs
```

| Setting | A-correct (×1) | B-wrong (×4, dominant) | C/D/E-wrong (×1 each) |
|---|---|---|---|
| GRPO          | +0.8750 | −0.1250 | −0.1250 |
| Poly-EPO n=2  | +0.3750 | −0.0536 | −0.0536 |
| Poly-EPO n=4  | +0.1018 | −0.0214 | −0.0054 |
| Poly-EPO n=6  | +0.0288 | −0.0069 | −0.0003 |

**Read:** This is the algorithm's **best case**. Standard GRPO punishes every
wrong answer equally (−0.125). Poly-EPO at $n=4$ punishes the dominant wrong
cluster B four times harder than the unique wrongs C/D/E (−0.0214 vs −0.0054).
At $n=6$ the ratio grows to ~25×. The unique wrongs are essentially uncoupled
from the gradient — this is the "optimistic exploration" the paper claims, and
it's most pronounced at intermediate-to-large $n$.

### S4: Common-correct + rare-correct (the question.md RQ1 scenario)

```
rewards:  [1, 1, 1, 1, 1, 0, 0, 0]
clusters: [A, A, A, A, B, C, D, E]   # 4x common-correct, 1x rare-correct, 3 unique wrongs
```

| Setting | A-correct (×4, common) | B-correct (×1, rare) | C/D/E-wrong (×1 each) |
|---|---|---|---|
| GRPO          | +0.3750 | +0.3750 | −0.6250 |
| Poly-EPO n=2  | +0.0536 | +0.2679 | −0.1607 |
| Poly-EPO n=4  | −0.0036 | +0.0839 | −0.0232 |
| Poly-EPO n=6  | −0.0030 | +0.0248 | −0.0043 |

**Read:** This is the punchline of the whole analysis. Under GRPO, both correct
clusters get the same push (+0.375). Under Poly-EPO at $n=4$, the **rare
correct cluster gets a positive push (+0.0839) while the common correct cluster
gets a slight negative push (−0.0036)**. The algorithm has flipped from "all
correctness is equal" to "redundant correctness is bad, rare correctness is
good." This is the core mechanism of the paper, expressed in concrete numbers.

### S5: Zero diversity (degenerate case)

```
rewards:  [1, 1, 1, 1, 0, 0, 0, 0]
clusters: [A, A, A, A, A, A, A, A]   # all same cluster
```

| Setting | correct (×4) | wrong (×4) |
|---|---|---|
| GRPO          | +0.5000 | −0.5000 |
| Poly-EPO n=2  | +0.1071 | −0.1071 |
| Poly-EPO n=4  | +0.0179 | −0.0179 |
| Poly-EPO n=6  | +0.0040 | −0.0040 |

**Read:** With zero cluster diversity, $|U|/n = 1/n$ regardless. Poly-EPO
collapses to a strictly smaller-magnitude GRPO. Ratios are preserved, no
new structure. The "diversity dilution" effect is visible: signal shrinks
roughly proportionally to $1/n$.

### S6: Maximum diversity (one cluster per generation)

```
rewards:  [1, 1, 1, 1, 0, 0, 0, 0]
clusters: [A, B, C, D, E, F, G, H]   # all unique
```

| Setting | each correct (×1) | each wrong (×1) |
|---|---|---|
| GRPO          | +0.5000 | −0.5000 |
| Poly-EPO n=2  | +0.2143 | −0.2143 |
| Poly-EPO n=4  | +0.0714 | −0.0714 |
| Poly-EPO n=6  | +0.0238 | −0.0238 |

**Read:** With maximum diversity, every subset has $|U|/n = 1$, so $f_{poly}$
reduces to $\bar r$. Same shape as GRPO, magnitudes scaled down by $\binom{N-1}{n-1}/\binom{N}{n} = n/N$.

---

## Part 2: Joint $(N, n)$ sweeps on a unified scenario

Held fixed across all sweeps: ~50% common-correct (one cluster), ~12.5%
rare-correct (one cluster), rest unique wrong clusters. Counts per row are
shown as `c/r/w` (common-correct / rare-correct / wrong).

The two key columns:
- **rare-common** = $A_{\text{rare}} - A_{\text{common}}$ — measures how much
  the algorithm differentiates rare-correct from common-correct (the diversity
  signal).
- **sum|A|** = $\sum_i |A_i|$ — measures the total per-batch gradient
  magnitude.

### Sweep 1: fix $n=4$, vary $N$ (compute scales; $n/N$ shifts)

```
   N   n   n/N       K  cnts(c/r/w)   A_common     A_rare    A_wrong   rare-common    sum|A|
   6   4  0.67      15        3/1/2    -0.0083    +0.0542    -0.0146       +0.0625    0.1083
   8   4  0.50      70        4/1/3    -0.0036    +0.0839    -0.0232       +0.0875    0.1679
  12   4  0.33     495        6/1/5    +0.0051    +0.1149    -0.0290       +0.1098    0.2904
  16   4  0.25    1820        8/2/6    -0.0003    +0.0955    -0.0314       +0.0959    0.3821
  20   4  0.20    4845       10/2/8    +0.0044    +0.1107    -0.0332       +0.1063    0.5317
```

**Read:** Diversity differentiation (rare-common) is **stable around 0.10**
across $N \in \{8, ..., 20\}$. Total signal magnitude **grows roughly linearly
with $N$** (0.17 at $N=8$ → 0.53 at $N=20$). This is the "correct" way to scale
compute up — you get more total signal without losing the diversity contrast.

### Sweep 2: fix $n/N = 0.5$, vary $N$ (compute and granularity scale; ratio held)

```
   N   n   n/N       K  cnts(c/r/w)   A_common     A_rare    A_wrong   rare-common    sum|A|
   4   2  0.50       6        2/1/1    +0.0000    +0.1667    -0.1667       +0.1667    0.3333
   6   3  0.50      20        3/1/2    -0.0056    +0.1167    -0.0500       +0.1222    0.2333
   8   4  0.50      70        4/1/3    -0.0036    +0.0839    -0.0232       +0.0875    0.1679
  12   6  0.50     924        6/1/5    -0.0012    +0.0517    -0.0089       +0.0530    0.1035
  16   8  0.50   12870        8/2/6    -0.0032    +0.0167    -0.0013       +0.0199    0.0667
  20  10  0.50  184756       10/2/8    -0.0020    +0.0132    -0.0008       +0.0152    0.0526
```

**Read:** This is the surprise. **Both the diversity gap AND the total signal
collapse as $N$ grows at fixed ratio**, by ~10× from $N=4$ to $N=20$. The
mechanism: with both $n$ and $N$ growing proportionally, each subset's
$f_{poly} = \bar r \cdot |U|/n$ concentrates around its expected value by the
law of large numbers, so the variance of set scores shrinks, and so do the
marginal advantages. The algorithmic regime ($n/N = 0.5$) is preserved in
spirit, but the *signal* is not. **Conclusion: $n/N$ is not a scale-invariant
parameter.**

### Sweep 3: fix $N=8$, vary $n \in \{1, ..., 7\}$ (compute fixed; pure regime change)

```
   N   n   n/N       K     A_common     A_rare    A_wrong   rare-common    sum|A|
   8   1  0.12       8      +0.3750    +0.3750    -0.6250       +0.0000    3.7500
   8   2  0.25      28      +0.0536    +0.2679    -0.1607       +0.2143    0.9643
   8   3  0.38      56      +0.0060    +0.1488    -0.0575       +0.1429    0.3452
   8   4  0.50      70      -0.0036    +0.0839    -0.0232       +0.0875    0.1679
   8   5  0.62      56      -0.0043    +0.0471    -0.0100       +0.0514    0.0943
   8   6  0.75      28      -0.0030    +0.0248    -0.0043       +0.0278    0.0496
   8   7  0.88       8      -0.0015    +0.0102    -0.0015       +0.0117    0.0204
```

**Read:** Tells the full story of $n$ as an algorithmic knob:
- $n=1$: standard GRPO exactly. No cluster awareness (gap = 0). Largest total magnitude.
- $n=2$: rare-correct boost peaks here in absolute terms (+0.27). Common still positive.
- $n \geq 4$: common-correct goes negative (regime change: active suppression).
- $n \to N$: signal collapses to 0.

**Sweet spot for the "diversity differentiation per unit of total signal" ratio
is $n \approx 4$–6.**

### Sweep 4: fix $N=16$, vary $n \in \{1, ..., 15\}$ (richer cluster pool)

```
   N   n   n/N       K     A_common     A_rare    A_wrong   rare-common    sum|A|
  16   1  0.06      16      +0.3750    +0.3750    -0.6250       +0.0000    7.5000
  16   2  0.12     120      +0.0625    +0.2625    -0.1708       +0.2000    2.0500
  16   4  0.25    1820      -0.0003    +0.0955    -0.0314       +0.0959    0.3821
  16   6  0.38    8008      -0.0041    +0.0395    -0.0077       +0.0436    0.1580
  16   8  0.50   12870      -0.0032    +0.0167    -0.0013       +0.0199    0.0667
  16  10  0.62    8008      -0.0021    +0.0063    +0.0007       +0.0084    0.0331
  16  12  0.75    1820      -0.0012    +0.0016    +0.0010       +0.0027    0.0186
  16  14  0.88     120      -0.0005    -0.0001    +0.0007       +0.0004    0.0082
  16  15  0.94      16      -0.0002    -0.0002    +0.0004       +0.0000    0.0044
```

**Read:** Same shape as Sweep 3 but at $N=16$. Confirms the regime structure
is robust to absolute scale. Note the late-stage **sign flip on wrong
clusters** at $n \geq 10$ — by $n=10$, $A_{\text{wrong}}$ is positive (+0.0007)
while $A_{\text{rare-correct}}$ is barely positive (+0.0063). At $n=14$ even
the rare-correct goes slightly negative. The "danger zone" of large-$n$
collapse is real and starts earlier than you'd guess.

---

## Quick reference: gradient update mechanics

For one prompt $x$ with $N$ generations $\{y_1, \dots, y_N\}$:

1. Score every size-$n$ subset $G$: $f_{poly}(G) = \bar r(G) \cdot |U(G)| / n$.
2. Baseline $\hat f$ = mean of all $K = \binom{N}{n}$ set scores.
3. Set advantage $\hat A^\#(G) = f_{poly}(G) - \hat f$.
4. Marginal advantage $A_i$ = mean of $\hat A^\#(G)$ over all subsets containing $y_i$.
5. Gradient step uses $A_i$ in place of GRPO's $r_i - \bar r$:
   ```
   theta_new = theta + lr * (1/N) * sum_i [ grad log pi(y_i | x) * A_i ]
   ```

So $A_i$ is literally the per-generation scalar weight on its log-probability
gradient. Positive → push policy to produce $y_i$ more; negative → less.
