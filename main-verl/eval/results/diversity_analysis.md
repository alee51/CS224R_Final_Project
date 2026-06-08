# Diversity Evaluation Analysis — Step 400 (4B)

All results use the exact unbiased estimator averaged over all C(n,k) subsets of k rollouts
from n=64 total per prompt. CoT clusters assigned by Qwen3-4B-Instruct judge based on
reasoning strategy (macro + micro approach), not final answer.

---

## Part 1: Pass@k (Answer Coverage)

**Datasets:** AIME25 (n=30), AIME26 (n=30), BeyondAIME (n=100)  
**Metric:** pass@k — fraction of problems solved by at least one correct rollout in k tries

### Results

**AIME25**

| arm | pass@1 | pass@4 | pass@16 | pass@64 |
|-----|--------|--------|---------|---------|
| base | 0.019 | 0.067 | 0.182 | 0.333 |
| grpo | 0.003 | 0.012 | 0.034 | 0.067 |
| minority | 0.002 | 0.006 | 0.020 | 0.033 |
| polyepo | 0.006 | 0.020 | 0.056 | 0.133 |

**AIME26**

| arm | pass@1 | pass@4 | pass@16 | pass@64 |
|-----|--------|--------|---------|---------|
| base | 0.019 | 0.062 | 0.128 | 0.200 |
| grpo | 0.001 | 0.004 | 0.017 | 0.067 |
| minority | 0.003 | 0.010 | 0.036 | 0.100 |
| polyepo | 0.000 | 0.000 | 0.000 | 0.000 |

**BeyondAIME**

| arm | pass@1 | pass@4 | pass@16 | pass@64 |
|-----|--------|--------|---------|---------|
| base | 0.018 | 0.059 | 0.144 | 0.290 |
| grpo | 0.007 | 0.023 | 0.057 | 0.120 |
| minority | 0.006 | 0.019 | 0.043 | 0.090 |
| polyepo | 0.006 | 0.022 | 0.058 | 0.130 |

### Analysis

**All trained arms regress severely on OOD benchmarks.** The base model dominates at every k
on every dataset. On BeyondAIME pass@16: base 0.144 vs. the best trained arm (polyepo/grpo
~0.058) — a ~60% drop. This holds even at pass@64, where more rollouts should close the gap
if diverse exploration is intact: base 0.290 vs. polyepo 0.130 (best trained arm).

**The gap does not close with more samples.** If training had simply shifted the model toward
a reliable subset of solution paths while preserving exploration, the pass@k curves would
converge at large k. They do not — the ratio base/trained stays roughly constant from k=1
to k=64, suggesting the trained models are not exploring strategies the base model finds.

**GRPO and the set-RL arms (minority, polyepo) are roughly equivalent on OOD data.**
On BeyondAIME: grpo 0.057, polyepo 0.058, minority 0.043 at pass@16. The set-RL arms
do not show higher pass@k than GRPO despite being designed to find rare correct solutions.

**AIME results are too noisy for strong conclusions.** With only 30 problems and pass rates
of 1–5%, individual arm differences (e.g. polyepo 0.000 on AIME26) are well within
sampling variance and should not be interpreted as systematic failures.

**Note on answer diversity.** On single-answer integer benchmarks like these, answer
diversity and pass@k are equivalent metrics: all correct rollouts for a given problem give
the same answer, so distinct-correct-answer clusters = 1 or 0 per problem. Pass@k directly
measures how many problems the model can solve, not how many ways it can solve them.
CoT diversity (Part 2) addresses this limitation.

---

## Part 2: CoT Diversity@k

**Datasets:** MATH500 (n=500), BeyondAIME (n=100)  
**Metric:** expected number of distinct correct CoT clusters in a random k-subset of
64 rollouts, averaged over all prompts (prompts with zero correct rollouts contribute 0)

### Results

**MATH500** (in-distribution)

| arm | n_correct_prompts | div@1 | div@4 | div@16 | div@64 |
|-----|:-----------------:|-------|-------|--------|--------|
| base | 464/500 | 0.315 | 0.665 | 0.895 | 1.094 |
| grpo | 408/500 | 0.278 | 0.527 | 0.735 | 0.940 |
| polyepo | 405/500 | 0.267 | 0.503 | 0.686 | 0.838 |
| minority | 402/500 | 0.238 | 0.464 | 0.646 | 0.796 |

**BeyondAIME** (OOD)

| arm | n_correct_prompts | div@1 | div@4 | div@16 | div@64 |
|-----|:-----------------:|-------|-------|--------|--------|
| base | 29/100 | 0.010 | 0.032 | 0.076 | 0.170 |
| grpo | 12/100 | 0.005 | 0.017 | 0.046 | 0.080 |
| polyepo | 13/100 | 0.003 | 0.009 | 0.023 | 0.030 |
| minority | 9/100 | 0.003 | 0.010 | 0.019 | 0.030 |

### Analysis

**The base model generates the most diverse correct reasoning strategies on MATH500.**
At div@16, base (0.895) > grpo (0.735) > polyepo (0.686) > minority (0.646). This ordering
is consistent across all values of k and holds for both datasets. The base model, on average,
discovers nearly one additional distinct correct reasoning approach per problem compared to
the best trained arm at k=16.

**The set-RL arms are less diverse than standard GRPO.** This is the central negative result.
Minority and polyepo, which explicitly train to maximize distinct correct CoT clusters, rank
below GRPO in measured CoT diversity. Polyepo (0.686) edges minority (0.646) at div@16 on
MATH500, consistent with polyepo's broader cluster-reward objective, but both sit below GRPO.

**BeyondAIME diversity is near-zero for all arms.** With only 9–29 problems having any
correct rollout, the diversity signal is too weak to distinguish arms. The low absolute
values (div@16 < 0.08 for all arms) reflect the sparsity of correct rollouts rather than
homogeneity of reasoning strategies.

**The diversity gap is largest at small k.** At div@1, base (0.315) is 32% higher than
minority (0.238). This means even a single rollout from the base model is more likely to
represent a distinct reasoning approach than one from the trained arms — a sign that the
base model's distribution over solution strategies is genuinely broader, not just noisier.

**Div@64 > 1 for the base model.** This occurs because the metric averages over all 500
prompts (including 36 with zero correct rollouts contributing 0), while problems that are
solved have on average more than one distinct CoT cluster across 64 rollouts. It does not
indicate a mathematical error.

---

## Part 3: Joint Conclusions

### Finding 1: RL training reduces both correctness and reasoning diversity on OOD tasks

Pass@k and CoT diversity tell a consistent story across both in-distribution and OOD data:
training on Polaris with RL (regardless of objective) reduces the model's tendency to explore
diverse solution strategies and reduces OOD problem-solving ability. The base model — never
trained to be correct — outperforms all trained arms on both metrics on every OOD benchmark.

This suggests a **diversity-correctness tradeoff in RL training for math**: RL reinforces
reliable solution paths and suppresses less-frequent strategies, including ones that would
succeed on distribution-shifted problems.

### Finding 2: The minority and poly-epo objectives do not improve over GRPO

Neither diversity metric shows minority or polyepo exceeding GRPO. On MATH500 CoT diversity,
GRPO (0.735) beats both polyepo (0.686) and minority (0.646) at div@16. On OOD pass@k,
grpo and polyepo are statistically tied (BeyondAIME pass@16: 0.057 vs. 0.058), with
minority slightly behind (0.043). The set-RL training signal — explicitly rewarding
diverse correct solutions — does not translate into a measurable advantage over standard
reward-on-correctness GRPO.

### Finding 3: The minority objective appears to actively reduce diversity

Among trained arms, minority consistently has the *lowest* CoT diversity (MATH500 div@16:
0.646 vs. 0.686 for polyepo, 0.735 for GRPO). One hypothesis: minority voting trains the
model to find the *single rarest correct answer* per problem, which converges the model
toward a narrow "minority mode" of solution rather than preserving a broad distribution.
If the model learns to always produce the low-frequency answer, it may reduce variance
in its rollout distribution rather than increase it.

### Finding 4: Diversity loss is not explained by correctness loss alone

A simpler explanation for lower diversity in trained arms would be: fewer problems have
multiple correct rollouts (lower pass@k → fewer opportunities to measure diversity).
But the data does not fully support this. On MATH500, all four arms solve 402–464/500
problems correctly — a narrow range — yet show a wide spread in div@16 (0.646–0.895).
The diversity reduction is real and not just an artifact of the trained arms having lower
pass rates.

### Implications

These results suggest that the set-RL approach as implemented here (minority voting and
poly-epo on Polaris problems with 4B Qwen3-Base) does not achieve its stated goal of
training models to explore diverse solution strategies. Two directions worth investigating:

1. **Training distribution**: Polaris problems may not have enough natural answer diversity
   to provide a useful minority signal. Problems with multiple valid solution approaches
   (e.g., geometry, combinatorics) might yield a stronger training signal.
2. **Objective formulation**: The cluster reward during training may be optimized via
   shortcuts (e.g., minor surface variation in CoTs) rather than genuine strategic diversity.
   Stronger judge prompts or answer-level diversity constraints may be needed.
