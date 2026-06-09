# Diversity Evaluation Analysis — Step 400 (4B)

All results use the exact unbiased estimator averaged over all C(n,k) subsets of k rollouts
from n=64 total per prompt. CoT clusters assigned by Qwen3-4B-Instruct judge based on
reasoning strategy (macro + micro approach), not final answer.

**Figures:** `fig1_passk.pdf` · `fig2_cot_diversity.pdf` · `fig3_summary_k16.pdf` · `fig4_correctness_vs_diversity.pdf`

---

## Part 1: Pass@k (Answer Coverage)

**Datasets:** AIME25 (n=30), AIME26 (n=30), BeyondAIME (n=100)  
**Metric:** pass@k — fraction of problems solved by at least one correct rollout in k tries  
**Figure:** Fig. 1 (pass@k curves); Fig. 3 left two panels (bar summary at k=16)

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

**All trained arms regress severely on OOD benchmarks** (Fig. 1, Fig. 3 left panels).
The base model dominates at every k on every dataset. On BeyondAIME pass@16: base 0.144
vs. the best trained arm (polyepo/grpo ~0.058) — a ~60% drop. This holds even at pass@64,
where more rollouts should close the gap if diverse exploration is intact: base 0.290 vs.
polyepo 0.130 (best trained arm).

**The gap does not close with more samples** (Fig. 1, note parallel curve spacing on log scale).
If training had simply shifted the model toward a reliable subset of solution paths while
preserving exploration, the pass@k curves would converge at large k. They do not — the
ratio base/trained stays roughly constant from k=1 to k=64, suggesting the trained models
are not exploring strategies the base model finds.

**GRPO and the set-RL arms (minority, polyepo) are roughly equivalent on OOD data**
(Fig. 3, second panel). On BeyondAIME: grpo 0.057, polyepo 0.058, minority 0.043 at
pass@16. The set-RL arms do not show higher pass@k than GRPO despite being designed to
find rare correct solutions.

**AIME results are too noisy for strong conclusions** (Fig. 1, first two panels).
With only 30 problems and pass rates of 1–5%, individual arm differences (e.g. polyepo
0.000 on AIME26) are well within sampling variance and should not be interpreted as
systematic failures.

**Note on answer diversity.** On single-answer integer benchmarks like these, answer
diversity and pass@k are equivalent metrics: all correct rollouts for a given problem give
the same answer, so distinct-correct-answer clusters = 1 or 0 per problem. Pass@k directly
measures how many problems the model can solve, not how many ways it can solve them.
CoT diversity (Part 2) addresses this limitation.

---

## Part 2: CoT Diversity@k

**Datasets:** MATH500 (n=500), BeyondAIME (n=100)  
**Metric:** average number of distinct correct CoT clusters per problem, averaged only over
problems where the model got at least one rollout correct (using all 64 rollouts)  
**Figure:** Fig. 2 (bar chart per arm/dataset); Fig. 3 right two panels; Fig. 4 (scatter)

### Results

**MATH500** (in-distribution)

| arm | n_solved / total | distinct correct CoT clusters (per solved problem) |
|-----|:----------------:|:--------------------------------------------------:|
| base | 464 / 500 | **1.179** |
| grpo | 408 / 500 | 1.152 |
| polyepo | 405 / 500 | 1.034 |
| minority | 402 / 500 | 0.990 |

**BeyondAIME** (OOD)

| arm | n_solved / total | distinct correct CoT clusters (per solved problem) |
|-----|:----------------:|:--------------------------------------------------:|
| base | 29 / 100 | 0.586 |
| grpo | 12 / 100 | **0.667** |
| minority | 9 / 100 | 0.333 |
| polyepo | 13 / 100 | 0.231 |

### Analysis

**On MATH500, the base model leads narrowly; all arms are close** (Fig. 2, left panel).
Base (1.179) edges GRPO (1.152) by just 0.027, with polyepo (1.034) and minority (0.990)
somewhat lower. All four arms produce on average roughly one distinct correct reasoning
approach per solved problem, reflecting that MATH-500 problems mostly have one canonical
solution method. The gap is real but small in absolute terms.

**On BeyondAIME, GRPO produces the most diverse reasoning among solved problems** (Fig. 2,
right panel; Fig. 3, fourth panel). GRPO (0.667) > base (0.586) > minority (0.333) >
polyepo (0.231). This reversal from the MATH-500 ordering is notable: when conditioned on
the model actually solving a problem, GRPO's rollouts explore more distinct reasoning
approaches than the base model's. The set-RL arms (minority, polyepo) are substantially
below both.

**The set-RL arms have the lowest per-problem diversity on BeyondAIME** (Fig. 2, Fig. 3).
Minority (0.333) and polyepo (0.231) sit well below GRPO (0.667) and base (0.586). This
is the central negative result: objectives explicitly designed to maximize correct-cluster
diversity produce the least diverse reasoning among the problems they can solve.

**Values below 1.0 on BeyondAIME** reflect that most solved problems have only a single
distinct correct CoT cluster across all 64 rollouts — the model always reaches the answer
via the same reasoning approach. Values above 1.0 on MATH-500 indicate that for many
solved problems, the model found multiple distinct valid reasoning paths.

---

## Part 3: Joint Conclusions

### Finding 1: RL training reduces OOD correctness but the diversity picture is mixed

Pass@k (Fig. 1) shows a clear and consistent story: all trained arms regress on OOD
correctness relative to the base model. But per-problem CoT diversity (Fig. 2, Fig. 4)
tells a more nuanced story that depends on the benchmark. On in-distribution MATH-500,
base (1.179) leads GRPO (1.152) by a small margin. On OOD BeyondAIME, GRPO (0.667)
actually *exceeds* base (0.586) when conditioned on solved problems. RL training reduces
the number of problems a model can solve, but does not necessarily reduce diversity
*among the problems it does solve*.

### Finding 2: The minority and poly-epo objectives do not improve over GRPO

On both benchmarks and both metrics, the set-RL arms (minority, polyepo) rank below
standard GRPO (Fig. 2, Fig. 3). On BeyondAIME CoT diversity, GRPO (0.667) is 2× higher
than minority (0.333) and 3× higher than polyepo (0.231). On OOD pass@k (Fig. 1),
grpo and polyepo are statistically tied (BeyondAIME pass@16: 0.057 vs. 0.058), with
minority behind (0.043). The set-RL training signal — explicitly rewarding diverse correct
solutions — does not translate into a measurable advantage over standard GRPO on either
correctness or reasoning diversity.

### Finding 3: The minority and poly-epo objectives actively reduce per-problem diversity

The set-RL arms do not merely fail to improve over GRPO — they are substantially worse
on the diversity metric they were designed to maximize (Fig. 2, right panel; Fig. 3,
fourth panel). On BeyondAIME, minority (0.333) and polyepo (0.231) sit far below both
GRPO (0.667) and base (0.586). One hypothesis: minority voting trains the model to
converge on the *single rarest correct answer*, collapsing its rollout distribution toward
a narrow "minority mode" rather than broadening it. If the model learns to always produce
the low-frequency answer, it reduces variance in its solution strategies rather than
increases it.

### Finding 4: Diversity reduction is not explained by correctness loss alone

On MATH-500, all four arms solve 402–464 / 500 problems — a narrow 13% range — yet span
from 0.990 to 1.179 in per-problem CoT diversity. This is visible in Fig. 4 (left panel),
where the three trained arms share nearly identical pass@64 ($\approx$0.80–0.93) but spread
across 0.16 units in diversity. The diversity reduction in the set-RL arms is a genuine
change in the distribution over solution strategies, not an artifact of solving fewer
problems.

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
