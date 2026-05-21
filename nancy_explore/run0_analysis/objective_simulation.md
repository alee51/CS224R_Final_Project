# Analysis C — Offline objective simulation

**Date:** 2026-05-21  
**Inputs:** `data/predictions_reparsed.jsonl` (4000 rollouts), `llm_clusters_summary.parquet` (Analysis A).  
**Outputs:** `objective_advantages.parquet`, `objective_corr_pearson.csv`, `objective_corr_spearman.csv`, `objective_scatter_grid.png`.

## Formula note

- Inverse-freq formula: `A_i = (1 / cluster_size_i) * (r_i - mean(r_p))` per design doc §C.2.
- `pilot/train/objectives.py:inverse_freq_weights` uses *normalized* weights (`n * c^-gamma / sum`, capped at `w_max=8`), which differs (production weights sum to ~N per prompt; the doc formula does not). Used the doc formula here; production behavior will differ in scale by the per-prompt normalization factor `n / sum_j (1/c_j)` and the cap.

## Sanity checks

- GRPO max |sum-by-prompt|: `0.00e+00` (expect ~0). PASS.
- IF_answer mean |sum-by-prompt|: `0.1755` (nonzero expected). PASS.
- GRPO max |adv| on all-correct/all-wrong prompts: `0.00e+00` (expect ~0). PASS.

## Pearson correlation matrix (4000 rollouts)

| | GRPO | IF_answer | IF_llm | fpoly_answer | fpoly_llm | worst |
|---|---|---|---|---|---|---|
| GRPO | +1.000 | **+0.924** | +0.875 | +0.200 | +0.202 | +0.122 |
| IF_answer | **+0.924** | +1.000 | +0.869 | -0.029 | -0.026 | +0.040 |
| IF_llm | +0.875 | +0.869 | +1.000 | +0.042 | +0.055 | +0.078 |
| fpoly_answer | +0.200 | -0.029 | +0.042 | +1.000 | **+0.955** | +0.260 |
| fpoly_llm | +0.202 | -0.026 | +0.055 | **+0.955** | +1.000 | +0.397 |
| worst | +0.122 | +0.040 | +0.078 | +0.260 | +0.397 | +1.000 |


## Spearman correlation matrix (4000 rollouts)

| | GRPO | IF_answer | IF_llm | fpoly_answer | fpoly_llm | worst |
|---|---|---|---|---|---|---|
| GRPO | +1.000 | **+0.997** | **+0.993** | -0.373 | -0.378 | +0.276 |
| IF_answer | **+0.997** | +1.000 | **+0.991** | -0.373 | -0.375 | +0.276 |
| IF_llm | **+0.993** | **+0.991** | +1.000 | -0.365 | -0.377 | +0.281 |
| fpoly_answer | -0.373 | -0.373 | -0.365 | +1.000 | **+0.993** | +0.309 |
| fpoly_llm | -0.378 | -0.375 | -0.377 | **+0.993** | +1.000 | +0.303 |
| worst | +0.276 | +0.276 | +0.281 | +0.309 | +0.303 | +1.000 |


*Bold = |r| > 0.9 (off-diagonal).*

## Disagreement table — top 3 most divergent pairs (opposite-sign rollouts)

Rows: `is_correct_v2 ∈ {0, 1}`; columns: `cluster_size_answer ∈ {1, 2, 3, 4+}`. Cell = #rollouts where the two objectives assign opposite-sign advantages.

### fpoly_answer vs worst — total opposite-sign: 1226

| r | cs=1 | cs=2 | cs=3 | cs=4+ |
|---|---|---|---|---|
| r=0 | 811 | 120 | 30 | 29 |
| r=1 | 83 | 66 | 87 | 0 |

### fpoly_llm vs worst — total opposite-sign: 1177

| r | cs=1 | cs=2 | cs=3 | cs=4+ |
|---|---|---|---|---|
| r=0 | 770 | 116 | 26 | 29 |
| r=1 | 83 | 66 | 87 | 0 |

### GRPO vs fpoly_answer — total opposite-sign: 990

| r | cs=1 | cs=2 | cs=3 | cs=4+ |
|---|---|---|---|---|
| r=0 | 811 | 120 | 30 | 29 |
| r=1 | 0 | 0 | 0 | 0 |

## Singleton-wrong mass under inverse_freq

Fraction of total `|adv|` mass concentrated on `(is_correct_v2=0, cluster_size=1)` rollouts.

| Substrate | sum |adv| on singleton-wrong | total sum |adv| | % |
|---|---|---|---|
| answer-hash (v2) | 179.6250 | 318.0000 | **56.49%** |
| LLM clusters | 114.3750 | 257.8176 | **44.36%** |

## Substrate sensitivity

Pearson r between answer-hash and LLM-cluster versions of the same objective (over 4000 rollouts):

| Objective | r(answer, llm) |
|---|---|
| inverse_freq | **+0.869** |
| f_poly | **+0.955** |

Sign-flip rate (opposite-sign rollouts under the two substrates): inverse_freq **0.00%**, f_poly **1.23%**.

## Can claim (per §C.4)

- On Run 0's empirical reward+cluster distribution, GRPO and `inverse_freq` advantages correlate at Pearson r = **+0.924** (under answer-hash) and **+0.875** (under LLM clusters).
- `inverse_freq` concentrates **56.5%** of its |advantage| mass on rare-wrong rollouts (r=0, cluster_size=1) under answer-hash clustering, vs **44.4%** under LLM clustering.
- `worst_subset` per-rollout advantages correlate with GRPO at r = **+0.122** ; `f_poly` (answer-hash) at r = **+0.200**.
- Substrate swap (answer-hash → LLM) changes inverse_freq advantages substantially: only r = **+0.869** between the two; f_poly is more stable at r = **+0.955**.

## Cannot claim (per §C.5)

- Anything about training trajectories. This is a one-step view on a fixed base-model rollout distribution. A correlation difference does not guarantee a trained-model accuracy difference; it is necessary-not-sufficient evidence that the objectives are distinguishable.
- That f_poly's substrate insensitivity implies it is the 'right' objective — it is more diluted because the cluster term only enters via `d(G) = |distinct|/n` (a small multiplier) and `mean_r(G)` dominates.