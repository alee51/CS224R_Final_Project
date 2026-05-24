# E1 — Set-score simulation (Run 0, cleaned labels)

**Runtime:** 18.7s  
**Inputs:** `data/cleaned_answers.parquet`, `analysis_a/llm_clusters_summary.parquet`  
**Labels:** `cleaned_correct`, `cleaned_cluster_id` only (no v2 parser fields).

## Formula notes

- Set-RL baseline: per-prompt mean of 70 subset scores; marginal adv = mean of `(f(G)-baseline)` over subsets containing rollout i (35 each).
- `f_poly(G) = mean(r in G) * (distinct cleaned_cluster_id in G) / 4`.
- `inverse_freq`: `(r_i - mean(r_p)) / cluster_size(cleaned_cluster_id)` (design doc; differs from normalized `objectives.py` weights).
- `*-rand`: 20 seeds; reported marginal adv = mean across seeds; `*_rand_std` columns in parquet.

## Sanity

- Rows: 4000 (500×8). GRPO max |sum-by-prompt|: `0.00e+00`.

## Q1 — Rand vs avg tie-break

| Pair | Pearson r | Spearman ρ |
|---|---|---|
| ans-rand vs ans-avg | **+0.994** | +0.867 |
| cot-rand vs cot-avg | **+0.995** | +0.927 |

Seed variance on marginal advantages (across 4000 rollouts):
- ans-rand std: mean `0.0115`, max `0.0581`
- cot-rand std: mean `0.0104`, max `0.0692`

## Q2 — Answer vs CoT cluster mode

| Pair | Pearson r | Spearman ρ |
|---|---|---|
| ans-avg vs cot-avg | **+0.525** | +0.329 |
| ans-rand vs cot-rand (mean) | **+0.519** | +0.324 |

## Q3 — Minority set scores vs f_poly

Pearson r between subset-level `f(G)` vectors (35,000 subset scores):

| Minority f | vs f_poly r |
|---|---|
| ans-avg | **+0.461** |
| cot-avg | **+0.627** |

Contingency: **high diversity** (4 distinct answer buckets in G) AND **minority f(G)=0**:

| Minority def | P(zero|high div) | count high∧zero / high div |
|---|---|---|
| ans-avg | **78.5%** | 13393 / 17064 |
| cot-avg | **84.8%** | 14462 / 17064 |

- f_poly=0 on 76.1% of subsets; high-div subsets with f_poly=0: 13393 / 17064

## Q4 — Distribution of f(G) on 70 subsets × 500 prompts

(ans-rand / cot-rand: pool 20 seeds × 35k subsets = 700k scores each.)

| Objective | frac f=0 | frac f=1 | mean f | std f |
|---|---|---|---|---|
| ans-avg | 0.836 | 0.013 | 0.062 | 0.165 |
| cot-avg | 0.846 | 0.014 | 0.065 | 0.177 |
| f_poly | 0.761 | 0.000 | 0.066 | 0.123 |
| ans-rand | 0.938 | 0.062 | 0.062 | 0.242 |
| cot-rand | 0.929 | 0.060 | 0.065 | 0.242 |

## Q5 — Single answer-bucket subsets (all 4 rollouts share cleaned_cluster_id)

- Fraction of subsets with one answer mode: **2.5%** (886 / 35000).
- In that case all four rollouts tie for rarest count; minority f(G) = mean(r in G) for ans-avg/ans-rand (intended).
- On single-mode subsets: mean ans-avg f = 0.217; mean f_poly = 0.054.

## Advantage correlation matrices (4000 rollouts)

### Pearson

| | grpo | inverse_freq | ans_avg | ans_rand | cot_avg | cot_rand | f_poly |
|---|---|---|---|---|---|---|---|
| grpo | +1.000 | **+0.916** | +0.447 | +0.440 | +0.609 | +0.608 | +0.867 |
| inverse_freq | **+0.916** | +1.000 | +0.632 | +0.626 | +0.621 | +0.620 | +0.833 |
| ans_avg | +0.447 | +0.632 | +1.000 | **+0.994** | +0.525 | +0.523 | +0.387 |
| ans_rand | +0.440 | +0.626 | **+0.994** | +1.000 | +0.521 | +0.519 | +0.381 |
| cot_avg | +0.609 | +0.621 | +0.525 | +0.521 | +1.000 | **+0.995** | +0.494 |
| cot_rand | +0.608 | +0.620 | +0.523 | +0.519 | **+0.995** | +1.000 | +0.493 |
| f_poly | +0.867 | +0.833 | +0.387 | +0.381 | +0.494 | +0.493 | +1.000 |


### Spearman

| | grpo | inverse_freq | ans_avg | ans_rand | cot_avg | cot_rand | f_poly |
|---|---|---|---|---|---|---|---|
| grpo | +1.000 | **+0.996** | +0.221 | +0.302 | +0.393 | +0.418 | +0.879 |
| inverse_freq | **+0.996** | +1.000 | +0.262 | +0.342 | +0.399 | +0.423 | +0.870 |
| ans_avg | +0.221 | +0.262 | +1.000 | +0.867 | +0.329 | +0.321 | +0.027 |
| ans_rand | +0.302 | +0.342 | +0.867 | +1.000 | +0.311 | +0.324 | +0.128 |
| cot_avg | +0.393 | +0.399 | +0.329 | +0.311 | +1.000 | **+0.927** | +0.263 |
| cot_rand | +0.418 | +0.423 | +0.321 | +0.324 | **+0.927** | +1.000 | +0.298 |
| f_poly | +0.879 | +0.870 | +0.027 | +0.128 | +0.263 | +0.298 | +1.000 |


*Bold = |r| > 0.9 off-diagonal.*

## Headline vs GRPO / inverse_freq

- GRPO ↔ ans-avg: Pearson **+0.447**
- GRPO ↔ f_poly: Pearson **+0.867**
- inverse_freq ↔ ans-avg: Pearson **+0.632**
- ans-avg ↔ f_poly: Pearson **+0.387**
