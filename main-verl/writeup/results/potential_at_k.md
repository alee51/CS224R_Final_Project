# Potential@k

## TL;DR

**What it measures.** Of the prompts that failed within the first k rollouts
(no correct answer in the first k), what fraction *would* be solved if you
kept sampling out to n=64. High potential@k means the failure mode is
"budget-bound" (more samples would help); low potential@k means it is
"quality-bound" (the policy is stuck).

**How to read.** Compare arms within a dataset. Base typically starts high
and falls off as k grows (its failures get progressively less recoverable as
the budget is exhausted). A trained arm with *low* potential@k across all k
is "fundamentally stuck" on those prompts; a trained arm with *high*
potential@k still has upside left on the table.

**Headline.** Base has the highest recoverable failure rate on
aime25/aime26/beyondaime/hmmt_feb25 — its losses are partly budget-bound.
On hmmt_nov25 the pattern *flips*: base saturates by pot@8=0.000, while
trained arms still have 0.038–0.138 potential left, consistent with the
hmmt_nov25 AUC@k crossover noted in `auc_at_k.md`. **polyepo/aime26 is 0.000
across all k** — consistent with the pass@k=0 collapse.

For each (arm, dataset, k): fraction of problems that failed in
the first k rollouts but were solved at least once across all
n rollouts. Higher means more recoverable failures (budget-bound).

## aime25

| arm | pot@1 | pot@4 | pot@8 | pot@16 | pot@32 |
|---|---|---|---|---|---|
| base | 0.310 | 0.259 | 0.259 | 0.231 | 0.130 |
| grpo | 0.034 | 0.034 | 0.034 | 0.034 | 0.034 |
| minority | 0.033 | 0.033 | 0.033 | 0.033 | 0.000 |
| polyepo | 0.133 | 0.103 | 0.103 | 0.037 | 0.000 |

## aime26

| arm | pot@1 | pot@4 | pot@8 | pot@16 | pot@32 |
|---|---|---|---|---|---|
| base | 0.200 | 0.200 | 0.172 | 0.077 | 0.040 |
| grpo | 0.067 | 0.034 | 0.034 | 0.034 | 0.034 |
| minority | 0.100 | 0.069 | 0.069 | 0.036 | 0.036 |
| polyepo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## beyondaime

| arm | pot@1 | pot@4 | pot@8 | pot@16 | pot@32 |
|---|---|---|---|---|---|
| base | 0.268 | 0.245 | 0.202 | 0.174 | 0.123 |
| grpo | 0.111 | 0.102 | 0.093 | 0.093 | 0.054 |
| minority | 0.081 | 0.071 | 0.071 | 0.052 | 0.032 |
| polyepo | 0.121 | 0.103 | 0.094 | 0.074 | 0.044 |

## hmmt_feb25

| arm | pot@1 | pot@4 | pot@8 | pot@16 | pot@32 |
|---|---|---|---|---|---|
| base | 0.172 | 0.172 | 0.172 | 0.143 | 0.040 |
| grpo | 0.067 | 0.067 | 0.067 | 0.067 | 0.034 |
| minority | 0.100 | 0.100 | 0.100 | 0.100 | 0.069 |
| polyepo | 0.167 | 0.167 | 0.138 | 0.107 | 0.074 |

## hmmt_nov25

| arm | pot@1 | pot@4 | pot@8 | pot@16 | pot@32 |
|---|---|---|---|---|---|
| base | 0.133 | 0.037 | 0.000 | 0.000 | 0.000 |
| grpo | 0.167 | 0.167 | 0.138 | 0.138 | 0.074 |
| minority | 0.167 | 0.074 | 0.074 | 0.038 | 0.000 |
| polyepo | 0.167 | 0.167 | 0.074 | 0.038 | 0.000 |

## How this was computed

- **Script**: `main-verl/eval/analysis/posthoc/potential_at_k.py`. For each
  prompt and each k in {1, 4, 8, 16, 32}: marks the prompt "failed at k" if
  all of the first k rewards are 0; counts it "recoverable" if
  `n_correct > 0` across the full n=64; potential@k =
  recoverable / failed.
- **Inputs**: same 20 probe JSONs as the AUC table.
- **Eval probe sampling**: as in `auc_at_k.md` (n=64, T=1.0).
- **Limitations / caveats**:
  - Denominator can shrink fast: as base solves more prompts in early k,
    fewer prompts remain in the "failed at k" pool, so pot@k values are
    noisier at large k for high-skill arms.
  - "Failed at k" uses `reward > 0.5` as the threshold; with the math
    grader this is functionally `reward == 1`.
  - pot@k = 0.000 means **either** zero prompts failed at k (so the
    ratio is 0/0 → reported as 0) **or** zero of the failed prompts were
    ever solved. Read alongside `auc_at_k.md` to disambiguate.
