# Results (live — pending cells flagged)

All numbers are step-400 checkpoints, `n_rollouts=16`, sampled at
`temperature=1.0` / `top_p=1.0` / `top_k=-1` / `max_tokens=4096`, scored with
`verl.utils.reward_score.math.compute_score` (Hendrycks `is_equiv` on
`last_boxed_only_string`). Cells marked `pending` correspond to evals not yet
landed at the time of writing — see §3 for status.

## 1. Headline pass@k panel

| dataset (n)        | k   | GRPO    | Poly-EPO-CoT | Minority-CoT |
|---|---|---|---|---|
| AIME-25 (30)       | 1   | 0.073   | 0.062        | pending      |
| AIME-25 (30)       | 4   | 0.179   | 0.159        | pending      |
| AIME-25 (30)       | 8   | **0.227** | 0.206      | pending      |
| AIME-25 (30)       | 16  | **0.267** | 0.233      | pending      |
| MATH-500 (500)     | 1   | 0.680   | 0.683        | pending      |
| MATH-500 (500)     | 4   | 0.825   | 0.832        | pending      |
| MATH-500 (500)     | 8   | 0.860   | **0.868**    | pending      |
| MATH-500 (500)     | 16  | 0.880   | **0.892**    | pending      |
| HMMT Feb 2025 (30) | 1   | pending | 0.008        | pending      |
| HMMT Feb 2025 (30) | 4   | pending | 0.029        | pending      |
| HMMT Feb 2025 (30) | 8   | pending | 0.047        | pending      |
| HMMT Feb 2025 (30) | 16  | pending | 0.067        | pending      |
| HMMT Nov 2025 (30) | 1   | pending | 0.042        | pending      |
| HMMT Nov 2025 (30) | 4   | pending | 0.092        | pending      |
| HMMT Nov 2025 (30) | 8   | pending | 0.125        | pending      |
| HMMT Nov 2025 (30) | 16  | pending | 0.167        | pending      |
| BeyondAIME (100)   | 1   | pending | 0.040        | pending      |
| BeyondAIME (100)   | 4   | pending | 0.099        | pending      |
| BeyondAIME (100)   | 8   | pending | 0.137        | pending      |
| BeyondAIME (100)   | 16  | pending | 0.190        | pending      |

Source: `main-verl/eval/results/comparison.md` (rescored offline through the
training grader on 2026-06-02).

## 2. Diagnostic — cluster-correctness inversion (Minority-CoT)

Pooled across 909 Minority-CoT training prompts (steps 100–380, sampled
every 10) — see `main-verl/eval/results/minority_diagnostic.md:25-39`.
For each prompt we rank the non-degenerate judge clusters by frequency
(rank 1 = most common) and ask: when a rollout sits in the rank-`r` cluster,
what fraction of those rollouts are correct?

| cluster rank  | P(rank-r cluster is correct) |
|---|---|
| 1 (most common)  | **0.596** |
| 2                | 0.209     |
| 3                | 0.142     |
| 4                | 0.099     |
| 5                | 0.111     |
| 6                | 0.081     |
| 7                | 0.071     |
| 8 (rarest possible) | 0.016  |

Aggregate hit rates:
- **most-common cluster ≡ correct cluster: 77.2 %**
- **rarest cluster ≡ correct cluster: 44.5 %**
- uniform chance with ≈ 3 distinct clusters per prompt: ≈ 33 %

The probability that a cluster is the correct cluster decreases
monotonically with rarity. Minority-CoT's objective therefore upweights the
*wrong* rare cluster more than half the time it engages. Poly-EPO-CoT
sidesteps this by rewarding *every* distinct cluster (including the
most-common one), which carries the strongest correctness signal.

## 3. Status of pending cells

- **GRPO**: AIME-25 and MATH-500 landed; `hmmt_feb25`, `hmmt_nov25`,
  `beyondaime` are queued and have not yet completed in
  `main-verl/eval/results/comparison.md`. Polaris-val is excluded from the
  headline panel (in-distribution; redundant with training validation).
- **Poly-EPO-CoT**: full 5-dataset panel landed.
- **Minority-CoT**: the eval auto-launches at training step 400; at the
  time `comparison.md` was last refreshed (2026-06-02), the minority arm
  had **no eval landed**. Per the project memory the minority training was
  in resume4 from step 380. All Minority-CoT cells are therefore `pending`,
  and the `hmmt_*` + `beyondaime` datasets for Minority-CoT have not yet been
  launched.

When the missing evals land, regenerate the cross-arm table with
`python3 main-verl/eval/analysis/compare.py` and re-paste into §1.

## 4. Diversity metrics (training-time, step-200 snapshot)

For context — these are the training-time diagnostics that motivate the
`|U_correct|@k` / coverage@k metrics defined in `eval.md`. From
`main-verl/eval/results/minority_diagnostic.md:19-23`:

| metric (per prompt, 8 rollouts) | Minority-CoT | GRPO | Poly-EPO-CoT |
|---|---|---|---|
| distinct parsed answers          | **5.02** | 4.72 | 4.74 |
| parsed-answer entropy (bits)     | **2.05** | 1.90 | 1.93 |
| distinct non-degenerate judge clusters | 2.67 | n/a | 2.59 |

Minority-CoT carries a small, real, but shrinking diversity advantage over
GRPO at the answer level. Token-level entropy (W&B `actor/entropy`) shows a
larger gap — see `minority_diagnostic.md:74-86` — but this is an
optimization-time signal and is not the headline poster metric.
