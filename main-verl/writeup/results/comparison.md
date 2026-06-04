# Cross-arm held-out eval — 4 arms × 6 OOD datasets

**Generated:** 2026-06-04 from 4 arms × 6 datasets (5 smallood + math500),
n=64 eval JSONs at
`main-verl/eval/probes/eval_4b/{base,grpo,minority,polyepo}_step400_<shard>_<ds>.json`.

Authoritative spec: `main-verl/writeup/eval.md`. Run plan: `eval_build.md`.
Detailed index of all derived analyses: [INDEX.md](INDEX.md).

Grader: `verl.utils.reward_score.math.compute_score` (Hendrycks `is_equiv`,
mathd ∨ sympy fallback).
Sampling: `temp=1.0`, `top_p=1.0`, `max_tokens=4096`, `n=64`, `logprobs=20`.

## Headline pass@k

### aime25 (n=30 prompts)

| arm | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|
| **base** | **0.019** | **0.067** | **0.116** | **0.182** | **0.254** | **0.333** |
| grpo | 0.003 | 0.012 | 0.021 | 0.034 | 0.049 | 0.067 |
| minority | 0.002 | 0.006 | 0.011 | 0.019 | 0.029 | 0.033 |
| polyepo | 0.006 | 0.020 | 0.035 | 0.055 | 0.083 | 0.133 |

### aime26 (n=30 prompts)

| arm | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|
| **base** | **0.019** | **0.062** | **0.096** | **0.128** | **0.158** | **0.200** |
| grpo | 0.001 | 0.004 | 0.008 | 0.017 | 0.033 | 0.067 |
| minority | 0.003 | 0.010 | 0.019 | 0.036 | 0.063 | 0.100 |
| polyepo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### beyondaime (n=100 prompts)

| arm | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|
| **base** | **0.018** | **0.058** | **0.095** | **0.144** | **0.209** | **0.290** |
| grpo | 0.007 | 0.023 | 0.037 | 0.057 | 0.084 | 0.120 |
| minority | 0.006 | 0.019 | 0.030 | 0.043 | 0.061 | 0.090 |
| polyepo | 0.006 | 0.022 | 0.037 | 0.058 | 0.085 | 0.130 |

### hmmt_feb25 (n=30 prompts)

| arm | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|
| **base** | **0.005** | **0.020** | **0.039** | **0.071** | **0.123** | **0.200** |
| grpo | 0.001 | 0.004 | 0.008 | 0.017 | 0.033 | 0.067 |
| minority | 0.003 | 0.010 | 0.020 | 0.038 | 0.067 | 0.100 |
| polyepo | 0.004 | 0.014 | 0.028 | 0.054 | 0.100 | 0.167 |

### hmmt_nov25 (n=30 prompts) — depth-vs-breadth crossover

| arm | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|
| **base** | **0.028** | **0.075** | 0.101 | 0.122 | 0.132 | 0.133 |
| grpo | 0.013 | 0.042 | 0.070 | 0.105 | 0.139 | **0.167** |
| minority | 0.013 | 0.044 | 0.074 | 0.110 | 0.141 | **0.167** |
| polyepo | 0.014 | 0.046 | 0.075 | 0.108 | 0.142 | **0.167** |

base saturates around 0.133 by k=32; trained arms cross over at k=32+ and
reach 0.167 at k=64 (one more unique prompt solved via stochastic
exploration). See [INDEX.md](INDEX.md#the-hmmt_nov25-crossover-explained-2026-06-04)
for the n_correct distribution analysis.

### math500 (n=500 prompts) — easy OOD

| arm | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|---|
| **base** | **0.358** | **0.541** | **0.704** | **0.806** | **0.864** | **0.902** | **0.928** |
| grpo | 0.299 | 0.427 | 0.542 | 0.636 | 0.711 | 0.769 | 0.816 |
| minority | 0.265 | 0.388 | 0.504 | 0.602 | 0.683 | 0.750 | 0.804 |
| polyepo | ⚠️ MISSING | — | — | — | — | — | — |

**Polyepo math500 generation crashed mid-JSON-write.** All 32000 rollouts
generated successfully (logs confirm "generated in 12597s") but the
script hung during the json.dump call. Only 85.3 MiB of the expected
50+ GB JSON was committed before the process was killed (Modal app
`ap-h8zHYGx8IuvDhiPOfYtITd`). Per-prompt data is unrecoverable from
the truncated file beyond prompt 2. Re-fire deferred — the 23 other
cells are sufficient for the v1 poster story. See [eval_pipeline_bugs.md](eval_pipeline_bugs.md)
for full diagnosis.

**Same pattern as smallood:** base wins at every k. The trained-arm gap
to base is *smaller* on math500 (~10–20 percentage points at most k) than
on hard-OOD (where trained arms underperform base by 50–95%). Both
GRPO and Minority converge to similar pass@64 (~0.81). The crossover
seen on hmmt_nov25 does NOT happen here — base saturates around 0.93,
trained arms saturate around 0.80, no overtaking.

⚠️ **math_dapo tripwire on base × math500 = 58.3% agreement** (well below
the 90% threshold). All 534 disagreements are `math+only` — the math
grader (Hendrycks `is_equiv`) accepts latex-equivalent formats
(e.g., `\\frac{14}{3}` matches `\frac{14}{3}`, `(3, \\frac{\\pi}{2})`
matches `\left( 3, \\frac{\\pi}{2} \right)`) that math_dapo's strict
string match rejects. **The math grader is internally consistent (100%
rescore-match) so cross-arm pass@k comparisons WITHIN this eval are
valid**, but math500 pass@k under strict scoring would be lower across
the board (likely 30-40 percentage points lower). See [grader_sanity_all.md](grader_sanity_all.md#math_dapo-tripwire-evalmd-8-step-4--5-cells).

## Bottom line

1. **Base wins on every (arm, dataset, k) for k ≤ 16.** All 3 trained arms
   strictly underperform base on aime25, aime26, beyondaime, hmmt_feb25 —
   and on hmmt_nov25 at low k.

2. **The only crossover** is hmmt_nov25 at k ≥ 32, where all 3 trained arms
   reach pass@64 = 0.167 while base saturates at 0.133. This is "trade
   depth for breadth" — trained arms solve 5 unique prompts each (shallowly,
   1-12 correct rollouts each), base solves 4 (deeply, 5-26 correct).

3. **polyepo / aime26 = 0/1920 is a real failure mode** — verified by
   spot-check, not a grader bug. The model gets stuck in step-numbered
   repetition loops ("### Step 47 / Step 48 / Step 49") or sentence loops
   and never reaches `\boxed{...}`. Same repetition signature as
   polyepo/hmmt_nov25 wait=1.296 in `reflective_actions.md`.

4. **Minority is NOT the most-diverse trained arm at eval time** —
   contradicts the training-time diagnostic in `minority_diagnostic.md`.
   On beyondaime unsolved partition, grpo diff@k=64 = 20.50 > polyepo 19.26
   > minority 18.37. The eval-time picture differs from training-time.

5. **All trained arms collapse lexical diversity** — `self_bleu.md` shows
   base distinct-3 ≈ 0.47, trained arms ≈ 0.20 (2× collapse). `coverage.md`
   shows base entropy + distinct_answers higher at every k.

See [INDEX.md](INDEX.md) for the full results map, audit notes, and links
to the underlying analyses.

## Related artifacts

- [auc_at_k.md](auc_at_k.md) — AUC@k scalar over k∈{1..64}
- [diff_at_k_split.md](diff_at_k_split.md) — distinct_answers@k by solved/unsolved
- [potential_at_k.md](potential_at_k.md) — recoverable failure fraction
- [reflective_actions.md](reflective_actions.md) — per-rollout reflective phrase counts
- [self_bleu.md](self_bleu.md) — Self-BLEU + distinct-n diversity
- [coverage.md](coverage.md) — coverage / distinct_answers / entropy / majority @ k
