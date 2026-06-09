# AUC@k (locked k ladder {1, 2, 4, 8, 16, 32, 64})

_Updated 2026-06-04 to include math500 column. polyepo / math500 missing
due to GEN failure (see [eval_pipeline_bugs.md](eval_pipeline_bugs.md))._


## TL;DR

**What it measures.** A single scalar summary of the pass@k curve over the
locked k-ladder, computed as `trapezoid(pass_at_k, ks)`. Rewards both
pass@1 quality and large-k coverage; arms with a higher curve at any k get a
proportionally larger AUC.

**How to read.** Larger = better. Compare arms within a column (same dataset).
The "Underlying pass@k points" section below the table shows the raw curve;
those values should be monotonically non-decreasing in k (a property of
pass@k).

**Headline.** Base dominates on 4 of 5 datasets (aime25, aime26, beyondaime,
hmmt_feb25), often by 2-4×. The exception is **hmmt_nov25**, where all three
trained arms slightly beat base (polyepo 7.998, minority 7.988, grpo 7.850 vs
base 7.685) — driven by base's pass@k curve flattening past k=16. **polyepo
collapses to AUC=0 on aime26** (0/30 solved across all 1920 rollouts).

| arm \ dataset | aime25 | aime26 | beyondaime | hmmt_feb25 | hmmt_nov25 | math500 |
|---|---|---|---|---|---|---|
| base | 14.566 | 9.360 | 12.180 | 7.322 | 7.685 | **54.799** |
| grpo | 2.826 | 2.133 | 4.932 | 2.133 | 7.850 | 46.288 |
| minority | 1.562 | 3.695 | 3.681 | 3.818 | 7.988 | 44.900 |
| polyepo | 5.088 | 0.000 | 5.115 | 5.951 | 7.998 | _MISSING_ |

**math500 column (easy OOD):** Base AUC = 54.8, ~9x its hard-OOD AUC.
GRPO and Minority sit around 45 — base's lead is smaller relative to the
absolute pass@k (since all arms saturate higher on easy problems) but
still unambiguous. No hmmt_nov25-style crossover here.

## Underlying pass@k points

- **base / aime25**: pass@1=0.019, pass@2=0.036, pass@4=0.067, pass@8=0.116, pass@16=0.182, pass@32=0.254, pass@64=0.333
- **base / aime26**: pass@1=0.019, pass@2=0.036, pass@4=0.062, pass@8=0.096, pass@16=0.128, pass@32=0.158, pass@64=0.200
- **base / beyondaime**: pass@1=0.018, pass@2=0.034, pass@4=0.058, pass@8=0.095, pass@16=0.144, pass@32=0.209, pass@64=0.290
- **base / hmmt_feb25**: pass@1=0.005, pass@2=0.010, pass@4=0.020, pass@8=0.039, pass@16=0.071, pass@32=0.123, pass@64=0.200
- **base / hmmt_nov25**: pass@1=0.028, pass@2=0.048, pass@4=0.075, pass@8=0.101, pass@16=0.122, pass@32=0.132, pass@64=0.133
- **grpo / aime25**: pass@1=0.003, pass@2=0.006, pass@4=0.012, pass@8=0.021, pass@16=0.034, pass@32=0.049, pass@64=0.067
- **grpo / aime26**: pass@1=0.001, pass@2=0.002, pass@4=0.004, pass@8=0.008, pass@16=0.017, pass@32=0.033, pass@64=0.067
- **grpo / beyondaime**: pass@1=0.007, pass@2=0.013, pass@4=0.023, pass@8=0.037, pass@16=0.057, pass@32=0.084, pass@64=0.120
- **grpo / hmmt_feb25**: pass@1=0.001, pass@2=0.002, pass@4=0.004, pass@8=0.008, pass@16=0.017, pass@32=0.033, pass@64=0.067
- **grpo / hmmt_nov25**: pass@1=0.013, pass@2=0.024, pass@4=0.042, pass@8=0.070, pass@16=0.105, pass@32=0.139, pass@64=0.167
- **minority / aime25**: pass@1=0.002, pass@2=0.003, pass@4=0.006, pass@8=0.011, pass@16=0.019, pass@32=0.029, pass@64=0.033
- **minority / aime26**: pass@1=0.003, pass@2=0.005, pass@4=0.010, pass@8=0.019, pass@16=0.036, pass@32=0.063, pass@64=0.100
- **minority / beyondaime**: pass@1=0.006, pass@2=0.011, pass@4=0.019, pass@8=0.030, pass@16=0.043, pass@32=0.061, pass@64=0.090
- **minority / hmmt_feb25**: pass@1=0.003, pass@2=0.005, pass@4=0.010, pass@8=0.020, pass@16=0.038, pass@32=0.067, pass@64=0.100
- **minority / hmmt_nov25**: pass@1=0.013, pass@2=0.024, pass@4=0.044, pass@8=0.074, pass@16=0.110, pass@32=0.141, pass@64=0.167
- **polyepo / aime25**: pass@1=0.006, pass@2=0.011, pass@4=0.020, pass@8=0.035, pass@16=0.055, pass@32=0.083, pass@64=0.133
- **polyepo / aime26**: pass@1=0.000, pass@2=0.000, pass@4=0.000, pass@8=0.000, pass@16=0.000, pass@32=0.000, pass@64=0.000
- **polyepo / beyondaime**: pass@1=0.006, pass@2=0.012, pass@4=0.022, pass@8=0.037, pass@16=0.058, pass@32=0.085, pass@64=0.130
- **polyepo / hmmt_feb25**: pass@1=0.004, pass@2=0.007, pass@4=0.014, pass@8=0.028, pass@16=0.054, pass@32=0.100, pass@64=0.167
- **polyepo / hmmt_nov25**: pass@1=0.014, pass@2=0.026, pass@4=0.046, pass@8=0.075, pass@16=0.108, pass@32=0.142, pass@64=0.167
- **base / math500**: pass@1=0.358, pass@2=0.541, pass@4=0.704, pass@8=0.806, pass@16=0.864, pass@32=0.902, pass@64=0.928
- **grpo / math500**: pass@1=0.299, pass@2=0.427, pass@4=0.542, pass@8=0.636, pass@16=0.711, pass@32=0.769, pass@64=0.816
- **minority / math500**: pass@1=0.265, pass@2=0.388, pass@4=0.504, pass@8=0.602, pass@16=0.683, pass@32=0.750, pass@64=0.804
- **polyepo / math500**: GEN failed mid-JSON-write — only 2/500 prompts recovered, pass@k not citable

## How this was computed

- **Script**: `main-verl/eval/analysis/posthoc/auc_at_k.py` (definition
  `AUC@k = trapezoid(pass_at_k_vector, ks)` over k ladder {1,2,4,8,16,32,64}).
- **Inputs**: the 20 `*_step400_smallood_*.json` probe files (4 arms ×
  5 datasets) under `/vol/probes/eval_4b/`. Each file holds the per-prompt
  rewards, predictions, and a saved `pass_at_k` dict; the script
  recomputes pass@k from `per_prompt[i].n_correct` if the saved dict is
  missing keys.
- **Eval probe sampling (re-used from `main-verl/eval/run_eval.py`)**:
  Qwen3-4B-Base + 3 trained-step-400 HF-merged checkpoints, B200:1,
  vLLM `enforce_eager=True`, `gpu_memory_utilization=0.95`,
  `max_model_len=5120`, `max_num_seqs=4096`,
  `temperature=1.0, top_p=1.0, max_tokens=4096`, **n=64 rollouts/prompt**,
  `logprobs=20`.
- **Grader**: `verl.utils.reward_score.math.compute_score` (Hendrycks
  `is_equiv`, mathd ∨ sympy fallback). Same grader used by the training
  reward signal.
- **Limitations / caveats**:
  - The trapezoid uses raw k as the x-axis (not log-k), so high-k bins
    dominate the area — a model that only wins at k=64 can match a model
    that wins at k=1..8.
  - AUC@k uses the unbiased pass@k estimator
    `1 - C(n-c, k) / C(n, k)` so partial-credit problems are smoothed.
    Zero AUC means **zero** correct rollouts across the full n=64
    (genuine collapse, not a thresholding artifact).

