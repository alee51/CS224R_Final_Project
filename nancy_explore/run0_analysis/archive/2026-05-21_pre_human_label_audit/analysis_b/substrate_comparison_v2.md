# Analysis B v2 — Qwen3-Embedding-0.6B substrate sweep

> ⚠ **ARCHIVED (v2-era).** **165** eligible / **14.55%** LLM minority reference used `is_correct_v2`. Human labels: **172** eligible, **14.53%** — see [`../../analysis_minority/minority_metrics.md`](../../analysis_minority/minority_metrics.md).

Re-embed all 4000 completions with `Qwen/Qwen3-Embedding-0.6B` (32K context, math-strong) and re-cluster per-prompt.

**Model:** `Qwen/Qwen3-Embedding-0.6B`  
**Why:** MiniLM (256-token cap) was truncating 91.6% of Run 0 completions. Qwen3-Embedding sees full completions.

**Reference (v2):** `llm_cluster_id` from Analysis A (LLM minority-correct prompt rate = 14.55% on **165** eligible prompts, `is_correct_v2`).

## Aggregate substrate metrics (new embedder)

| Substrate | Mean ARI [95% CI] | Mean V-measure [95% CI] | Mean \|Δn_clusters\| | Minority-rate | Concordance acc (eligible) |
|---|---|---|---|---|---|
| `completion_embedding@0.05_qwen3_0p6b` | 0.109 [0.087, 0.131] | 0.762 [0.746, 0.778] | 1.990 | 18.79% | 0.873 |
| `completion_embedding@0.1_qwen3_0p6b` | 0.063 [0.048, 0.080] | 0.502 [0.480, 0.524] | 2.424 | 11.52% | 0.836 |
| `completion_embedding@0.15_qwen3_0p6b` | 0.044 [0.030, 0.059] | 0.303 [0.280, 0.326] | 3.178 | 4.85% | 0.867 |
| `completion_embedding@0.2_qwen3_0p6b` | 0.028 [0.016, 0.042] | 0.195 [0.175, 0.217] | 3.586 | 2.42% | 0.842 |
| `completion_embedding@0.25_qwen3_0p6b` | 0.018 [0.007, 0.029] | 0.138 [0.120, 0.157] | 3.802 | 2.42% | 0.842 |
| `completion_embedding@0.3_qwen3_0p6b` | 0.015 [0.005, 0.027] | 0.109 [0.093, 0.126] | 3.910 | 2.42% | 0.842 |
| `completion_embedding@0.4_qwen3_0p6b` | 0.013 [0.004, 0.024] | 0.088 [0.073, 0.104] | 3.984 | 2.42% | 0.842 |
| `completion_embedding@0.5_qwen3_0p6b` | 0.011 [0.001, 0.021] | 0.074 [0.060, 0.089] | 4.032 | 1.82% | 0.848 |
| `completion_embedding@0.6_qwen3_0p6b` | 0.014 [0.004, 0.024] | 0.060 [0.046, 0.074] | 4.098 | 1.82% | 0.848 |

## Confusion matrices vs LLM (eligible prompts: has ≥1 correct)

| Substrate | TP | FP | FN | TN |
|---|---|---|---|---|
| `completion_embedding@0.05_qwen3_0p6b` | 17 | 14 | 7 | 127 |
| `completion_embedding@0.1_qwen3_0p6b` | 8 | 11 | 16 | 130 |
| `completion_embedding@0.15_qwen3_0p6b` | 5 | 3 | 19 | 138 |
| `completion_embedding@0.2_qwen3_0p6b` | 1 | 3 | 23 | 138 |
| `completion_embedding@0.25_qwen3_0p6b` | 1 | 3 | 23 | 138 |
| `completion_embedding@0.3_qwen3_0p6b` | 1 | 3 | 23 | 138 |
| `completion_embedding@0.4_qwen3_0p6b` | 1 | 3 | 23 | 138 |
| `completion_embedding@0.5_qwen3_0p6b` | 1 | 2 | 23 | 139 |
| `completion_embedding@0.6_qwen3_0p6b` | 1 | 2 | 23 | 139 |

## Headline

- Best mean ARI under Qwen3 embeddings: **`completion_embedding@0.05_qwen3_0p6b`** with mean ARI = **0.109** [0.087, 0.131].
- Compared to v1 best (MiniLM@0.2, ARI 0.074) and the overall v1 best across all substrates (`answer_strict`, ARI 0.188).
- See `substrate_comparison.md` (v1) for the original side-by-side substrate table.
