# Phase 1 — Run 0 cluster readout
**Generated:** 2026-05-26  
**Ground truth:** `data/cleaned_answers.parquet`  
**LLM clusters:** `analysis_a/llm_clusters_summary.parquet` (degenerate `100` → `-1`; all `-1` in a prompt = one cluster)  

## Pass@k (human-verified correctness)

| Metric | Value | 95% bootstrap CI |
|---|---:|---:|
| Prompts | 500 | — |
| Rollouts | 4000 | — |
| **Pass@1** | **9.03%** (361/4000) | — |
| **Pass@8** | **34.40%** | [30.20%, 38.40%] |

Pass@8: Chen et al. unbiased Pass@k per prompt (k=8), prompt-level bootstrap (1000 resamples, seed=0).

## Methods — distinct clusters per prompt
For each of 500 prompts (8 rollouts), define a per-prompt count = |{distinct cluster IDs among rollouts in that row's scope}|.
- **Answer-hash:** `cleaned_cluster_id` (canonical human-verified answer string).
- **LLM reasoning:** `llm_cluster_id` (`100` → `-1`; all `-1` on a prompt counts as one cluster).

**Row scopes (prompt cohorts are disjoint for correct vs all-incorrect):**
1. **All prompts** — all 8 rollouts (n=500).
2. **≥1 correct prompt** — only the 172 prompts with ≥1 `cleaned_correct` rollout; count distinct clusters among **correct rollouts only** on that prompt (wrong rollouts on the same prompt are ignored).
3. **All-incorrect prompt** — the complementary 328 prompts with **zero** correct rollouts; count distinct clusters among **all 8** rollouts (every rollout is incorrect). No prompt has all 8 correct on this run.

Median / mean / range are taken over prompts in that cohort. Mixed prompts (some correct, some wrong) appear only in row 2 for the correct-rollout count; their incorrect rollouts are **not** included in row 3.

## Distinct clusters per prompt

| Stratum | n prompts | Substrate | Median | Mean | Range |
|---|--:|---|---:|---:|---:|
| All prompts (8 rollouts each) | 500 | Answer-hash | **6** | 5.70 | 1–8 |
| All prompts (8 rollouts each) | 500 | LLM reasoning | **5** | 5.28 | 1–8 |
| ≥1 correct prompt (correct rollouts only) | 172 | Answer-hash | **1** | 1.00 | 1–1 |
| ≥1 correct prompt (correct rollouts only) | 172 | LLM reasoning | **1** | 1.31 | 1–4 |
| All-incorrect prompt (8 rollouts each) | 328 | Answer-hash | **6** | 5.77 | 1–8 |
| All-incorrect prompt (8 rollouts each) | 328 | LLM reasoning | **5** | 5.36 | 1–8 |
