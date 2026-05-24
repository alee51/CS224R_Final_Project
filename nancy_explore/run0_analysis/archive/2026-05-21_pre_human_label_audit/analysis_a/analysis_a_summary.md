# Analysis A — LLM reasoning clusters (summary)

> ⚠ **ARCHIVED (v2-era).** Eligible **165** and rate **14.55%** used `is_correct_v2`. Current summary: [`../../../analysis_a/analysis_a_summary.md`](../../../analysis_a/analysis_a_summary.md) — eligible **172**, **14.53%** under `cleaned_correct`.

**Generated:** 2026-05-21  
**Provider / tier / model:** `google` / `cheap` / `gemini-3.1-flash-lite`  
**Cache dir:** `pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters`  

## Headline: minority_correct_prompt_rate_llm
Among prompts with ≥1 correct rollout (v2 parser), fraction where correct rollouts span ≥2 LLM clusters and at least one correct cluster is not the largest (same definition as `has_minority_correct_cluster` in `pilot/train/run_proxy.py`).

| Metric | Value |
|---|---:|
| Prompts attempted | 500 |
| Prompts with successful parse | 500 |
| Prompts with ≥1 correct (eligible) | 165 |
| **minority_correct_prompt_rate_llm** | **14.55%** (95% CI [9.09%, 20.00%]) |

| Rollouts in degenerate cluster (`cluster_id == -1`) | 16.95% of parsed assignments |

## Next steps (manual)
- Hand-check 10 prompts → [`llm_clusters_handcheck.md`](llm_clusters_handcheck.md) (see design doc §A.7)
- Optional cheap↔moderate ARI on 50 prompts → `llm_judge_cross_tier.md`
