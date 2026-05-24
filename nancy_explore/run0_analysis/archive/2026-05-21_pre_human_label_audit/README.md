# Pre–human-label audit archive (2026-05-21)

Outputs and scripts from the v1/v2 parser era. **Do not use for metrics or paper claims** unless explicitly comparing to historical runs.

## Eligible prompt count (common confusion)

| Label source | Prompts with ≥1 correct rollout | LLM minority rate (same cluster IDs) |
|--------------|--------------------------------:|-------------------------------------:|
| v2 `is_correct_v2` (archive) | **165** / 500 | **14.55%** (24/165) |
| Human `cleaned_correct` (canonical) | **172** / 500 | **14.53%** (25/172) |

Canonical sources: `data/cleaned_answers.parquet`, `analysis_minority/minority_metrics.md`, `analysis_a/analysis_a_summary.md`.

## Where stale **165** appears

- `analysis_a/analysis_a_summary.md`, `analysis_a_full_run.log`
- `analysis_b/substrate_comparison.md`, `substrate_comparison_v2.md`, `analysis_b_run.log`
- `analysis_d/baseline_metrics.json` (`n_eligible` fields)
- `overnight_workflow_log.md`

All of these are labeled or bannered as v2-era; do not copy into active docs.
