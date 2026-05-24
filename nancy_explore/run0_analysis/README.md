# Run 0 analysis

Offline analyses on Run 0 proxy rollouts. **Execution plan:** `[run0_exec_plan.md](run0_exec_plan.md)`. Historical v2-era spec + overnight log: `[archive/2026-05-21_pre_human_label_audit/](archive/2026-05-21_pre_human_label_audit/)`.

> **2026-05-21 reset.** All prior parser outputs (v1, v2) and the analyses that depended on them have been moved to `archive/2026-05-21_pre_human_label_audit/`. The canonical answer-extraction artifact going forward is `**data/cleaned_answers.parquet`**, built from human-verified labels in `labels/rollout_labels.jsonl`. See `nancy_explore/narrative/timeline.md` (entry 2026-05-21) for context.

## Folder layout


| Folder        | Role                                                                             |
| ------------- | -------------------------------------------------------------------------------- |
| `data/`       | Raw rollouts, prompts, gold answers, and the canonical `cleaned_answers.parquet` |
| `labels/`     | Human-verified answer-extraction labels (`rollout_labels.jsonl`, 4000 rows)      |
| `labeling/`   | Blind-A/B + dispute-resolution pipeline archive (provenance for `labels/`)       |
| `config/`     | LLM judge prompts + model yaml                                                   |
| `analysis_a/` | LLM reasoning-cluster artifact (script + `llm_clusters_summary.parquet`)         |
| `analysis_b/` | Long-context completion embeddings cache (Qwen3-Embedding-0.6B, 1024-dim)        |
| `dashboard/`  | Local browser view (cleaned answers + LLM clusters per prompt)                   |
| `archive/`    | Outputs from the pre-human-label-audit era; not part of current analyses         |


## Data


| Path                              | Role                                                                                                                                                                                                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/raw_predictions.jsonl`      | Symlink → `pilot/artifacts/run0_proxy/20260519T190202Z/raw_predictions.jsonl`                                                                                                                                                                                                                           |
| `data/prompt_inputs.jsonl`        | 500 prompts (problem + gold)                                                                                                                                                                                                                                                                            |
| `data/cleaned_answers.parquet`    | **Canonical.** 4000 rows: `cleaned_answer`, `cleaned_state`, `cleaned_correct`, `cleaned_cluster_id` — derived from human labels                                                                                                                                                                        |
| `data/predictions_reparsed.jsonl` | Source of `completion` text for each rollout. ⚠ Contains stale v1/v2 parser fields (`parsed_answer`, `correct`, `cluster_id`, `parsed_answer_v2`, `canonical_v2`, `is_correct_v2`, `cluster_id_v2`, `extract_path_v2`, `parser_clean_v2`) — **do not use these; use `cleaned_answers.parquet` instead** |


### Schema of `data/cleaned_answers.parquet`


| Column               | Type | Meaning                                                              |
| -------------------- | ---- | -------------------------------------------------------------------- |
| `prompt_id`          | str  |                                                                      |
| `rollout_idx`        | int  | 0..7 within prompt                                                   |
| `rollout_key`        | str  | `{prompt_id}#{rollout_idx}`                                          |
| `gold`               | str  | Gold answer from `prompt_inputs.jsonl`                               |
| `cleaned_answer`     | str  | Human-verified extracted answer (empty for runon/no_answer)          |
| `cleaned_state`      | str  | One of `extracted`, `runon`, `no_answer`                             |
| `cleaned_correct`    | bool | `normalize(cleaned_answer) == normalize(gold)` AND state=`extracted` |
| `cleaned_cluster_id` | int  | SHA8 of canonicalized cleaned_answer (0 when no answer)              |


Baseline summary under cleaned labels: **Pass@1 = 9.03%** (361/4000), **Pass@8 = 34.40%** (172/500 prompts with ≥1 correct). Minority LLM rate **14.53%** (25/172 eligible). Archive v2-era tables used **165** eligible — see `archive/2026-05-21_pre_human_label_audit/README.md`.

## Labels


| Path                          | Role                                                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `labels/rollout_labels.jsonl` | 4000 rows. Blind dual-AI extraction (`result_A`, `result_B`) + human dispute resolution (`human_result`). Final adjudicated value in `result`. **Source of truth** for `cleaned_answers.parquet` |


## Scripts (active)


| Script                                  | Role                                                                                   |
| --------------------------------------- | -------------------------------------------------------------------------------------- |
| `analysis_a/analysis_a_llm_clusters.py` | LLM reasoning-cluster judge (Gemini cheap-tier); writes `llm_clusters_summary.parquet` |
| `dashboard/build.py`                    | Build `data.js` for the local cluster-review dashboard                                 |


### Analysis A (`analysis_a_llm_clusters.py`)

Offline LLM judge clustering for Run 0 (design doc §A). Groups 8 rollouts per `prompt_id`, calls Google Gemini at the configured tier, caches raw JSON per prompt, and writes summary artifacts.

**Dependencies** (not in `pilot/requirements.txt`; install in your venv):

```bash
pip install pyyaml pyarrow google-genai python-dotenv
```

Uses `**google-genai**` (`from google import genai`) when installed; falls back to `**google-generativeai**` if only the legacy package is present.

**Environment**


| Variable                             | Role                                                                             |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google AI Studio / Gemini API key                                                |
| `.env` in `run0_analysis/`           | Optional; loaded via `python-dotenv` if available, else simple `KEY=value` parse |


**Model config:** `config/llm_judge_models.yaml` — default provider `google`, tier `cheap` → `gemini-3.1-flash-lite` (AI Studio “3.1 flash”; API id is not `gemini-3.1-flash`). Prompt templates: `config/analysis_a_prompt.md`.

**Paths**


| Path                                                                        | Role                                                                                |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `data/predictions_reparsed.jsonl`                                           | Input rollouts (only `prompt_id` + `completion` are used; v1/v2 fields are ignored) |
| `data/prompt_inputs.jsonl`                                                  | Problem + gold per prompt                                                           |
| `pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters/{prompt_id}.json` | Per-prompt cache (idempotent skip)                                                  |
| `analysis_a/llm_clusters_summary.parquet`                                   | Per-rollout LLM cluster assignment (`100` → `-1` degenerate; all `-1` in a prompt = one cluster) |


**Usage** (from repo root):

```bash
# Build prompts only (no API); writes cache JSON with system/user prompts
python nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py --dry-run --pilot

# 5-prompt pilot API run
python nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py --pilot --tier cheap

# Full 500 prompts (~$5 cheap tier), 10 concurrent workers
python nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py --tier cheap --provider google

# Re-label despite cache
python nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py --force --limit 10
```

**CLI flags:** `--pilot` (5 prompts), `--limit N`, `--tier cheap|moderate|expensive`, `--provider google`, `--dry-run`, `--force`, `--workers` (default 10).

Manual follow-ups after a full run (not automated here): `llm_clusters_handcheck.md` (§A.7), optional `llm_judge_cross_tier.md` (cheap vs moderate on 50 prompts).

## Archived labeling pipeline

Everything used to **produce** `rollout_labels.jsonl` lives under `**labeling/`** (chunks, blind outputs, spawn prompts, merge scripts). See `[labeling/README.md](labeling/README.md)`.