# Run 0 analysis

Offline analyses on Run 0 proxy rollouts. Design doc: `**[run0_offline_analyses_20260521.md](run0_offline_analyses_20260521.md)**`.

## Config


| Path                           | Role                                                                       |
| ------------------------------ | -------------------------------------------------------------------------- |
| `config/llm_judge_models.yaml` | Analysis A — tier → API model ID; Google `cheap` = `gemini-3.1-flash-lite` |
| `.env`                         | `GOOGLE_API_KEY` (+ optional `GEMINI_MODEL`) — see `.env.example`          |
| `config/analysis_a_prompt.md`  | Analysis A — system/user prompt templates                                  |


## Data


| Path                              | Role                                                                             |
| --------------------------------- | -------------------------------------------------------------------------------- |
| `data/raw_predictions.jsonl`      | Symlink → `pilot/artifacts/run0_proxy/20260519T190202Z/`                         |
| `data/prompt_inputs.jsonl`        | Same run, 500 prompts                                                            |
| `data/predictions_reparsed.jsonl` | v1 fields + `parsed_answer_v2`, `canonical_v2`, `is_correct_v2`, `cluster_id_v2` |


## Labels (canonical)


| Path                              | Role                                                    |
| --------------------------------- | ------------------------------------------------------- |
| `**labels/rollout_labels.jsonl**` | 4000 rows — tail-only `result` per rollout (`extracted` |


Human-style labels are best-effort (dual blind A/B + manual disputes), not ground truth.

## Scripts (active)


| Script                       | Role                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `reparse_rescore.py`         | Prerequisite 0b — C2/C3 re-score → `predictions_reparsed.jsonl`, `reparse_diff.md` |
| `analysis_a_llm_clusters.py` | **Analysis A** — LLM reasoning clusters per prompt (8 rollouts)                    |
| `audit_1024_token_labels.py` | Qwen tokenizer audit vs 1024 cap                                                   |


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


| Path                                                                        | Role                                                           |
| --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `data/predictions_reparsed.jsonl`                                           | Input rollouts (v2 parse fields)                               |
| `data/prompt_inputs.jsonl`                                                  | Problem + gold per prompt                                      |
| `pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters/{prompt_id}.json` | Per-prompt cache (idempotent skip)                             |
| `llm_clusters_summary.parquet`                                              | Per-rollout LLM cluster assignment                             |
| `analysis_a_summary.md`                                                     | Headline `minority_correct_prompt_rate_llm` + bootstrap 95% CI |


**Usage** (from repo root):

```bash
# Build prompts only (no API); writes cache JSON with system/user prompts
python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --dry-run --pilot

# 5-prompt pilot API run
python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --pilot --tier cheap

# Full 500 prompts (~$5 cheap tier), 10 concurrent workers
python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --tier cheap --provider google

# Re-label despite cache
python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --force --limit 10
```

**CLI flags:** `--pilot` (5 prompts), `--limit N`, `--tier cheap|moderate|expensive`, `--provider google`, `--dry-run`, `--force`, `--workers` (default 10).

Manual follow-ups after a full run (not automated here): `llm_clusters_handcheck.md` (§A.7), optional `llm_judge_cross_tier.md` (cheap vs moderate on 50 prompts).

## Archived labeling pipeline

Everything used to **produce** `rollout_labels.jsonl` lives under `**labeling/`** (chunks, blind outputs, spawn prompts, merge scripts). See `[labeling/README.md](labeling/README.md)`.