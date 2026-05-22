# Run 0 cluster-review dashboard

Per-prompt browser of Run 0's **human-verified cleaned answers** alongside **Analysis A LLM clusters**. Static HTML; no server-side code.

## Quick start

```bash
python build.py
./serve.sh   # http://localhost:8766
```

`build.py` reads:
- `../data/cleaned_answers.parquet` — canonical answer-extraction (`cleaned_answer`, `cleaned_state`, `cleaned_correct`, `cleaned_cluster_id`)
- `../data/predictions_reparsed.jsonl` — completion text only (v1/v2 parser fields ignored)
- `../data/prompt_inputs.jsonl` — problem text + gold
- `../analysis_a/llm_clusters_summary.parquet` — `llm_cluster_id` per rollout (`100` → `-1` degenerate; all `-1` in a prompt = one cluster)

Writes `data.js`. Re-run after any of those inputs change.

## What the dashboard shows

**Per prompt:**
- Correct rollouts (e.g. 3/8) under cleaned ground truth
- LLM cluster count + size of the largest cluster containing a correct rollout
- Cleaned-answer cluster count + same
- Minority-correct flag under both clusterings (LLM and cleaned)

**Per rollout (collapsible):**
- Cleaned answer + correctness mark (or `(runon)` / `(no_answer)` if the model didn't state one)
- LLM cluster id + cluster size (degenerate tagged)
- Cleaned-answer cluster id + cluster size
- Cleaned state (extracted / runon / no_answer)
- Full completion (KaTeX-rendered)

## Filters

- All
- Has correct / No correct / Partial (1–7)
- Minority-LLM (correct rollouts span ≥2 LLM clusters with at least one minority)
- Minority-cleaned (same on the cleaned-answer clustering)

Keyboard: `j`/`↓` next, `k`/`↑` prev, `l` expand all rollouts on the current prompt.
