# Run 0 analysis plan (merged)

**Date:** 2026-05-21 (post human-label reset)  
**Workspace:** `nancy_explore/run0_analysis/`  
**Run artifact:** `pilot/artifacts/run0_proxy/20260519T190202Z/`  
**Constraint:** No new training, no GPU. Phase 1 is laptop-only (+ existing LLM cluster cache). Phase 2 may use CPU embeddings and archived script patterns; no new Analysis A judge calls unless explicitly re-run.

**Single spec** for Run 0 offline analysis (replaces `run0_offline_analyses.md` and `milestone_analysis_plan.md`).

---

## Table of contents

1. [2026-05-21 reset and archive](#2026-05-21-reset-and-archive)
2. [Scope](#scope)
3. [Canonical data and headline numbers](#canonical-data-and-headline-numbers)
4. [Completed infrastructure](#completed-infrastructure)
5. [Phase 1 — Q II milestone (immediate)](#phase-1--q-ii-milestone-immediate)
6. [Phase 2 — Redo analyses A metrics / B / C / D on cleaned labels](#phase-2--redo-analyses-a-metrics--b--c--d-on-cleaned-labels)
7. [Can claim / cannot claim](#can-claim--cannot-claim)
8. [Execution order](#execution-order)
9. [Output paths (consolidated)](#output-paths-consolidated)
10. [Related docs](#related-docs)
11. [Explicitly out of scope this round](#explicitly-out-of-scope-this-round)

---

## 2026-05-21 reset and archive

Human-verified tail labels (`labels/rollout_labels.jsonl`) are now ground truth for answer extraction and correctness. Automated parser v1/v2 fields remain in `data/predictions_reparsed.jsonl` for audit only; v2-vs-human presence agreement was **78%**, so both parsers were retired for scoring.

| What moved | Where |
|------------|--------|
| Full A/B/C/D design | [`archive/2026-05-21_pre_human_label_audit/`](archive/2026-05-21_pre_human_label_audit/) |
| Overnight session log | [`archive/.../overnight_workflow_log.md`](archive/2026-05-21_pre_human_label_audit/overnight_workflow_log.md) |
| Archived spec (v2-era) | [`archive/.../run0_offline_analyses_20260521.md`](archive/2026-05-21_pre_human_label_audit/run0_offline_analyses_20260521.md) |
| B/C/D scripts and v2 outputs | `archive/.../analysis_b/`, `analysis_c/`, `analysis_d/` |
| Parser re-score (0b) | `archive/.../prereq_0b_reparse/` |

**Do not cite** archived headline numbers (v1 Pass@1 8.10%, v2 8.25%, `minority_correct_prompt_rate_v2` 0%) or `overnight_workflow_log.md` without the cleaned-label banner. **Analysis A LLM cluster assignments are parser-independent** — cache is reused; only **downstream metrics** must re-join `cleaned_correct` / `cleaned_cluster_id`.

Context: `nancy_explore/narrative/timeline.md` (entry 2026-05-21).

---

## Scope

| In scope | Out of scope (deferred) |
|----------|-------------------------|
| **Phase 1 — Q II:** minority-voting readout, distributions, qual | **Q I** until Phase 2 B completes on cleaned labels |
| Metrics on **human-verified** answers (`cleaned_*`) | Parser v1/v2 comparison, `reparse_diff` claims |
| Reuse **completed** LLM reasoning clusters (Analysis A cache) | In-loop judge replacement claims |
| **Phase 2 — redo** B/C/D (and A metric refresh) on cleaned inputs | New training, AIME/HMMT generalization |
| Bootstrap CIs, dashboard forensics | Numbers from pre-reset handoff without cleaned banner |

---

## Canonical data and headline numbers

### Data table

| Path | Role |
|------|------|
| `labels/rollout_labels.jsonl` | Source of truth (4000 rows, blind A/B + human disputes) |
| `data/cleaned_answers.parquet` | **`cleaned_answer`, `cleaned_correct`, `cleaned_cluster_id`, `cleaned_state`** — all correctness and answer-hash clustering |
| `data/predictions_reparsed.jsonl` | **`completion` text only** — v1/v2 fields stale |
| `data/prompt_inputs.jsonl` | 500 prompts (problem + gold) |
| `analysis_a/llm_clusters_summary.parquet` | **`llm_cluster_id`** per rollout — cluster IDs valid; join `cleaned_correct` at metric time |
| `pilot/artifacts/.../llm_clusters/{prompt_id}.json` | Analysis A judge cache (500 prompts, Poly-EPO §A.1) |
| `analysis_b/embedding_ids_qwen3_0p6b.parquet` | Qwen3-Embedding-0.6B completion embeddings (infrastructure for Phase 2 B) |
| `config/llm_judge_models.yaml`, `config/analysis_a_prompt.md` | Analysis A judge config |

**Schema (`cleaned_answers.parquet`):** see [`README.md`](README.md). Join key: `rollout_key` or (`prompt_id`, `rollout_idx`).

**Stale fields (never use for metrics):** `parsed_answer_v2`, `is_correct_v2`, `cluster_id_v2`, `canonical_v2`, `extract_path_v2`, and all v1 parser columns in `predictions_reparsed.jsonl`.

### Headline numbers (cleaned labels + existing LLM clusters)

| Metric | Value |
|--------|-------|
| Pass@1 | **9.03%** (361/4000 rollouts) |
| Pass@8 | **34.40%** (172/500 prompts) |
| Eligible prompts (≥1 correct rollout) | **172** |
| `minority_correct_prompt_rate` (answer-hash / `cleaned_cluster_id`) | **0%** |
| `minority_correct_prompt_rate_llm` | **~14.5%** (25/172; bootstrap 95% CI in Phase 1) |

Under answer-only clustering, minority-correct structure is absent on Run 0; under LLM reasoning clustering, a minority of eligible prompts show correct rollouts in non-dominant clusters — the gate metric Q II cares about.

---

## Completed infrastructure

### Prerequisite 0a — Human rollout labels — **DONE**

`labels/rollout_labels.jsonl` → `data/cleaned_answers.parquet`. Provenance: `labeling/` (archived pipeline).

### Prerequisite 0b — Parser re-score — **DONE, audit only**

`data/predictions_reparsed.jsonl` retains v2 fields for audit. Not used for metrics. Diff: `archive/2026-05-21_pre_human_label_audit/prereq_0b_reparse/reparse_diff.md`.

### Analysis A — LLM reasoning clusters — **DONE (cache); metrics join cleaned labels**

- Script: `analysis_a/analysis_a_llm_clusters.py`
- One-shot offline judge (Poly-EPO §A.1); does not see gold or parsed answers
- **Reuse** `pilot/artifacts/.../llm_clusters/*.json` and `analysis_a/llm_clusters_summary.parquet` — no `--force` unless re-labeling
- **Phase 1 / Phase 2:** recompute `minority_correct_prompt_rate_llm`, Pass@*, Cover@τ, etc. by joining `cleaned_correct` from `cleaned_answers.parquet` (ignore any correctness columns baked into old summary exports)
- Optional housekeeping: re-run script without `--force` to refresh `analysis_a/analysis_a_summary.md` with cleaned-label wording
- Quality artifacts from first run (if still needed): `archive/.../analysis_a/llm_clusters_handcheck.md`, `llm_degenerate_sanity.md`; optional cross-tier probe per archived §A.7

### Dashboard — **DONE**

`dashboard/build.py`, `dashboard/serve.sh` — rebuild `data.js` after parquet changes.

---

## Phase 1 — Q II milestone (immediate)

**Question:** On Run 0 proxy rollouts, does minority-correct structure exist under cheap vs LLM substrates?  
**No** new judge or embedding API calls in Phase 1.

### Deliverables

#### 1. Headline gate metric: `minority_correct_prompt_rate`

Among prompts with ≥1 correct rollout (**n_eligible = 172**), fraction where correct rollouts span ≥2 clusters and at least one correct cluster is not the largest.

| Substrate | Column | Expected (verified) |
|-----------|--------|---------------------|
| Cleaned answer-hash | `cleaned_cluster_id` | **0%** |
| LLM reasoning | `llm_cluster_id` from `analysis_a/llm_clusters_summary.parquet` | **~14.5%** (25/172) |

Also report Pass@1 **9.03%**, Pass@8 **34.40%**. Prompt-level bootstrap 95% CIs (1000 resamples).

**Definition (per prompt, substrate S):**

1. Restrict to rollouts with `cleaned_correct == true`.
2. If none, prompt not eligible.
3. Group by cluster ID under S (`cleaned_cluster_id` or `llm_cluster_id`; map LLM degenerate `100` → `-1` if needed).
4. Minority-correct iff ≥2 distinct correct clusters and ∃ correct cluster whose size &lt; max correct-cluster size.

**Output:** `analysis_minority/minority_metrics.md` (+ optional bootstrap table CSV).

#### 2. Distribution view (~1 page)

Per-prompt histograms:

- Number of `cleaned_cluster_id` buckets (answer-hash)
- Number of `llm_cluster_id` buckets
- Size of largest correct cluster (each substrate)

**Output:** `analysis_minority/minority_distributions.png` (or `.pdf`).

#### 3. Qualitative forensics

~25 prompts with `minority_correct_llm` (dashboard filter **Minority-LLM**). Screenshot 3–5 examples: minority correct cluster vs largest cluster; brief reasoning note.

```bash
python nancy_explore/run0_analysis/dashboard/build.py
./nancy_explore/run0_analysis/dashboard/serve.sh
```

### Phase 1 implementation status

| Item | Status |
|------|--------|
| Human labels → `cleaned_answers.parquet` | **Done** |
| LLM clusters (500 prompts) | **Done** |
| Dashboard | **Done** (rebuild `data.js` after parquet changes) |
| `analysis_minority/` script + metrics + plots | **Not started** |

**Suggested script:** `analysis_minority/minority_readout.py` (new) — reads `cleaned_answers.parquet` + `analysis_a/llm_clusters_summary.parquet`, writes Phase 1 outputs.

---

## Phase 2 — Redo analyses A metrics / B / C / D on cleaned labels

Archived implementations used v2 parser fields; **redo on cleaned inputs** using formulas from [`archive/.../run0_offline_analyses_20260521.md`](archive/2026-05-21_pre_human_label_audit/run0_offline_analyses_20260521.md) with substitutions below. Adapt or port scripts from `archive/.../analysis_{a,b,c,d}/` — do not read v2 columns.

**Global substitutions**

| Archived (stale) | Cleaned (use) |
|------------------|---------------|
| `is_correct_v2`, `reward_v2` | `cleaned_correct` |
| `cluster_id_v2`, `canonical_v2` | `cleaned_cluster_id`, `cleaned_answer` |
| `answer_strict` / `answer_loose` (v1/v2 parsers) | Single substrate: **`answer_hash_cleaned`** = `cleaned_cluster_id` |
| `parsed_answer_v2` in metrics | `cleaned_answer` (labels only; not sent to judge) |

---

### Phase 2A — Analysis A metric refresh (no re-judge)

**Status:** Cache **done**; recompute downstream tables only.

| Task | Input | Output |
|------|-------|--------|
| Join cleaned correctness | `cleaned_answers.parquet` + `llm_clusters_summary.parquet` | Updated `analysis_a/analysis_a_summary.md` (Pass@*, minority LLM rate, degenerate rate vs `cleaned_correct`) |
| Optional §A.7 | Existing cache + hand-check doc in archive | `analysis_a/llm_clusters_handcheck.md` (copy/update if audit requires fresh 10-prompt read) |
| Optional cross-tier | 50-prompt subsample | `analysis_a/llm_judge_cross_tier.md` |

**Cannot re-run judge** unless human approves API spend; default is metric-only refresh.

---

### Phase 2B — Cheap-substrate comparison (Q I)

**Question:** Does any substrate cheaper than an LM judge approximate the LLM clustering well enough?

**Reference:** `analysis_a/llm_clusters_summary.parquet` (`llm_cluster_id`).

**Inputs**

- `data/cleaned_answers.parquet` — answer-hash + `cleaned_correct`
- `data/predictions_reparsed.jsonl` — `completion` only (for embedding / features)
- `analysis_a/llm_clusters_summary.parquet`
- Optional: `analysis_b/` Qwen embeddings parquet (1024-dim) instead of re-embedding with MiniLM

**Substrates (per prompt, 8 rollouts)**

1. **`answer_hash_cleaned`** — `cleaned_cluster_id`. Replaces archived `answer_strict` + `answer_loose` (single human-verified answer bucket).
2. **`completion_embedding`** — embed full `completion`; per-prompt agglomerative clustering, cosine distance. Archived spec: `all-MiniLM-L6-v2` with threshold sweep {0.2, 0.3, 0.4, 0.5}; **or** cluster precomputed Qwen3-0.6B vectors in `analysis_b/` with documented threshold(s).
3. **`completion_features`** — rule-based tags per rollout (e.g. has_boxed, has_sympy_code, has_repetition, has_code_fence, keyword flags, `cleaned_state`, numeric/latex flags from `cleaned_answer`). Cluster ID = tag tuple within prompt.

**Metrics (per substrate vs LLM, aggregate over 500 prompts)**

- Adjusted Rand Index (ARI): mean + 95% CI
- V-measure (homogeneity + completeness)
- Cluster count agreement: mean |n_substrate − n_llm|
- Minority-correct concordance: confusion matrix vs LLM minority-correct flag (using `cleaned_correct` + respective cluster columns)

**Mandatory qual:** 5 prompts per substrate with largest ARI gap vs LLM — hand-read vignettes.

**Outputs**

| Path | Content |
|------|---------|
| `analysis_b/substrate_comparison.md` | Substrates × metrics table + CIs |
| `analysis_b/substrate_comparison.png` | Bar: `minority_correct_prompt_rate` per substrate; LLM reference line |
| `analysis_b/substrate_disagreement_vignettes.md` | ~20 prompts (5 × substrate) |

**Script (new or port):** `analysis_b/analysis_b_substrate_cleaned.py` — do not use archived `*_v2.py` without stripping v2 paths.

---

### Phase 2C — Offline objective simulation

**Question:** On real Run 0 data, are GRPO, `inverse_freq`, `f_poly`, and `worst_subset` distinguishable in assigned advantages?

**Inputs**

- `cleaned_correct` as reward `r_i`
- Cluster column: **`cleaned_cluster_id`** (pass 1) and **`llm_cluster_id`** (pass 2)

**Formulas (per rollout i within prompt; from archived §C.2)**

1. **GRPO:** `A_i = r_i − mean(r_prompt)`
2. **inverse_freq:** `A_i = (1 / cluster_size_i) × (r_i − mean(r_prompt))` — align with `pilot/train/objectives.py`
3. **f_poly (set-level):** subsets G of size n=4 from 8 rollouts (70 per prompt); `f_poly(G) = mean_r(G) × d(G)`, `d(G) = (# distinct clusters in G) / n`; `A_i^set = mean_{G∋i} f_poly(G) − global_mean`
4. **worst_subset:** same subsets; `f_worst(G) = min_r(G)`; `A_i^worst = mean_{G∋i} f_worst(G) − global_mean`

**Outputs**

| Path | Content |
|------|---------|
| `analysis_c/objective_simulation.md` | Narrative + singleton-wrong mass under `inverse_freq` |
| `analysis_c/objective_corr_pearson.csv` | 4×4 Pearson |
| `analysis_c/objective_corr_spearman.csv` | 4×4 Spearman |
| `analysis_c/objective_scatter_grid.png` | Pairwise advantage scatters (color: `cleaned_correct`, cluster size) |
| `analysis_c/objective_disagreement_by_bucket.md` | Opposite-sign pairs by (`cleaned_correct`, cluster_size) |

**Two passes required:** answer-hash vs LLM clusters. Report how substrate choice shifts GRPO vs `inverse_freq` correlation and rare-wrong mass.

**Script (port):** `analysis_c/analysis_c_objective_sim_cleaned.py` from `archive/.../analysis_c_objective_sim.py`.

---

### Phase 2D — Frozen-eval base-model baseline

**Question:** Proxy base model floor on declared metrics under human-verified extraction.

**Inputs:** `cleaned_answers.parquet`, `analysis_a/llm_clusters_summary.parquet`, best embedding threshold from Phase 2B (if Cover@τ uses embeddings).

**Metrics (bootstrap 95% CI, prompt-level, 1000 resamples)**

| Metric | Definition |
|--------|------------|
| Pass@1 | Mean `cleaned_correct` over rollouts |
| Pass@8 | Unbiased Pass@k estimator (k=8; note `preflight_lock.json` may say k=16 — actual rollouts = 8) |
| Cover@τ (τ=0.15) | Among prompts with ≥1 correct rollout, fraction where largest **correct** cluster mass ≥ τ — compute under **`cleaned_cluster_id`**, **embedding clusters (B)**, **LLM clusters** |
| worst_subset_accuracy | Mean per-prompt Pass@1 on worst 25% of prompts (by per-prompt Pass@1) |

**No v1/v2 parser side-by-side** — single cleaned-label column only.

**Outputs**

| Path | Content |
|------|---------|
| `analysis_d/baseline_metrics.md` | Table + CIs |
| `analysis_d/baseline_metrics.json` | Machine-readable metrics |

**Script (port):** `analysis_d/baseline_metrics_cleaned.py` from archive `baseline_metrics.py`.

---

## Can claim / cannot claim

### Can claim (after Phase 1)

- Run 0 proxy base model Pass@1 / Pass@8 under **human-verified** answer extraction.
- `minority_correct_prompt_rate_llm` on Run 0 with bootstrap CI (eligible n=172).
- Answer-hash clustering yields **0%** minority-correct on this run — diversity is in reasoning (LLM clusters), not final-string buckets.
- Qualitative examples of what a minority objective would upweight (LLM substrate).

### Can claim (after Phase 2, when complete)

- Phase 2A: LLM-cluster minority rate and Pass@* **under cleaned correctness** (recompute on 172 eligible; expect ~14.5% if cluster IDs unchanged).
- Phase 2B: Best cheap substrate vs LLM reference (ARI, minority concordance) on **cleaned-labeled** rollouts — tentative Q I evidence only on base-model offline data.
- Phase 2C: Objective advantage correlations and disagreement rates on empirical Run 0 distribution (two cluster substrates).
- Phase 2D: Cleaned-label baseline table (Pass@*, Cover@τ, worst_subset) any training arm must clear on this prompt set.

### Cannot claim

- LLM clusters are philosophical ground truth (reference substrate only).
- High substrate–LLM ARI implies a cheap judge works **in RL** (correlational on base rollouts only).
- Training gains, AIME/HMMT generalization, or parser-fix narrative from v2 tables.
- Numbers from `archive/.../overnight_workflow_log.md` or pre-reset handoff Pass@ without cleaned-label banner.
- That offline Phase 2 results select a final training objective or replace Poly-EPO's in-loop judge.

---

## Execution order

| Order | Work | Depends on | Est. effort |
|-------|------|------------|-------------|
| 1 | **Phase 1** — `analysis_minority/` metrics + distributions + qual | Cleaned parquet + LLM summary (done) | ~2–4 h |
| 2 | **Phase 2A** — Analysis A metric refresh / summary markdown | Phase 1 minority definitions aligned | ~30 min |
| 3 | **Phase 2B** — Substrate comparison + vignettes | 2A reference stable; embeddings optional | ~1.5–2 h |
| 4 | **Phase 2C** — Objective simulation (2 cluster passes) | `cleaned_*` + LLM clusters | ~1.5 h |
| 5 | **Phase 2D** — Baseline metrics (Cover@τ needs B threshold) | 2B if embedding Cover@τ used | ~30 min |

**If time-constrained:** do not skip Phase 1; defer Phase 2D first (easiest to reproduce). Do not skip Phase 2B vignettes if claiming Q I. Do not re-run Analysis A judge without cause.

**Historical overnight order (archived):** 0b → A → A QC → B → C → D — superseded by cleaned-label redo above.

---

## Output paths (consolidated)

| Phase | Path | Status |
|-------|------|--------|
| 1 | `analysis_minority/minority_metrics.md` | Not started |
| 1 | `analysis_minority/minority_distributions.png` | Not started |
| 1 | `analysis_minority/minority_readout.py` (suggested) | Not started |
| 2A | `analysis_a/analysis_a_summary.md` | Exists (archive copy); refresh with cleaned joins |
| 2A | `analysis_a/llm_clusters_handcheck.md` | Optional |
| 2A | `analysis_a/llm_judge_cross_tier.md` | Optional |
| 2B | `analysis_b/substrate_comparison.md` | Redo |
| 2B | `analysis_b/substrate_comparison.png` | Redo |
| 2B | `analysis_b/substrate_disagreement_vignettes.md` | Redo |
| 2C | `analysis_c/objective_simulation.md` | Redo |
| 2C | `analysis_c/objective_corr_*.csv`, scatter figure | Redo |
| 2D | `analysis_d/baseline_metrics.md`, `.json` | Redo |
| — | `data/cleaned_answers.parquet` | Done |
| — | `analysis_a/llm_clusters_summary.parquet` | Done |
| — | `pilot/artifacts/.../llm_clusters/*.json` | Done |

---

## Related docs

| Doc | Role |
|-----|------|
| [`README.md`](README.md) | Folder layout, schemas, script usage |
| [`archive/2026-05-21_pre_human_label_audit/run0_offline_analyses_20260521.md`](archive/2026-05-21_pre_human_label_audit/run0_offline_analyses_20260521.md) | Full A/B/C/D design (v2-era; redo formulas) |
| [`pilot/docs/archive/RUN0_HANDOFF_FOR_REVIEW.md`](../../pilot/docs/archive/RUN0_HANDOFF_FOR_REVIEW.md) | Pre-reset Run 0 narrative (historical) |
| [`../narrative/timeline.md`](../narrative/timeline.md) | Project chronology |

---

## Explicitly out of scope this round

- New training or smoke runs on policy checkpoints.
- In-loop training judge (Poly-EPO RL); offline labeling only for Analysis A.
- Final objective selection from Phase 2C alone.
- Claim that any cheap substrate replaces Poly-EPO's LM judge **in the training loop** — only "matches LLM clustering on base-model rollouts at ARI=X" after Phase 2B.
- Re-litigating parser v1 vs v2 except in archive audit trails.
