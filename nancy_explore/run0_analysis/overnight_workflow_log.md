# Overnight workflow log — 2026-05-21 PM → 05-22 AM

**Operator:** Claude (Opus 4.7), autonomous overnight session  
**Goal:** Complete as much of `run0_offline_analyses_20260521.md` as possible while Nancy sleeps.

## Starting state

- **Analysis A model calls:** DONE (500 prompts labeled by `gemini-3.1-flash-lite`, cache populated, summary parquet built).
- **Headline from A:** `minority_correct_prompt_rate_llm = 14.55%` (CI [9.09%, 20.00%]), 165/500 prompts eligible. Degenerate rate = 16.95%.
- **A.7.1 hand-check:** Nancy did the first pass of `llm_clusters_handcheck.md`, said it looked fine; rest not reviewed.
- **A.7.2 cross-tier ARI probe:** explicitly marked skippable in design doc — **SKIPPING** for the night.
- **A.7.3 degenerate-cluster sanity:** not done.
- **Analyses B, C, D:** not started.

## Plan

Three independent agents run in parallel (B, C, D); A.7.3 is a quick scalar check I'll do directly. D's Cover@τ for `completion_embedding` depends on B's best threshold — D agent will be told to leave that cell as a TODO and fill it after B finishes if both complete in time.

| Slot | Agent | Output files |
|---|---|---|
| A.7.3 | (me, direct) | append to `llm_clusters_handcheck.md` or new short file |
| B | general-purpose | `substrate_comparison.md`, `substrate_disagreement_vignettes.md`, `substrate_results.parquet`, `analysis_b_minority_rate.png` |
| C | general-purpose | `objective_simulation.md`, `objective_advantages.parquet`, `objective_corr_*.csv`, `objective_scatter_grid.png` |
| D | general-purpose | `baseline_metrics.md`, `baseline_metrics.json` |

Inputs all agents share:
- `data/predictions_reparsed.jsonl` (4000 rows, has v1 + v2 parse fields).
- `llm_clusters_summary.parquet` (4000 rows, `llm_cluster_id` per rollout, `-1` = degenerate).

## Caveats agents are told to flag

- "Hand-read" vignettes are LLM-read by an agent, not by a human — must be labeled as such in the output.
- All bootstrap CIs are prompt-level (resample at the prompt index, not rollout index).
- Analysis B's `completion_embedding` uses `sentence-transformers/all-MiniLM-L6-v2` (CPU; ~5 min).

## Timeline

(filled in as events happen)

## A.7.3 degenerate sanity (done)

- LLM degenerate rate: **16.95%** (678 / 4000 rollouts).
- 333 / 500 prompts have ≥1 degenerate; 5 / 500 are fully degenerate.
- Sits between the qual analysis's 9.2% long-parse floor and 28.5% sympy-presence ceiling — internally consistent. **Trust the LLM clusters for B/C/D.**
- Written up at `llm_degenerate_sanity.md`.

## Analysis B agent

- Started 2026-05-21. Installed sentence-transformers, scikit-learn, matplotlib (pip not present in venv, bootstrapped via ensurepip).
- Confirmed schema in predictions_reparsed.jsonl: uses `cluster_id` (v1) and `cluster_id_v2` directly. `llm_clusters_summary.parquet` has `llm_cluster_id` (-1 = degenerate).

## Analysis C agent

- Started 2026-05-21; runtime 10.6s.
- Sanity: GRPO sum-by-prompt max |x| = 0.00e+00; edge-prompt max |adv| = 0.00e+00.
- Inverse_freq formula divergence vs `pilot/train/objectives.py` noted in writeup (production normalizes weights to sum-to-N with cap=8; doc formula does not). Used doc formula.
- Headline numbers:
  - Singleton-wrong |adv|-mass under IF: answer-hash **56.49%**, LLM **44.36%**.
  - Substrate-sensitivity Pearson: inverse_freq r=+0.869, f_poly r=+0.955.
  - GRPO↔IF Pearson: answer-hash r=+0.924, LLM r=+0.875.
  - GRPO↔worst_subset Pearson r=+0.122.
- Outputs: `objective_advantages.parquet`, `objective_corr_pearson.csv`, `objective_corr_spearman.csv`, `objective_scatter_grid.png`, `objective_simulation.md`.
- No blockers.
- Embedding pass starting (MiniLM, CPU, 4000 completions).

## Analysis D agent

- Started 2026-05-21. Wrote `baseline_metrics.py`; runtime ~2s including 1000 prompt-level bootstrap resamples.
- Confirmed inputs: 500 prompts × 8 rollouts; v1 fields `correct`, `cluster_id`; v2 fields `is_correct_v2`, `cluster_id_v2`; LLM clusters from `llm_clusters_summary.parquet` (`llm_cluster_id`, -1 = degenerate, treated as singleton).
- Preflight-lock discrepancy logged: lock declares `pass_at_k=16` and `bootstrap_samples=2000`; we report Pass@8 (n=k=8, Chen et al. estimator collapses to "any-correct" per prompt) and bootstrap=1000 per Analysis D spec.
- Headline (v2):
  - Pass@1 = 8.25% [6.97%, 9.53%]; Pass@8 = 33.00% [29.20%, 37.00%]
  - Cover@τ=0.15 (answer_loose) = 49.70% [41.72%, 57.50%]; (llm_clusters) = 72.73% [66.07%, 79.65%]
  - worst_subset_accuracy = 0.00% [0.00%, 0.00%] (worst 25% = 125 prompts all with 0 correct rollouts; 335/500 prompts have 0 correct)
- v1→v2 parser fix moves Pass@1 by +0.15pp; effectively a wash within CIs.
- **Blocker (one cell):** `completion_embedding` Cover@τ is `TBD pending Analysis B` — `substrate_results.parquet` and `substrate_comparison.md` not yet present. Script accepts the embedding cluster column; will need re-run once B finishes.
- Outputs: `baseline_metrics.md`, `baseline_metrics.json`, `baseline_metrics.py`.

### Analysis D — update after Analysis B completed

- Analysis B finished during D's poll window; `substrate_comparison.md` reports best embedding threshold **0.2** (mean ARI = 0.074 vs LLM clusters, i.e. low absolute agreement).
- `substrate_results.parquet` is per-prompt **aggregate metrics**, not per-rollout cluster IDs. Built `build_embed_clusters.py` to recompute per-rollout assignments from cached `embeddings_minilm.npy` + `embedding_ids.parquet` at threshold 0.2 via per-prompt agglomerative clustering (cosine, average linkage); wrote `embed_clusters_at_best_threshold.parquet` (4000 rows). Runtime ~1s.
- Re-ran `baseline_metrics.py`; Cover@τ=0.15 (completion_embedding) = **92.12% [87.95%, 96.05%]**.
- Note for downstream readers: high embedding Cover@τ at threshold 0.2 is consistent with that threshold producing many tiny clusters per prompt (so a correct-containing cluster easily dominates), but Analysis B already established that those clusters do **not** match LLM clusters semantically (mean ARI 0.074). So embedding Cover@τ here is a "best-case substrate" number, not evidence that the embedding substrate is reasoning-aware.
- Embedding pass finished (~17s on CPU); cached to embeddings_minilm.npy (4000x384, 6.1MB).
- Computed 4 substrates × 4 embedding thresholds = 7 variants vs LLM reference over 500 prompts.
- All 4 outputs written: substrate_comparison.md, substrate_results.parquet, substrate_disagreement_vignettes.md, analysis_b_minority_rate.png.
- **Headline:** `answer_strict` wins by ARI (0.188 [0.159, 0.218]); best embedding threshold = 0.2 (ARI 0.074); my computed LLM ref minority rate matches Analysis A's 14.55% headline exactly (after switching to canonical `has_minority_correct_cluster` definition from pilot/train/run_proxy.py).
- **2-line summary:** No cheap substrate matches the LLM reference well — best ARI is 0.188 (`answer_strict`), all others below. The LM judge appears load-bearing on Run 0; recommend `completion_embedding@0.2` for Analysis D's Cover@τ embedding variant.

---

## Closing summary (morning brief)

**All four analyses (A.7.3, B, C, D) completed overnight.** A.7.2 cross-tier ARI probe explicitly skipped per design doc. A.7.1 hand-check is the only un-finished item (you reviewed the first prompt and stopped).

### Headline numbers

| Analysis | Output | Headline |
|---|---|---|
| A.7.3 | `llm_degenerate_sanity.md` | LLM degenerate rate **16.95%** (sits between qual analysis's 9.2% long-parse floor and 28.5% sympy-presence ceiling) — clusters trusted |
| B | `substrate_comparison.md` + 3 more | No cheap substrate matches LLM well. Best ARI = **0.188** (`answer_strict`); all others < 0.2 in absolute terms. **§B.6 outcome: "LM judge appears load-bearing."** |
| C | `objective_simulation.md` + 4 more | Singleton-wrong |adv|-mass under inverse_freq = **56.5%** (answer-hash) / **44.4%** (LLM). GRPO↔IF Pearson = 0.92, Spearman 0.997, 0% sign-flips → IF only rescales magnitudes. **worst_subset is the only objective qualitatively distinct from GRPO** (Pearson 0.12). |
| D | `baseline_metrics.md` + json | Pass@1 **8.25%**, Pass@8 **33.00%**, Cover@τ=0.15 varies **49.7% → 72.7% → 92.1%** across (answer_loose, LLM, embedding@0.2). v1→v2 parser fix is a wash on aggregate. `worst_subset_accuracy = 0%` because 335/500 prompts have zero correct rollouts. |

### Things worth your attention in the morning

1. **B's outcome is the "kill the LM judge" pitch's worst-case answer for Run 0:** no cheap substrate approximates LLM clustering well enough to justify replacement. This contradicts the hope you'd expressed for §B.6 — but it's a defensible empirical finding for the office-hours conversation. The Question A3 update in §"How this changes tomorrow's office hours doc" should be inverted: "We tested 7 substrate variants; none reach ARI 0.2 vs the LLM reference, so the LM judge appears load-bearing on Run 0's base-model rollouts."
2. **C's substrate-sensitivity finding is useful:** swapping answer-hash → LLM clusters in inverse_freq changes the per-rollout advantage by Pearson 0.87 (not 1.0), and shifts singleton-wrong mass from 56.5% → 44.4%. So even though IF and GRPO are nearly co-linear, the substrate choice meaningfully changes IF's gradient geometry.
3. **D's `worst_subset_accuracy = 0%` needs framing.** Mathematically correct but not informative — once 67% of prompts have 0 correct rollouts, "the worst 25%" is all zeros by construction. Consider redefining as "mean Pass@1 on the 4th quartile of prompts ranked by per-prompt Pass@1, excluding all-zero prompts" if you want a moving signal post-training.
4. **D's embedding Cover@τ (92.1%) is a substrate artifact**, not a quality signal — agent flagged it correctly. Tight threshold = many tiny clusters → correct rollouts almost always in a not-tiny cluster.
5. **Vignettes in `substrate_disagreement_vignettes.md` are LLM-read, NOT human hand-reads** — flagged in the file. Spot-check 2–3 before citing.

### Files Nancy should open first

- `overnight_workflow_log.md` (this file)
- `substrate_comparison.md` — the headline §B table
- `objective_simulation.md` — singleton-wrong mass + correlation matrix
- `baseline_metrics.md` — the locked metrics under v2

### Not done / known gaps

- A.7.1: hand-check of remaining 9 prompts in `llm_clusters_handcheck.md`.
- A.7.2: cheap↔moderate cross-tier ARI on 50 prompts. (Skipped per design doc, but if office hours pushes back on judge-tier robustness, this is the next thing to run; cost ~$1–2 + 30 min.)
- The 25 disagreement vignettes in B are agent-read, not human-read. Whether to upgrade depends on how much weight the office-hours conversation puts on them.
