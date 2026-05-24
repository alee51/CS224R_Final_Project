# Run 0 offline analyses — design doc (archived)

> ⚠ **ARCHIVED — v2-era design.** **Current plan:** [`../../run0_analysis_plan.md`](../../run0_analysis_plan.md).
>
> **HISTORICAL — 2026-05-21 reset.** This doc was the spec for the original A/B/C/D analyses, which depended on the v1/v2 parsers. Subsequent v2-vs-human validation showed only 78% presence agreement (see `nancy_explore/narrative/timeline.md`, entry 2026-05-21), so we promoted the §0a human labels to ground truth. The canonical answer-extraction artifact is now `data/cleaned_answers.parquet` (schema in `README.md`). All references in this document to `parsed_answer_v2`, `is_correct_v2`, `cluster_id_v2`, `canonical_v2`, `extract_path_v2`, and `reparse_diff.md` are **stale**: substitute `cleaned_answer`, `cleaned_correct`, `cleaned_cluster_id`, `cleaned_state` from `cleaned_answers.parquet`. References to the `prereq_0b_reparse/` folder now point to `archive/2026-05-21_pre_human_label_audit/prereq_0b_reparse/`. The doc is preserved for context on why Analyses A–D were structured the way they were; the headline numbers it cites (v1 Pass@1 8.10%, v2 8.25%, `minority_correct_prompt_rate_v2` 0%, eligible **165**, LLM minority **14.55%**) are superseded — under human labels: Pass@1 **9.03%**, Pass@8 **34.40%**, eligible **172**, LLM minority **14.53%** (see `README.md` in this archive folder).

**Date:** 2026-05-21 (updated: 0b done, Analysis A script + Poly-EPO judge prompt)  
**Workspace:** `nancy_explore/run0_analysis/`  
**Artifact under analysis:** `pilot/artifacts/run0_proxy/20260519T190202Z/`  
**Constraint:** No new training, no GPU. Everything below runs on a laptop + an external LLM API (any provider with a key).  
**Audience:** human executor (or agent under human supervision).

This doc specifies four analyses on Run 0's 4000 rollouts. Each section includes inputs, method, outputs, **can claim**, and **cannot claim**.


| Path                              | Role                                                               |
| --------------------------------- | ------------------------------------------------------------------ |
| `data/*.jsonl`                    | Symlinks to Run 0 proxy jsonl                                      |
| `data/predictions_reparsed.jsonl` | 0b output — v1 + v2 parse fields                                   |
| `config/llm_judge_models.yaml`    | Analysis A tier → API model ID                                     |
| `config/analysis_a_prompt.md`     | Poly-EPO §A.1 judge prompt (verbatim)                              |
| `analysis_a_llm_clusters.py`      | Analysis A runner (`--pilot`, `--force`, `--tier cheap`)           |
| `labels/rollout_labels.jsonl`     | Human-style tail labels (4000 rows) — canonical for substrate work |
| `labeling/`                       | Archived blind A/B pipeline — not needed downstream                |


A note on terminology: the term "chain-of-thought analysis" was previously used in this project for analyses that never called an LLM. In this doc, **only Analysis A** ("LLM reasoning-cluster ground truth") qualifies as CoT analysis. The other substrate-based clusterings are explicitly named "completion-text embedding" or "structural feature clustering" and must be described that way in any writeup.

### LLM judge tiers (provider-agnostic)

Analysis A refers to **tiers**, not vendor names. The executor picks a **provider** and maps tiers to that provider's API model IDs. Config: `config/llm_judge_models.yaml`.

**Default provider for this round:** Google AI Studio. Poly-EPO used `gemini-2.0-flash`; on the current **free tier** that model hits quota limits, so this run uses `**gemini-3.1-flash-lite`** for the `cheap` tier (AI Studio “3.1 flash” class; API id is not `gemini-3.1-flash`). Override via `.env` `GEMINI_MODEL=`. Record `provider`, `tier`, and resolved `model` in each cached `llm_clusters/{prompt_id}.json`.


| Tier          | Role in this doc                    | Google (this run)       | Anthropic           | OpenAI         |
| ------------- | ----------------------------------- | ----------------------- | ------------------- | -------------- |
| **cheap**     | Primary judge — all 500 prompts     | `gemini-3.1-flash-lite` | `claude-haiku-4-5`  | `gpt-4o-mini`  |
| **moderate**  | Optional 50-prompt robustness probe | `gemini-2.5-flash`      | `claude-sonnet-4-6` | `gpt-4.1-mini` |
| **expensive** | Not used this round                 | `gemini-2.5-pro`        | `claude-opus-4-6`   | `gpt-4.1`      |


**Tier semantics:** **cheap** = same cost/role class as Poly-EPO's in-loop judge (flash / mini; pennies per call). Do not use **expensive** for the 500-prompt primary run — stronger than the paper's setup and weakens the "faithful cheap judge" comparison. **moderate** is only for the optional ARI sanity check (same or different provider).

---

## 0. Prerequisites (plumbing, not analyses)

### 0a. Human rollout labels — **DONE**

Tail-only labels for all **4000** rollouts (priority subset + 8 non-priority prompts), produced by blind dual-agent labeling + CSV dispute resolution.


| Output           | Path                                                 |
| ---------------- | ---------------------------------------------------- |
| Canonical labels | `labels/rollout_labels.jsonl`                        |
| Pipeline archive | `labeling/` (`chunks/`, `blind/`, `spawn/`, scripts) |


Each row: `rollout_key`, `gold`, `result_A`, `result_B`, `result` (agreed label or human override). Values: extracted answer string, or `runon`  `no_answer`. **Not ground truth** — tail-only, best-effort; use for substrates that need “did the model state an answer?” vs run-on.

Optional audit: `python audit_1024_token_labels.py` (Qwen 1024-token cap vs labels).

### 0b. Parser-fix re-score — **DONE**

One-shot re-parse with PILOT_REDESIGN C2/C3 logic via `pilot/train/answer_clean.py` (offline; production `answer_parse.py` unchanged). Not the same as 0a.


| Output             | Path                                                                    |
| ------------------ | ----------------------------------------------------------------------- |
| Re-parsed rollouts | `data/predictions_reparsed.jsonl` (4000 rows; v1 fields preserved + v2) |
| Diff report        | `reparse_diff.md`                                                       |
| Script             | `reparse_rescore.py`                                                    |


**Headline (see `reparse_diff.md`):** v1 Pass@1 **8.10%** → v2 **8.25%** (+6 rollouts); 514 parsed changed (12.9%); `minority_correct_prompt_rate_v2` **0.0%** [0%, 0%] bootstrap CI — proceed to Analysis A for substrate-controlled minority metric.

This is data hygiene. Analyses A–D are the science.

---

## Analysis A — LLM reasoning-cluster ground truth

**The most important analysis in this doc.** Every other substrate analysis is judged against it.

### A.1 Motivation

Ifdita's Poly-EPO uses a `**cheap`-tier** LM judge (flash-class; per-call pennies) to cluster *whole responses* (N rollouts per prompt) by reasoning strategy — macro- and micro-strategy inferred from the math in each completion — in one judge call per prompt. Run 0's substrate is exact-match on final answer strings. These are not the same operation. The whole question "is the LM judge load-bearing?" presupposes that we can compare an LM judge's clustering to a cheap substrate's clustering — but we don't currently have an LM judge clustering of Run 0 to compare against. This analysis produces that reference clustering.

The output of this analysis is the **reference** that Analysis B's cheap substrates get scored against.

**Clarification on what "expensive LM judge" means in Poly-EPO.** The paper's judge is `**cheap`-tier** per call. When Poly-EPO frames the judge as expensive or limiting, that is about **training-loop overhead** — the judge is invoked on every rollout set on every gradient step, gating each policy update on an external API call, accumulating cost and latency across thousands of training steps and introducing an availability dependency. It is **not** about per-call labeling cost for a one-shot offline pass like this analysis. The "kill the LM judge" framing in our project should mean "replace the in-training-loop LM judge with a cheap local substrate" — not "we cannot afford to call an LM judge ever." This analysis (one-shot offline labeling of 500 prompts) costs on the order of a few dollars at `**cheap`** tier and is fully consistent with the "kill the in-loop judge" pitch.

### A.2 What we are doing (and not doing)

- **Doing:** for each of the 500 prompts, send the problem + the 8 full completions to the LLM judge (`cheap` tier) with the Poly-EPO §A.1 prompt; one judge call returns one `cluster_id` per rollout (JSON keys `"1"`…`"8"`). Per-prompt clustering, not cross-prompt.
- **Not doing:** replicating Poly-EPO's in-loop judge on every training step (the paper's **850 steps** are optimization steps, not reasoning-step segmentation). Cross-prompt clustering. Training-time judge behavior on policy checkpoints — this pass labels base-model Run 0 rollouts offline only.

### A.3 Inputs

- `data/prompt_inputs.jsonl` — problem text per `prompt_id`.
- `data/predictions_reparsed.jsonl` — 8 rollouts per prompt; **full `completion` text** sent to the judge (no per-rollout truncation; ~4% of rollouts exceed 3.2k chars, max ~5k).
- **Not sent to the judge:** gold answer, `parsed_answer_v2`, or `is_correct_v2` (those stay in our tables for downstream metrics only — the paper judge clusters from reasoning text alone).

### A.4 Model and protocol

- **Primary judge:** `**cheap`** tier (see tier table above). One provider for all 500 prompts; resolve model ID from `config/llm_judge_models.yaml` (override via `.env` `GEMINI_MODEL=`). Same role class as Poly-EPO's paper judge — do not promote to `**moderate`** or `**expensive`** for the main run.
- **Optional cross-tier / cross-provider sanity probe:** `**moderate`** tier on a 50-prompt random subsample (same provider, or a different provider — note which in `llm_judge_cross_tier.md`). Not the centerpiece. Compute ARI between `**cheap`** and `**moderate`** clusterings per prompt; if mean ARI ≥ 0.7, judge clusterings are robust to tier; if < 0.5, report as a caveat (numbers conditional on judge tier/provider). Skip if time-constrained.
- **Prompt structure** (faithful to Poly-EPO §A.1; full text in `config/analysis_a_prompt.md`, extracted in `pilot/docs/analysis/0519_poly_epo_methodology.md`):
  - **System:** paper instruction block — macro/micro strategy clustering rules; degenerate → `**cluster_id: 100`** (gibberish, off-topic, non-mathematical/code-only).
  - **User:** `**Context:`** + problem; `**Responses:**` numbered 1–8, each the full rollout completion.
  - **Output (paper JSON):** keys `"1"` … `"8"`, each value `{"chain_of_thought": "Macro: … Micro: …", "cluster_id": int}`. Script maps `100` → `-1` for `minority_correct_prompt_rate_llm` and parquet.
- **Runner:** `python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --tier cheap --workers 2 --purge-stale` (drops any cache without `prompt_format: poly_epo_paper_a1` + `parse_ok`; global ~4.1s throttle for 15 RPM free tier; `--force` to redo valid caches). Env: `nancy_explore/run0_analysis/.env` with `GOOGLE_API_KEY` and `GOOGLE_API_KEY_2` (script auto-switches on daily quota).
- **Caching:** `pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters/{prompt_id}.json` — idempotent skip on `parse_ok`; stores `provider`, `tier`, `model`, `prompt_format: poly_epo_paper_a1`.

### A.5 Judge prompt source (not a separate custom instruction)

The clustering criteria are **only** the Poly-EPO §A.1 block above (macro-strategy + micro-strategy; not final answer; arithmetic errors do not split clusters). Do not use the older project-specific “cluster by reasoning approach” paraphrase — it diverged from the paper.

**Judge scope:** §A.1 asks the judge to infer macro/micro strategy from *key intermediate steps within* each response; the clustering unit is still one rollout per JSON key (same as the paper's one-call-per-prompt setup). Real deltas vs Poly-EPO: judge model (Gemini `cheap` tier vs paper's Qwen-Instruct), offline one-shot vs in-loop on every gradient step, base-model Run 0 outputs not training checkpoints.

### A.6 Cost and time estimate

- Per call: variable input (problem + 8 full completions + instructions; median rollout ~1.9k chars, max ~5k) + ~800 output tokens.
- `**cheap`** tier, 500 prompts: order of **~$5** total (provider-dependent; check current list pricing).
- Optional `**moderate`** cross-check, 50 prompts: order of **~$1–2** additional.
- Wall time: ~30–60 minutes with modest concurrency (10 in-flight calls).
- Use provider prompt/context caching on the system + instruction block where available — cuts repeated input cost.

### A.7 Quality checks (mandatory before using these clusters downstream)

1. **Hand-read 10 prompts.** Pick: 3 prompts with all-correct rollouts, 3 with mixed correctness, 4 with no-correct rollouts. For each, read the 8 completions and the LLM's cluster assignments. Does the clustering match what a human would do? Document disagreements in `llm_clusters_handcheck.md`.
2. **Optional: cross-tier sanity probe (50-prompt subsample).** Run `**moderate`** on 50 prompts already labeled at `**cheap`**. Compute ARI between the two tier clusterings per prompt; report mean and distribution. Skip if time-constrained — nice-to-have, not a blocker.
3. **Degenerate-cluster rate sanity.** What fraction of rollouts land in `cluster_id: 100` (paper) / `-1` (downstream)? Cross-check against the qual analysis's tagged garbage rate (~28% sympy derailment + ~25% repetition + ~9% long parse). If LLM's degenerate rate is wildly different, investigate before trusting clusters.

### A.8 Outputs

- `llm_clusters/{prompt_id}.json` × 500 (raw model output, cached).
- `llm_clusters_summary.parquet` — per-rollout LLM cluster assignment.
- `llm_clusters_handcheck.md` — 10-prompt hand audit.
- `llm_judge_cross_tier.md` — `**cheap`** vs `**moderate`** ARI on 50-prompt subsample (if cross-check run); record provider(s) used.
- **The headline number:** `minority_correct_prompt_rate_llm` — under LLM clusters, fraction of prompts with ≥1 correct rollout that have correct rollouts spanning ≥2 LLM-clusters with at least one not the largest. Bootstrap 95% CI.

### A.9 Can claim

- "Under LLM-based reasoning clustering on Run 0's base-model rollouts, minority-correct prompts exist at rate X% (CI […, …]). This is the substrate-controlled value for the gate metric."
- If `**cheap`**↔`**moderate`** ARI is high (cross-check run): "LM-judge clusterings are robust to judge tier on this data — `**cheap**` is not a quality compromise vs `**moderate**` here."
- If `**cheap**`↔`**moderate**` ARI is low (cross-check run): "LM-judge clusterings are tier-dependent on this data; reported numbers are conditional on judge tier/provider; Poly-EPO's reproducibility depends on judge availability."

### A.10 Cannot claim

- That LLM clusters are "correct." They are a reference, not ground truth in the philosophical sense. A human-labeled subsample (the 10 hand-checked prompts) is the only thing closer to ground truth, and it's tiny.
- That this replicates Poly-EPO's in-loop judge setup: different judge model, offline vs training-time invocation, base-model rollouts not policy checkpoints.
- Anything about training-time behavior. This is offline labeling of base-model outputs.

---

## Analysis B — Cheap-substrate comparison against the LLM reference

**What it answers:** Does any substrate cheaper than an LM judge approximate the LLM clustering well enough that Question I has a positive answer? This is the empirical core of the "kill the LM judge" pitch.

### B.1 Inputs

- `predictions_reparsed.jsonl` (for substrate computation).
- `llm_clusters_summary.parquet` from Analysis A (the reference).

### B.2 Substrates to evaluate

For each of the substrates below, compute per-prompt cluster assignments over the 8 rollouts. All cluster assignments are local to a single prompt.

1. `**answer_strict`** — hash of original `canonical` field. The current code's substrate. Baseline.
2. `**answer_loose`** — hash of `canonical_v2` (parser-fixed). Tests whether the parser alone explains the 0% finding.
3. `**completion_embedding`** — embed each completion's full text with `sentence-transformers/all-MiniLM-L6-v2` (~80MB, runs CPU in ~5 min for 4000 completions). Per-prompt cluster the 8 embeddings using agglomerative clustering with cosine distance. **Hyperparameter sweep, not a single value:** distance threshold ∈ {0.2, 0.3, 0.4, 0.5}. Report results at each threshold.
4. `**completion_features`** — rule-based feature tagging per rollout (these tags already exist in the qual analysis): `{has_boxed, has_sympy_code, has_repetition, has_code_fence, has_modular_arithmetic_keywords, has_coordinate_method_keywords, parsed_is_numeric, parsed_is_latex_fraction}`. Cluster ID = tuple of binary tags within prompt. Cheapest possible substrate.

### B.3 Metrics computed per substrate

For each prompt, compare the substrate's cluster assignment against the LLM cluster assignment from Analysis A. Aggregate over the 500 prompts:

- **Adjusted Rand Index (ARI)**: mean and 95% CI over prompts. Measures cluster-agreement corrected for chance.
- **V-measure** (homogeneity + completeness harmonic mean): mean over prompts.
- **Cluster count agreement**: per-prompt, does substrate's `n_clusters` match LLM's `n_clusters`? Report mean absolute difference.
- **Minority-correct concordance**: per-prompt, do substrate and LLM agree on whether the prompt has minority-correct structure? Report confusion matrix.

### B.4 Disagreement diagnostic

For each substrate, identify the 5 prompts with the largest ARI gap vs LLM clustering. Hand-read these. For each, write a 2-3 sentence note: is the substrate missing real reasoning structure, or is the LLM hallucinating structure that isn't there? File these in `substrate_disagreement_vignettes.md`.

This is the qualitative companion to the quantitative ARI numbers and is mandatory — ARI alone can mislead.

### B.5 Outputs

- `substrate_comparison.md` with one table (4 substrates × 4 metrics, with CIs).
- One bar chart: `minority_correct_prompt_rate` per substrate, with the LLM value as a reference line.
- `substrate_disagreement_vignettes.md` (20 hand-read prompts: 5 per substrate).

### B.6 Can claim

- "Among substrates evaluated, X has highest ARI vs the LLM reference (mean ARI = Y, CI […, …]). This is the candidate cheap substrate for replacing the LM judge."
- If ARI is high across multiple substrates: "Cheap substrates substantially recover the LLM clustering on Run 0 — Question I has tentative positive evidence."
- If ARI is uniformly low: "No cheap substrate we tested approximates LLM clustering — the LM judge appears load-bearing, and the original Poly-EPO design choice is justified."

### B.7 Cannot claim

- That a high-ARI cheap substrate would *work as well as* an LM judge inside an RL training loop. ARI on base-model rollouts ≠ downstream policy gradient behavior. This is correlational, not causal.
- That `completion_embedding` is "CoT clustering." It is text embedding clustering. Different operation, different semantics.

---

## Analysis C — Offline simulation of candidate objectives

**What it answers:** On real Run 0 data, are GRPO, `inverse_freq`, `f_poly`, and `worst_subset` actually distinct in the advantages they assign? This replaces the previous toy sims with measurements on 500 actual prompts.

### C.1 Inputs

- `predictions_reparsed.jsonl` — only the `reward_v2 = is_correct_v2` field and `cluster_id_v2` are used.
- Choice of cluster substrate: run this analysis twice, once with `cluster_id_v2` (answer-hash) and once with the LLM clusters from Analysis A. The comparison between the two passes is itself informative — it shows how much the substrate choice affects the objective shape.

### C.2 Objectives and formulas

All advantages are computed per-rollout per-prompt. No model gradients are involved; only the advantage values that a training step would *send* to the model.

1. **GRPO**: `A_i = r_i - mean(r_prompt)`.
2. `**inverse_freq`**: `A_i = (1 / cluster_size_i) × (r_i - mean(r_prompt))`. Matches `pilot/train/objectives.py`.
3. `**f_poly` (set-level Poly-EPO substrate-swap)**:
  - Enumerate all subsets G of size n=4 from each prompt's 8 rollouts: C(8,4) = 70 subsets per prompt.
  - Per subset: `f_poly(G) = mean_r(G) × d(G)`, where `d(G) = (# distinct clusters in G) / n`.
  - Per rollout i: `A_i^set = mean over G ∋ i of f_poly(G) - global_mean_f_poly`.
4. `**worst_subset`**:
  - Same subset enumeration.
  - Per subset: `f_worst(G) = min_r(G)`. (Alternative: 25th-percentile reward — pick `min` for n=4 since the lower quartile is the min.)
  - Per rollout i: `A_i^worst = mean over G ∋ i of f_worst(G) - global_mean_f_worst`.

### C.3 Outputs

- A 4×4 correlation matrix (Pearson + Spearman) of the four advantage vectors over all 4000 rollouts.
- 4×4 scatter plot grid (advantages of each pair, color-coded by `is_correct_v2` and by cluster size).
- **Disagreement table:** for each pair of objectives, count rollouts where the two assign opposite-sign advantages, bucketed by `(is_correct_v2, cluster_size_v2)`.
- **Singleton-wrong mass claim:** under `inverse_freq`, the percentage of total |advantage| mass on rollouts with `(r=0, cluster_size=1)`. This is the quantitative version of the "inverse_freq reweights wrong singletons" critique from `briefs/pilot_strategy_20260520.md`.
- Two passes: once with `answer_loose` clusters, once with LLM clusters. Report both.

### C.4 Can claim

- "On Run 0's empirical reward+cluster distribution, GRPO and `inverse_freq` advantages correlate at r=X (under answer-hash) and r=Y (under LLM clusters). The substrate choice changes the objective comparison by Δ=…"
- "`inverse_freq` directs N% of its advantage mass to rare-wrong rollouts (r=0, cluster_size=1) under answer-hash clustering, vs M% under LLM clustering."
- "`worst_subset` and `f_poly` produce per-rollout advantages that differ from GRPO on Z% of rollouts; these rollouts are concentrated in [bucket]."

### C.5 Cannot claim

- Anything about training trajectories. This is a one-step view on a fixed base-model rollout distribution. A correlation difference doesn't guarantee a trained-model accuracy difference; it is necessary-not-sufficient evidence that the objectives are distinguishable.

---

## Analysis D — Frozen-eval base-model baseline

**What it answers:** What does the proxy base model score on the metrics declared in `preflight_lock.json`, with parser fixes applied? This is the floor that any future training arm must clear.

### D.1 Inputs

- `predictions_reparsed.jsonl`.

### D.2 Metrics

- **Pass@1**: mean `is_correct_v2` over rollouts.
- **Pass@k with k=8**: unbiased estimator (Chen et al. 2021). Note: `preflight_lock.json` says k=16 but the actual rollout count is 8; we report Pass@8 and flag the discrepancy.
- **Cover@τ with τ=0.15**: among prompts with ≥1 correct rollout, fraction where the largest correct cluster has mass ≥ τ. Compute under three substrates: `answer_loose`, `completion_embedding` (at its best threshold from Analysis B), LLM clusters from Analysis A.
- `**worst_subset_accuracy`**: mean Pass@1 on the worst-performing 25% of prompts (ranked by per-prompt Pass@1).
- All metrics with bootstrap 95% CIs (1000 resamples, prompt-level).

### D.3 Output

A small markdown table with v1 (original parser) vs v2 (fixed parser) side-by-side, and the three substrate variants of Cover@τ.

### D.4 Can claim

- "Proxy base model achieves these values on the Run 0 prompt set under the corrected parser; any training arm needs to clear these to be a real gain on this distribution."

### D.5 Cannot claim

- Generalization to AIME-25 / HMMT / Minerva. Run 0 used DaPO-3k proxy prompts. This is a training-distribution baseline only.

---

## Execution order if time-constrained before office hours (2026-05-21 PM)

1. ~~**Prerequisite 0b** — parser-fix re-score.~~ **Done** — see `reparse_diff.md`.
2. **Analysis A** — LLM clustering, 500 prompts, `gemini-3.1-flash-lite`, Poly-EPO §A.1 prompt, full completions, `--force` if caches predate prompt change. ~~30–60 min, ~$5. Optional `**moderate`** cross-check on 50 prompts (+~~$2).
3. **Analysis A quality checks** — hand-read 10 prompts. ~45 min. Do not skip — this is what makes the LLM clusters defensible as a reference.
4. **Analysis B** — substrate comparison. ~1.5 hours including the disagreement vignettes.
5. **Analysis C** — objective simulation. ~1.5 hours.
6. **Analysis D** — base-model baseline. ~30 min.

If something must be cut: drop D first (it's the easiest to produce later and the least conversation-altering for tomorrow). Do not cut A's hand-check.

---

## What we are explicitly NOT doing in this round

- **No training.** Smoke is unverified; any training comparison would be confounded with parser and substrate issues that Analyses 0, A, B are *about*.
- **No in-loop training judge.** Poly-EPO calls the LM judge during RL (850 optimization steps); this round labels base-model rollouts offline only.
- **No final objective selection.** Analysis C produces evidence for the conversation; it does not by itself choose `worst_subset` over cheap `f_poly`.
- **No claim that any cheap substrate approximates Poly-EPO's LM judge in the training loop.** The closest claim we can make is "matches LLM clustering on base-model rollouts at ARI=X."

---

## How this changes tomorrow's office hours doc

If Analyses A and B return useful numbers, several questions in `briefs/ta_office_hours_20260521.md` can be tightened or removed:

- **Question A3** (pivot to CoT clustering?) — answered with data, not asked. Replace with: "We ran LLM-based clustering on Run 0; the substrate-controlled minority-correct rate is X%, vs 0% under answer-hash. We propose [substrate Y] for Stage 1 because it matches LLM clusters at ARI=Z."
- **Question A2** (how to formalize minority voting?) — still open, but Analysis C provides empirical evidence about which formalizations are even distinguishable on real data. Strengthens the conversation.
- **Question A4** (is GRPO an acceptable majority baseline?) — Analysis C shows quantitatively how different GRPO is from the alternatives on real data; the conversation has numbers to anchor it.

