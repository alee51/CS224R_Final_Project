# Run 0 offline analyses — design doc

**Date:** 2026-05-21
**Artifact under analysis:** `pilot/artifacts/run0_proxy/20260519T190202Z/`
**Constraint:** No new training, no GPU. Everything below runs on a laptop + Anthropic API.
**Audience:** human executor (or agent under human supervision).

This doc specifies four analyses to run on Run 0's existing 4000 rollouts before tomorrow's office hours. Each analysis section includes inputs, parameters, method, outputs, what the analysis **can claim**, and what it **cannot claim** (the prior round of analysis conflated several things — this section is the guardrail against repeating that).

A note on terminology: the term "chain-of-thought analysis" was previously used in this project for analyses that never called an LLM. In this doc, **only Analysis A** ("LLM reasoning-cluster ground truth") qualifies as CoT analysis. The other substrate-based clusterings are explicitly named "completion-text embedding" or "structural feature clustering" and must be described that way in any writeup.

---

## 0. Prerequisite: parser-fix re-score (plumbing, not an analysis)

This is shared setup that every downstream analysis reads from. It is not a hypothesis-driven analysis — it is a one-shot data correction, plus a brief audit of what changed.

### 0.1 Inputs
- `pilot/artifacts/run0_proxy/20260519T190202Z/raw_predictions.jsonl` (4000 rollouts)
- `pilot/artifacts/run0_proxy/20260519T190202Z/prompt_inputs.jsonl` (500 prompts, gold answers)
- Current `pilot/train/answer_parse.py` (`extract_answer`, `canonicalize_answer`, `is_correct`)

### 0.2 Actions
1. Implement the parser fixes scoped in `pilot/docs/operations/PILOT_REDESIGN.md` items C2 and C3:
   - **C2 (`extract_answer`):** brace-balanced `\boxed{...}` matching instead of shallow regex (fixes the `\frac{1190` truncation class).
   - **C3 (`canonicalize_answer`):** normalize `\( … \)` wrappers, strip trailing `%`, strip trailing units like ` degrees` / ` square units`, strip extraneous LaTeX wrappers.
2. Apply the fixed parser to all 4000 rollouts. Produce `raw_predictions_reparsed.jsonl` with new fields side-by-side with originals:
   - `parsed_answer_v2`, `canonical_v2`, `is_correct_v2`, `cluster_id_v2 = hash(canonical_v2)`
3. Do **not** overwrite the original fields. Downstream analyses can compare v1 vs v2.

### 0.3 Audit deliverable
A short markdown report `reparse_diff.md` containing:
- Total rollouts with `is_correct` changed (split: false→true vs true→false).
- Cluster merges and splits per prompt (count distribution).
- Count of rollouts still unparseable (no boxed, no `Answer:`).
- The new headline numbers: rollout accuracy v2, prompt accuracy v2, `minority_correct_prompt_rate_v2`, each with bootstrap 95% CI (1000 resamples over prompts).
- Same numbers v1 for comparison.

This is data hygiene, not science. The analyses below are the science.

---

## Analysis A — LLM reasoning-cluster ground truth

**The most important analysis in this doc.** Every other substrate analysis is judged against it.

### A.1 Motivation

Ifdita's Poly-EPO uses an LM judge (Gemini 2.0 Flash) to cluster *reasoning steps*. Run 0's substrate is exact-match on final answer strings. These are not the same operation. The whole question "is the LM judge load-bearing?" presupposes that we can compare an LM judge's clustering to a cheap substrate's clustering — but we don't currently have an LM judge clustering of Run 0 to compare against. This analysis produces that reference clustering.

The output of this analysis is the **reference** that Analysis B's cheap substrates get scored against.

**Clarification on what "expensive LM judge" means in Poly-EPO.** Gemini 2.0 Flash is per-call cheap (pennies; Haiku-tier). When the Poly-EPO paper frames the judge as expensive or limiting, that is about **training-loop overhead** — the judge is invoked on every rollout set on every gradient step, gating each policy update on an external API call, accumulating cost and latency across thousands of training steps and introducing an availability dependency. It is **not** about per-call labeling cost for a one-shot offline pass like this analysis. The "kill the LM judge" framing in our project should mean "replace the in-training-loop LM judge with a cheap local substrate" — not "we cannot afford to call an LM judge ever." This analysis (one-shot offline labeling of 500 prompts) costs a few dollars and is fully consistent with the "kill the in-loop judge" pitch.

### A.2 What we are doing (and not doing)
- **Doing:** for each of the 500 prompts, send the problem + the 8 completions to Claude and ask it to cluster the 8 rollouts by reasoning approach. Per-prompt clustering, not cross-prompt.
- **Not doing:** segmenting completions into reasoning steps and clustering steps across prompts (Poly-EPO does this with 850 steps). That is more expensive and not required for the Question I comparison. If Ifdita pushes back on the cheaper design, we can scope a step-level version after office hours.

### A.3 Inputs
- `raw_predictions_reparsed.jsonl` (so the LLM sees re-parsed answers, which matters for completions with broken `\boxed`).
- For each prompt: the problem statement + the 8 completion texts + the gold answer + the parsed answer per rollout + the `is_correct_v2` flag per rollout.

### A.4 Model and protocol
- **Primary model:** Haiku 4.5 (`claude-haiku-4-5`). This is the Anthropic-side analog of Gemini 2.0 Flash (the Poly-EPO paper's judge) — same cost tier, same role. Using a stronger model than the paper would *over-spend* relative to her setup and would be a less faithful mimic. Pick one judge and roll with it.
- **Optional cross-model sanity probe:** Sonnet 4.6 (`claude-sonnet-4-6`) on a 50-prompt random subsample. This is *not* the centerpiece — it's a small robustness check. Compute ARI between Haiku and Sonnet per prompt; if mean ARI is high (≥0.7), the LM-judge approach is robust to model tier. If low (<0.5), report as a caveat — clusterings would be model-dependent. Optional because exhaustive judge-robustness analysis strays from the project's core question.
- **Prompt structure** (per prompt):
  - System: brief instruction explaining the task (cluster rollouts by reasoning approach, not by final answer).
  - User: the problem, the gold answer, then a numbered list of the 8 rollouts (each truncated to ~800 tokens if needed; flag truncation).
  - Output format: structured JSON with fields `clusters: [{cluster_id: int, member_rollouts: [int], reasoning_signature: str}]`, plus a per-rollout `cluster_assignment: {rollout_idx: cluster_id}`.
- **Caching:** save raw API responses to `pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters/{prompt_id}.json`. Idempotent — re-running skips prompts already labeled.

### A.5 Clustering instructions to the LLM (verbatim seed; tune after a 5-prompt pilot)

> Group these 8 attempted solutions by **reasoning approach**, not by whether they reach the correct answer. Two solutions are in the same cluster if they use substantially the same method (e.g., both use modular arithmetic via Fermat's little theorem; both set up coordinates and use vectors; both invoke SymPy code execution). Two solutions are in different clusters if they pursue distinct mathematical strategies, even if they happen to reach the same numeric answer.
>
> A solution that derails into unrelated content (a different problem, code with no math, repetition loops) should be placed in a "degenerate" cluster labeled `cluster_id: -1`.
>
> Return JSON with: (1) a list of clusters, each with a short signature describing the approach; (2) per-rollout assignment.

### A.6 Cost and time estimate
- Per call: ~5800 input tokens (problem + 8 completions + instructions) + ~800 output tokens.
- Sonnet 4.6 at $3/MTok input, $15/MTok output: ~$15 total for 500 prompts.
- Haiku 4.5 validation on 50 prompts: ~$0.50.
- Wall time: ~30-60 minutes with modest concurrency (10 in-flight calls).
- Use prompt caching on the system prompt + instruction block (these don't change across prompts) — should cut input cost meaningfully.

### A.7 Quality checks (mandatory before using these clusters downstream)
1. **Hand-read 10 prompts.** Pick: 3 prompts with all-correct rollouts, 3 with mixed correctness, 4 with no-correct rollouts. For each, read the 8 completions and the LLM's cluster assignments. Does the clustering match what a human would do? Document disagreements in `llm_clusters_handcheck.md`.
2. **Cross-model agreement (50-prompt subsample).** Run Haiku 4.5 on a random 50 prompts. Compute Adjusted Rand Index (ARI) between Sonnet and Haiku clusterings per prompt; report mean and distribution.
3. **Degenerate-cluster rate sanity.** What fraction of rollouts land in `cluster_id: -1`? Cross-check against the qual analysis's tagged garbage rate (~28% sympy derailment + ~25% repetition + ~9% long parse). If LLM's degenerate rate is wildly different, the prompt needs tuning.

### A.8 Outputs
- `llm_clusters/{prompt_id}.json` × 500 (raw model output, cached).
- `llm_clusters_summary.parquet` — per-rollout LLM cluster assignment.
- `llm_clusters_handcheck.md` — 10-prompt hand audit.
- `llm_judge_cross_model.md` — Sonnet vs Haiku ARI on 50-prompt subsample.
- **The headline number:** `minority_correct_prompt_rate_llm` — under LLM clusters, fraction of prompts with ≥1 correct rollout that have correct rollouts spanning ≥2 LLM-clusters with at least one not the largest. Bootstrap 95% CI.

### A.9 Can claim
- "Under LLM-based reasoning clustering on Run 0's base-model rollouts, minority-correct prompts exist at rate X% (CI […, …]). This is the substrate-controlled value for the gate metric."
- If Sonnet↔Haiku ARI is high: "Even a cheap LM judge produces clusterings consistent with a stronger one — the LM judge in Poly-EPO need not be expensive."
- If Sonnet↔Haiku ARI is low: "LM-judge clusterings are model-dependent on this data; the substrate is less determinate than the Poly-EPO paper implies."

### A.10 Cannot claim
- That LLM clusters are "correct." They are a reference, not ground truth in the philosophical sense. A human-labeled subsample (the 10 hand-checked prompts) is the only thing closer to ground truth, and it's tiny.
- That this replicates Poly-EPO's judge. Poly-EPO clusters reasoning steps; we cluster whole rollouts. Different granularity.
- Anything about training-time behavior. This is offline labeling of base-model outputs.

---

## Analysis B — Cheap-substrate comparison against the LLM reference

**What it answers:** Does any substrate cheaper than an LM judge approximate the LLM clustering well enough that Question I has a positive answer? This is the empirical core of the "kill the LM judge" pitch.

### B.1 Inputs
- `raw_predictions_reparsed.jsonl` (for substrate computation).
- `llm_clusters_summary.parquet` from Analysis A (the reference).

### B.2 Substrates to evaluate

For each of the substrates below, compute per-prompt cluster assignments over the 8 rollouts. All cluster assignments are local to a single prompt.

1. **`answer_strict`** — hash of original `canonical` field. The current code's substrate. Baseline.
2. **`answer_loose`** — hash of `canonical_v2` (parser-fixed). Tests whether the parser alone explains the 0% finding.
3. **`completion_embedding`** — embed each completion's full text with `sentence-transformers/all-MiniLM-L6-v2` (~80MB, runs CPU in ~5 min for 4000 completions). Per-prompt cluster the 8 embeddings using agglomerative clustering with cosine distance. **Hyperparameter sweep, not a single value:** distance threshold ∈ {0.2, 0.3, 0.4, 0.5}. Report results at each threshold.
4. **`completion_features`** — rule-based feature tagging per rollout (these tags already exist in the qual analysis): `{has_boxed, has_sympy_code, has_repetition, has_code_fence, has_modular_arithmetic_keywords, has_coordinate_method_keywords, parsed_is_numeric, parsed_is_latex_fraction}`. Cluster ID = tuple of binary tags within prompt. Cheapest possible substrate.

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
- `raw_predictions_reparsed.jsonl` — only the `reward_v2 = is_correct_v2` field and `cluster_id_v2` are used.
- Choice of cluster substrate: run this analysis twice, once with `cluster_id_v2` (answer-hash) and once with the LLM clusters from Analysis A. The comparison between the two passes is itself informative — it shows how much the substrate choice affects the objective shape.

### C.2 Objectives and formulas

All advantages are computed per-rollout per-prompt. No model gradients are involved; only the advantage values that a training step would *send* to the model.

1. **GRPO**: `A_i = r_i - mean(r_prompt)`.
2. **`inverse_freq`**: `A_i = (1 / cluster_size_i) × (r_i - mean(r_prompt))`. Matches `pilot/train/objectives.py`.
3. **`f_poly` (set-level Poly-EPO substrate-swap)**:
   - Enumerate all subsets G of size n=4 from each prompt's 8 rollouts: C(8,4) = 70 subsets per prompt.
   - Per subset: `f_poly(G) = mean_r(G) × d(G)`, where `d(G) = (# distinct clusters in G) / n`.
   - Per rollout i: `A_i^set = mean over G ∋ i of f_poly(G) - global_mean_f_poly`.
4. **`worst_subset`**:
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
- `raw_predictions_reparsed.jsonl`.

### D.2 Metrics
- **Pass@1**: mean `is_correct_v2` over rollouts.
- **Pass@k with k=8**: unbiased estimator (Chen et al. 2021). Note: `preflight_lock.json` says k=16 but the actual rollout count is 8; we report Pass@8 and flag the discrepancy.
- **Cover@τ with τ=0.15**: among prompts with ≥1 correct rollout, fraction where the largest correct cluster has mass ≥ τ. Compute under three substrates: `answer_loose`, `completion_embedding` (at its best threshold from Analysis B), LLM clusters from Analysis A.
- **`worst_subset_accuracy`**: mean Pass@1 on the worst-performing 25% of prompts (ranked by per-prompt Pass@1).
- All metrics with bootstrap 95% CIs (1000 resamples, prompt-level).

### D.3 Output
A small markdown table with v1 (original parser) vs v2 (fixed parser) side-by-side, and the three substrate variants of Cover@τ.

### D.4 Can claim
- "Proxy base model achieves these values on the Run 0 prompt set under the corrected parser; any training arm needs to clear these to be a real gain on this distribution."

### D.5 Cannot claim
- Generalization to AIME-25 / HMMT / Minerva. Run 0 used DaPO-3k proxy prompts. This is a training-distribution baseline only.

---

## Execution order if time-constrained before office hours (2026-05-21 PM)

1. **Prerequisite 0** — parser-fix re-score. ~1 hour. Blocking for everything else.
2. **Analysis A** — LLM clustering on all 500 prompts via Sonnet 4.6, plus Haiku 4.5 validation on 50 prompts. ~1-1.5 hours wall time (API-bound, mostly waiting). ~$15-20.
3. **Analysis A quality checks** — hand-read 10 prompts. ~45 min. Do not skip — this is what makes the LLM clusters defensible as a reference.
4. **Analysis B** — substrate comparison. ~1.5 hours including the disagreement vignettes.
5. **Analysis C** — objective simulation. ~1.5 hours.
6. **Analysis D** — base-model baseline. ~30 min.

If something must be cut: drop D first (it's the easiest to produce later and the least conversation-altering for tomorrow). Do not cut A's hand-check.

---

## What we are explicitly NOT doing in this round

- **No training.** Smoke is unverified; any training comparison would be confounded with parser and substrate issues that Analyses 0, A, B are *about*.
- **No step-level reasoning clustering.** Poly-EPO clusters reasoning *steps*; we cluster whole *rollouts*. If Ifdita pushes back on the granularity, that's a scope conversation, not something to add before tomorrow.
- **No final objective selection.** Analysis C produces evidence for the conversation; it does not by itself choose `worst_subset` over cheap `f_poly`.
- **No claim that any cheap substrate approximates Poly-EPO's LM judge in the training loop.** The closest claim we can make is "matches LLM clustering on base-model rollouts at ARI=X."

---

## How this changes tomorrow's office hours doc

If Analyses A and B return useful numbers, several questions in `briefs/ta_office_hours_20260521.md` can be tightened or removed:

- **Question A3** (pivot to CoT clustering?) — answered with data, not asked. Replace with: "We ran LLM-based clustering on Run 0; the substrate-controlled minority-correct rate is X%, vs 0% under answer-hash. We propose [substrate Y] for Stage 1 because it matches LLM clusters at ARI=Z."
- **Question A2** (how to formalize minority voting?) — still open, but Analysis C provides empirical evidence about which formalizations are even distinguishable on real data. Strengthens the conversation.
- **Question A4** (is GRPO an acceptable majority baseline?) — Analysis C shows quantitatively how different GRPO is from the alternatives on real data; the conversation has numbers to anchor it.
