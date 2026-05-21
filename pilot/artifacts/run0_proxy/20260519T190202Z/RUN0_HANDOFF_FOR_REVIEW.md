# Run 0 — detailed results handoff (for downstream review)

**Artifact run:** `20260519T190202Z`  
**Run ID:** `run0_proxy`  
**Purpose of this doc:** Freeze what was actually measured, where it lives, and what is *not* measured — so a reviewer can compare answer-only clustering to Poly-EPO-style reasoning clusters (which Run 0 did **not** build).

**Do not treat prior gate language (`minority_correct_prompt_rate ≥ 15%`) as sacred.** The team now believes that metric may be the wrong object for answer-only clustering.

---

## 0. What Run 0 was supposed to be (design intent)

Sources:

- `pilot/docs/operations/RUNBOOK.md` — Run 0 definition, gates (historical)
- `pilot/docs/operations/PILOT_REDESIGN.md` — Stage 1 matrix, substrate, mechanism diagnostics
- `nancy_explore/narrative/context.md` — project framing (“kill the LM-judge”, `inverse_freq`)
- `pilot/train/objectives.py` — what `inverse_freq` actually computes

**Run 0 was not training.** It was a single forward sampling pass:

| Field | Value |
|-------|--------|
| Model | `Qwen/Qwen3-1.7B-Base` |
| Prompts | Rows **0–499** of `pilot/data/dapo_slice_3k.jsonl` (500 prompts) |
| Rollouts / prompt | **8** |
| Total rollouts | **4000** |
| Sampling | `temperature=1.0`, `top_p=0.95`, `max_new_tokens` **2048** in config (execute path clamped to **1024** at time of run — see `pilot/infra/execute.py`) |
| Training | **None** (`mode: proxy_rollout_only`) |
| Budget | ~$15.86 actual (`cost.json`), cap $24 |
| Wall time | ~6.35 GPU-hr (`train.log` tail: `completed 500/500`, `gpu_seconds=22849`) |

**Stated research bet (Stage 1):** Replace Poly-EPO’s LM judge for clustering with **cheap exact-match canonicalization** on extracted `\boxed{}` answers, then run set-RL with **`inverse_freq`** (upweight rollouts in *smaller answer clusters* within each prompt’s 8 samples).

**What Run 0 was meant to validate before spend on Run 1–3:**

1. The **substrate** produces usable within-prompt structure (many clusters / modes).
2. Historically also: `minority_correct_prompt_rate` — fraction of prompts where **correct** rollouts fall in **more than one answer cluster** (see §2 — team now questions this metric).

---

## 1. Primary data files (immutable + derived)

All paths relative to:  
`pilot/artifacts/run0_proxy/20260519T190202Z/`

### 1.1 Raw Run 0 outputs (do not edit)

| File | Lines | Description |
|------|------:|-------------|
| `raw_predictions.jsonl` | 4000 | One row per rollout. Fields: `prompt_id`, `parsed_answer`, `correct`, `cluster_id`, `completion` (full model text). |
| `prompt_inputs.jsonl` | 500 | `prompt_id`, `problem`, `gold_answer`. |
| `metrics.json` | 1 object | Aggregates from **stored** labels at write time. |
| `config.snapshot.yaml` | — | Frozen config (`clustering: exact_canonical`, etc.). |
| `cost.json` | — | `gpu_seconds`, `estimated_usd`. |
| `train.log` | append-only | Progress + HF load noise; full run ends `Run0 done: minority_correct_prompt_rate=0.000`. |

**Join key:** `prompt_id` + rollout order (8 per prompt). Safer join for audits: `(prompt_id, completion)` — used in `review/build_review_dashboard.py`.

### 1.2 Cleaned relabel pass (completions unchanged)

| File | Description |
|------|-------------|
| `cleaned/predictions.jsonl` | Same 4000 completions + `parsed_answer_clean`, `correct_clean`, `cluster_id_clean`, `extract_path_clean`, `is_runon_fallback`, `canon_clean`, `semantic_bucket`. |
| `cleaned/metrics.json` | Aggregates on clean labels. |
| `cleaned/prompt_stats.jsonl` | Per-prompt clean stats. |
| `cleaned/delta_vs_raw.md` | Parse/correct flips vs stored. |
| `cleaned/manual_review_weird_cases.md` | Human spot-checks (truncated `\boxed`, run-on, flips). |
| `cleaned/signal_investigation.md` | Substrate viability writeup (answer-cluster lens). |
| `cleaned/signal_stats.json` | Machine-readable recomputation. |
| `cleaned/signal_compute.py` | Repro script. |

Cleaner implementation: `pilot/train/answer_clean.py`  
Batch script: `pilot/scripts/clean_run0_artifacts.py`

### 1.3 Analysis & UI (read-only lenses)

| File | Role |
|------|------|
| `analysis_v2_quant.md` | Completion-aware re-audit of stored fields. |
| `analysis_v2_qual.md` | Failure taxonomy + vignettes. |
| `_audit_parse_cluster.md` | Parser/cluster bugs on stored pipeline. |
| `review/` | HTML dashboard; `;` toggles RAW vs CLEAN; build via `build_review_dashboard.py`. |

### 1.4 Per-prompt quick stats (stored labels)

`_prompt_level_stats.jsonl` — 500 rows: `n_distinct_parsed`, `n_distinct_clusters`, `n_correct_rollouts`, `n_wrong_clusters` (stored only).

---

## 2. Labeling pipeline (what “cluster” and “correct” mean here)

Code paths:

- Rollout loop: `pilot/infra/execute.py` → `run0_proxy()`
- Extract: `pilot/train/answer_parse.py` — `extract_answer()`, `is_correct()` (**requires exactly one shallow-regex `\boxed{...}`** for `is_correct`; int-only auto-clean in shallow boxed)
- Cluster: `pilot/train/canonicalize.py` — `canonicalize_answer()` → `cluster_id = hash(canon) % 2**31`
- Legacy gate metric: `pilot/train/run_proxy.py` — `has_minority_correct_cluster()` among **correct** rollouts only

### 2.1 Critical definitional point (team concern)

If **cluster ID is a function of canonicalized final answer string**, then **all rollouts with the same correct parsed answer necessarily share one cluster**.

Therefore `has_minority_correct_cluster` (multiple clusters **among correct rollouts**) is **impossible** whenever “correct” implies “same canonical answer as gold.”

That does **not** by itself prove Poly-EPO’s mechanism is irrelevant — Poly-EPO clusters **reasoning traces** (via an LM judge), not just final numeric answers. Run 0 **does not cluster CoT**; it only clusters **parsed_answer**.

**The right question for a downstream reviewer may be:**  
*Under answer-only clustering, does `inverse_freq` approximate “upweight diverse reasoning,” or only “upweight rare wrong strings / singleton answer clusters”?*  
*Is there enough **completion-level** diversity **conditional on** same answer or same cluster to motivate a CoT-aware substrate?*

---

## 3. Headline aggregates (stored / raw labels)

Source: `metrics.json`

```json
{
  "run_id": "run0_proxy",
  "minority_correct_prompt_rate": 0.0,
  "n_prompts": 500,
  "n_rollouts_per_prompt": 8,
  "fraction_with_correct": 0.326,
  "mean_distinct_clusters": 7.176
}
```

| Metric | Value | Notes |
|--------|------:|-------|
| Rollout-level `correct` (stored) | **324 / 4000 = 8.10%** | Strict boxed path in `is_correct()` at run time |
| Prompt-level ≥1 correct | **163 / 500 = 32.6%** | |
| `minority_correct_prompt_rate` | **0.0% (0/500)** | See §2.1 |
| Mean distinct `cluster_id` / prompt | **7.18** | Among 8 rollouts |
| Mean distinct `parsed_answer` / prompt | **7.19** (v2 recompute) | Almost same as clusters |
| Max correct rollouts on one prompt | **7** (one prompt) | **0** prompts with 8/8 correct |

### 3.1 Distribution: correct rollouts per prompt (stored)

From `cleaned/signal_stats.json` → `dist_correct_per_prompt_stored`:

| n_correct (of 8) | # prompts | % |
|------------------|----------:|--:|
| 0 | 337 | 67.4% |
| 1 | 81 | 16.2% |
| 2 | 35 | 7.0% |
| 3 | 28 | 5.6% |
| 4 | 10 | 2.0% |
| 5 | 6 | 1.2% |
| 6 | 2 | 0.4% |
| 7 | 1 | 0.2% |
| 8 | 0 | 0.0% |

### 3.2 Distribution: distinct clusters per prompt (stored)

| n_clusters | # prompts |
|------------|----------:|
| 8 | 272 |
| 7 | 126 |
| 6 | 49 |
| 5 | 32 |
| 4 | 14 |
| 3 | 6 |
| 2 | 1 |

---

## 4. Headline aggregates (cleaned labels)

Source: `cleaned/metrics.json`, `cleaned/signal_stats.json`

| Metric | Stored | Clean | Δ |
|--------|-------:|------:|---|
| Rollout correct | 324 (8.10%) | 330 (8.25%) | +6 flips, 0 lost |
| Prompt ≥1 correct | 163 (32.6%) | 165 (33.0%) | +2 prompts |
| `minority_correct` (clean recompute) | 0% | 0% | — |
| Mean distinct `cluster_id_clean` / prompt | 7.18 | 6.86 | Merges + run-on drops |

**Correct flips (6 rollouts, 5 prompts):** all false→true; **parsed string often unchanged** — canon/LaTeX normalization only.  
Prompt IDs: `cfc7b48f-94bf-429f-b1c9-a7ac15e86b80`, `22063de2-a7a2-4214-895f-e015e0b78f87`, `65da7224-5f07-48e3-9b01-3c9ea1dfb036`, `2e690d58-de84-4003-a33f-fbebdb71dae5`, `70aabfd8-5728-4d08-8363-94e175fc0632`.  
Details: `cleaned/delta_vs_raw.md`, `cleaned/manual_review_weird_cases.md`.

**Parse changes:** 514 / 4000 (12.85%) — 426 `runon_rejected`, 88 brace-balanced `\boxed` fixes; **0** correct-only changes from parse text alone.

---

## 5. Diversity on three axes (raw completions) — **key for Poly-EPO comparison**

These stats were recomputed from `raw_predictions.jsonl` for this handoff (not in `metrics.json`).

### 5.1 Axis A — Full completion text (exact-string; not CoT-clustered)

| Stat | Value |
|------|--------|
| Mean distinct **completions** / prompt (exact string) | **8.0** |
| Median | **8.0** |
| Prompts with **all 8 completions distinct** | **500 / 500 (100%)** |
| Mean completion length | **~1877 chars** (~469 tok @ chars/4) |
| Completions containing `\boxed{` | **2023 / 4000 (50.6%)** |

**Interpretation:** At the **trace level**, the model almost never repeats an identical rollout verbatim. Every prompt exhibits maximum completion diversity among 8 samples.

### 5.2 Axis B — Parsed final answer string (`parsed_answer`)

| Stat | Value |
|------|--------|
| Mean distinct `parsed_answer` / prompt | **7.19** |
| Prompts with 8 distinct parsed | **274** (v2); **272** with 8 distinct `cluster_id` |

**Interpretation:** Collapsing trace → extracted answer loses **one** degree of freedom on average (8 completions → ~7.2 answer strings).

### 5.3 Axis C — Cluster ID (`hash(canonicalize(parsed))`)

| Stat | Value |
|------|--------|
| Mean distinct `cluster_id` / prompt | **7.18** |
| Same as parsed for most prompts | canon duplicates rare |

**Cluster size among 4000 rollouts** (how many rollouts share each cluster within a prompt):

| Cluster size (of 8) | # rollouts |
|---------------------|----------:|
| 1 | 3319 |
| 2 | 364 |
| 3 | 153 |
| 4 | 84 |
| 5 | 55 |
| 6 | 18 |
| 7 | 7 |

**`inverse_freq` implication:** **83%** of rollouts sit in a **singleton** answer cluster (size 1). Inverse-frequency weighting **upweights most rollouts equally** when most clusters are size 1 — weighting is flat among singletons.

### 5.4 Same final answer / same cluster, different completion strings

Among prompts where **≥2 rollouts share the same `parsed_answer`**:

- **226 / 500** prompts have **≥2 different completions** for the same parsed string.

Among prompts where **≥2 rollouts share the same `cluster_id`**:

- **228 / 500** prompts have **≥2 different completions** in that cluster.

Among prompts with **≥2 correct rollouts** (stored `correct=True`):

- **82** prompts have multiple correct rollouts.
- **82 / 82** have **≥2 distinct completions** among those correct rollouts (same gold answer, different full text).

Example prompt IDs (same parsed among correct, different completions):  
`7ec6f22e-5008-43cf-8218-ea0c4ce775ac`, `6137f3cc-cd8e-43dd-9213-2e4c3784c96e`, `a6bce30d-9781-402b-95ae-882c43e72b79`, `ddd26788-…` (7/8 correct, single cluster `n:2` / same parsed).

**Interpretation:** **Distinct completion strings** exist even when answer-cluster diversity among correct rollouts does not. Run 0 did not cluster or judge reasoning — only parsed answers.

### 5.5 Correct rollout cluster sizes (stored)

When a rollout is marked correct, how many of the 8 rollouts share its cluster?

| Cluster size | # correct rollouts |
|--------------|-------------------:|
| 1 | 81 |
| 2 | 70 |
| 3 | 84 |
| 4 | 40 |
| 5 | 30 |
| 6 | 12 |
| 7 | 7 |

So correct mass often sits in clusters that also contain **wrong** rollouts (same wrong answer string as correct? or same cluster from canon collision — reviewer should spot-check in dashboard).

---

## 6. Extraction path breakdown (cleaned, completion-aware)

From `cleaned/signal_stats.json` → `extract_path_clean`:

| Path | Rollouts | % |
|------|----------:|--:|
| `boxed_balanced` | 2016 | 50.4% |
| `answer_line` | 853 | 21.3% |
| `last_line` | 705 | 17.6% |
| `runon_rejected` | 426 | 10.7% |

**Qual failure modes** (`analysis_v2_qual.md`, approximate): ~85–92% wrong math; ~49% no usable `\boxed`; ~25% repetition; ~28% code/SymPy derailment; 8 truncated `\boxed`; nested-boxed regex issues (88).

---

## 7. What `inverse_freq` does in this codebase (not Poly-EPO verbatim)

`pilot/train/objectives.py`:

- Base: `A_i = r_i - mean(r)` (GRPO-style per-prompt baseline).
- `inverse_freq`: multiply by weight ∝ `(cluster_size)^{-γ}`, normalized, capped (`w_max=8`).
- **Cluster size** = count of rollouts sharing the same `cluster_id` **among the 8**, regardless of correct/wrong.
- **Does not** use completion text, LM judge, or “minority among correct only.”

**Implication for Run 0:** High cluster diversity mostly reflects **diverse wrong answers**, not **diverse correct minorities**. Weighting still affects **wrong** rollouts in rare string clusters.

---

## 8. Poly-EPO vs this pilot (conceptual anchor)

Read: `nancy_explore/narrative/context.md`, `nancy_explore/agents/outputs/final_decision.md`, `PILOT_REDESIGN.md` §1.

| Dimension | Poly-EPO (related work) | Run 0 / Stage 1 pilot |
|-----------|-------------------------|------------------------|
| Clustering substrate | LM judge on reasoning | Exact-match on extracted answer |
| What “minority” means | Rare **reasoning equivalence classes** | Rare **answer-string** buckets (mostly wrong) |
| Correctness axis | Separate from cluster definition | Cluster derived from answer → collapses correct |
| Completion-text diversity | Explicitly clustered (Poly-EPO) | **§5.1, §5.4:** distinct strings only; **not** clustered or LM-judged |

---

## 9. Infrastructure context (why Run 0 completed but matrix failed)

Not the focus of scientific conclusions, but explains artifact quality:

- First matrix launch failed on cost, resume, logging (`pilot/docs/incidents/0519-*`, `PILOT_REDESIGN.md`).
- Run 0 completed detached; artifacts pulled to this directory.
- `train.log` is **concatenated across smokes** — filter lines after `2026-05-19 19:02:16` for full 500-prompt job.

---

## 10. Suggested spot-checks for reviewer (concrete)

Use `review/` dashboard (`./serve.sh` → http://localhost:8765):

1. **High completion diversity, single correct cluster:** `ddd26788-…` — 7/8 correct, one parsed/cluster; read 7 completions.
2. **Correct flip (format only):** `cfc7b48f-…` — RAW vs CLEAN (`;`).
3. **Many clusters, all wrong:** pick from `analysis_v2_qual.md` vignettes, e.g. `01677f18-…`.
4. **Run-on rejected (clean):** filter CLEAN + `run-on:` label rows.
5. **Same parsed, different completions:** `a6bce30d-9781-402b-95ae-882c43e72b79` — 4 correct, 4 distinct completion strings (same parsed).

Repro stats:

```bash
python pilot/artifacts/run0_proxy/20260519T190202Z/cleaned/signal_compute.py
python pilot/scripts/clean_run0_artifacts.py --artifact-dir pilot/artifacts/run0_proxy/20260519T190202Z
```

---

## 11. Analyst notes (Cursor agent — not authoritative)

These are **working thoughts** for the downstream model; challenge them.

1. **The team is right** that `minority_correct_prompt_rate` on answer clusters is **near-tautological** if correct ⇒ same canonical answer ⇒ same cluster. Using that metric as a Run 0 gate was **misaligned** with the metric definition, not necessarily proof that “minority voting is dead.”

2. **Poly-EPO’s reason for minority upweighting** (diversity of reasoning on an axis **orthogonal** to “did the final answer string match”) **was not emulated** in Run 0. Run 0 only measured answer-string clustering.

3. **There IS a separate signal in the raw data:** **100%** prompt-level distinct full completions; **82** prompts with multiple correct rollouts and **distinct completion strings** per correct answer — counted by exact text only, not reasoning labels — **invisible** to `inverse_freq` (answer `cluster_id` only).

4. **`inverse_freq` on answer clusters** mostly upweights **singleton wrong answers** (83% rollouts in size-1 clusters). That is a **different hypothesis** from mentor’s minority-voting story: “optimize worst-case reasoning paths,” not “upweight rare wrong integers.”

5. **Cleaning** matters for fair **accuracy** and interpretability; it does **not** create multi-cluster correct structure (still 0%). It **does** matter for whether Run 1+ labels match training prompt (`\boxed{}`).

6. **Open question** is not “did Run 0 pass a broken gate?” but:
   - Should Stage 1 test **answer-cluster `inverse_freq`** at all?
   - Is a **CoT-aware cluster** (embeddings, judge, trajectory hash) required for the project to be “minority voting” in the Poly-EPO sense?
   - Can **answer-only `inverse_freq`** still help Pass@k/Cover@τ for other reasons (upweight rare outcomes regardless of correctness)?

7. **Do not rerun Run 0** hoping for different answer-cluster minorities. **Do** use existing jsonl to study **completion-level** diversity if proposing a new substrate.

---

## 12. Prompt for downstream reviewer (smarter model)

Use everything above and the cited files. The human team’s latest position:

> We obviously never have correct answers in more than one answer cluster — that’s by definition. Poly-EPO had a reason to upweight based on diversity of chain-of-thought on a separate axis from correctness. We’re unsure if our setup has that same reason, or if that’s even the right question.

**Your task:**

1. **Restate** what Run 0 actually measured vs what Poly-EPO-style minority voting needs (be explicit about axes: completion, parsed answer, cluster ID, correctness).

2. **Answer:** Given **only** the artifacts in `20260519T190202Z`, is there empirical support for **any** minority-upweighting story that is **faithful** to the mentor pitch (`nancy_explore/narrative/context.md`)? Distinguish:
   - (A) Minority **correct reasoning paths** (Poly-EPO-like)
   - (B) Minority **answer strings** mostly among **wrong** rollouts (`inverse_freq` as implemented)
   - (C) No viable minority signal — pivot recommendations

3. **Evaluate** whether `inverse_freq` in `pilot/train/objectives.py` is **mechanistically aligned** with §5 statistics (singleton cluster dominance, etc.). Quantify if possible from the jsonl.

4. **Recommend** concrete changes to **Stage 1** (`PILOT_REDESIGN.md`): what to keep (runs, budget, infra), what to drop or reframe (Run 0 gate, run2 interpretation), and whether a **new Run 0b** (e.g. completion-level clustering sample) is worth cost **without** full re-rollout.

5. **Verdict** (one line): `CONTINUE_STAGE1_AS_IS` | `REFRAME_STAGE1` | `NEW_SUBSTRATE_REQUIRED` | `PIVOT_OBJECTIVE` — plus 5 bullets the team should put in the milestone writeup.

**Constraints:** Cite specific numbers and file paths from this handoff. If you spot-check completions, name `prompt_id`s. Do not treat `minority_correct_prompt_rate` as a load-bearing pass/fail without addressing §2.1. Be direct if the current “kill the LM judge + exact match” path cannot test the mentor’s minority-voting hypothesis.

---

*Generated for external review. Raw completions remain in `raw_predictions.jsonl`; no further GPU spend required for questions in §5–6.*
