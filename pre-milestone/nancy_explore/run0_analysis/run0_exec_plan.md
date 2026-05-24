# Run 0 — execution plan

**Workspace:** `nancy_explore/run0_analysis/`  
**Rollouts:** `pilot/artifacts/run0_proxy/20260519T190202Z/` (500 prompts × 8)  
**Constraints:** No training, no GPU. No new LLM judge calls unless approved.

**How to read this doc:** [Objective](#objective) → [Questions](#questions) → [Experiments](#experiments) → [Data rules](#data-rules) → [Already done](#already-done).

> **Formulas:** plain text / code (no LaTeX) so this file renders in Cursor, GitHub, and basic Markdown viewers. For pretty math, paste blocks into a LaTeX editor or enable a Markdown math extension (e.g. VS Code “Markdown Math”).

---

## Objective

**Set-RL framework** (same as Poly-EPO): for each prompt, take all 4-rollout subsets (70 per prompt). Score each subset, then turn set scores into per-response advantages.

**Rollout reward:** `r_i = cleaned_correct` (0 or 1).

**Set score `f(G)`:** reward of the **minority** response(s) in that subset — as an average over rollouts in the minority mode (see 2×2 below). This is **not** Poly-EPO’s `mean(r in G) * diversity(G)`.

**Minority in subset G:** mode(s) with the **smallest count** among the 4 rollouts (ties = several modes tied for smallest count).

### Advantages (standard set-RL)

1. Compute `f(G)` for each of the 70 subsets.
2. **Baseline** = mean of all 70 set scores for that prompt.
3. **Set advantage** = `f(G) - baseline`.
4. **Marginal advantage of a response** = average set advantage over all subsets that include that response.

(E1 should use per-prompt baseline; note in writeup if you also try a global baseline.)

### Four set-score definitions (2×2)

**Axis 1 — tie-break among rarest modes** (Q1; **use rand** — see Questions)

- **rand (default):** pick one tied-rarest mode uniformly at random, then average `r_i` within that mode only (one rollout → its `r_i`)
- **avg:** average `r_i` over every rollout in **every** tied-rarest mode (E1: ~same marginal advantages as rand)

**Axis 2 — mode type** (Q2)

- **Answer:** `cleaned_cluster_id`
- **Cluster (CoT):** `llm_cluster_id` (degenerate: all `-1` on one prompt = one cluster)

**Definitions:**

- **ans-avg:** `f(G)` = mean of `r_i` over rollouts in G whose answer bucket is tied-rarest in G.
- **ans-rand:** pick one tied-rarest answer bucket `b` at random; `f(G)` = mean `r_i` over rollouts in G with bucket `b`.
- **cot-avg:** same with tied-rarest `llm_cluster_id`(s) in G.
- **cot-rand:** pick one tied-rarest cluster `c` at random; `f(G)` = mean `r_i` over rollouts in G in cluster `c`.

**E1 contrast only (not training):** Poly-EPO `f_poly(G) = mean(r in G) * (distinct clusters in G) / 4`.

**Training vs Run 0:** answer buckets are cheap (`cleaned_cluster_id`). For `cot-*`, offline E1 uses cached LLM clusters; **cheap embed/features are not a viable CoT substitute** (archived Analysis B: best mean ARI vs LLM **≤0.19**, embeddings **≤0.07**).

---

## Questions

What we needed to decide before training. Each item has a **status**; numbers live in `analysis_c/set_score_simulation.md` or `analysis_minority/minority_metrics.md`.

| Status | Meaning |
|--------|---------|
| **Settled** | Run 0 offline analysis is enough; use this in the writeup |
| **Open — training** | Not worth more offline work; compare arms when we actually train |
| **Settled (Phase 1)** | Answered before E1 |

---

### Q1 — When two “rarest” groups tie, do we pick one at random or average over all of them?

**Status: Settled**

They give almost the same training signal (correlation ~0.99 between the two ways of breaking ties). **Use random tie-break** — slightly simpler and matches what we’ll implement (`ans-rand`, `cot-rand`).

---

### Q2 — Should “minority” mean same final answer, or same reasoning style (LLM clusters)?

**Status: Open — training**

**What Run 0 showed:** The two definitions disagree a fair amount. If you score every rollout with both objectives, the resulting advantages correlate only ~**0.52** — much lower than the random-vs-average tie-break (~0.99). So this is a real fork, not a rounding difference.

**What we are not doing:** More offline metrics (e.g. subset rank disagreement). Cheap text embeddings do **not** substitute for LLM CoT clusters (archived Analysis B).

**Plan:** Treat **answer-only vs LLM-CoT** as a **training experiment** (e.g. `ans-rand` vs `cot-rand`), not another E1-style analysis. Phase 1 already showed answer-hash “minority” is 0% on eligible prompts while LLM clusters see ~14.5% — that’s why CoT mode matters for the story.

---

### Q3 — Is our minority set-score actually different from Poly-EPO’s diversity objective?

**Status: Settled** (whether it’s *different*). **Open — training** (answer vs CoT *variant*, same as Q2).

No — they are not the same in practice. Correlations with `f_poly` are moderate (~0.4–0.6 depending on variant), and with plain GRPO ~0.44–0.61. GRPO and `f_poly` look much more alike (~0.87). Minority set-RL is a distinct objective, not a rename of Poly-EPO.

**Implication:** If we train with minority scoring, we should expect different credit assignment than GRPO / Poly-EPO — not a tiny tweak.

---

### Q4 — Do individual rollouts actually get a non-zero training signal?

**Status: Settled**

Most subsets score zero, but the objective still assigns signal at the rollout level. After averaging the 35 subsets that include each rollout:

- ~**66%** of rollouts: advantage ≈ 0 (no real push)
- ~**34%** of rollouts: non-zero advantage (mix of positive and negative)
- ~**10–11%** of rollouts: advantage has the **opposite sign** from GRPO

Ignore “84% of subsets have f=0” as a headline; that’s the wrong level of aggregation.

---

### Q5 — What if all four rollouts in a subset share the same answer?

**Status: Settled**

Happens on only **~2.5%** of subsets. Behavior is sensible (minority score = average reward in the subset). Each rollout still appears in 35 other subsets, so this edge case doesn’t break the method.

---

### Q6 — How strong is the base model on this prompt set?

**Status: Settled (Phase 1)**

Under human-verified labels (`cleaned_correct`), on 500 prompts × 8 rollouts:

- **Pass@1:** 9.03% (361 / 4000 rollouts correct)
- **Pass@8:** 34.40% of prompts have at least one correct rollout [30.2%, 38.4%] CI

Source: `analysis_minority/minority_metrics.md`. Use these as the Run 0 floor when claiming training improved on the same data.

---

### Summary — what’s left before training

| Topic | Status |
|-------|--------|
| Tie-break: random vs average | **Settled** → random |
| Minority objective vs Poly-EPO / GRPO | **Settled** → it’s different |
| Enough per-rollout signal? | **Settled** → yes for ~34% of rollouts |
| Single-answer subset edge case | **Settled** → rare, OK |
| Base model Pass@k | **Settled** → see Q6 |
| **Answer-only vs LLM-CoT minority** | **Open — training** |
| **Which of the four `f(G)` variants to run** | **Open — training** (after Q2; default tie-break is rand) |

---

## Experiments

**E1 only** for set-RL objective choice (Q1–Q5). **Q6** (eval floor) is already in Phase 1 — see [Already done](#already-done).

### E1 — Set-score simulation · answers **Q1–Q5** · **Done**

**Method:** Per prompt, enumerate C(8,4). For each subset, compute `ans-avg`, `ans-rand`, `cot-avg`, `cot-rand`, plus contrast `f_poly`. Build marginal advantages via set-RL steps above. Compare to GRPO and `inverse_freq`. For `*-rand`, multiple RNG seeds (e.g. 20); report mean ± std of advantages.

**Inputs:** `data/cleaned_answers.parquet`, `analysis_a/llm_clusters_summary.parquet`

**Outputs:** `analysis_c/set_score_simulation.py`, `set_score_simulation.md`, `objective_advantages.parquet` (4000 rows, `adv_*` + rand seed std), `objective_corr_pearson.csv`, `objective_corr_spearman.csv`. ~19s on laptop; cleaned labels only.

See [Questions](#questions) for readable answers. Full tables: `analysis_c/set_score_simulation.md`.

---

## Data rules

### Use

| What | Path / notes |
|------|----------------|
| Ground truth | `labels/rollout_labels.jsonl` → **`data/cleaned_answers.parquet`** (`cleaned_correct`, `cleaned_cluster_id`, `cleaned_answer`, `cleaned_state`) |
| Join key | `rollout_key` or (`prompt_id`, `rollout_idx`) |
| LLM cluster IDs | `analysis_a/llm_clusters_summary.parquet`; cache `pilot/artifacts/.../llm_clusters/{prompt_id}.json` |
| Completions | `data/predictions_reparsed.jsonl` — **`completion` only** |
| Prompts | `data/prompt_inputs.jsonl` |
| Qual | `dashboard/` — `build.py` then `serve.sh` |
| v2 → cleaned | `cleaned_correct`, `cleaned_cluster_id`, `cleaned_answer` — see [`README.md`](README.md) |

### Do not use (metrics / claims)

| What | Why |
|------|-----|
| `archive/2026-05-21_pre_human_label_audit/` outputs as-is | v1/v2 parser scoring |
| Parser columns in `predictions_reparsed.jsonl` | Stale |
| v1/v2 headline Pass@ / minority rates | Use `analysis_minority/minority_metrics.md` |
| [`run0_analysis_plan.md`](run0_analysis_plan.md) | Superseded — use **this file** |
| [`overnight_workflow_log.md`](archive/2026-05-21_pre_human_label_audit/overnight_workflow_log.md) | Pre-reset log |

---

## Already done

| Artifact | What it is |
|----------|------------|
| `analysis_minority/minority_metrics.md` | **Q6 eval floor:** Pass@1 **9.03%**, Pass@8 **34.40%** [30.2%, 38.4%]; legacy `has_minority_correct_cluster` LLM **14.53%** (25/172) |
| `analysis_minority/minority_readout.py`, composition PNGs | Viz / legacy gate — not minority **set** score |
| `analysis_a/llm_clusters_summary.parquet` | LLM assignments (reuse in E1) |
| `analysis_a/analysis_a_summary.md` | LLM summary under cleaned correctness |
| `dashboard/data.js` | Per-prompt forensics |
| `archive/.../analysis_b/substrate_comparison.md` | **Cheap vs LLM CoT — settled.** Best mean ARI vs LLM: answer-strict **0.19**, Qwen/MiniLM embed **≤0.07**; no cheap substrate viable for `cot-*`. |

**Not doing again:** legacy minority gate as milestone, cheap-clustering / embed proxy, Cover@τ, E2 / `analysis_d/` baseline export (worst-subset), archived Phase 4 `f_poly` only.

---

## Scope

| In | Out |
|----|-----|
| E1 + Phase 1 metrics on 500 prompts | Training runs |
| Train with `*-rand` tie-break; compare `ans-rand` vs `cot-rand` in training | Claim training improves AIME/HMMT |
| LLM CoT clusters for `cot-*` (or ans-only) | Cheap embed/features as CoT substitute |
| Human-label correctness | In-loop LM judge (without approval) |
| | v2 parser narrative |
