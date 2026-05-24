# Run 0 cleaned-label signal investigation

**Artifact:** `20260519T190202Z` · **Run:** `run0_proxy` (substrate validity, not training)  
**Inputs:** `cleaned/predictions.jsonl`, `cleaned/metrics.json`, prior `analysis_v2_*.md`, `delta_vs_raw.md`  
**Recompute:** `cleaned/signal_compute.py` → `cleaned/signal_stats.json`

## Executive summary

Cleaning (`answer_clean.py`) **improves measurement** (fewer garbage parses, +6 rollout-level correct via normalization) but **does not change the scientific conclusion** for Run 0’s core question: whether exact-match clustering supports **minority-weighted set RL** when correct answers occasionally land in **distinct canonical clusters**.

After clean labels, **`minority_correct_prompt_rate` remains 0.0%** (0/500). Among **165** prompts with ≥1 clean-correct rollout in this artifact’s `answer_clean` pass (human-verified labels → **172** eligible; see `nancy_explore/run0_analysis/analysis_minority/minority_metrics.md`) and high wrong-answer diversity, **zero** prompts have ≥2 distinct **correct** answer modes (semantic bucket or `canon_clean`). The substrate shows **high wrong-mode diversity**, not **multi-cluster correct minorities**.

---

## 1. Run 0 anchor (what we are testing)

Per `pilot/docs/operations/RUNBOOK.md`, Run 0 checks whether cheap exact-match clustering on `Qwen3-1.7B-Base` rollouts (500 prompts × 8) yields:

1. **Within-prompt diversity** (many distinct wrong parses / clusters).
2. **Occasional correct answers in distinct clusters** on the same prompt — the substrate for `inverse_freq` / Poly-EPO-style minority upweighting without an LM judge.

Original gate: `minority_correct_prompt_rate` ≥ 15%. Team has deprioritized gating, but the **research question** persists: is this clustering substrate viable for “kill the LM judge”?

---

## 2. Clean vs stored correctness

| Metric | Stored (raw) | Clean (`answer_clean`) | Δ |
|--------|-------------:|----------------------:|---:|
| Rollout correct | 324/4000 (**8.10%**) | 330/4000 (**8.25%**) | +6 |
| Prompts with ≥1 correct | 163/500 (**32.6%**) | 165/500 (**33.0%**) (`answer_clean`; human labels **172/500**) | +2 prompts |
| `minority_correct_prompt_rate` | **0.0%** | **0.0%** | — |

**Per-prompt correct-rollout distribution** shifts slightly (e.g. 81→83 prompts with exactly 1 correct); no prompt reaches 8/8.

Cleaning **does not** materially change headline accuracy; it is not a path to “more correct mass” at scale.

---

## 3. Minority-correct clusters (recomputed)

Definition (`pilot/train/run_proxy.py`): among rollouts marked correct, correct answers must fall in **≥2 clusters** with at least one correct cluster strictly below majority frequency among correct rollouts.

| Labeling | Prompts with ≥1 correct | `has_minority_correct_cluster` |
|----------|------------------------:|-------------------------------:|
| Stored `correct` + `cluster_id` | 163 | **0** |
| `correct_clean` + `cluster_id_clean` | 165 | **0** |
| `correct_clean` + `canon_clean` groups | 165 | **0** |

**Among 82 prompts with ≥2 clean-correct rollouts:** every correct rollout shares **one** `cluster_id_clean` and **one** `canon_clean` (100% single correct cluster).

**Interpretation:** When the model is right more than once on a prompt, it almost always repeats the **same formatted integer** (same cluster). There is **no** “right answer hiding in a rare cluster” pattern for minority weighting to exploit.

---

## 4. Semantic bucket diversity (clean parses)

Buckets use `semantic_bucket_clean(parsed)` from `answer_clean.py` (int → `n:`, frac → `frac:`, else `s:…`), excluding `empty`.

| Stat | Stored raw (v2 completion-aware) | Clean parse buckets |
|------|----------------------------------|---------------------|
| Mean distinct buckets / prompt | 7.09 | **6.30** |
| Median | 8.0 | **6.0** |
| Prompts with >1 bucket | 500 (100%) | **500 (100%)** |

Clean labels **reduce** apparent diversity vs v2 (run-on rejection empties 426 tails; clusters merge slightly: mean 7.18 → 6.86 distinct `cluster_id_clean`).

### Actionable subsets

| Subset | Rule | Count | Notes |
|--------|------|------:|-------|
| **A. High diversity + any correct** | ≥1 `correct_clean` AND ≥2 semantic buckets (all rollouts) | **165** | Diversity is **wrong-answer modes**; correct sits in one bucket |
| **B. Multi-mode correct (RL target)** | ≥1 `correct_clean` AND ≥2 buckets **among correct only** | **0** | No minority-correct substrate |
| **C. Multi-canon correct** | ≥1 `correct_clean` AND ≥2 `canon_clean` among correct | **0** | Same |

**Example (subset A, not B):** `cfc7b48f` — gold `50`; 2/8 clean-correct (`n:50`); other buckets include `n:16`, `frac:16/π`, garbage strings. Correct mass is **one** cluster; diversity does not split correct answers.

**Example (many correct, still one cluster):** `ddd26788` — 7/8 stored/clean correct, all `n:2` / same cluster.

---

## 5. `extract_path_clean` distribution

| Path | Rollouts | % |
|------|----------:|--:|
| `boxed_balanced` | 2016 | 50.4% |
| `answer_line` | 853 | 21.3% |
| `last_line` | 705 | 17.6% |
| `runon_rejected` | 426 | 10.7% |

- **Run-on rejection (10.7%):** drops long prose/code tails from clustering — **fair** per manual review (`manual_review_weird_cases.md`); reduces spurious cluster IDs, not unlock minority correct.
- **Brace-balanced boxed (50.4%):** fixes nested-`\boxed` truncation vs shallow regex (88 rollouts flagged nested mismatch); mostly **wrong** answers still wrong after relabel.

---

## 6. Raw vs clean delta — what changes the story?

### 6.1 Correctness flips (6 rollouts, 5 prompts)

**All six:** `parsed_answer == parsed_answer_clean`; flip is **`is_correct_clean`** (normalized canon equality) vs stored strict boxed-int `correct`.

| `prompt_id` | Mechanism | Science impact |
|-------------|-----------|----------------|
| `cfc7b48f`, `65da7224` | `\( 50 \)` → canon `50` | **Measurement only** — merges format variants; correct cluster already unified |
| `22063de2` (×2) | `20%` / `$20\%$` → canon `20` | **Measurement only** — percent forms |
| `2e690d58` | `\(100\)` → `100` | **Measurement only** |
| `70aabfd8` | `1250\%` → `1250` | **False positive risk** — likely wrong math marked correct via percent peel |

**Story:** Flips **do not** reveal hidden multi-cluster correct minorities; they fix **grading unfairness** on a handful of format variants.

### 6.2 Parse changes (514 rollouts, 12.85%)

| Category | Count | Science impact |
|----------|------:|----------------|
| `runon_rejected` (empty parse) | 426 | **Measurement** — removes garbage clusters; qual review: fair |
| `boxed_balanced` (parse text change) | 88 | **Measurement** — fixes truncated `\frac{…}`; rarely changes correctness |
| Other paths | 0 | — |

**100% of parse-only changes** leave `correct` unchanged. Cluster partitions change on **500/500 prompts** (hash + canon), but **minority-correct structure stays absent**.

### 6.3 What would *not* change the story (noise)

- Hash vs SHA-256 cluster IDs, run-on counts without correct-cluster linkage, restating ~8% accuracy alone.

---

## 7. Qualitative synthesis — guidance before Run 1–3

1. **Do not expect `inverse_freq` signal from exact-match clusters on this base model slice** — minority-correct prompts are **structurally 0%** after clean relabel, not a parser artifact.
2. **High cluster/bucket counts are misleading** — median ~6–7 modes/prompt are dominated by **wrong integers, code garbage, and format splits among incorrect rollouts**; qual v2 (`01677f18`, `56a368fe`) shows 8 clusters ≠ 8 reasoning strategies.
3. **Cleaning is still worth deploying for Run 1+ metrics** — run-on rejection and balanced boxed improve **cluster interpretability** and fairer correct counts (+0.15 pp rollouts), but treat as **hygiene**, not a new experimental axis.
4. **Adopt clean `is_correct_clean` + `cluster_id_clean` for offline analysis** — but align training prompt contract (require one `\boxed{integer}`) or training will not match eval.
5. **If the program stays “kill the LM judge” via clustering:** need a **different substrate** (embedding clusters, semantic canon, or model that produces multiple correct *representations*) — not more parser tweaks on Run 0.
6. **If minority structure is required:** consider **worst-subset / Cover@τ** pivot (`PIVOT_WORST_SUBSET` in RUNBOOK) or **token-surprise minority** (orthogonal to answer-frequency clusters) — Run 0 does not validate answer-cluster minority voting here.
7. **Optional cheap check before spend:** on prompts with ≥2 clean-correct rollouts (n=82), manually inspect whether any second “correct” is a **canon false positive** (e.g. `70aabfd8`); even if a few exist, rate is far below 15% gate.
8. **Run 1–3 are not blocked on parser** for the *minority-cluster hypothesis* — they test training objectives, but **substrate validity for that hypothesis is already negative** on this rollout distribution.

---

## 8. Verdict

### `SUBSTRATE_NOT_VIABLE`

**For Run 0’s stated intent** — cheap exact-match clustering supporting minority-weighted set RL without an LM judge:

| Criterion | Result |
|-----------|--------|
| Within-prompt diversity | **Present** (wrong-answer modes) |
| Correct in distinct clusters (minority correct) | **Absent** (0/500 prompts, clean or stored) |
| Parser fix unlocks minority structure | **No** (0 multi-mode correct prompts) |

Cleaning moves the verdict from “maybe parser-limited” to **“structurally absent on this model + slice.”** Proceeding with Run 1–3 as a test of **inverse_freq on exact-match clusters** would optimize a signal that **does not exist** in Run 0 rollouts; parser work remains useful for **fair measurement**, not for resurrecting minority-cluster viability.

**Not recommended:** `FIX_PARSER_THEN_REASSESS` as primary path for the minority-cluster hypothesis — reassessment already done; further parser edits should be scoped to **training/eval alignment**, not expecting minority-correct rates to jump.

---

## Appendix: reproducibility

```bash
python pilot/artifacts/run0_proxy/20260519T190202Z/cleaned/signal_compute.py
```

Key outputs in `cleaned/signal_stats.json`.
