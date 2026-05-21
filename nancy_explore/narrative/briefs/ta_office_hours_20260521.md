# Office hours brief — Ifdita (2026-05-21)

Short doc for tomorrow: what we tried, what we learned from rollouts, where we paused, and **what we might do next**.

**Also read:** [`../timeline.md`](../timeline.md) · Run 0 dashboard: `pilot/artifacts/run0_proxy/20260519T190202Z/review/index.html`

---

## One-sentence project

We are trying to instantiate **minority voting optimization** (set-RL that improves worst-case / rare-mode performance) vs **majority voting**, on Qwen-1.7B + DaPO, and study generalization on hard reasoning sets — per your pitch and our 2026-05-07 meeting.

---

## Calendar pressure (CS224R)

| Deadline | Date | Our status |
|----------|------|------------|
| **Milestone report** (5%, 1 page) | **5/22/26** | Proposal was **QC-GRPO** (since killed). Post-proposal GPU work = **Run 0 proxy only** + failed matrix (≤1 training step). Need your input on whether that counts as “one experiment” vs we must finish a short training run first. |
| Poster | 6/3/26 | — |
| Final report | 6/8/26 | — |

Milestone must include **AI Tools Disclosure** (we used agents heavily for infra, docs, analysis).

---

## What we’ve tried so far

### 1. Killed directions (before GPU)

- **QC-GRPO** (first proposal): quantile baselines collapse under binary RLVR → abandoned.
- **Poly-EPO scaling / schedule replication**: toy sims show **n≥4 regime change** and **n/N not scale-invariant** — not a defensible center given you co-authored Poly-EPO and our compute ceiling.

### 2. Exploration (no GPU)

- ~12 directions mapped; locked **Tier 1**: “kill the LM judge” + **`inverse_freq`** vs GRPO vs F-GRPO (we now doubt this lineup — see below).
- Exploration also flagged **`dual_head`** and **`worst_subset`** as stronger formalizations; neither was in the first matrix code path.

### 3. First pilot (2026-05-19) — what we actually ran

| Run | Result |
|-----|--------|
| **Run 0** — 500 prompts × 8 rollouts, eval/proxy only (~$16, ~6.4 GPU-hr) | **Completed** — main scientific signal |
| **run1–run3** — GRPO / `inverse_freq` / F-GRPO training (100-step plan) | **No comparison** — ≤1 GRPO step; preemption; no resume; matrix stopped |
| **run1b** — GRPO seed 43 (noise replicate) | Partial; **different prompt batch** than run1/2/3 (seed 42) — not comparable |

**Lessons from this pilot (one issue per bullet — guardrails for the next experiment):**

- **Budgeted from step count, not measured step time** — ~99 min/step × 100 steps × 4 runs ≈ **$1,275** vs ~**$210** intended; always extrapolate from a timed smoke before a matrix.
- **No per-step checkpoint / resume** — Modal preemption → replay step 1; **`raw_predictions.jsonl` wiped** on cold restart.
- **Volume commits only at job end** — mid-run pulls showed stale artifacts; operators couldn’t see real progress.
- **Logging blind spots** — long rollouts with little heartbeat; progress milestone math wrong after OOM patch (`done % 25` with micro-batch 8).
- **Blocking Modal client** — first Run 0 killed when client aborted; use **`--detach`** + pull logs separately.
- **Launched training matrix before Run 0 was analyzed** — spent on objectives before we knew the substrate story.
- **`inverse_freq` is not set-RL** — per-rollout reweighting in code, not subset enumeration / marginal set advantages like Poly-EPO.
- **No explicit majority-voting training objective** — only GRPO mean baseline; mentor pitch asks for both algorithms instantiated.
- **Cross-run comparability** — run1b used seed 43 vs 42 elsewhere; **never compare rewards across unmatched prompt sets**.
- **`canonicalize_answer` broken** — strips `}`; splits `"12"` vs `"\\( 12 \\)"` into different clusters; fix before any cluster-based objective.
- **Train vs eval token cap mismatch** — `execute.py` clamped rollouts/eval to **1024** while training config said **2048**; distorts length/completion stats.
- **Parser strictness** — `is_correct` requires exactly one shallow `\boxed{}`; many numerically right completions graded wrong.
- **Detached launch log timeout** — spawn succeeded but local logs truncated; easy to think a job failed when it was still running.
- **Mechanism tripwire** — checks formula on “minority-correct” prompts; with 0% such prompts it **vacuously passes** (not a science check).

### 4. Run 0 — what we learned from rollouts

**Setup:** Qwen3-1.7B-Base, DaPO rows 0–499, 8 samples/prompt, temperature 1.0, proxy only (no weight updates).

**Analysis mistake (one line):** We initially designed gates around **multiple correct answer-clusters per prompt**. Clusters are **only** canonicalized final answers; under RLVR there is **exactly one correct answer string** per prompt — so that gate was the wrong object. We are not treating 0% on that metric as proof the project is dead.

**Run 0 analysis only:** parsed answers + exact-string completion stats (length, distinct full text). **Not done:** LM reasoning judge, CoT/strategy clustering, embedding clusters.

#### Issues / pitfalls (substrate, parsing, measurement)

- **Answer-only clustering cannot see trace-level differences in cluster ID** — same gold answer ⇒ one correct cluster.
- **~49% of rollouts** lack a single clean shallow `\boxed{}` — low accuracy mixes model failure with **format** failure.
- **~3%** stored parse ≠ re-extract on audit; cleaned pass helped slightly (+6 correct) but **did not** change the clustering story.
- **Run-on / truncated completions** (~11% flagged in clean pass) — long completions without a scorable tail.
- **`inverse_freq` on this substrate** — ~**83%** of rollouts are **singleton** answer-clusters → weights mostly tie; signal is “rare wrong string,” not rare correct answer modes.
- **1024-token effective cap** on Run 0 execute path — may understate long-completion behavior on hard math.
- **Stored vs clean labels** — dashboard has RAW/CLEAN toggle; any future gate must say which pipeline it uses.

#### Interesting results (still useful)

- **~8%** rollout accuracy; **~33%** of prompts have ≥1 correct rollout (8.1% any-correct @ 8 samples).
- **500/500** prompts: all **8 completions distinct** (full traces differ).
- **82/82** prompts with ≥2 correct rollouts: **different completions**, same parsed correct answer (diversity lives in text, not cluster ID).
- **Wrong-answer diversity is high** — ~7 distinct wrong modes/prompt on average; among prompts with any correct, **0** have multiple distinct **correct** answer modes (expected given single gold answer).
- **Cheap proxy is affordable** — ~$16 for 4k rollouts; substrate diagnostics can be run often before training burns.
- **Review tooling** — HTML dashboard + exemplar prompt IDs for “same answer, different trace” spot-checks during meetings.

### 5. Where we are today (2026-05-20)

We **redesigned the pilot** after the first matrix failure (caps, checkpointing, smoke gate, logging) and **partially implemented** that in code. We **started smoke launches** on Modal (detached; local artifacts not fully pulled).

Then we **stepped back**: the rebuilt plan still centered **`inverse_freq`**, but rollout analysis + [`pilot_strategy_20260520.md`](pilot_strategy_20260520.md) made us ask **what experiment we are actually running** — judge ablation vs minority objective vs something else — and whether the mentor’s verbal pitch even maps to one loss function.

**Current state:** infra improvements exist and may be reusable, but **no new training comparison**; **matrix on hold** until we align on research question and objective with you (and possibly rewrite the trainer for **set-RL**, not per-rollout scalers).

---

## What we’re currently considering (main section)

We think the project split into **two orthogonal questions**. The first pilot tried to answer both with **`inverse_freq`**, which belongs to neither cleanly.

| | Question | In plain language |
|---|----------|-------------------|
| **I** | **Is the LM judge load-bearing?** | Poly-EPO clusters **reasoning** with a strong model. We used **exact-match on final answers**. Does cheap clustering still get Poly-EPO-like gains, or was the judge essential? |
| **II** | **Does minority-style set-RL beat majority-style set-RL?** | Same data, same model, same substrate — only the **set objective** changes (worst-case tail vs mean/diversity vs standard GRPO). |

**`inverse_freq` (what we coded):** multiply GRPO advantages by **1 / (answer-cluster size)** within each prompt. On Run 0 this mostly upweights **singleton wrong answers** (large negative advantage), not “reward the minority correct vote.” Structurally **per-rollout**, not set-RL — close to GRPO+DIV reweighting in your paper, not a new set aggregator.

Below: **candidates**, what each would mean, and a **sample phased plan** if we committed (all assume fixed parser, matched seeds, smoke-timed budget, reuse redesign infra where it still fits).

---

### Option A — Headline **Question I** (“kill the LM judge”)

**Core idea:** Replicate Poly-EPO’s **set-RL** (`f_poly = mean reward × diversity`) but swap the LM judge for **cheap exact-match answer clusters** (Stage 1) or a stronger substrate later (embeddings / n-gram / small LM).

**What we’d compare:**

- **GRPO** — per-rollout mean baseline (not literal majority voting, but standard RLVR control).
- **Cheap Poly-EPO** — implement real subset sampling (e.g. n=4 from N=8), marginal set advantages, `f_poly` on answer clusters.

**Sample plan:**

1. **Phase 0 — Substrate + infra** — Fix parser/canonicalize; smoke timed step; one seed; confirm checkpoint/resume.
2. **Phase 1 — Minimal comparison (~2 runs)** — GRPO vs cheap Poly-EPO, DaPO 3k slice, ~25 steps, same rollouts/prompt. Mechanism: does diversity term correlate with cluster spread?
3. **Phase 2 — Readout** — AIME-25 + HMMT Nov (integer + symbolic where needed); Pass@k; optional 64-sample **qual** CoT review (human or LM — not done in Run 0).
4. **Phase 3 (if signal)** — Substrate sweep: exact match vs embedder vs LM-judge on **one** winning objective; then Stage 2 scale (17k, 400 steps).

**Milestone story:** “Poly-EPO without the judge” — negative result is still publishable if judge is load-bearing.

---

### Option B — Headline **Question II** (minority vs majority **set objective**)

**Core idea:** Hold substrate fixed (cheap clusters Stage 1), implement **set-level** objectives that differ only in how subsets of n rollouts are scored.

**Candidates for `f_minority`:**

- **`worst_subset` / CVaR** — For each prompt, sample subsets of size n from N rollouts; score subset by mean correctness (or minority-vote indicator); optimize **lower tail** (worst subset or bottom-quantile average). Matches “improve **worst-case** performance” in your pitch; math in `../agents/outputs/depth/02_depth_worst_subset.md`. **Risk:** without diversity term, may over-penalize hard prompts or collapse Pass@1; need comparison to Poly-EPO/PKPO-style tail metrics.

- **Smallest-cluster reward** — Within each subset, upweight rollouts in the **rarest answer cluster** (or reward if the smallest cluster is correct). Closer to “less popular answer” wording. **Risk:** on answer-only substrate, rare cluster often means **rare wrong string**; needs derivation before we trust gradients.

- **Cheap `f_poly` as “majority-side” set-RL** — Not minority voting, but the **paired baseline** for Q II: diversity × mean reward at set level (your paper’s structure, cheap clusters).

**What we’d compare (3 arms):**

- GRPO (per-rollout baseline)
- Cheap **Poly-EPO** (`f_poly`) — majority-leaning set-RL with diversity
- Cheap **worst-subset** (or your chosen `f_minority`)

**Sample plan:**

1. **Phase 0 — Implement set-RL trainer** — Subset enumeration, marginal advantages, shared clip/KL; drop `inverse_freq` unless you explicitly want it as a fourth ablation.
2. **Phase 1 — Three-run matrix** — Matched prompts/seeds; ~25 steps; mechanism logs (tail gradient mass, cluster stats).
3. **Phase 2 — Eval** — Tail metrics: worst-subset accuracy, Cover@τ; Pass@1 regression cap; hard sets (Beyond-AIME, HMMT).
4. **Phase 3** — Promote winner to 400-step / 17k; add **majority@k**-style eval if you want literal voting readout at inference.

**Milestone story:** “Which set aggregator matches minority voting?” — needs your sign-off on **worst-subset vs smallest-cluster**.

---

### Option C — **`dual_head`** (both algorithms in one model)

**Core idea:** Shared trunk; **majority head** trained with GRPO / Pass@1-style signal; **minority head** with tail or rare-cluster signal; interpolate at inference. Only exploration direction that literally trains **both** methods in one run (`better_stage2_synthesis.md`).

**Sample plan:**

1. **Phase 0** — Architecture + loss wiring (two heads, shared rollouts).
2. **Phase 1** — Single run vs separate-run matrix (higher eng risk, lower run count).
3. **Phase 2** — Pareto sweep inference interpolation; eval tail + Pass@1.
4. **Phase 3** — Scale only if Phase 2 beats separate-run baseline on compute-adjusted terms.

**Tradeoff:** Best match to “instantiate both algorithms,” but **most implementation scope**; probably not for 5/22 unless milestone frames it as design + Run 0.

---

### Option D — **Substrate-first** (defer objective fight)

**Core idea:** Run 0 showed answer clustering is cheap but only sees final answers (completion strings differ, but we never clustered them). Before set-RL spend, ablate **clustering substrate** on the same 500×8 harness: exact match vs embedding vs LM judge vs n-gram fingerprint.

**Sample plan:**

1. **Phase 1** — 4 proxy passes (~$15–20 each) — cluster statistics, stability, cost.
2. **Phase 2** — Pick substrate; **then** choose Q I or Q II with mentor.
3. **Phase 3** — Training matrix with chosen substrate + objective.

**Milestone story:** “What is the right cluster definition for minority voting on reasoning models?”

---

### What we’re **not** planning to run as-is

- **`inverse_freq` matrix** — unless you explicitly want it as a **negative control** (expected ≈ GRPO+DIV).
- **Old pre-registered gate** (15% minority-correct on answer clusters) — see footnote below.

**Related work anchors (if narrative shifts):** **PKPO** (Pass@k / tail), **SetPO** (set diversity, not minority vote), **F-GRPO** (hard-prompt reweighting — we had this as run3, not a majority baseline).

---

## If minority voting doesn’t work — what we’d still claim

1. **Substrate:** LM judge vs cheap clustering — judge load-bearing or not.
2. **Objective:** head-to-head set aggregators with honest eval harness.
3. **Mechanistic null:** answer-level frequency ≠ minority voting; format/parser confounds; distinct completion strings exist but aren’t used for clustering.

---

## Questions for you tomorrow

### A. Research direction

1. **Milestone headline — Q I, Q II, substrate-first (D), or dual_head (C)?**
2. **Formalize minority voting:** worst-subset / CVaR, smallest-cluster reward, or something else from your whiteboard?
3. **Is GRPO enough as “majority” baseline**, or do you want explicit majority-voting training + Majority@k eval?
4. **CoT / reasoning clustering:** required for Stage 1, or answer-level OK for a first training pass with qual CoT eval later?
5. **`dual_head` vs separate-run matrix** — worth the eng cost for this quarter?

### B. Milestone (5/22)

6. Is **Run 0 + failed matrix postmortem + redesign + this rethink** enough as “one experiment since proposal”?
7. Which **single** option (A/B/C/D) would you fund for the **next** ~$150 GPU burst?

### C. Quick ops

8. Minimum rollouts (8 vs 16) and train slice (3k vs 17k) you expect before a 400-step Stage 2?

---

## Compute & team ops

- ~**$1,400** Modal total; **personal workspaces**; artifacts via pull + HF Hub / drive. wandb: **`cs224r-minority-voting`**.
- **We will not launch a training matrix** until direction is aligned; infra-only smoke is optional to validate resume/checkpointing.

---

## References

| Topic | Path |
|-------|------|
| Timeline | [`../timeline.md`](../timeline.md) |
| Mentor meeting | [`../../reference/mentor_meeting_20260507.md`](../../reference/mentor_meeting_20260507.md) |
| Run 0 handoff | [`../../../pilot/artifacts/run0_proxy/20260519T190202Z/RUN0_HANDOFF_FOR_REVIEW.md`](../../../pilot/artifacts/run0_proxy/20260519T190202Z/RUN0_HANDOFF_FOR_REVIEW.md) |
| Strategic critique | [`pilot_strategy_20260520.md`](pilot_strategy_20260520.md) |
| Poly-EPO vs pilot | [`../../../pilot/docs/analysis/0519_poly_epo_methodology.md`](../../../pilot/docs/analysis/0519_poly_epo_methodology.md) |
| Worst-subset depth | [`../../agents/outputs/depth/02_depth_worst_subset.md`](../../agents/outputs/depth/02_depth_worst_subset.md) |

---

## Footnote — old frozen eval & gate (likely abandoned)

*Relic of the pre-0520 plan; kept for traceability only.*

We pre-registered `pilot/preflight_lock.json` + `pilot/eval/gate.py`: Cover@τ (τ=0.15), worst-subset **prompt** accuracy, Pass@k (k=16 in lock vs 8 rollouts in configs), bootstrap CIs, tier-1 gate on 60 prompts (AIME-25 Nov + HMMT Nov). Auto `gate_decision.json` said **`PIVOT_WORST_SUBSET`** after Run 0’s 0% minority-correct gate. We are **not** executing that decision tree unless you want to revive it after tomorrow’s alignment.
