# Final Decision — Initial Exploration Plan

This doc is the working artifact for choosing which of the top four directions to commit to. It supersedes open-ended exploration in `01_design_space.md`, `stage2_synthesis.md`, and `better_stage2_synthesis.md` for the purpose of starting initial experiments.

The plan is structured so that **a single short pilot matrix** discriminates between the top contenders and tells us which Tier 2 direction (if any) to escalate next.

---

## Tiering

### Tier 1 — primary path (start here)

Cleanest single-model "majority vs minority" comparison. Causally interpretable, cheap, falsifiable.

| Component | Choice | Role |
|---|---|---|
| Majority/baseline objective | **GRPO** | Reference behavior; majority comparator |
| Minority objective | **`inverse_freq`** (per-prompt normalized inverse cluster-frequency reweighting) | Direct minority-voting instantiation; cheapest implementation; clearest credit-assignment story |
| Closest-neighbor baseline | **F-GRPO** | Mandatory novelty separator — if inverse_freq ≈ F-GRPO, story collapses to "variant" |
| Clustering substrate | **Exact-answer canonicalization** | Cheapest, deterministic, no extra dependency. Defer embedding/LM-judge substrates until minority signal exists |
| Model | Qwen-1.7B-Base | Mentor-locked |
| Data | DaPO subset (~2–5k prompts) | Mentor-locked; subset for pilot speed |
| Eval | Pass@1, Pass@k, Cover@τ, worst-subset accuracy | Pre-registered headline metrics |
| Hard set | Subset of Beyond-AIME or HMMT | Pre-registered OOD generalization probe |

**Hypothesis under test:** under binary RLVR at Qwen-1.7B scale, per-prompt inverse-frequency advantage weighting moves Cover@τ and/or worst-subset accuracy on hard OOD sets vs both GRPO and F-GRPO, without unacceptable Pass@1 regression.

### Tier 2 — secondary, only if Tier 1 produces signal

Each Tier 2 option is gated by a specific Tier 1 outcome (see Decision Rules below).

| Direction | Trigger to escalate | Role |
|---|---|---|
| **`worst_subset`** | Tier 1 shows `inverse_freq ≈ GRPO` (frequency signal inert) | Alternate, sharper minority objective via lower-tail subset risk |
| **`embedder_clustering`** | Tier 1 shows `inverse_freq` beats GRPO AND `Pass@1` regression is from formatting/proxy noise | Substrate ablation: does cheap exact-match recover most of LM-judge's gain? |
| **`dual_head`** | Tier 1 shows both majority and minority objectives produce distinctly useful behavior in separate runs | Architecture story + inference-time interpolation frontier |

### Tier 3 — deprioritized

Do not build initial experiments around these. Revisit only if Tier 1 + Tier 2 fail entirely.

- **`cover_at_tau`** — high PKPO-collision risk, sparse-reward instability.
- **`token_uncertainty`** — closest neighbor (UCAS) is direct prior work; likely reframing.
- **`prompt_curriculum`** — agent's own novelty rating is medium-low; F-GRPO/DAPO already cover similar ground.

---

## Initial runs (the pilot matrix)

All runs share identical scaffolding so comparisons are clean. Differences are isolated to the **objective** only.

### Shared setup (locked across all runs)

- **Model:** Qwen-1.7B-Base
- **Data:** DaPO subset, ~2–5k prompts (pick a fixed slice — same slice across all runs)
- **Training length:** ~100 steps (pilot scale — not full 400-step run)
- **Rollouts per prompt (N):** 8
- **Optimizer / KL / clipping:** standard GRPO settings; identical across runs
- **Seed:** 1 seed for pilot (multi-seed only on Tier 1 winners)
- **Clustering for "answer frequency":** exact-answer canonicalization (normalize whitespace, math form). No embeddings, no LM judge.
- **Verifier:** standard binary RLVR verifier
- **Logging:** training loss, KL, clip fraction, reward variance, per-step cluster counts, per-step gradient mass to rare-correct clusters

### Run 0 — proxy validity check (no training)

**Purpose:** sanity-check that frequency-based minority signal even exists in base-model rollouts before running any training.

- Sample ~500 prompts from DaPO.
- Roll out N=8 from base Qwen-1.7B-Base on each.
- Canonicalize answers (exact match).
- For each prompt, compute: number of distinct answer clusters, frequency-of-correct-cluster distribution, fraction of prompts where ≥1 correct cluster exists AND it is a minority cluster.
- **Decision gate:** if minority-correct clusters appear in non-trivial frequency (e.g., ≥15% of prompts with a non-empty rollout set), proceed with Tier 1. If they barely exist, start with `worst_subset` instead of `inverse_freq` (rarity proxy is dead-on-arrival).
- **Cost:** ~1 GPU-hour.

### Run 1 — GRPO baseline (majority comparator)

- Standard GRPO with the shared setup above.
- No minority weighting.
- **Purpose:** establish pipeline; reference training curves; reference eval numbers.
- **Output artifacts:** checkpoint at step 100, training curves, eval table on Pass@1/Pass@k/Cover@τ/worst-subset for AIME-25 (sanity) + hard OOD subset (Beyond-AIME or HMMT).

### Run 2 — inverse_freq minority

- Same trainer/data/steps as Run 1.
- Only change: multiply per-trajectory advantage by per-prompt-normalized inverse-cluster-frequency weight w_i (see `02_depth_inverse_freq.md` for exact formulation).
- Hyperparameters: γ = 1.0 (default), per-prompt normalization on, weight cap w_max (set conservatively).
- **Purpose:** does minority weighting move tail metrics vs Run 1?

### Run 3 — F-GRPO (closest-neighbor baseline)

- Same trainer/data/steps as Run 1.
- Only change: prompt-level difficulty reweighting (F-GRPO formulation).
- **Purpose:** novelty separator. If Run 2 ≈ Run 3, our minority-by-rarity story collapses to "this is just F-GRPO."

### Total pilot cost

Roughly 3 × short runs + 1 proxy check. Estimated well under 15% of $1,400 budget. Leaves the bulk of budget for the Tier 2 escalation + final headline runs.

---

## Decision rules (pre-registered)

These are the kill/escalate criteria. Decide **before** looking at results which path each outcome triggers — so the pivot is not post-hoc storytelling.

| Pilot outcome | Interpretation | Next step |
|---|---|---|
| Run 2 (inverse_freq) beats Run 1 (GRPO) on Cover@τ or worst-subset, **and** matches or beats Run 3 (F-GRPO), **and** Pass@1 regression ≤ 1–2 pts | Minority signal is real and distinct from F-GRPO. Story is alive. | **Escalate:** scale to 400 steps + 2 seeds. Then add **`embedder_clustering`** substrate ablation as Tier 2. |
| Run 2 ≈ Run 1 (no tail metric movement) | Frequency-based minority signal is inert at this scale. | **Pivot to `worst_subset`**: re-run pilot with worst-subset / CVaR-style minority objective instead of inverse_freq. |
| Run 2 ≈ Run 3 (gains explained by F-GRPO) | Novelty collapses; reweighting is not distinct from prompt-difficulty weighting. | Two options: (a) pivot to **`dual_head`** as architecture-centered story; (b) pivot to **`embedder_clustering`** as substrate-centered story. Choose based on remaining budget and which story is cleaner. |
| Run 2 Pass@1 collapses, tail metrics flat or worse | Frequency proxy is misallocating gradient mass — likely formatting noise inside "exact-match" clusters. | **Try embedder_clustering substrate** to fix the proxy. If that also fails, drop frequency-based minority and consider `worst_subset` or `token_uncertainty`. |
| All three runs ≈ each other on all metrics | Scale/data limitation — methods do not differentiate at 1.7B / ~100 steps / DaPO subset. | Either (a) scale to 400 steps before pivoting (one of Run 1 or Run 2 only), or (b) reframe as negative-result paper with mechanistic diagnostics. Do not silently keep exploring. |

---

## What's still uncertain (worth a low-cost check before/during Run 1–3)

- **Proxy validity** for frequency-based minority — Run 0 above addresses this.
- **Clustering sensitivity** — do exact-match clusters merge or split distinct reasoning chains? Spot-check ~50 prompts manually after Run 0.
- **Eval pipeline correctness** — make sure Cover@τ and worst-subset implementations are audited once and reused identically across Runs 1/2/3. This is the single biggest correctness risk.
- **Hard-set selection** — pick which hard OOD subset is the primary headline metric *now*, not after seeing results. Avoids cherry-picking.
- **`F-GRPO` correctness** — Run 3 only works as a novelty separator if F-GRPO is implemented faithfully. Worth a separate small audit before reading any conclusions from it.

---

## Notes / important context

- **Mentor-fit reminder:** the literal pitch is "instantiate training algorithms for both majority and minority voting and compare." Runs 1 + 2 already satisfy this — we do not *need* `dual_head` to be on-pitch. `dual_head` is a Tier 2 architectural bet, not a requirement.
- **Why not start with `dual_head`:** shared-trunk dynamics introduce gradient-conflict confounds and make causal attribution harder. We do not yet know if a minority signal exists in clean single-model runs; betting on shared-trunk dynamics before that is putting architecture in front of evidence.
- **Why not start with `embedder_clustering`:** it is infrastructure, not a voting objective. Without a working minority objective on top of it, there is nothing to ablate. Once Tier 1 produces signal, it becomes the natural Tier 2 ablation.
- **Why not `cover_at_tau` first:** PKPO is a direct neighbor; sparse-reward instability is real; if it does not beat PKPO the contribution evaporates. Strictly higher-risk than `inverse_freq` for the same milestone slot.
- **Pre-registration matters:** lock the headline metric, hard-set choice, and Pass@1 regression threshold in this doc *before* running. Otherwise any positive result is suspect.
- **Budget discipline:** cap the pilot at ~15% of the $1,400 Modal budget. If any single run exceeds 1.5× its estimated cost, stop and debug, do not extend.
- **Negative results are still publishable:** every Tier 1 outcome above maps to a coherent paper story (positive Cover@τ frontier; "frequency proxy fails, semantic substrate needed"; "minority objectives at this scale do not differentiate from F-GRPO"). The pilot is not lose-able if pre-registered properly.
- **Graceful degradation order:** if everything fails, the salvage order is (1) substrate ablation as standalone paper → (2) head-to-head minority-objective comparison (`inverse_freq` vs `worst_subset`) as standalone paper → (3) pre-registered negative-result paper with mechanistic diagnostics.
- **Do not add more directions during the pilot.** Tier 3 stays Tier 3 unless Tier 1 + Tier 2 are both exhausted.

---

## What to hand to the planning agent

A planning agent reading this doc should produce:

1. A concrete implementation spec for the **shared trainer scaffold** (GRPO + advantage-weighting hook, exact-answer canonicalization, eval harness for Pass@1/Pass@k/Cover@τ/worst-subset).
2. A concrete spec for each of **Run 0, Run 1, Run 2, Run 3** including config files, expected wall-clock, and explicit deliverable artifacts.
3. A list of unit-test-level invariants the implementation must satisfy (e.g., per-prompt weights sum to N after normalization; eval Cover@τ matches a hand-computed reference on a 10-prompt fixture).
4. The exact go/no-go decision rules from this doc, restated as the evaluation script's output checks.

The builder agent that follows should implement strictly to that spec, not re-derive direction.
