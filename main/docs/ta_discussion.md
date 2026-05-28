# TA office-hours agenda

**Date:** 2026-05-28 (poster due 2026-06-03; experiments freeze ~2026-05-31).
**Goal:** Get a TA read on (a) the right paper framing now that the set-based hypothesis is falsified at 1.7B-unfiltered convergence, (b) which of the still-viable experiments (filtered retrain at 1.7B, 4B retrain on filtered Polaris, CoT-clustering arms) is worth the remaining compute and wall-time, and (c) what counts as a passable contribution.

**1-line context:** Late-checkpoint eval landed 2026-05-28 AM (see [`checkpoint_eval_morning_2026-05-28.md`](./checkpoint_eval_morning_2026-05-28.md)): **GRPO wins on every decision-grade slice** at convergence, in both the LR=3e-6 redo family and the LR=1e-6 family. The mid-training "poly_epo +5 pp BeyondAIME" headline from the prior version of this doc is withdrawn (BeyondAIME deemed not arm-rankable at n=100, pass@16 in single digits). The diagnosis (gradient starvation at our model scale) is well-cited and still the strongest paper-grade artifact we have. Question shifts from "what does the paper say about the 1.7B-unfiltered table" to "do we bet remaining compute on filtered / 4B / CoT variants where set-based might still separate."

---

## 1. Headline result (the chart we'd put on the poster)

Three RL arms on Qwen3-1.7B-Base / Polaris-51K, arm-C prompt (`hybrid_answer_boxed`), Rank-2 parser, mathd∨sympy grader, GRPO core (KL=0, N=8, bs=64, 799-step budget). Decision-grade eval at matched train/eval prompt, `n_rollouts=16` (n=8 on the prior mid-training table; doubled here to sharpen).

**Primary table — LR=3e-6 redo family, 3 arms vs base** (`grpo_lr3e6_s59`, `minority_lr3e6_s54`, `poly_epo_lr3e6_s39`):

| Slice | base | GRPO Δ | minority_answer Δ | poly_epo_answer Δ |
|-------|------|--------|-------------------|-------------------|
| Polaris stratified 2k pass@8 (training distribution) | 30.8% | **+2.1 pp** | +0.9 pp | +1.2 pp |
| DAPO 2k pass@8 (easier OOD) | 29.9% | **+2.0 pp** | +2.0 pp | +1.1 pp |
| MATH-500 pass@16 (medium OOD, n=500) | 78.8% | **+4.4 pp** | +2.0 pp | +2.4 pp |

**Confirmation table — LR=1e-6 converged checkpoints** (`grpo_s519`, `minority_s239`, `poly_s133`, no base column re-run): Polaris 31.8% / 31.0% / 31.4%; DAPO 31.7% / 31.3% / 30.6%; MATH-500 81.6% / 80.0% / 78.2%. Same ordering: **GRPO ≥ minority ≥ poly_epo on every decision-grade slice in both LR regimes.**

**The load-bearing claim:** **GRPO wins on every decision-grade slice at convergence.** Set-based reweightings (minority_answer, poly_epo_answer) trail GRPO by 0.5–3.4 pp at the converged LR=1e-6 checkpoints and trail or tie GRPO at the LR=3e-6 redo checkpoints. The mid-training fair-prompt table that previously showed poly_epo winning 3/3 reflected incomplete training plus BeyondAIME pass@16 noise (n=100, SE ≈ 0.04 — the prior +5 pp headline did not survive at n_rollouts=16 across the rest of the panel). **The "set-based RL beats GRPO at 1.7B-unfiltered" hypothesis is falsified at our model scale and data regime.**

**Methodology notes for honesty in the writeup:**
- BeyondAIME (n=100, pass@16) is **dropped** from the decision-grade panel. After re-eval at n_rollouts=16 the prior +5 pp poly_epo lead did not hold; pass@16 in low single digits at n=100 is not arm-rankable. Treat the previous +5 pp number as withdrawn.
- AIME-25 (n=30, pass@16) is shown only as sanity-check (single-digit pass@16 means ±3 pp ≈ 1 problem). Not used for ranking.
- MATH-500 (n=500) substituted as the medium-OOD slice — cleaner statistics, less noise, broadly accepted in the literature.
- LR=3e-6 redo GRPO at s59 ≈ LR=1e-6 GRPO at s519 (slightly better on Polaris/MATH-500). **A side finding** — GRPO compute efficiency, not the paper's hypothesis. Worth a §discussion paragraph.

Full eval pull commands and per-band breakdowns: [`checkpoint_eval_morning_2026-05-28.md`](./checkpoint_eval_morning_2026-05-28.md). Prior delta summary kept for archive value: [`fair_prompt_eval_summary_2026-05-27.md`](../data/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/fair_prompt_eval_summary_2026-05-27.md).

---

## 2. The diagnosis (why the deltas are small)

**Signal starvation at our model scale.** On our 51K Polaris training set, base Qwen3-1.7B-Base passes at exactly 0/8 on **66%** of prompts and at 8/8 on a small fraction; only **34%** of prompts produce non-degenerate centered advantages. That's the regime where vanilla GRPO has near-zero gradient on most batches and where set-based reweightings have nothing to amplify.

- Empirically corroborated by our `random_fullgold_n800` probe (33% mixed-reward yield).
- Matches the published unfiltered GRPO baseline (Nie et al. 2026, [arxiv:2605.07689](https://arxiv.org/abs/2605.07689): 0.69 degeneracy at GS=4 on GSM8K ≈ 31% productive groups).
- Polaris-53K was calibrated by Deepseek-R1-distill-**Qwen-7B**; HKU NLP's own published recipe refilters 53K→~30K specifically for their 4B model via rollout-pass-rate filtering ([Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/), [ChenxinAn-fdu/POLARIS](https://github.com/ChenxinAn-fdu/POLARIS)). We are running unfiltered 51K on a model **2.3× smaller** than the refiltered-recipe's target.
- Live verification: a 200-step LR=3e-6 probe on all three arms (launched 2026-05-27 20:53 PDT) showed **`train/mean_advantage ≈ 0` across all three arms** at step ~25–60. Higher LR amplifies signal but does not create it.

**Why this matters for our specific hypothesis.** Minority_answer and poly_epo_answer are set-based *reweightings* of the GRPO advantage. They have nothing to reweight when most groups in a batch have zero centered advantage. The 34% signal-density regime is where the published literature shows GRPO baselines struggle, and where set-based RL has no upstream signal to differentiate from GRPO. The converged-checkpoint result that GRPO *beats* (not just matches) the set-based arms suggests the failure may be slightly worse than "no signal" — set-based reweightings may be actively de-emphasizing the small amount of useful signal we do get.

Full synthesis with citations and verification flags: [`timeline.md`](./timeline.md) entry *structural diagnosis: model/data mismatch, signal-density benchmark, LR first principles* (2026-05-27 late).

---

## 3. The decision (paper framing first, experiments second)

### Framing options

**Framing X — "Set-based RL beats GRPO."** Headline = the separation number. Requires one of Paths A / C / D to land a positive result. Highest upside, highest risk of empty headline if no separation appears under any regime.

**Framing Y — "Well-motivated hypothesis falsified at this scale; here's the failure mode."** Honest, on-time, $0 additional compute. Risk: reads as "did not engage with the diagnosed problem."

**Framing Z — "Measured the gradient-starvation failure mode at our model scale + tested whether the published fix opens room for set-based RL."** Contribution is the measurement + the controlled comparison, not the headline delta. Survives any of:

- Set-based separates after filtering (or at 4B, or with CoT) → "Set-based RL needed the right regime to show its lift."
- Set-based still ties or loses to GRPO after the fix → "Even with the published fix applied, set-based clustering does not separate — failure is structural at this model class, not signal density alone."
- Filter / scale / CoT meaningfully changes the curves either way → method-paper-grade controlled comparison; the diagnosis chart carries it.

Under Framing Z, the 51K base rollout pass produces a *load-bearing* chart (the pass-rate distribution figure) regardless of what we do downstream. Under Framing X or Y the same pass is a $120–150 hedge.

### Experimental paths — quick comparison

| Path | Cost | Wall | Fits poster window? | Best-case headline | P(separation) |
|---|---|---|---|---|---|
| **A** — Filter + retrain at 1.7B (poly_epo + GRPO) | $400–700 | 18–26 h | Yes, easily | "Set-based RL needs filtered regime to lift" | 20–30% |
| **A+** — Phase 1 only (51K base rollout pass) | $120–150 | 5–8 h | Yes, today | Diagnosis chart (Framing Z load-bearing) | n/a |
| **B** — No more training | $0 | 0 h | Trivially | Framing Y or Z | n/a |
| **C** — Qwen3-4B + model-calibrated filter + retrain | $900–1,100 | 36–48 h | Yes if launched by 2026-05-29 PM | "At Polaris-designed model scale, set-based separates" | 40–55% |
| **D** — CoT-clustering arms at 1.7B (current data) | $900–1,300 | 3–4 days | Tight; needs judge sidecar bring-up to start 2026-05-29 AM | "CoT-clustering finds signal answer-clustering missed" | 30–45% |

P(separation) numbers are honest priors, not measurements — bring them up at the meeting and ask the TA to reweight.

### Path details

**Path A — Filter then retrain at 1.7B.** Reuses staged pipeline ([`plans/option_a_filter_retrain_2026-05-27.md`](./plans/option_a_filter_retrain_2026-05-27.md)). One **1.7B** base rollout pass over 51K → filter → retrain on the surviving subset (size **TBD after Phase 1**; n800 smoke suggests ~⅓ retention). Retrain poly_epo + GRPO (optionally + minority). Filter script tested on n800; **Key concern:** at 1.7B-unfiltered, set-based arms *trail* GRPO at convergence; filtering raises mixed-reward density but may help GRPO at least as much as poly_epo. P(separation) ~20–30%.

**Path A+ — Phase 1 only (1.7B diagnosis pass).** One **1.7B** base rollout pass over 51K, no retrain commitment. Gives the pass-rate distribution figure for the paper and a filtered manifest for Path A (size TBD). Path C would need a **separate 4B rollout pass** on the same manifest — not reusable from A+. Cheapest non-zero-information action; **default regardless of framing** if we want the diagnosis chart. Retrain decision waits on TA input + histogram.

**Path B — No more training.** Take the falsified-hypothesis result as the result; lean on the diagnosis chart + Nie et al. + Polaris-recipe citations for the failure mode; spend the rest of the week on writing + optional diversity panel. Cheapest path to a publishable poster. **Risk:** TA / grader read as "did not engage with the diagnosed problem" — Framing Z partially mitigates by treating the diagnosis itself as the contribution.

**Path C — Qwen3-4B + model-calibrated filter + retrain.** Pivots to the model scale Polaris actually RL-trained (Qwen3-4B-Base; their published 53K→~30K refilter was *for* 4B, not 7B). **Filtering (informed by their recipe, not yet run):**

1. **Offline pass on our 51K manifest** with **Qwen3-4B-Base** (not 1.7B): N=8 rollouts/prompt, production grader, `hybrid_answer_boxed` — same protocol as Path A but the rollout model matches the train model, mirroring how Polaris re-calibrated 7B-difficulty 53K down to ~30K for 4B ([Polaris blog §1](https://hkunlp.github.io/blog/2025/Polaris/)).
2. **Static filter on pass rate:** drop prompts where the 4B base never succeeds (0/8) or always succeeds (8/8) — the degenerate groups GRPO gets no gradient from. Polaris's published 7B ablation shows removing all 8/8-easy prompts helps; cutting everything above 4/8 hurt; they do not publish the exact 4B cutoff, so we pick cutoffs from the Phase 1 pass-rate histogram (default: strict `0 < pass_rate < 1`).
3. **Retrain** poly_epo + GRPO on the filtered manifest (~1 epoch). We are **not** replicating their full recipe (dynamic acc>0.9 drops, rollout rescue, in-batch substitution) in v1 — static filter only.

**Upside:** 4B should land more prompts in the mixed-reward band natively → real signal for set-based reweightings; closest match to published Polaris RL conditions. P(separation) ~40–55%. **Downside:** ~$1K; 4B B200 smoke first (~$20–50); ~2× per-step cost vs 1.7B. **Open question for the TA:** is 1.7B → 4B ex-post acceptable given PLAN named 1.7B?

**Path D — CoT-clustering arms at 1.7B.** Tests whether clustering on CoT trajectories (rather than final-answer sets) finds signal that answer-clustering missed. Independent of the signal-density argument — this is the hypothesis test for *"the clustering mechanism, not the data regime, is the limit."* **Upside:** publishable either way (positive → new contribution; negative → reinforces the diagnosis). **Downside:** judge sidecar is unimplemented per PLAN §5; bring-up risk is real (1–2 days of engineering, gating the rest of the path). **Mitigation worth exploring:** offline CoT labeling on persisted rollouts (cluster post-hoc rather than in-loop) — sidesteps the sidecar entirely if rollouts can be re-emitted. Open question for the TA: does offline clustering meet the spirit of the CoT-clustering hypothesis, or does it have to be in-loop?

### Specific questions for the TA (priority-ordered)

1. **Framing.** Given the set-based hypothesis is falsified at 1.7B-unfiltered convergence, is **Framing Z** ("measured the failure mode + tested the fix") a defensible main contribution at the poster level? Or — if we want a positive separation result in the paper — should we commit to Framing X and bet the remaining ~5 days on **Path A** (filtered retrain at 1.7B), **Path C** (4B + filtered, Polaris-recipe-matched), or **Path D** (CoT-clustering)? Which of those three, in your read, has the highest probability of producing a separation worth defending?
2. **Conceding on 1.7B-unfiltered.** Both set-based arms (minority_answer, poly_epo_answer) underperform GRPO at convergence on every decision-grade slice. Is it acceptable to (a) concede the 1.7B-unfiltered headline straight, then (b) frame at least one of Paths A / C / D as "testing whether the failure is signal-density or structural"? Or does the PLAN-named primary hypothesis (`minority_answer`) need an explicit "rejected" statement in the writeup before we move on?
3. **What's a passable contribution if the headline doesn't separate?** Concretely: if Path A retrain produces poly_epo ≈ GRPO on filtered data, what does the paper need (beyond what we have) to be defensible at the poster level? Any specific RL / alignment papers as models for "well-motivated hypothesis, falsified at this scale, here's the failure mode"?
4. **Diversity panel.** Worth the ~$50 + half-day plumbing to compute cluster diversity for minority_answer (PLAN consolation criterion: `matches GRPO on pass@1 AND improves cluster diversity`)? Or is `pass@k` consistency enough evidence for the poster?

---

## 4. What we want to walk out with

- **Framing decision** (X / Y / Z). Highest-leverage TA input — determines the next 5 days of work.
- **Path decision** (A / A+ / B / C / D, or combination) — conditional on framing. **A+ should likely happen this afternoon regardless;** the question is what (if anything) follows.
- **Decision on whether 1.7B → 4B (Path C) is an acceptable ex-post scope shift,** given the PLAN named 1.7B.
- **Decision on whether offline CoT labeling counts for Path D** (vs requiring an in-loop judge sidecar).
- **Diversity panel: yes/no/conditional.**

Pre-meeting status: late-ckpt eval has landed (see §1); LR=3e-6 probes effectively superseded by the redo runs that produced the morning eval. The previous "tonight plan" is largely executed. Open execution call for this afternoon: launch A+ Phase 1 (51K rollout pass) immediately, or wait for meeting input?

---

## 5. Out of scope for this meeting

- Implementation details (clustering primitives, trainer loop, configs) — not blocked.
- Probe re-runs — frozen.
- Compute/efficiency tuning — at acceptable speed; sleep+gc_off explored and ruled out.

---

## 6. What we tried (paper-writeup history)

This section exists so the TA can see the experiment was real and the failure mode wasn't from negligence on any single axis. It's also the source material for the Methods / Ablations sections of the paper.

**Prompt format ablation (2026-05-25).** A/B/C ablation across three answer-extraction prompts (DAPO-style boxed, hybrid-prefix-boxed, etc). Arm C (`hybrid_answer_boxed`) won on `parse_ok` (56% → 88%). Locked as canonical. Timeline entry: *Afternoon — parser concern → diagnosis → A/B/C ablation*.

**Parser rank (2026-05-25).** Rank-1 vs Rank-2 answer extraction. Rank-2 (last-boxed-then-last-numeric) raised `parse_ok` ~17 pp on hold-out, no regression on parser-easy prompts. Locked as canonical.

**Grader choice (2026-05-26).** mathd-only vs mathd∨sympy. Sympy fallback recovers ~3% of true-positive numeric matches that mathd's symbol-table chokes on. Adopted mathd∨sympy (DeepScaleR / rLLM convention). Timeline entry: *Evening — train grader: mathd OR sympy*.

**Training data: DAPO vs Polaris (2026-05-26).** DAPO-17k vs Polaris-53K head-to-head on a small training run. Polaris won on arm-C numbers; we then prompt-filtered 53K → 51,139 to drop multi-answer / non-final-answer prompts (full filter spec in `timeline.md` 2026-05-27 entry *Polaris prompt filter*). **Locked: Polaris-51K.**

**Batch size sweep (2026-05-26).** bs=128 OOMs on B200 at our seq-length / N=8; bs=64 fits comfortably. Locked.

**GPU class A/B (2026-05-26 → 2026-05-27).** H100 OOM on initial smoke; switched to H200; later B200 won on $/step (smoke ladder green). All production runs on B200.

**FA2 enablement debug (2026-05-26 → 2026-05-27).** FA2 deferred initially due to forward-pass instability with our token_budget packing; re-enabled 2026-05-27 after the underlying issue was narrowed. No measured speedup at our seq-length so we did not chase further.

**Three-arm full B200 training (2026-05-27).** GRPO, minority_answer, poly_epo_answer launched at LR=1e-6, total_steps=799. All three completed mid-training checkpoints (`grpo_s299/359`, `minority_s133/159`, `poly_epo_s133`). Canonical mid-training eval table in [`timeline.md`](./timeline.md) *B200 three-arm checkpoint eval (canonical)*; this is the table whose headline was later superseded by §1 above.

**Canonical eval (2026-05-27).** Three eval slices (Polaris 2k, DAPO 2k, BeyondAIME pass@16, AIME-25 exploratory). Initial eval used `dapo_answer_v1` prompt across all slices → BeyondAIME showed −5 to −6 pp regression for all trained arms. Flagged as suspicious because the trained models were trained on `hybrid_answer_boxed`.

**Fair-prompt rerun (2026-05-27 evening).** Re-ran BeyondAIME pass@16 and DAPO 2k at `hybrid_answer_boxed`. Resolved the BeyondAIME prompt-mismatch artifact, but the rerun itself was at n_rollouts=8 on BeyondAIME — which is what produced the now-withdrawn "+5 pp poly_epo" number that the morning re-eval (n_rollouts=16) failed to reproduce.

**LR=3e-6 probe (2026-05-27 → 2026-05-28).** Three arms × 200 steps at 3× the canonical LR. Mid-probe (step ~25–60): `mean_advantage ≈ 0` on all three arms, no separation. Stopped 2026-05-28 AM; superseded by the LR=3e-6 redo runs that produced the morning canonical eval. Confirms LR is not the constraint at our model scale.

**Late-checkpoint re-eval (landed 2026-05-28 AM).** Re-evaluated the final stopped LR=1e-6 ckpts (`grpo_s527`, `minority_s242`, `poly_s230`) plus the LR=3e-6 redo ckpts on the canonical 3-slice eval (Polaris 2k, DAPO 2k, MATH-500) at n_rollouts=16. **Key result:** GRPO wins on every decision-grade slice in both LR families at convergence; the prior mid-training "+5 pp poly_epo BeyondAIME" did not survive the higher-n_rollouts panel. Full numbers + pull commands in [`checkpoint_eval_morning_2026-05-28.md`](./checkpoint_eval_morning_2026-05-28.md). **This is what flipped the §1 headline.**

**Things we considered and deprioritized originally (some now reconsidered):**

- **Increase training-time N (8 → 16).** Square-root effect on degeneracy escape rate; 2× rollout cost halves effective step count under fixed budget. Net not worth it at training time. (Eval-time N was bumped 8→16 in the morning re-eval — that's a separate decision and the right call for ranking precision.)
- **SFT cold start on Polaris solutions before RL.** No SFT pipeline exists; ≥2 days of bring-up; out of timeline.
- **Switch base model (Qwen3-4B or instruct-tuned variant).** Originally deprioritized because of compute cost and the need to redo every probe. **Now reconsidered as Path C** — the falsification of the 1.7B-unfiltered hypothesis changed the cost-benefit; 4B is now the model scale at which the literature predicts set-based RL should work, and it shares filtering infrastructure with Path A.
- **CoT-clustering arms.** Originally deferred to "after answer-clustering arms land." **Now reconsidered as Path D** — answer-clustering arms have landed and been falsified, which makes CoT-clustering the natural next hypothesis test rather than a follow-on.
- **Diversity panel (PLAN consolation criterion).** Requires re-running rollouts with per-rollout persistence; the existing eval harness writes summaries only. ~$50 + half a day of plumbing. **Pending — would be `pass@k` consistency for poly_epo + cluster-diversity quantification for minority.** TA question 4 above.

---

## Notes from meeting

- [fill in during meeting]

## Action items

- [ ] [Owner — what — by when]
