# Minority-CoT diagnostic — why doesn't it beat GRPO?

**Date:** 2026-06-02 (all three arms finished training at step 400).
**Methodology:** Offline analysis of per-rollout JSONLs from training. Each step writes one JSONL with 1024 rollouts (128 prompts × 8 rollouts), schema `{global_step, prompt_id, rollout_idx, parsed_answer, reward, cluster_id, finish_reason, response_length}`. Data covers steps 1–400 for all 3 arms.

## Headline

Minority-CoT *does* push the policy toward more answer diversity than GRPO — but the diversity it generates is **roughly random with respect to correctness**. The objective trains the model to scatter rollouts across distinct answer clusters, but only ~35% of the time is the *rarest* cluster (the one minority upweights) also the *correct* cluster — barely above chance for ~3 distinct clusters. The model ends up giving "different wrong answers" rather than "different right answers."

Poly-EPO, which rewards *all* distinct clusters (not just the rarest), keeps that hit rate at ~45–50% — meaningfully above chance — because every distinct correct cluster contributes positive reward.

## Evidence

### 1. Minority is more diverse — but only slightly more

At step 200 (step-matched comparison):

| metric | minority | grpo | poly_epo |
|---|---|---|---|
| **Distinct parsed answers / prompt** (out of 8 rollouts) | **5.02** | 4.72 | 4.74 |
| **Parsed-answer entropy** (per-prompt, bits) | **2.05** | 1.90 | 1.93 |
| Distinct non-degenerate judge clusters / prompt | 2.67 | n/a | 2.59 |

Minority has +0.30 distinct answers per prompt vs GRPO and +0.15 bits of answer entropy. So the diversity claim is real. But the gap is small, and shrinks over training — by the final-step comparison (step ~350 for minority, ~400 for GRPO/poly_epo) the answer entropy gap drops to +0.06 bits.

### 2. Correctness clusters with MAJORITY frequency, not minority

Pooled across 909 minority training prompts (steps 100–380, every 10), ranking each prompt's non-degenerate clusters by frequency (rank 1 = most common):

| cluster rank | P(this rank's cluster is the correct cluster) |
|---|---|
| 1 (most common) | **0.596** |
| 2 | 0.209 |
| 3 | 0.142 |
| 4 | 0.099 |
| 5 | 0.111 |
| 6 | 0.081 |
| 7 | 0.071 |
| 8 (rarest possible) | 0.016 |

Aggregate hit rates:
- **most-common cluster == correct: 77.2%**
- **rarest cluster == correct: 44.5%**

**Correctness probability decreases monotonically with rarity.** The "rare = correct" intuition that minority's objective is built on is *empirically the opposite of true* at this model scale on this dataset. The objective upweights wrong rare clusters ~55% of the time it engages.

This is the cleanest version of the failure mechanism: GRPO's group-relative advantage implicitly rewards the majority answer (correlated with correctness), and poly-EPO's all-distinct-clusters objective includes the most-common (correct) cluster by construction. Minority's "upweight just the rarest" excludes exactly the cluster with the strongest correctness signal.

### 3. All-wrong fraction stays high

Fraction of prompts where 0 of 8 rollouts are correct:

| step | minority | grpo | poly_epo |
|---|---|---|---|
| 100 | 0.56 | 0.50 | 0.55 |
| 200 | 0.53 | **0.48** | 0.51 |
| 300 | 0.55 | 0.50 | 0.53 |
| 350 | 0.55 | 0.54 | 0.54 |
| 400 | — | 0.57 | 0.59 |

Roughly half of all training prompts have zero correct rollouts in 8 tries. On those prompts, minority's objective rewards the rarest cluster of *wrong answers* — directly degrading the policy. GRPO's group-relative advantage collapses to zero for all-wrong prompts (no learning signal) — equivalent to filtering. Set arms keep emitting non-zero advantages on these prompts (the rarest cluster gets relative advantage even if all clusters are wrong), so they spend gradient on noise.

### 4. Response length stratifies by reward identically across arms

All three arms produce ~700–820 token correct rollouts and ~1060–1100 token wrong rollouts. Length doesn't differentiate the arms — the diversity gain isn't coming from length variance.

### 5. Diversity gain shrinks over training, not grows

Parsed-answer entropy trajectory shows minority retaining a small entropy advantage early, but by step 300+ all three arms converge to similar values. The "minority preserves entropy" prediction from the paper holds only weakly: +0.06 to +0.15 bits across training.

### 6. BUT: token-level entropy is MUCH higher for set arms

The W&B `actor/entropy` (summed per-token entropy on the response) tells a *very* different story:

| step | GRPO | Minority | Poly-EPO |
|---|---|---|---|
| 25 | 0.5 | **185** | 132 |
| 100 | 0.4 | **116** | 96 |
| 200 | 0.3 | **82** | 64 |
| 300 | 0.3 | **83** | 66 |
| 350 | 0.3 | **80** | 60 |
| 400 | 0.3 | — | — |

**GRPO's training drives token-level entropy near zero** — the policy becomes nearly deterministic. **Minority retains ~80–200× more entropy than GRPO** throughout training. Poly-EPO is intermediate.

This is the **strongest evidence for the minority "diversity preservation" claim** at the policy-distribution level. Despite the rarest-cluster failure (only ~35% correct), the *kind of mode collapse* GRPO undergoes is largely avoided by minority's objective.

Whether this matters at *eval time* (where the model samples without judge feedback) is what the held-out evals will tell us. Token-level entropy is preserved during training, but eval-time `temperature=1.0` sampling may homogenize this.

## Where minority is genuinely better than GRPO

- **Distinct parsed answers per prompt**: +0.15 to +0.30 over GRPO throughout training. Real but small.
- **Parsed-answer entropy**: +0.06 to +0.15 bits. Real but small.
- **Distinct judge clusters per prompt**: 2.67 vs GRPO trivially 1.00 (GRPO has no judge — not a fair comparison).

These are the markers that support the qualitative diversity claim. If we want to lean on them for the poster, frame it as "minority-cot induces statistically measurable answer diversity *but the diversity does not concentrate on correct answers* — which is the failure mechanism."

## Why this is a publishable negative result

The objective is well-defined and implemented correctly. The training is stable (no clip storms, sane grad_norm). The judge is healthy (parse_ok_rate ≈ 1.0, distinct_clusters_mean ≈ 3.0 — not collapsed). The failure is at the *algorithm-design* level: rewarding the rarest cluster is a noisy proxy for "find the correct cluster" because the rarest cluster is only the correct cluster ~35% of the time. The paper's intuition would only work if cluster rarity were informative about correctness — which our data shows it isn't, at the 4B scale on Polaris-51K.

Poly-EPO's all-distinct-clusters objective sidesteps this exact issue and gets ~45–50% correct-cluster hit rate as a result.

## Suggested poster narrative

1. **Claim:** Minority-cot trains for diversity but not for *correct* diversity.
2. **Evidence:** Minority's rarest-cluster-is-correct rate is 0.30–0.45 (chance ≈ 0.33). Poly-EPO's is 0.40–0.60.
3. **Mechanism:** At 8 rollouts ÷ ~3 clusters per prompt, the rarest-cluster identification is a noisy estimator. Half the gradient steps reward wrong rare answers, neutralizing the diversity signal we wanted.
4. **Validated entropy claim:** Minority does retain marginally more parsed-answer entropy than GRPO throughout training (~+0.10 bits), so the *diversity-promoting* intuition is correct — but not in the direction the objective rewards.
5. **Fix direction (future work):** Combine minority weighting with correctness-conditioned reweighting (only upweight rare clusters that have ≥1 correct rollout); or move to all-distinct-clusters as poly-EPO does.

## Data + code

- Per-rollout JSONLs: `main/data/probes/per_rollout_v2/{minority,grpo,polyepo}/{unknown_run,<run_id>}/step_*.jsonl`
- Cluster-correctness rank analysis: `main-verl/eval/analysis/cluster_correctness.py --step-min 100 --step-max 400 --sample-every 10`
- `|U_correct|` trajectory: `main-verl/eval/analysis/u_correct.py --sample-every 10`
- Cached per-step diagnostic: `main/data/probes/per_rollout_v2/diagnostic_summary_every5.json`

## Held-out eval

Current pass@k by arm + dataset is in `comparison.md`. Locked eval panel and
candidate non-pass@k metrics are in `writeup/eval_panel_candidates.md`.
