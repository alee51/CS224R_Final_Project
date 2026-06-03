# Minority-CoT diagnostic — why doesn't it beat GRPO?

**Date:** 2026-06-02 (all three arms finished training at step 400).
**Methodology:** Offline analysis of per-rollout JSONLs from training. Each step writes one JSONL with 1024 rollouts (128 prompts × 8 rollouts), schema `{global_step, prompt_id, rollout_idx, parsed_answer, reward, cluster_id, finish_reason, response_length}`. Data covers steps 1–400 for all 3 arms.

## Headline

Minority-CoT *does* push the policy toward slightly more answer diversity than GRPO, but the rarest-cluster identifier minority upweights is **uncorrelated with correctness**. In the regime where the question is well-posed — a unique (untied) smallest cluster on a prompt where geometry permits the correct cluster to be smallest — P(rarest = correct) sits at chance for ~3 clusters. The objective's gradient signal at the rarest cluster is approximately random with respect to whether that cluster is right.

Poly-EPO sidesteps this by rewarding *all* distinct clusters, so any correct cluster contributes positive reward regardless of its frequency rank.

## Evidence

### 1. Minority is more diverse — but only slightly more

At step 200 (step-matched comparison):

| metric | minority | grpo | poly_epo |
|---|---|---|---|
| **Distinct parsed answers / prompt** (out of 8 rollouts) | **5.02** | 4.72 | 4.74 |
| **Parsed-answer entropy** (per-prompt, bits) | **2.05** | 1.90 | 1.93 |
| Distinct non-degenerate judge clusters / prompt | 2.67 | n/a | 2.59 |

Minority has +0.30 distinct answers per prompt vs GRPO and +0.15 bits of answer entropy. So the diversity claim is real. But the gap is small, and shrinks over training — by the final-step comparison (step ~350 for minority, ~400 for GRPO/poly_epo) the answer entropy gap drops to +0.06 bits.

### 2. Rarest cluster is uncorrelated with correctness in the well-posed regime

Cluster sizes per prompt are heavily skewed toward singletons: many prompts have multiple clusters tied at the smallest size. Two stratifications are needed for a fair measurement:

**(a) Unique vs tied rarest.** Restrict to prompts where exactly one cluster has the strictly smallest size, so "rarest" is a single cluster rather than a tied set:

|  | unique rarest = correct |
|---|---|
| minority | 58/354 = **0.164** |
| polyepo | 66/368 = **0.179** |

**(b) Geometry by n_correct.** At n_correct ≥ 5 (5+ of 8 rollouts correct), the correct cluster *cannot* be the unique-smallest cluster by arithmetic — the correct mass is too large to fit in a strictly smallest cluster while every other cluster is strictly larger. Those rows are geometric zeros, not measurements.

The well-posed slice is unique-rarest with n_correct ≤ 2:

| n_correct | minority | polyepo |
|---|---|---|
| 1 | 32/98 = **0.327** | 33/89 = **0.371** |
| 2 | 16/51 = 0.314 | 23/57 = 0.404 |

Chance for ~3 clusters ≈ 0.333. So when there's a real rarest cluster and the problem is hard enough for it to plausibly be the correct one, **rarity carries no signal about correctness**. The minority objective is rewarding the rarest cluster approximately at random.

CoT clustering itself is informative — 37% (minority) / 34% (polyepo) of size≥2 clusters are mixed (same reasoning chain, divergent final answers), so the judge is genuinely separating rollouts by reasoning structure. The clustering just doesn't separate clusters along a rare↔correct axis.

Poly-EPO's all-distinct-clusters objective is unaffected by this: it rewards every distinct cluster, so any correct cluster contributes positive reward regardless of frequency.

Full analysis: `main-verl/writeup/results/cluster_correctness.md`.

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

### 6. Minority retains more token-level entropy than poly-EPO

W&B `actor/entropy` (summed per-token entropy on the response), comparing the two set arms:

| step | Minority | Poly-EPO |
|---|---|---|
| 25 | **185** | 132 |
| 100 | **116** | 96 |
| 200 | **82** | 64 |
| 300 | **83** | 66 |
| 350 | **80** | 60 |

Minority's token-level entropy runs ~25–40% above poly-EPO throughout training. Since both arms share the same advantage structure modulo the cluster reweighting coefficient, this difference is attributable to minority's rare-cluster upweighting amplifying the policy distribution's per-token uncertainty.

Whether this carries to *eval time* (where the model samples without judge feedback) is what the held-out evals will tell us — `temperature=1.0` sampling may homogenize this.

## Where minority is genuinely better than GRPO

- **Distinct parsed answers per prompt**: +0.15 to +0.30 over GRPO throughout training. Real but small.
- **Parsed-answer entropy**: +0.06 to +0.15 bits. Real but small.

These are the markers that support the qualitative diversity claim. If we want to lean on them for the poster, frame it as "minority-cot induces statistically measurable answer diversity *but the diversity does not concentrate on correct answers* — which is the failure mechanism."

## Why this is a publishable negative result

The objective is well-defined and implemented correctly. The training is stable (no clip storms, sane grad_norm). The judge is healthy (parse_ok_rate ≈ 1.0, distinct_clusters_mean ≈ 3.0 — not collapsed). The failure is at the *algorithm-design* level: rewarding the rarest cluster assumes cluster rarity carries information about correctness. At 4B scale on Polaris-51K, in the regime where the question is well-posed, it doesn't — rarity is uncorrelated with correctness, so the minority gradient signal is approximately random.

Poly-EPO sidesteps this by rewarding *every* distinct cluster: any correct cluster contributes positive reward regardless of its rank in the frequency distribution.

## Suggested poster narrative

1. **Claim:** Minority-cot trains for diversity but not for *correct* diversity.
2. **Evidence:** On hard prompts where a unique-rarest cluster could plausibly be the correct one (n_correct ≤ 2 of 8), P(rarest = correct) is 0.31–0.40 — indistinguishable from the ~0.33 chance rate for ~3 clusters. Rarity is uncorrelated with correctness.
3. **Mechanism:** The rarest-cluster identifier carries no information about which cluster is correct, so minority's gradient signal at the rarest cluster is approximately random. The objective is correctly implemented; its underlying assumption (rare → correct) doesn't hold at this scale.
4. **Validated entropy claim:** Minority does retain marginally more parsed-answer entropy than GRPO throughout training (~+0.10 bits) and ~25–40% more W&B `actor/entropy` than poly-EPO, so the *diversity-promoting* intuition is correct — but the diversity it produces is not concentrated on correct answers.
5. **Fix direction (future work):** Combine minority weighting with correctness-conditioned reweighting (only upweight rare clusters that have ≥1 correct rollout); or move to all-distinct-clusters as poly-EPO does.

## Data + code

- Per-rollout JSONLs: `main/data/probes/per_rollout_v2/{minority,grpo,polyepo}/{unknown_run,<run_id>}/step_*.jsonl`
- Cluster-correctness rank analysis: `main-verl/eval/analysis/training/cluster_correctness.py --step-min 100 --step-max 400 --sample-every 10`
- `|U_correct|` trajectory: `main-verl/eval/analysis/training/u_correct.py --sample-every 10`
- Cached per-step diagnostic: `main/data/probes/per_rollout_v2/diagnostic_summary_every5.json`

## Held-out eval

Current pass@k by arm + dataset is in `main-verl/writeup/results/comparison.md`.
Locked eval spec (datasets, metrics) is in `main-verl/writeup/eval.md`; run
plan in `main-verl/writeup/eval_build.md`.
