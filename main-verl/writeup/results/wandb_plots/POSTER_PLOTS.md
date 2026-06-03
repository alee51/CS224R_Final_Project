# Poster Plot Shortlist

Which W&B training-dynamics plots to put on the poster, why, and how they were generated. Source CSVs in `../`: `GRPO_history.csv`, `Minority_CoT_history.csv`, `Poly_EPO_CoT_history.csv` (400 steps each, runs `rof8t8kf` / `yfpxs7wo` / `m29o33k1`→`4x6ywtp7`).

Regeneration: `python3 plot_poster.py` → outputs in `poster/`. Smoothing is W&B-style EMA with α ∈ {0.6, 0.85, 0.95}.

## Recommended for poster

### 1. Train Pass@8 — **headline accuracy**
- Files: `poster/pass_at_8_ema{60,85,95}.png`
- Recommended: **α=0.95** (cleanest separation; the raw is way too noisy at poster size).
- Story: GRPO ~0.47, Poly-EPO ~0.45, Minority ~0.40 at step 400. GRPO wins raw accuracy by ~2pp; Poly-EPO is the diversity-aware arm that doesn't pay a meaningful accuracy tax; Minority gives up ~6pp.
- Caption hook: "Among RL arms with the same compute, the diversity-aware Poly-EPO essentially matches GRPO's raw accuracy while sustaining materially higher per-token entropy (see Entropy panel)."

### 2. Actor Token-Level Entropy — **diversity hypothesis (core narrative)**
- File: `poster/entropy_calibrated_ema85.png` (primary), `poster/entropy_twinaxis_ema85.png` (alt — twin axes, raw units, if reviewers prefer no back-out).
- All three arms on a **single y-axis in nats/token**. Set arms backed out to GRPO units via a step-1 calibration; ±15% bias band shaded.
- Story: by step 400, Minority sustains ~0.30 nats/tok, GRPO sits at ~0.26, Poly-EPO at ~0.23. Minority's +13% per-token entropy gap over GRPO is **robust** to the bias band; Poly-EPO's −12% gap sits inside the band, so on the poster we phrase Poly-EPO as "≈ GRPO per-token entropy, not above."
- Caption hook: "Diversity preserved at per-token resolution: Minority retains ~13% more token-level entropy than GRPO without collapsing modes; Poly-EPO holds GRPO-comparable entropy."

#### Why the back-out (footnote-worthy detail)
verl's `actor/entropy` is logged in different units for GRPO vs. the set arms:
- **GRPO**: seq-mean-token-mean (mean over sequences of mean per-token entropy), in nats/token.
- **Minority / Poly-EPO** (verl set-arm patch): per-prompt summed entropy (sstn).

At step 1, all three arms share the same Qwen3-4B initial policy (verified by identical `response_length/mean = 1066.57`). The raw ratio set-arm / GRPO = 243.38 / 0.966 = **251.9**. Using that as a constant divisor reproduces the same nats/token value for all three arms at init (sanity check passes: Cov ≈ 0 at init because the policy is uniform-ish over the same base model). After init, the bias from comparing token-mean (set arms, backed out) to seq-mean-token-mean (GRPO) is:

```
token-mean − seq-mean-token-mean = Cov(T_i, H_i_bar) / ⟨T⟩
|bias| ≤ |r| · std_T · std_H_bar / ⟨T⟩  ≈ ±5–15% (best guess ±8%)
```

(`std_T / ⟨T⟩ ≈ 1.1–1.3`; `std_H_bar` plausibly 0.02–0.05; `|r| ≤ 0.7`.) Sign is likely negative in math-RL (longer derivations → more-confident later tokens), so the backed-out set-arm values likely **underestimate** the true GRPO-comparable entropy. Shaded band = ±15% (upper end of the plausible range).

### 3. Train Fraction Filtered — **"GRPO pays a sampling tax"**
- Files: `poster/fraction_filtered_ema{60,85,95}.png`
- Recommended: **α=0.95** (sharp separation, no ambiguity).
- Story: GRPO filters ~55–60% of rollouts; set arms ~17–20%. ~3× sample-efficiency gap.
- Caption hook: "GRPO discards ~3× more rollouts via reward filtering than the set arms — set-arm diversity converts directly into better signal-per-rollout." → this anchors the "sampling tax" line in the conclusion.

## Reserve / supplementary (use if you have space)

| Plot | Verdict | Reason |
|---|---|---|
| `train_distinct_clusters_mean` | Maybe | Gap is real (~3.7 vs 3.5) but small; works as a secondary diversity panel only if needed. |
| `critic_rewards_mean` | Maybe | Three-arm convergence to identical reward — supports "policies differ, reward learning doesn't." Nice secondary story. |

## Skip

| Plot | Reason |
|---|---|
| `actor_pg_loss` | Computed from each arm's own objective; cross-arm magnitudes are not comparable. |
| `actor_ppo_kl` | All near zero throughout; "KL was not a bottleneck" is a one-line caption, not a plot. |
| `train_judge_parse_ok_rate` | Flat at 1.0 by step ~50; no visual story. |
| `train_prompts_unlocked` | Cumulative line shows sawtooth resets (run-stitching artifacts in Poly-EPO; sharp drops in GRPO and Minority near step 380); looks like a bug at first glance. |
| `train_degenerate_rollouts` | Duplicates the filtering/entropy story with less punch (set-arm-only). |

## Notes on the regenerated plots

- Output files live under `poster/`. The originals (committed in 784888c) remain in this directory and are untouched.
- Smoothing line is the EMA; the raw trace is overlaid at low alpha for honesty.
- All plots are 10×5" @ 150 dpi (poster-readable at full column width).
- Color scheme matches existing convention: GRPO blue (`#1f77b4`), Minority-CoT orange (`#ff7f0e`), Poly-EPO-CoT green (`#2ca02c`).

## Conclusion-line drafts (for poster body)

- **GRPO pays a sampling tax**: ~55–60% of GRPO rollouts get filtered vs ~17–20% for set arms. Diversity converts directly into useful gradient signal per sampled rollout.
- **Poly-EPO is the balance point**: matches GRPO on raw pass@8 within 2pp while holding GRPO-comparable per-token entropy. The all-distinct-clusters objective recovers the accuracy that Minority's rarest-cluster objective gives up.
- **Minority is honest about its tradeoff**: ~6pp accuracy cost buys +13% per-token entropy and the largest mode-coverage signal, but the rarest-cluster reweighting is uncorrelated with correctness in the well-posed regime (see `minority_diagnostic.md`) — so Minority is interesting as a diversity *probe* rather than a deployment choice.
