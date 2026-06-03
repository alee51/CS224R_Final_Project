# Poster Plot Shortlist

Which W&B training-dynamics plots to put on the poster, why, and how they were generated. Source CSVs in `../`: `GRPO_history.csv`, `Minority_CoT_history.csv`, `Poly_EPO_CoT_history.csv` (400 steps each, runs `rof8t8kf` / `yfpxs7wo` / `m29o33k1`→`4x6ywtp7`).

Regeneration: `python3 plot_poster.py` → outputs in `poster/`. Smoothing is W&B-style EMA, α=0.95 for the metric plots.

## Poster: two plots (maybe one)

### 1. Train Pass@8 — **headline accuracy**
- File: `poster/pass_at_8_ema95_square.png`.
- Story: GRPO ~0.47, Poly-EPO ~0.45, Minority ~0.40 at step 400. GRPO wins raw accuracy by ~2pp; Poly-EPO is the diversity-aware arm that doesn't pay a meaningful accuracy tax; Minority gives up ~6pp.
- Caption hook: "Among RL arms with the same compute, Poly-EPO essentially matches GRPO's raw accuracy; Minority pays a ~6pp accuracy tax in exchange for stronger rare-cluster reweighting."

### 2. Train Fraction Filtered — **"GRPO pays a sampling tax"**
- File: `poster/fraction_filtered_ema95_square.png`.
- Story: GRPO filters ~55–60% of rollouts; set arms ~17–20%. ~3× sample-efficiency gap.
- Caption hook: "GRPO discards ~3× more rollouts via reward filtering than the set arms — set-arm diversity converts directly into better signal-per-rollout." → anchors the "sampling tax" line in the conclusion.

If we end up needing to drop one for space, pass@8 stays — it's the headline metric. Fraction filtered moves to body text as a one-line stat.

## Deferred to the paper: entropy

Currently generated but **not on the poster**:
- `poster/entropy_calibrated_ema85.png` — all 3 arms on one nats/token y-axis (set arms backed out, ±15% bias band).
- `poster/entropy_twinaxis_ema85.png` — raw units, twin axes (set arms left, GRPO right).

**Why deferred:**
- verl logs `actor/entropy` in *different units* for GRPO (seq-mean-token-mean nats/tok) vs. set arms (per-prompt summed sstn). We do not have nats/token directly logged for the set arms.
- The poster plot uses an inferred back-out: divide set-arm sstn by 251.9 (the step-1 calibration constant where all three arms share the same initial Qwen3-4B policy, verified by identical `response_length/mean = 1066.57`). The ±15% shaded band reflects the residual `Cov(T_i, H̄_i) / ⟨T⟩` bias when comparing token-mean to seq-mean-token-mean.
- This is defensible but **inferential**, not directly measured. We'd rather not put a number on the poster that we'd have to footnote-defend.
- **Train-time validation gives us a directly-measured entropy reading**: each held-out validation rollout during training is logged with `logprobs=20`, so for the paper we'll compute per-token entropy directly from the saved logprobs (`H = -Σ p_i log p_i` over the top-20, a tight lower bound for a sharpened post-RL policy). That gives us ground-truth nats/token at step 400 for all three arms on the same validation distribution — no back-out, no bias band.
- For the paper we'll also have a 3rd cross-check: re-running the same validation probe under different temperatures isolates *exploration* entropy (sampling-time) from *policy* entropy (logit-level).

**Plan for the paper:**
- Replace the training-time back-out plot with train-time-validation direct-measurement nats/token.
- Keep the training-time trajectory as a supplementary figure with the back-out + band, so the reader can see entropy decay over training under the same lens.
- Story remains the same in spirit (Minority preserves more per-token entropy; Poly-EPO ≈ GRPO per-token but has structural/modal diversity) but is now grounded in a directly-measured number rather than an inferred one.

### Back-out math (for future reference)
At step 1, true per-token entropy is provably identical across arms (same init policy, same generations). Raw ratio: set-arm 243.38 / GRPO 0.966 = 251.9 → that's the exact unit conversion at step 1. For steps > 1, the bias is bounded by:

```
token-mean − seq-mean-token-mean = Cov(T_i, H̄_i) / ⟨T⟩
|bias| ≤ |r| · std_T · std_H̄ / ⟨T⟩  ≈ ±5–15% (best guess ±8%)
```

`std_T / ⟨T⟩ ≈ 1.1–1.3`; `std_H̄` plausibly 0.02–0.05; `|r| ≤ 0.7`. Sign is likely negative in math-RL (longer derivations → more confident later tokens), so backed-out set-arm values likely **underestimate** the true GRPO-comparable entropy. Band = ±15% (upper end of plausible range).

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

- Output files live under `poster/`. Originals (committed in 784888c) live under `archive/`.
- Smoothing line is the EMA; the raw trace is overlaid at low alpha for honesty.
- Metric plots are 6×5.5" @ 150 dpi (square, for poster cells); entropy plots are 10×5" wide.
- Color scheme matches existing convention: GRPO blue (`#1f77b4`), Minority-CoT orange (`#ff7f0e`), Poly-EPO-CoT green (`#2ca02c`).

## Conclusion-line drafts (for poster body)

- **GRPO pays a sampling tax**: ~55–60% of GRPO rollouts get filtered vs ~17–20% for set arms. Diversity converts directly into useful gradient signal per sampled rollout.
- **Poly-EPO is the balance point**: matches GRPO on raw pass@8 within 2pp; the all-distinct-clusters objective recovers the accuracy that Minority's rarest-cluster objective gives up.
- **Minority is honest about its tradeoff**: ~6pp accuracy cost in exchange for stronger rare-cluster reweighting and the largest mode-coverage signal at the rollout level; the rarest-cluster reweighting is uncorrelated with correctness in the well-posed regime (see `minority_diagnostic.md`) — so Minority is interesting as a diversity *probe* rather than a deployment choice.
- **(Paper, not poster)** Per-token entropy story: Minority preserves more, Poly-EPO ≈ GRPO at the token level despite generating structurally distinct rollouts — modal vs. token-level diversity. Deferred to the paper because we'll have a directly-measured train-time validation entropy reading rather than the training-time back-out.
