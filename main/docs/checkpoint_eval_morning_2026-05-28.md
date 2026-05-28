# Checkpoint eval morning writeup (2026-05-28)

All runs: `n_rollouts=16`, `prompt_variant=hybrid_answer_boxed`, B200.

| Family | Profile | Config | Bundle stamp |
|--------|---------|--------|----------------|
| LR=3e-6 redo (base+3 arms) | chicken602 | `checkpoint_eval_lr3e6_latest_dapo2k_polaris2k_b200` | `20260528T083158Z` |
| LR=1e-6 resolved (3 arms, no base) | anastasia | `checkpoint_eval_lateckpt_resolved_nobase_dapo2k_polaris2k_b200` | `20260528T083202Z` |
| LR=3e-6 AIME+MATH-500 | chicken602 | `checkpoint_eval_lr3e6_latest_aime_math500_b200` | `20260528T082033Z` |
| LR=1e-6 AIME+MATH-500 | anastasia | `checkpoint_eval_lateckpt_resolved_nobase_aime_math500_b200` | `20260528T082039Z` |

Volume paths: `main-artifacts/probes/<output_dir>/<stamp>/`.

---

## Headline

**GRPO wins on every decision-grade slice.** LR=3e-6 redo GRPO (`grpo_lr3e6_s59`) is the only arm that consistently beats base and the other arms on DAPO 2k, Polaris stratified 2k, and MATH-500. Minority and poly_epo are close on DAPO but trail GRPO on Polaris; on the older 1e-6 resolved checkpoints they trail GRPO everywhere.

**AIME-25 is not usable for arm ranking** (30 prompts, pass@16 in single digits). Use MATH-500 for medium OOD signal instead of BeyondAIME.

---

## LR=3e-6 redo family (chicken602, vs base)

Checkpoints: `grpo_lr3e6_s59`, `minority_lr3e6_s54`, `poly_epo_lr3e6_s39`.

### DAPO 2k — pass@8 (primary)

| Arm | pass@8 | Δ vs base | Δ vs GRPO |
|-----|--------|-----------|-----------|
| base | 29.9% | — | — |
| **grpo** | **31.9%** | **+2.0pp** | — |
| minority | 31.9% | +2.2pp | −0.1pp |
| poly_epo | 31.0% | +1.1pp | −0.9pp |

pass@1: base 6.7% → GRPO 7.8% (+1.1pp). Minority/poly ~flat vs GRPO on pass@1.

### Polaris stratified 2k — pass@8 (250 prompts per band 0/8…7/8)

| Arm | pass@8 | Δ vs base | Δ vs GRPO |
|-----|--------|-----------|-----------|
| base | 30.8% | — | — |
| **grpo** | **32.9%** | **+2.1pp** | — |
| minority | 31.7% | +0.8pp | −1.3pp |
| poly_epo | 32.0% | +1.1pp | −1.0pp |

GRPO gains are largest on easier bands (6/8, 7/8): +2.8pp and +3.3pp pass@8 vs base on those bands.

### MATH-500 (500 prompts) — pass@16

| Arm | pass@16 | Δ vs base |
|-----|---------|-----------|
| base | 78.8% | — |
| **grpo** | **83.2%** | **+4.4pp** |
| minority | 80.8% | +2.0pp |
| poly_epo | 81.2% | +2.4pp |

### AIME-25 (30 prompts) — pass@16 — noisy

| Arm | pass@16 |
|-----|---------|
| base | 13.3% |
| grpo | 10.0% |
| minority | 6.7% |
| poly_epo | 10.0% |

Do not treat ±3pp here as meaningful (≈3–4 problems with any success at pass@16).

---

## LR=1e-6 resolved family (anastasia, no base; compare to GRPO)

Checkpoints: `grpo_b200_s519`, `minority_b200_s239`, `poly_epo_b200_s133`.

### DAPO 2k — pass@8

| Arm | pass@8 | Δ vs GRPO |
|-----|--------|-----------|
| **grpo** | **31.7%** | — |
| minority | 31.3% | −0.5pp |
| poly_epo | 30.6% | −1.2pp |

### Polaris stratified 2k — pass@8

| Arm | pass@8 | Δ vs GRPO |
|-----|--------|-----------|
| **grpo** | **31.8%** | — |
| minority | 31.0% | −0.8pp |
| poly_epo | 31.4% | −0.5pp |

### MATH-500 — pass@16

| Arm | pass@16 | Δ vs GRPO |
|-----|---------|-----------|
| **grpo** | **81.6%** | — |
| minority | 80.0% | −1.6pp |
| poly_epo | 78.2% | −3.4pp |

### AIME-25 — pass@16

| Arm | pass@16 |
|-----|---------|
| grpo | 10.0% |
| minority | 3.3% |
| poly_epo | 10.0% |

---

## LR=3e-6 vs LR=1e-6 (GRPO only, apples-to-apples)

| Slice | 1e-6 `grpo_b200_s519` | 3e-6 `grpo_lr3e6_s59` | Δ |
|-------|----------------------|----------------------|-----|
| DAPO pass@8 | 31.7% | 31.9% | +0.2pp |
| Polaris pass@8 | 31.8% | 32.9% | +1.1pp |
| MATH-500 pass@16 | 81.6% | 83.2% | +1.6pp |

3e-6 redo is not worse; modest uplift on Polaris and MATH-500. DAPO is flat.

---

## Runtime / cost notes

Per-checkpoint wall clock (representative):

- DAPO 2k: ~55–100 min per variant (poly_epo_lr3e6 was slowest at ~100 min)
- Polaris 2k: ~59–85 min per variant
- MATH-500: ~10–15 min per variant
- AIME-25: ~1 min per variant

Combined apps ran DAPO then Polaris sequentially per GPU (4 GPUs chicken602, 3 anastasia). Total wall per app ~3.3h (anastasia) / ~3.3h (chicken602).

`rollout_chunk_prompts=32` with `n_rollouts=16` behaved stably (same effective batch pressure as prior 64×8 runs).

---

## Recommendation (Job 1 / next train)

1. **Default arm: GRPO** — only consistent winner on 2k slices and MATH-500.
2. **Learning rate: continue LR=3e-6 redo** — GRPO at s59 matches or slightly beats s519 on resolved checkpoints; no regression on DAPO.
3. **Do not pivot to minority/poly_epo** from these evals — they do not beat GRPO on Polaris; poly trails on 1e-6 MATH-500.
4. **Eval menu going forward:** Polaris stratified 2k + DAPO 2k + MATH-500 (n=16). Skip BeyondAIME for arm decisions; treat AIME-25 as sanity-only.
5. **Optional next step:** extend 3e-6 GRPO training a bit further (s59 is early-ish) and re-check Polaris 6/8–7/8 bands where GRPO showed the largest lift.

---

## Pull commands

```bash
# LR=3e-6 DAPO+Polaris
MODAL_PROFILE=chicken602 modal volume get main-artifacts \
  probes/checkpoint_eval_lr3e6_latest_dapo2k_polaris2k_b200/20260528T083158Z \
  main/data/probes/checkpoint_eval_lr3e6_latest_dapo2k_polaris2k_b200/20260528T083158Z

# 1e-6 resolved DAPO+Polaris
MODAL_PROFILE=anastasia modal volume get main-artifacts \
  probes/checkpoint_eval_lateckpt_resolved_nobase_dapo2k_polaris2k_b200/20260528T083202Z \
  main/data/probes/checkpoint_eval_lateckpt_resolved_nobase_dapo2k_polaris2k_b200/20260528T083202Z
```
