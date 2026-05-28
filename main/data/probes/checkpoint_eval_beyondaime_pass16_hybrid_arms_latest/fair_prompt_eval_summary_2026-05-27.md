# Fair-prompt eval — BeyondAIME + DAPO 2k summary (2026-05-27)

**TL;DR.** The prompt mismatch was masking a real hypothesis-positive signal for **poly_epo_answer**. Under matched `hybrid_answer_boxed` prompts, `poly_epo_answer` is the best arm on all three eval slices and the **BeyondAIME −5–6 pp regression disappears entirely**.

## Source artifacts

- BeyondAIME (n=100, seed=42, 16 rollouts/prompt, `hybrid_answer_boxed`): `/vol/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/20260528T023940Z/results.json` (Modal volume `main-artifacts`, alee72; mirrored to `/tmp/beyondaime_hybrid_results.json` local).
- DAPO 2k (n=2000, seed=43, 8 rollouts/prompt, `hybrid_answer_boxed`): `/vol/probes/checkpoint_eval_2k_dapo_hybrid_arms_latest/20260528T024146Z/results.json`; mirrored to `/tmp/dapo_hybrid_results.json` local.
- Polaris 2k baseline (already fair prompt all along): canonical entry in `main/docs/timeline.md`.

## Combined picture (Δ vs base, all under `hybrid_answer_boxed`)

| Slice | n | metric | base | GRPO Δ | minority Δ | **poly_epo Δ** |
|-------|---|--------|------|--------|------------|-----------------|
| Polaris 2k (training distribution) | 2000 | pass@8 | 0.306 | +1.1 pp | +0.1 pp | **+1.3 pp** |
| DAPO 2k (easier OOD) | 2000 | pass@8 | 0.313 | −0.5 pp | −0.65 pp | **+0.8 pp** |
| BeyondAIME (hard OOD) | 100 | pass@16 | 0.070 | +1.0 pp | +1.0 pp | **+5.0 pp** |

Checkpoints: `grpo_b200_s359` (Polaris 2k uses `s299`), `minority_b200_s159` (Polaris uses `s133`), `poly_epo_b200_s133`. All from the `train_real_b200/` H200→B200 fresh lineage, not the LR=3e-6 chicken602 line that's currently running.

## Diff vs original (`dapo_answer_v1`) eval

| Slice | Arm | dapo_answer_v1 Δ | hybrid_answer_boxed Δ | Prompt-mismatch effect |
|-------|-----|------------------|------------------------|------------------------|
| BeyondAIME | base | (0.130 raw) | (0.070 raw) | base **inflated +6 pp** under mismatched prompt |
| BeyondAIME | GRPO | −6.0 pp | +1.0 pp | **+7 pp swing** |
| BeyondAIME | minority | −6.0 pp | +1.0 pp | **+7 pp swing** |
| BeyondAIME | poly_epo | −5.0 pp | +5.0 pp | **+10 pp swing** |
| DAPO 2k | base | (0.248 raw) | (0.313 raw) | base **deflated −6.5 pp** under mismatched prompt |
| DAPO 2k | GRPO | +2.4 pp | −0.5 pp | **−2.9 pp swing** |
| DAPO 2k | minority | +0.5 pp | −0.65 pp | **−1.2 pp swing** |
| DAPO 2k | poly_epo | +1.5 pp | +0.8 pp | **−0.7 pp swing** |

Direction of the prompt-mismatch artifact differed between the two slices (BeyondAIME inflated base + deflated trained; DAPO did the opposite). Net effect under matched eval: **the "BeyondAIME hard-OOD regression" was the artifact, and the "DAPO 2k GRPO win" was also partially the artifact.** Both swings make the *trained vs base* comparison less favorable to the trained arms on DAPO 2k and more favorable on BeyondAIME.

## What this changes

1. **Hypothesis-positive signal for poly_epo_answer.** Wins all three slices. The +5 pp on BeyondAIME is the largest effect, though noisy at n=100 (see significance below).
2. **minority_answer still flat-to-negative** at this scale. +0.1 pp / −0.65 pp / +1.0 pp across the three slices is not a hypothesis confirmation.
3. **GRPO baseline is genuinely flat-to-marginal**, not the +1–2 pp the mismatched eval suggested.
4. **The synthesis-entry §1 BeyondAIME claim ("regression is mostly real") needs to be retracted** — it's almost entirely the prompt artifact.

## Significance caveats

- **BeyondAIME (n=100)**: pass@16 differences of ~5 pp are roughly 1σ under independent-proportion approximation (`SE ≈ √(0.07·0.93/100 + 0.12·0.88/100) ≈ 0.041`). Not significant in isolation. Combined with the other slices, the consistent direction (poly_epo > GRPO ≥ minority across all three) is the load-bearing observation.
- **DAPO 2k (n=2000)**: SE ≈ 0.015. None of the trained arms are individually significantly different from base. poly_epo > GRPO consistency is the signal.
- **Polaris 2k (n=2000)**: same SE ≈ 0.015. poly_epo > GRPO not individually significant.

Across-slice consistency (poly_epo best on 3/3, minority worst on 2/3) is more informative than any single number. A diversity panel on saved Polaris 2k rollouts would add a fourth orthogonal axis and is offline (~$0).

## Action items

- Update **`main/docs/timeline.md`** canonical-eval entry: add hybrid_answer_boxed rows to BeyondAIME and DAPO 2k tables alongside the dapo_answer_v1 rows (don't delete the old rows — they're the prompt-mismatch reference).
- Update **`main/docs/timeline.md`** synthesis entry (`structural diagnosis: model/data mismatch ...`) §1 second bullet to retract the "BeyondAIME regression is mostly real" claim.
- Update **`main/docs/ta_discussion.md`** §1 table with the fair-prompt BeyondAIME row; §3.Q2 should now read "rerun shows regression was the prompt artifact; poly_epo +5 pp."
- Compute diversity panel on saved Polaris 2k rollouts (`/vol/probes/checkpoint_eval_2k_polaris_arms_latest/20260527T213611Z/rollouts/`) to test PLAN consolation criterion for minority + poly.
- **Reconsider the Option A (filter-then-retrain) vs Option B (own-the-null) decision** in the synthesis entry: with poly_epo showing hypothesis-positive signal, the value of a longer / more-signal-rich retrain just went up.
