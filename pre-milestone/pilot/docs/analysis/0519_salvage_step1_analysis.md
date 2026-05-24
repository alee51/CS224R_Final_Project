# Salvaged step-1 GRPO rollout analysis (2026-05-19)

**Scope:** One training step of base-model rollouts (32 prompts × 8 completions = 256 rows per run) from the killed pilot pull `20260519T214807Z_final_pull`. Rewards are binary RLVR: 1.0 if `correct`, else 0.0. `run1_grpo` predictions were wiped (bootstrap `write_text("")`); not analyzed.

**Configs:** `run1b_grpo` seed **43**; `run2_inverse_freq` / `run3_f_grpo` seed **42**. Same model, verifier, clustering, and rollout hyperparameters otherwise.

---

## 1. Per-run summary


| Metric                                                                     | run1b_grpo                    | run2_inverse_freq             | run3_f_grpo  |
| -------------------------------------------------------------------------- | ----------------------------- | ----------------------------- | ------------ |
| Prompts / completions                                                      | 32 / 256                      | 32 / 256                      | 32 / 256     |
| Unique exact-match answer clusters (Σ per-prompt distinct `parsed_answer`) | 225 (μ≈7.0/prompt)            | 207 (μ≈6.5)                   | 207          |
| Reward min / mean / median / max / std                                     | 0 / **0.062** / 0 / 1 / 0.242 | 0 / **0.172** / 0 / 1 / 0.377 | same as run2 |
| Prompts with ≥1 correct (Pass@8 proxy)                                     | **8/32 (25%)**                | **15/32 (46.9%)**             | 15/32        |
| Prompts with minority-correct cluster*                                     | **2/32 (6.2%)**               | 2/32 (6.2%)                   | 2/32         |


Correct answer appears in a cluster strictly smaller than the plurality wrong cluster.

**run1b** — Hardest batch: 24/32 prompts have 0/8 correct (e.g. `9e6520c5-0699-41a8-9848-a09a68e462c5`, mean reward 0). Best prompts cap at 3/8: `ac30fdd3-d3b4-44a5-88f9-20d375206d73` (rewards 1,1,1,0,… → mean **0.375**), `40da682c-a68f-4d2d-ae7f-ef06d4d80d69` (**0.375**). Illustrative partial success: `71fb6079-2cab-4682-8015-8915cd11e52b` — 2/8 correct on canonical answer `"12"` vs wrong clusters `"11"`, `"0"`, `"\\( 12 \\)"` (formatting split). Minority-correct examples: `a5d4dc54-…` (1/8 correct, wrong plurality `"72"`), `b84654f5-…` (1/8 correct vs plurality `"576"`).

**run2 / run3** — Per-prompt histogram: 17 zeros, four at 0.5, two at **0.75** (`40a686fc-c7c3-457e-b80a-52ee961cbf37`, `0db0e2d7-0db7-4409-9e72-bb191583853c`). Shared easy prompt `0b4478a7-8a73-4d82-8a1b-1a6b7ff27196`: 3/8 correct on `"0"`. Minority-correct: `b7590bfe-37d7-4de3-850d-92522f4d1904`, `c67a46a6-a612-49f1-95b5-432975155f15` (each 1/8 correct).

`train.log`: all runs completed step-1 **rollout build** (256 completions); none logged `step 1/100 done` in this pull (pilot killed mid-step). Salvaged `run1_grpo` log elsewhere shows `mean_reward=0.172` at step 1 with seed 42 — matches recomputed run2/3.

---

## 2. Cross-run comparison

**Prompt sets differ.** Zero overlap between `run1b` (seed 43) and `run2`/`run3` (seed 42). **No per-prompt reward deltas** run1b vs run2/3.

**run2 vs run3:** Identical on all 256 lines (`prompt_id`, `parsed_answer`, `correct`); per-prompt mean rewards match exactly (max |Δ|=0). Expected: step 1 is pre-update rollout generation; `inverse_freq` / `f_grpo` only change advantages after rewards. `cluster_id` hashes differ but do not affect rewards.

**Comparable aggregates only:** cohort-level Pass@8 proxy, minority-correct rate, reward variance. Not comparable for Tier-1 decision rules that require the **same DaPO slice** across runs (`final_decision.md` shared setup).

---

## 3. Diagnosing run1b mean_reward = 0.062 vs 0.172


| Hypothesis                                 | Verdict                      | Evidence                                                                                                                                                                                                              |
| ------------------------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(a) Seed 43 → harder prompt batch**      | **Supported**                | +7 all-fail prompts (24 vs 17); lower ceiling (max per-prompt mean 0.375 vs 0.75); fewer solvable prompts (25% vs 47% with ≥1 correct). Recomputed JSONL mean 0.062 matches expected batch effect.                    |
| **(b) Different GRPO advantage code path** | **Rejected** for this metric | `mean_reward` is mean binary rollout reward, not advantage-weighted. run2 ≡ run3 rollouts despite different objectives.                                                                                               |
| **(c) Reward verifier bug on run1b**       | **Unlikely**                 | Same `binary_rlvr` + `exact_canonical`; failures look like model/format noise (e.g. `71fb6079` split), not systematic misgrading. Wiped `run1_grpo` (seed 42, vanilla GRPO) reported 0.172 — same as run2, not run1b. |


**Conclusion:** The gap is almost entirely **prompt sampling (seed 43)**, not a run1b-specific training bug.

---

## 4. Recommendation (vs `final_decision.md`)

**Run 0 gate:** Minority-correct prompts are **6.2%** (2/32), below the pre-registered “≥15%” proxy-validity threshold. On this salvaged slice alone, `final_decision.md` would suggest treating frequency-based minority signal as weak and considering `worst_subset` — but *n*=32 and seed mismatch limit confidence.


| Use                                              | Verdict                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(a) Smoke regression fixture**                 | **Partial yes.** Use **run2** (or run3) JSONL as a golden step-1 rollout file: expect mean reward **0.172**, 256 rows, 15 prompts with any correct. Add **run1b** as a second fixture for seed-43 batching (mean **0.062**, 24 all-fail prompts). Assert parser, verifier, and reward aggregation — not cross-run equality. |
| **(b) Publishable reward-distribution snapshot** | **No.** Single step, *n*=32, two seeds, incomplete training, wiped run1, run2≡run3 duplicate. Violates locked “same slice across runs” for pilot comparisons.                                                                                                                                                               |
| **(c) Neither**                                  | **Default for science claims**; salvage is diagnostic only.                                                                                                                                                                                                                                                                 |


**Next pilot discipline:** Align all matrix runs to **one seed** (42) and one frozen 32-prompt index file before interpreting Run 1–3 decision table outcomes.

---

*Artifacts:* `pilot/artifacts/{run1b_grpo,run2_inverse_freq,run3_f_grpo}/20260519T214807Z_final_pull/*/raw_predictions.jsonl`