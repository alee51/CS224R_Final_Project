# Project status

**Updated:** 2026-06-01 (post-launch)

**Also read:** [`human-remaining-work.md`](human-remaining-work.md) · [`verl_migration_plan.md`](verl_migration_plan.md)

## Deadlines

| When | Milestone |
|------|-----------|
| ~~Fri 2026-05-29 EOD~~ | ~~Training launched~~ (slipped to 2026-06-01 07:47 UTC) |
| **Mon 2026-06-01 → Tue 2026-06-02 ~1 AM PT** | Training **done** (per W&B ETA) |
| **Tue 2026-06-02 4:00 PM** | Poster to print |
| **Wed 2026-06-03 9:00 AM** | Poster due |

## Active path

| What | Where |
|------|--------|
| Training (in flight) | [`../`](../) (`main-verl/`) |
| Eval tooling (canonical) | [`../../main/probes/checkpoint_rollout_eval.py`](../../main/probes/checkpoint_rollout_eval.py) + [`../../main/configs/checkpoint_eval_*`](../../main/configs/) |
| Frozen 1.7B baseline + eval results | [`../../main/docs/checkpoint_eval_morning_2026-05-28.md`](../../main/docs/checkpoint_eval_morning_2026-05-28.md) |

---

## What is actually running

Three Stage 8 arms, **launched 2026-06-01 ~07:47 UTC**, each on its own Modal account, all calling a shared judge on `stonedpinecones`:

| Arm | Run ID | Account | Ckpt dir (on `/vol/`) | Avg step (excl. step 1) | ETA |
|-----|--------|---------|------------------------|-------------------------|-----|
| GRPO | `rof8t8kf` | anastasia | `/vol/checkpoints/main-verl/grpo_train_4b_1epoch_lr3e6` | 157 s/step | Mon 6/1 ~6:15 PM PT |
| minority_cot | `yfpxs7wo` | emma | `/vol/checkpoints/main-verl/minority_cot_train_4b_1epoch_lr3e6` | 213 s/step | Tue 6/2 ~12:30 AM PT |
| poly_epo_cot | `m29o33k1` | stonedpinecones | `/vol/checkpoints/main-verl/poly_epo_cot_train_4b_1epoch_lr3e6` | 214 s/step | Tue 6/2 ~12:45 AM PT |

**W&B workspace:** <https://wandb.ai/224r-project/cs224r-minority-voting/workspace?nw=vqymgsruo5> · tag filter `verl AND production AND lr3e6`.

**Run configuration (all three arms):**

- **Base model:** `Qwen/Qwen3-4B-Base`
- **Train data:** `polaris_train.parquet` (51,139 problems, 1 epoch)
- **Val data (in-training):** `polaris_val.parquet` (5k held-out) + `aime_val.parquet` (AIME-25, held-out)
- **Steps:** 400 cap (= 1 epoch @ `train_batch_size=128`, dataset gives 399 full steps)
- **`save_freq=10`, `test_freq=50`** → ckpts every 10 steps, in-training eval every 50 steps
- **LR:** `3e-6` symmetric across all 3 arms (final relaunch after v2 at `1e-6` learned too slowly — see [Bring-up history](#bring-up-history))
- **`loss_agg_mode`:** GRPO = `seq-mean-token-mean`; set arms = `seq-mean-token-sum-norm`
- **DAPO knobs:** asymmetric clip `0.20 / 0.28`, `ppo_mini_batch_size=64`, `entropy_coeff=0`, KL=0
- **Judge:** Qwen3-4B-Instruct-2507, served by `stonedpinecones`, batch=64, concurrency=2, 2 containers. Prompt = `main-verl/judge/prompts/poly_epo_a1.md` with both paper §A.1 few-shot examples restored.
- **Fork:** `chicken602/maxrl @ 33873ec9` (per-rollout logging, GRPO W&B parity, finish_reason aggregator, final-step ckpt forcing).

---

## What's known to work / what to expect

**Trustworthy from 1.7B baseline** (frozen, evals canonical in [`checkpoint_eval_morning_2026-05-28.md`](../../main/docs/checkpoint_eval_morning_2026-05-28.md)):
- GRPO wins all slices at convergence.
- Background panel: Polaris 2k, DAPO 2k, BeyondAIME @ n=16, MATH-500 @ n=16.

**Open from 4B / VeRL** (this run answers):
- Does `minority_cot` separate from `GRPO` at 4B + CoT (vs the 1.7B finding that GRPO dominates)?
- Does `poly_epo_cot` learn at all under our config, given Ifdita confirmed set arms are inherently slower?
- Whether the few-shot–restored judge prompt produces meaningful cluster diversity (`distinct_clusters_mean > 1.5`) vs the prior collapsed runs.

---

## Eval handoff (Tue 6/2)

**Eval panel — same as 5/28 1.7B run** (so 4B results are directly comparable):
- Polaris 2k (in-distribution)
- DAPO 2k (in-distribution, different prompt style)
- MATH-500 @ n=16 (pass@k story)
- AIME-25 already covered by in-training `test_freq=50` evals — pull from W&B; no separate launch needed unless the held-out curve looks weird.

**Launch path:**
- Configs: `main/configs/checkpoint_eval_lr3e6_latest_dapo2k_polaris2k_b200.yaml` is the template — swap ckpt paths to the three `/vol/checkpoints/main-verl/*_lr3e6/global_step_<final>/` dirs once each arm finishes.
- Driver: `main/scripts/launch_checkpoint_eval.sh` → `main/probes/checkpoint_rollout_eval.py`.
- **Launch per-arm as each ckpt lands** — do not wait for all three. GRPO finishes ~6 h before the set arms; that's 6 h of eval head start to spend.
- Modal account for eval: any free account (chicken602 or abao have the most balance left — see `account_balances.md`).

**Outputs:** JSON under `main/data/probes/checkpoint_eval_*_arms_latest/`, same layout as 5/28 so the same analysis scripts work.

**Optional / nice-to-have if time:**
- 51K pass-rate histogram if the merge utility is ready.
- Per-step diversity readout from `/vol/per_rollout/<run_id>/step_*.jsonl` on the CoT arms (post-hoc cluster diagnostics).
- Mid-training ckpt eval (any saved `global_step_*` dir, e.g. step 200) if a curve is interesting in W&B.

---

## Poster framing

Background (locked, can write now): 1.7B GRPO-wins-all table; method draft for CoT + 4B; diagnosis of why this was hard (judge prompt fewshot gap, v1/v2 LR too low — see [Bring-up history](#bring-up-history)).

Conclusion (locked from eval outcomes Tue):

| Outcome | Direction |
|---------|-----------|
| **minority_cot beats GRPO at 4B** | Lead with separation: 1.7B/GRPO-only result was scale-limited; CoT minority voting unlocks at 4B |
| **minority_cot flat or loses** | Analysis: why minority voting fails (cluster collapse, judge noise, all-wrong fraction); 1.7B finding holds at 4B |
| **poly_epo_cot weak vs paper / 1.7B** | Inconclusive at 1 epoch + LR ablation; cite Ifdita's "set arms slower" confirmation |

---

## Stage checklist (final)

| # | Stage | Status |
|---|--------|-------|
| 1 | Modal image + verl + Ray | ☑ |
| 2 | GRPO bring-up smoke (1.7B) | ☑ |
| 3a | `minority_cot` + mock clusters | ☑ |
| 3b | Real judge wired (no silent mock fallback) | ☑ |
| 4 | Judge on Modal (Qwen3-4B-Instruct-2507, S4.5 v2 100/100) | ☑ |
| 5 | `poly_epo_cot` registered + unit-tested | ☑ |
| 5.5 | Judge sanity gate (paper §A.1 fewshot examples restored in prompt) | ☑ |
| 6 | 4B fit check (formal OOM ladder dropped — minority_cot+judge ladder 1d confirms 4B fits at `gpu_memory_utilization=0.65`) | ☑ |
| 7 | `finish_reason="length"` wiring + per-rollout JSONL + GRPO W&B parity | ☑ |
| **8** | **3-arm 1-epoch retrain — launched 2026-06-01 07:47 UTC, training in flight** | **🟡 running** |

---

## Bring-up history (compressed — read if eval results look weird)

- **2026-05-31 morning crashed runs** — all three arms had wrong `loss_agg_mode` (GRPO needed `seq-mean-token-mean`, not `sum-norm`); judge prompt was missing the paper §A.1 fewshot examples (Memory: `project_judge_prompt_fewshot_gap`); minority subset score was rewarding the degenerate (-1) cluster.
- **2026-05-31 v2 relaunch** — fixed all of the above, plus DAPO knobs (asymmetric clip, mini-batch 64, KL=0). Reward learned but slowly: poly_epo +0.015%/step vs GRPO +0.047%/step at LR=1e-6 (Ifdita confirmed her set arms also slower). Minority *regressed* pre-fix; cluster-100 (-1) exclusion landed in `objective_poly_epo.py` and `objective_minority.py`.
- **2026-06-01 v3 lr3e6 relaunch (this run)** — LR `1e-6` → `3e-6` symmetric across all 3 arms; same fork (`33873ec9`); fresh ckpt dirs (`_lr3e6` suffix) to prevent resume.
- Per migration plan, **1 epoch only.** No second epoch.

## What not to do

- No new training in `main/`.
- No second epoch — there is no time, and a second epoch was never the plan.
- No new training experiments on this branch — eval and poster only.
- No `pre-milestone/` cleanup.
