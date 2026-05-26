# Group A probe — results readout

**Runs:**

| Scale | Wandb | Artifacts |
|-------|-------|-----------|
| 200 prompts | [`t33091vc`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/t33091vc) | `probes/05-24/group_a/` — [`05-24_group_a.pointer.json`](./artifacts/05-24_group_a.pointer.json) |
| **800 prompts (n800)** | [`mu8kj4ll`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/mu8kj4ll) | `probes/05-25/group_a_n800/` |

The **200-run** tables below are the first readout. **n800** tightens band stats and is the source for **rollout vs judge timing** (same pipeline, 4× scale). Early 200-run logging bugs: [`issues.md`](../issues.md). n800 used fixed git SHA passthrough and VRAM tracking.

---

## What this probe was for

Group A was a **standalone rollout + judge cost probe** meant to unblock PLAN.md before locking the training matrix. Per [`05-24_probe_plan.md`](./05-24_probe_plan.md), it targets open questions in **§5 (optimization / judge)**, **§2 (sampling)**, and **§7 (sizing & cost)** — without needing the trainer skeleton.

One script, two Modal phases:

1. **Phase 1** — Qwen3-1.7B-Base rollouts on 200 Polaris problems (25/band × 8 bands, 8 rollouts each)
2. **Phase 2** — Qwen3-4B-Instruct batched judge on all 200 problems (Poly-EPO §A.1)

Soft expectations were not hard gates: ~90% `parse_ok`, monotone pass rate vs difficulty, judge cost informing Minority-CoT go/no-go.

---

## Question-by-question readout

| Open question (from probe plan) | Answered? | Result | Implication |
|---|---|---|---|
| **`max_response_length` cap** (PLAN §5 / §7) | **Yes** | p50=565, p90=1319, p95=1974, **p99=4096**; 20/1600 (1.25%) hit length cap at 4096; ≥3072 = 2.3% | **4096 locked.** Most completions are well under cap; only 1.25% of rollouts hit it. Length table below is observational distribution only. |
| **Reward parser sanity** (`parse_ok` rate) | **Yes (200-run); superseded** | **55.9%** Minerva-only on 200-run (`has_answer_line` 56.4%, `has_boxed` 34.6%) | Failed soft gate on Rank-1. **Resolved 2026-05-25:** Rank-2 parser + hybrid prompt (arm C). See addendum below — do not use 56% as train-time expectation. |
| **Pass rate by difficulty band** (PLAN §2) | **Partially** | Overall pass **2.9%**; per band 1.5–4.5%. **Not monotone** (e.g. 7/8 = 4.0% > 6/8 = 1.5%) | Base model is very weak everywhere; bands are noisy at this sample size. Can't cleanly decide "drop 8/8-easy problems" yet — need post-parser re-run or larger N. |
| **Mixed-reward rate by band** (PLAN §2 / §7 reward density) | **Yes** | **30/200 prompts (15%)** have mixed correct/incorrect rollouts; per band 8–24% | Minority-answer signal exists on ~15% of prompts — thin but non-zero. GRPO also gets signal wherever any rollout passes (~similar density). |
| **vLLM tokens/sec on H100** (PLAN §7 step time) | **Yes** | **~4,198 tok/s mean** (Phase 1 wall clock 271s for 1600 rollouts); last batch logged ~2,632 tok/s | H100 throughput is solid for 1.7B policy rollouts. Use for §7 step-time estimates; Group B still needed for full step decomposition. |
| **Judge wall-clock / VRAM / $/call** (PLAN §5 judge hosting; CoT go/no-go) | **Mostly yes** | p50 **1.30s**, p99 1.68s; median **$0.00143/call** (~$0.29 for 200); output tokens p50=556; **0 truncated**; **96.5% judge JSON parse ok** (7/200 failed); cluster_count mean 2.6; **VRAM = 0 (logging bug on this run)** | Judge is **fast and cheap per call**. Sidecar vLLM looks feasible. Minority-CoT **not blocked on latency/cost**, but re-run VRAM with the fix before committing same-GPU vs sidecar GPU. |
| **Drop easy problems where 8/8 correct?** (PLAN §2) | **No** | Pass rates too low to identify "solved" problems | Defer until parser fixed and/or after some RL steps. |
| **GPU class lock** (PLAN §7) | **Partially** | Confirmed H100 works; ~4.2k tok/s policy, ~1.3s/judge call | H100 is validated for Group A workload. Group B still needed for collocated train+vLLM VRAM. |

---

## Phase 2 judge details

From the wandb Table (200 rows):

- **193/200** successful JSON parses (7 failures; logs show at least problem_ids 169, 198)
- **0/200** truncated inputs
- Total judge GPU time ≈ **262s** for 200 batched calls
- **191/200** prompts had cluster_id=100 (degenerate bucket) hits — judge is producing clusters, but the degenerate bucket is heavily used (worth watching during training)

---

## Per-band breakdown (Phase 1)

| Band | parse_ok | pass rate | mixed-reward prompts |
|---|---|---|---|
| 0/8 | 56.5% | 2.5% | 2/25 |
| 1/8 | 55.0% | 4.0% | 5/25 |
| 2/8 | 52.0% | 4.5% | 4/25 |
| 3/8 | 53.5% | 3.5% | 5/25 |
| 4/8 | 49.5% | 1.5% | 3/25 |
| 5/8 | 60.5% | 2.0% | 2/25 |
| 6/8 | 58.5% | 1.5% | 3/25 |
| 7/8 | 62.0% | 4.0% | 6/25 |

**Aggregate:** parse_ok_rate=55.9%, pass_rate=2.9%, mixed_reward_prompts=30/200 (15.0%), length_cap_hits=20/1600.

**Response length distribution (200-run):**

| Cap | Rollouts at/above |
|---|---|
| ≥2048 | 74/1600 (4.62%) |
| ≥3072 | 37/1600 (2.31%) |
| ≥4096 (actual cap) | 20/1600 (1.25%) |

---

## Why we report rollout vs judge wall time

Group A measures **inference only** (no HF backward, no `update_weights`). That is still useful for PLAN.md because several open items depend on **how expensive the judge subsystem is relative to policy rollouts** — before Group B adds training time:

- **§5 / §3:** Judge hosting (sidecar vs same GPU vs API) and whether **Minority-CoT** stays in scope.
- **§7:** Partial **$/step** and wall-clock budget — rollout slice is known; judge slice was the big unknown.
- **§5:** Async rollout/train overlap is deferred, but **serial judge-after-rollout** would ~double GPU-active time if judge runs every step; this probe quantifies that risk.

We report n800 timing here so the training matrix can treat **CoT arms as ~2× inference GPU** (rollout + judge) vs **~1×** for GRPO / minority-answer / poly-epo-answer, without waiting for the full trainer probe.

---

## n800 timing — rollout vs judge (H100)

**Source:** wandb `mu8kj4ll` — 800 prompts × 8 rollouts (6400 completions), 800 batched judge calls (4B), separate Modal containers, `modal_price_per_sec: 0.001097`.

### Wall time and cost

| Phase | Model | Active GPU time (sum of logged calls) | Approx $ |
|-------|-------|--------------------------------------|----------|
| Phase 1 rollouts | Qwen3-1.7B-Base | **~993 s** (~16.5 min) | ~$1.09 |
| Phase 2 judge | Qwen3-4B-Instruct | **~1003 s** (~16.7 min) | ~$1.10 |
| **Ratio** | | **~1.01×** (judge ≈ rollout) | ~50/50 split |

Other n800 summary panels: `tokens_per_sec_mean ≈ 4502` (policy); judge median **~1.25 s/call**, **~$0.00137/call**; `judge_vram_gb_used ≈ 70.3` GB; `phase2_json_parse_ok_rate ≈ 96.1%`; 0 truncated judge inputs. vLLM rollout **decode cost** is driven by tokens actually generated, not `max_tokens` for every sequence; `max_model_len` still sets the KV memory budget.

### Extrapolation to one training step (rollout + judge only)

Probe shape matches one step if each of `P` prompts gets 8 rollouts and **one** judge call (8 completions batched per prompt):

| Prompts / step `P` | Rollout GPU (est.) | Judge GPU (est.) | Notes |
|--------------------|--------------------|------------------|-------|
| 32 | ~40 s | ~40 s | Linear scale from n800 |
| 64 | ~79 s | ~80 s | |
| 128 | ~159 s | ~160 s | Poly-EPO-ish batch size |

Formula: `rollout_s ≈ (P×8/6400)×993`, `judge_s ≈ (P/800)×1003`. **Train/backward not included** — Group B still gates full §7 $/step.

### What this answers in PLAN.md

| Open item | n800 timing readout |
|-----------|---------------------|
| Judge cost / CoT go-no-go | **In scope** on latency and $/call; **serial in-loop judge ≈ doubles** GPU-active time vs rollout-only arms at similar `P`. |
| Judge hosting | Each engine fits **one H100** alone (~71 GB policy, ~70 GB judge). **Collocated** train+vLLM+judge not measured here. |
| vLLM throughput | **~4.5k tok/s** mean on 1.7B — use for rollout portion of step estimates. |
| Full step time / $/arm | **Not answered** — need Group B (backward, weight sync, collocated VRAM). |
| GPU H100 vs H200 $/throughput | **Not answered** — H100 only on this run. |

### Wandb `_step` axis (not “one rollout = one step”)

| Region | `_step` | What was logged | Count |
|--------|---------|-----------------|-------|
| Phase 1 | 128 … **6400** | Batch scalars (`vllm_tokens_per_sec`, `wall_clock_s`) | **50** logs (batch size 128) |
| Gap | **6401–7399** | **Nothing** — intentional buffer | 999 empty steps |
| Phase 2 | **7401–8200** | Per-prompt `judge_wall_clock_s` | **800** logs |

Phase 2 starts at `n_rollouts + 1000` (= 7400) so judge scalars do not collide with Phase 1’s max step (6400). The gap is not missing work — it is unused chart space (`_phase2_step_offset` in `group_a_rollout_judge.py`).

---

## Addendum (2026-05-25): n800, Rank-2, prompt A/B/C, arm C locked

**Context:** 200-run readout above used DAPO arm A + Minerva-only `parse_ok`. Same day: offline Rank-2 rescore on saved completions, 800-prompt rerun (arm A), and parallel Modal rollouts for arms B/C (`probes/05-25/prompt_b/`, `prompt_c/`). Full design: [`prompt_probe.md`](./prompt_probe.md); narrative: [`../timeline.md`](../timeline.md).

### Rank-2 on arm A (offline rescore, n=6400)

| Metric | n=200 | n=800 | Notes |
| --- | --- | --- | --- |
| `parse_ok_rank2` | 83.8% | **84.8%** | Boxed-first ∪ Minerva (`main/train/reward.py` `extract_rank2`) |
| `parse_ok_minerva` | 55.9% | 60.3% | Old Group A headline metric |
| `mixed_reward` (rank2) | 15.0% | **26.5%** | GRPO signal density under rank2 reward |

**Read:** Low Minerva `parse_ok` was **format compliance**, not a broken parser. Rank-2 is locked for training.

### Prompt A/B/C (offline rank2 on each arm’s rollouts, n=6400 each)

| Metric | A (DAPO) | B (VeRL MATH) | C (Hybrid) |
| --- | --- | --- | --- |
| `has_answer_line` | 60.7% | 2.2% | 42.9% |
| `has_boxed` | 33.7% | 89.3% | **90.2%** |
| **`parse_ok_rank2`** | 84.8% | 79.0% | **87.6%** |
| pass_rate | 6.0% | 8.6% | 8.3% |
| **`mixed_reward`** | 26.5% | 30.9% | **33.9%** |
| residual (`extract_path: none`) | 15.2% | 21.0% | **12.4%** |

**Decision:** Lock **`hybrid_answer_boxed` (arm C)** for training and Group B. Config key `prompt_variant` in `main/train/prompts.py`. Fallbacks if convergence issues: `dapo_answer_v1`, `verl_math_boxed`.

**PLAN §7 signal density (arm A n800):** **73.4%** of prompts are all-wrong (0/8 correct) → only ~27% of batch contributes under GRPO zero-advantage filter. Arm C lowers all-wrong to **~65.9%** (~30% more effective prompts). Flag: DAPO-style dynamic sampling or curriculum (see PLAN §7).

**Conservative §5 rule in `prompt_probe.md` would default to A** (C did not hit +5pp parse_rank2 threshold); team adopted C on RL-relevant metrics anyway.

---

## What we still can't decide from Group A alone

1. **§2 sampling** — band-level pass rates still noisy/low to finalize train subset size and “drop 8/8-easy” filter.
2. **§7 full collocated step cost** — rollout + judge slices are in n800 timing above; **Group B** (in flight) gates backward, `update_weights`, microbatch, collocated VRAM, $/step.
3. **GPU SKU** — H100 only here; H100 vs H200 $/step not compared (optional thin re-run, not blocking).

---

## Recommended next moves

1. ~~**Fix parser** — Rank-2 shipped; hybrid prompt locked.~~
2. ~~**Update PLAN.md** with length cap, H100 rollout/judge $, parser/prompt lock~~ — see PLAN §5/§7 updates (2026-05-25).
3. **Group B readout** → lock microbatch, collocated `gpu_memory_utilization`, step time, async go/no-go.
4. **§2 freeze** — materialize `polaris_train.jsonl` once band/size decision is made.
