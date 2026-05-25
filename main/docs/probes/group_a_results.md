# Group A probe — results readout

**Run:** [probe-A_nancy_05-25-2207](https://wandb.ai/224r-project/cs224r-minority-voting/runs/t33091vc) (`t33091vc`)  
**Artifacts:** Modal volume `main-artifacts` → `probes/05-24/group_a/` (1600 rollouts, 200 judged)  
**Pointer:** [`artifacts/05-24_group_a.pointer.json`](./artifacts/05-24_group_a.pointer.json)  
**Modal app:** `ap-c7FATv5JQ8K4BhL5UFlNVM` (full run; some logging bugs — see [`issues.md`](../issues.md))

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
| **`max_response_length` cap** (PLAN §5 / §7) | **Yes** | p50=565, p90=1319, p95=1974, **p99=4096**; 20/1600 (1.25%) hit length cap at 4096; ≥3072 = 2.3% | Most completions are well under 4096. **4096 is safe**; lowering to **3072** would truncate ~2% of rollouts — possible savings knob, not urgent. |
| **Reward parser sanity** (`parse_ok` rate) | **Yes — failed soft gate** | **55.9%** parse_ok (has_answer_line 56.4%, has_boxed 34.6%) | Far below ~90% target. **Escalate to Rank 2 parser** per probe plan before trusting reward signal. Do not lock `reward.py` as-is. |
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

**Length cap sensitivity:**

| Cap | Rollouts at/above |
|---|---|
| ≥2048 | 74/1600 (4.62%) |
| ≥3072 | 37/1600 (2.31%) |
| ≥4096 (actual cap) | 20/1600 (1.25%) |

---

## What we still can't decide from this run alone

1. **Parser** — 56% `parse_ok` is the biggest blocker; re-probe or implement Rank 2 before locking reward.
2. **§2 sampling** — band-level pass rates too noisy/low to finalize train subset.
3. **Judge VRAM** — logged 0 on this run; need one re-run with the fix in `f2c304f`.
4. **§7 full step cost** — Group B (trainer skeleton end-to-end step) still required for microbatch, collocated VRAM, and $/step.

---

## Recommended next moves

1. **Fix parser** (Rank 2 multi-path or investigate why DAPO `Answer:` + Minerva only gets 56% — many completions may lack a parseable `Answer:` line).
2. **Update PLAN.md** §2 / §5 / §7 with the numbers above (especially length cap = keep 4096, parser escalation, H100 throughput, judge ~$0.0014/call).
3. **Optional short re-run** only if you need judge VRAM + git SHA on a clean wandb run; Phase 1 numbers are already complete on `t33091vc`.
