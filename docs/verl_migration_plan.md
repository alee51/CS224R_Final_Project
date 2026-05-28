# VeRL migration plan

**Status:** plan (2026-05-28)
**Companion docs:** [`verl.md`](./verl.md) (framework overview, knob reference) · [`../main/docs/verl_move_ta_meeting.md`](../main/docs/verl_move_ta_meeting.md) (raw TA notes) · [`../main/docs/ta_discussion.md`](../main/docs/ta_discussion.md) (paper framing, Paths A/C/D) · [`../main-verl/README.md`](../main-verl/README.md) (target codebase layout)

## 0. Framing

We're not rewriting the project on verl — we're **lifting the custom logic** (minority-voting advantage, CoT cluster assignment, judge) **onto a trusted training stack** so that (a) the TA-named coding bar is met, (b) bs=128 and 4B actually fit, (c) we stop paying engineering tax on answer extraction / FA2 / weight sync.

Estimates below are in **GPU-hours and agent-sessions**, not wall-days. A "session" = one focused agent-driven block (~1–3 hours of human attention, agent does the heavy lifting). We expect the full migration through Path C parity in **3–5 sessions**, not a week.

Compute is denominated in **B200-hours on Modal**, $/hour ≈ $6 (recent observed); $/step ≈ $0.15 at our seq-length / N=8 / bs=64. 4B roughly 2× that.

## 1. Stages and gates

Each stage has a smoke that **must pass** before the next stage starts. If a smoke fails the kill criterion, fall back rather than spending more on bring-up.

| # | Stage | Smoke gate | Budget | Kill criterion | Owns |
|---|---|---|---|---|---|
| 1 | Modal image + Ray + verl bring-up | `hello_verl.py` loads Qwen3-1.7B on B200, prints rollout | ~1 B200-hr, 1 session | >3 image rebuild cycles for cu128/vllm pin churn | `main-verl/infra/` |
| 2 | GRPO parity smoke | 50-step GRPO on Polaris, mean_reward + prompt_coverage within ±10% of `main/` at step 50 | ~3 B200-hr per attempt, expect 2 attempts | Reward contract incompatible, parity gap >20% after 2 fixes | `main-verl/configs/` + parity wrapper |
| 3 | **Port minority-voting objective** | Unit tests from `main/tests/test_objective_minority.py` pass against verl's group buffer | ~0 GPU (CPU unit tests) + 1 short verl run (~2 B200-hr) | Verl hook surface doesn't expose per-group rollout tensors cleanly | `main-verl/train/objective_minority.py` |
| 4 | Judge service on Modal | OpenAI-compatible `/v1/chat/completions` up, async client does 64-way semaphore fan-out from trainer | ~2 B200-hr (judge GPU) + 1 session | Judge latency >1s/call at 64-way concurrency | `main-verl/judge/` |
| 5 | **CoT-clustering arm** | Cluster IDs from judge → minority advantage path, end-to-end 50 steps | ~4 B200-hr | Judge load on training loop adds >25% per-step latency even on separate GPU | `main-verl/train/objective_minority_cot.py` |
| 6 | 4B fit check (Path C) | Qwen3-4B-Base loads with bs=128 on ≤4× B200, micro_batch tuned | ~2 B200-hr (1× 4B), then ~4 B200-hr (4× 4B) | VRAM OOM at any micro_batch ≥1 with FSDP offload | new config |
| 7 | Filter manifest + retrain | Decision on Polaris-53K vs 51K-filtered vs 4B-recalibrated (see §4); retrain GRPO + minority on chosen manifest | ~$150 base rollout pass + ~$400–700 retrain | (none — this is the experiment) | new manifest + configs |

**Stage 3 is the project.** The other stages are bounded plumbing; this one is where the coding-component grade lives and where verl's hook surface might fight us. Spend the agent attention proportionally.

## 2. Stage 3 deep-dive: porting the minority-voting objective

This is the most important part of the migration. Three things have to be true after Stage 3:

1. **Correctness.** The advantages computed by the verl-hosted `minority_answer` match the custom-trainer outputs on a fixed `(rollouts, rewards, cluster_ids)` fixture to within float tolerance. Unit tests from `main/tests/test_objective_minority.py` are the oracle — port them verbatim, swap the import.
2. **Verl hook is clean.** We want to extend the advantage-estimator path, not monkey-patch the trainer loop. Two candidate hooks (from `verl/trainer/ppo/core_algos.py`):
   - **Custom `adv_estimator`** registered via Hydra (`algorithm.adv_estimator: minority_answer`). Cleanest if verl exposes the post-reward, pre-clip tensor of shape `(batch, n_rollouts)` along with per-rollout cluster IDs.
   - **Reward-fn-returns-advantage pattern.** Compute the minority advantage *inside* a custom reward function and route it through. Hackier; only fall back to this if the adv_estimator surface is private.
   Investigate which exists in our pinned verl version *before* writing the port — this is the highest-uncertainty piece of the migration.
3. **Cluster IDs are passable.** For `minority_answer` the cluster ID is `hash(extracted_answer)` — computable from rollout strings + grader output, no judge needed. For `minority_cot` (Stage 5) we need to thread judge outputs through to the same advantage path. Design the cluster-ID input to Stage 3 with Stage 5 in mind: take cluster IDs as an input tensor, not recompute them inside the estimator.

**Ports to write (in order):**

1. `main-verl/train/cluster_answer.py` — pure function, `rollouts → cluster_ids`. Lift from `main/train/clustering.py`. CPU-only, no verl deps. Should be ~30 lines.
2. `main-verl/train/objective_minority.py` — the advantage estimator. Take `(rewards, cluster_ids)`, return `advantages`. Lift from `main/train/objective.py`. The math is settled; the only new code is the verl hook adapter.
3. `main-verl/tests/test_objective_minority.py` — copy fixtures from `main/tests/`, point at the new module.
4. Verl integration glue (Hydra registration, type adapter for the rollout tensor shape).

**Things to NOT port:**
- Our custom Rank-2 hybrid parser (`main/train/reward.py`). Stage 2 decides whether to use verl's `math_dapo` straight or wrap mathd∨sympy. Either way, the parser doesn't follow `objective.py` into verl.
- `main/train/weight_sync.py`, `main/train/trainer.py`, `main/train/rollout.py`. Verl owns these.
- `main/train/loss.py`. Verl's GRPO loss is the reference; we only override the advantage computation.

**Stage 3 done when:**
- All unit tests green.
- One short verl run (~50 steps, GRPO arm + minority arm) logs `train/mean_advantage` and `train/distinct_clusters` per step.
- Sanity: minority arm's advantage distribution differs from GRPO's on the same batch (proves the override actually ran).

## 3. Stage 1–2 deep-dive: smoke + parity

Plumbing, but parity is what tells us we can trust subsequent science.

**Stage 1 — `hello_verl.py`:**
- Modal image: verl + pinned vllm (verl ≥0.7 expects vllm ≥0.8.2; our `main/` is on 0.9.0 + cu128 for B200, so the pin may align or may need a verl version bump — check first).
- B200-specific verl flags from `verl.md` §6.2: `enforce_eager=True`, `model_dtype=bfloat16`, `ray_init.num_gpus=N` explicit. **Don't skip these** — they're known foot-guns on Blackwell.
- Smoke: load Qwen3-1.7B-Base, do one rollout, print. ~1 B200-hr including image build.

**Stage 2 — GRPO parity gate:**
- Config: GRPO, Qwen3-1.7B-Base, Polaris (whichever manifest we pick — see §4), KL=0 (override verl's default `use_kl_loss: True`), `rollout.n=8`, bs=64 single GPU first, then bs=128 multi-GPU.
- 50 steps. Compare to `main/` wandb run at the same step (we have `grpo_s59` from the LR=3e-6 redo as the cleanest comparator).
- Metrics that must agree to within ±10%: `train/mean_reward`, `train/prompt_coverage` (or its verl equivalent), `train/mean_response_length`.
- **Reward contract decision happens here.** Two options, pick at smoke time:
  - **Use verl `math_dapo` reward straight.** TA-aligned, zero custom code, but checkpoints aren't comparable to `main/` B200 runs (different parser → different reward signal).
  - **Wrap our Rank-2 hybrid + mathd∨sympy as a `custom_reward_function`.** Comparable, but adds ~100 lines and partly defeats the "use what verl ships" point.
  Recommendation: try `math_dapo` first (5 min), check parity. If it's within ±10% of `main/`'s reward curve at step 50, ship it and don't write the wrapper. If it diverges, wrap.

**Parity smoke fails → don't proceed to Stage 3.** Wrong reward contract or wrong KL means the minority-objective port will be ported against the wrong baseline.

## 4. Open question: Polaris-53K vs Polaris-51K-filtered vs 4B-recalibrated

This is unresolved and affects every retrain decision downstream.

**The data (band drop analysis of our 51K filter):**

| Band | Normalized drop rate | Obs / expected | Read |
|---|---|---|---|
| 0/8 | 0.997 | 619 / 621 | **Proportional — our filter is NOT predominantly removing gradient-starvers.** |
| 1/8 | 0.961 | 270 / 281 | proportional |
| 2/8 | **1.361** | 300 / 220 (+80) | **Over-filtered — we're losing mixed-reward signal.** |
| 3/8 | **1.268** | 263 / 207 (+56) | **Over-filtered — same.** |
| 4/8 | 0.969 | 204 / 210 | proportional |
| 5/8 | 0.839 | 160 / 191 (−31) | under-filtered |
| 6/8 | 0.978 | 158 / 162 | proportional |
| 7/8 | **0.684** | 178 / 260 (−82) | **Under-filtered — we're keeping near-trivial prompts.** |

**The tension:**
- The 51K filter caught real **gold-leak** prompts (answer-in-prompt cases that would silently inflate rewards). That's load-bearing — can't just revert.
- But the filter's *band* selectivity is wrong for set-RL: it disproportionately removed 2/8 and 3/8 prompts (the densest signal for set-based reweightings) while keeping 7/8 (near-trivial). For minority/poly_epo arms specifically, the 51K we trained on may be **worse** than raw 53K despite the gold-leak win.
- For Path C (4B), the right move per Polaris's own recipe is a **model-calibrated** rollout-pass filter — run an N=8 pass with Qwen3-4B-Base, drop the 0/8 and 8/8 tails (or stricter), keep mid-band. That's a different filter than either 53K or 51K.

**Options for verl runs:**

| Option | Manifest | When it's right |
|---|---|---|
| **A** | Polaris-53K raw | If we want max coverage and trust verl's reward to penalize gold-leak completions on the fly. Risk: gold-leak still poisons advantage estimates. |
| **B** | Polaris-51K (our current filter) | Default — preserves continuity with `main/` results, gold-leak removed. Risk: under-filters the high-pass-rate tail; mixed-reward density is worse than it should be. |
| **C** | Polaris-51K + 4B rollout-pass filter (strict `0 < pass_rate < 1`) | Right for Path C. Costs a Phase 1 rollout pass on 4B (~$120–150) but produces the manifest the paper wants. |
| **D** | Polaris-53K + gold-leak filter only, no pass-rate filter | If we believe band-skew was the real cost of the 51K filter. Cheap to produce (just re-run the gold-leak heuristic on 53K). |

**Recommendation as a default:** **Option B for Stage 2 parity** (so we're comparing apples to apples vs `main/` B200 checkpoints), then **Option C as the manifest for Path C retrain** in Stage 7. Option D is worth ~30 min to produce as a sanity check — if band-skew flips on the re-filtered 53K, that's directly useful for the writeup's data section.

This decision doesn't need to be made before Stage 1. It does need to be made before Stage 7.

## 5. GPU and credit allocation across Modal accounts

Three Modal accounts: **nancy** (least credits), **anastasia**, **emma**. Reported cap of **10 B200s per account** at our previous usage peak — assume that's still the ceiling. Theoretical max concurrent: 30 B200s across accounts, but each verl job runs in one account at a time (Ray cluster can't span Modal accounts).

**Allocation rule of thumb:**

| Workload | Account | Why |
|---|---|---|
| Stages 1–3 smokes (≤4× B200, <1 day total) | **nancy** | Burn least-credit account on bring-up; if something goes wrong, the loss is bounded. |
| Stage 4 judge service (1× B200, long-lived) | **nancy** or **anastasia** | Judge runs alongside training, low marginal cost. |
| Stage 2 parity production runs | **anastasia** | Cleaner credits, OK to run for several hours. |
| Stage 6–7 4B + retrain (large, expensive) | **emma** | Most credits → big runs. |
| Parallel arms (GRPO + minority + cot in parallel) | one per account | Verl can't batch arms; running them on three accounts simultaneously is the only way to parallelize. |

**Parallel arms is the one place 3 Modal accounts is a real speedup** — for the final Stage 7 retrain we should fire all three arms simultaneously across accounts. Single-account would serialize them, which is what blew up wall-clock in `main/`.

**GPU layout per stage (per `verl.md` §6, §7):**
- Stages 1–3: 1× B200 sufficient.
- Stage 5 (CoT arm): 4× B200 train + 1× B200 judge → 5 B200s in one account (within cap).
- Stage 6 (4B fit): start with 1× B200, scale to 4× B200 once fit confirmed.
- Stage 7 retrain: 4× B200 per arm; if running three arms in parallel across three accounts, that's 12 B200s of concurrent burn but bounded wall-clock.

## 6. Cost estimates from initial smokes

Don't trust the numbers below until Stage 1 produces real `$/step` measurements. Plan structure for cost estimation:

**After Stage 1 (hello smoke):** record image build time + cold-start latency. These are sunk per session, not per step, so they distort short runs.

**After Stage 2 (parity smoke):** record `$/step` at bs=64 single-GPU and bs=128 multi-GPU. This is the **load-bearing number** — it tells us:
- Is verl actually cheaper than `main/` per step at bs=128? (Hypothesis: yes, by 30–50%, because FSDP path beats our per-seq logprob loop.)
- What does a full 799-step retrain cost on verl? (Multiply.)
- Is Stage 7 across three accounts in budget? (Three arms × 4× B200 × 799 steps × $/step.)

**After Stage 4 (judge up):** record judge `$/call` at 64-way concurrency. Determines whether CoT arm is feasible at all.

**After Stage 6 (4B fit):** record `$/step` for 4B; multiply by Path C retrain step count.

Rough priors (replace with real numbers as smokes land):
- 1.7B GRPO step at bs=64 on 1× B200: ~$0.15/step (our `main/` data).
- 1.7B GRPO step at bs=128 on 4× B200 via verl: expect ~$0.20–0.25/step (4× cost / 2× throughput ≈ 2× per-step, but 2× fewer steps per epoch).
- 4B step on 4× B200 via verl: expect ~2× the 1.7B 4× cost ≈ $0.40–0.50/step.
- Judge: depends on model size; smallest reasonable judge (Qwen2.5-7B-Instruct) on 1× B200 is ~$6/hour serving capacity = lots of calls/hour, probably not the bottleneck.

Update this section after each smoke lands. Make `$/step` the single number the plan tracks against budget.

## 7. What's NOT in this plan

- **Pre-milestone code cleanup.** Out of scope; the migration is additive to `main-verl/`.
- **Eval harness rewrite.** Verl checkpoints are FSDP-format, not `step_*.pt`. We need a small loader shim in `main/eval/passk.py` to read them. ~30 min of work, add to Stage 3 close-out.
- **Re-running `main/` ablations on verl.** The ablations in `main/docs/timeline.md` (prompt format, parser rank, grader choice, data source) are settled and cited in the paper as-is. No need to redo on verl.
- **Multi-node verl (>8 GPUs in one job).** Modal's clustered API is private beta. Stay single-node; if we need more parallelism, use multiple Modal accounts (§5).

## 8. Open decisions for the next session

- [ ] Stage 2 reward contract: `math_dapo` straight, or wrap mathd∨sympy? (Decide after parity smoke.)
- [ ] §4 manifest for Stage 7: B + C (recommended), or also produce D as sanity check?
- [ ] Stage 5 judge model: Qwen2.5-7B-Instruct (cheap, fast, possibly under-capable), or step up to a 14B/32B? Decide after a quick eval of judge-on-CoT agreement vs human spot-check on ~50 examples.
- [ ] Whether to keep `minority_answer` on the verl run list at all, or skip straight to `minority_cot` (TA: answer clustering → reward hacking). Cost of skipping: lose the apples-to-apples comparison with the falsified 1.7B-unfiltered baseline.
