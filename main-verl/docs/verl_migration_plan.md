# VeRL migration plan

**Status:** plan v3 (2026-05-29) — 1-epoch default, poster framing outcome-dependent. v2 (2026-05-28, post-OH).  
**Epistemic note:** This is a *proposed* migration path from preliminary VeRL survey + TA direction. Stages, smoke gates, and kill criteria are how we learn what actually works — nothing below is guaranteed until the corresponding smoke passes.  
**Companion docs:** [`verl-reference.md`](./verl-reference.md) (survey notes + knobs — hypotheses, not facts) · [`../../main/docs/verl_move_ta_meeting.md`](../../main/docs/verl_move_ta_meeting.md) (raw TA notes) · [`../../main/docs/ta_discussion.md`](../../main/docs/ta_discussion.md) (paper framing, Paths A/C/D) · [`../README.md`](../README.md) (codebase layout) · [`human-remaining-work.md`](./human-remaining-work.md) (due dates + deliverables)

## 0. Framing

The science question is fixed: **does set-based RL with CoT-clustered cluster IDs find signal that answer-clustering missed at our model scale?** The remaining work is engineering + execution. This plan describes our *intended* path from empty `main-verl/` toward a 3-arm full retrain on Qwen3-4B-Base, Polaris-51K filtered, **1 epoch** (2 only if time permits after Mon deadline — unlikely; see [`human-remaining-work.md`](./human-remaining-work.md)). We may need to fall back (1.7B, fewer arms, etc.) if bring-up smokes fail — see kill criteria in §2.

**Stack:** Training runs on the **VeRL tree inside [tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl)** — an earlier pinned snapshot, **not** `pip install verl` from upstream. We do **not** run the repo’s **MaxRL algorithm** (`adv_estimator=maxrl`); baseline arm is **GRPO**, science arms are **minority_cot** / **poly_epo_cot**. Use the repo for fork stability + custom `adv_estimator` / reward examples (TA OH 2026-05-28).

**Poster narrative is not locked** until VeRL results land. If minority_cot beats GRPO → lead with separation. If not → analysis (why minority voting fails, collapse diagnostics). If poly_epo_cot underperforms vs the paper or our 1.7B runs, treat as inconclusive at 1 epoch (would be more alarming at 2). The 1.7B falsification table stays background either way.

**Schedule:** [`human-remaining-work.md`](./human-remaining-work.md) — training must **start Fri 2026-05-29 EOD**; finish by Mon 2026-06-01 11:59 PM.

Estimates are in **B200-hours and agent-sessions**, not wall-days. A "session" = one focused agent-driven block (~1–3 hours of human attention). Bring-up through Stage 7 is **~12 hours of focused code time** (Fri); Stage 8 is up to **3 days** wall if started on schedule.

**Integration principles (same as [`verl-reference.md`](./verl-reference.md)):**
- **Fork VeRL first** (maxrl repo) — config + hooks over re-porting `main/train/*` plumbing.
- **`main/` code** — algorithm/fixture reference for minority math and tests only; wire through VeRL extension points.
- **`main/` numbers** — not parity targets or cost priors; re-measure $/step and curves on VeRL smokes.

## 1. TA-resolved decisions (policy locked; implementation still to prove)

These are **project/policy** choices from TA discussion — not proof that VeRL + Modal + our hooks will work as written. Plan stages below assume we try to honor them; reopen if a smoke gate forces a fallback.

| Decision | Resolution | Source |
|---|---|---|
| Framework | **VeRL via [tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl)** — vendored fork, not upstream `verl` package. TA: cleaner than verl-project proper; good `@register_adv_est` / reward examples. **Not** the MaxRL training method itself. | OH 2026-05-28 |
| Manifest | **Polaris-51K filtered** (current). Band-skew of the filter is a paper analysis item, not a re-filter trigger. | OH 2026-05-28 |
| Online vs offline | **Online RL.** Round-based / ReST-style is ruled out — judge HTTP service is required infra. | OH 2026-05-28 |
| Reward / extraction | VeRL built-in **MathReward + `\boxed{}`** — no Rank-2 hybrid wrapper. | OH 2026-05-28 |
| Clustering substrate | **CoT, not answer.** Answer-clustering deprioritized to "only if time" (likely won't happen). | OH 2026-05-28 |
| Arms | **GRPO + minority_cot + poly_epo_cot** (the 3rd arm is the TA's call — "could lead to cool analysis"). | OH 2026-05-28 |
| Epochs | **1 epoch** (TA noted 2 can help convergence; **2 only if time** after Mon — unlikely) | OH 2026-05-28; schedule 2026-05-29 |
| Model | **Qwen3-4B-Base** if it fits ("which we should be able to fit"). 1.7B is fallback only. | prior TA + OH |
| Batch size | **128**, split across GPUs (we *expect* verl to handle this — unverified). | prior TA |
| Judge architecture | 1 judge instance, **async, 32–64 concurrent calls via `asyncio.Semaphore`**. Modal-hosted as separate function. | prior TA + OH |
| Mid-run eval | Yes. `trainer.log_val_generations=10` + AIME-25 + 2K Polaris periodically. | OH 2026-05-28 |
| If arms don't separate | Analysis-as-contribution + collapse diagnostics (TA: first to try minority voting at scale). Poster angle chosen after results — see [`human-remaining-work.md`](./human-remaining-work.md). | OH 2026-05-28 |

## 2. Stages and gates

Each stage has a smoke that must pass before the next. Kill criterion = fall back rather than spend more on bring-up.

| # | Stage | Smoke gate | Budget | Kill criterion |
|---|---|---|---|---|
| 1 | Modal image + maxrl repo + Ray bring-up | `hello_verl.py`: `pip install -e .` from maxrl clone; loads Qwen3-1.7B on B200, prints one rollout | ~1 B200-hr | >3 image rebuild cycles for pin churn (fork pins torch 2.6 / vllm 0.8.4 vs our B200 stack — may need overrides) |
| 2 | GRPO bring-up smoke (1.7B, MathReward) | 50 steps; completes without OOM; reward/length metrics sane. **Not** numeric match to `main/`'s `grpo_s59` (different grader + stack) | ~3 B200-hr per attempt | Cannot complete 50 steps after 2 config fixes — escalate |
| 3a | **`minority_cot` skeleton with mock cluster IDs** | Unit tests pass (math fixtures from `main/tests/test_objective_minority.py`); 50-step run logs `train/mean_advantage` differing from GRPO | ~2 B200-hr (judge mocked) | `adv_estimator` hook cannot register or lacks per-group tensors — redesign hook (see fork’s `core_algos.py` examples) or escalate |
| 4 | Judge service on Modal | OpenAI-compatible `/v1/chat/completions` up; async client does 64-way semaphore fan-out; <1s/call median at batch | ~2 B200-hr (judge GPU) | Latency >2s/call at 64-way — re-size judge or restructure |
| 3b | Wire real judge → `minority_cot` end-to-end | 50-step run with live judge; `train/distinct_clusters` non-trivial; no per-step latency increase >25% | ~3 B200-hr | Step time >2× Stage 3a — investigate async path |
| 5 | **`poly_epo_cot`** | Unit tests pass (algorithm fixtures); 50-step run logs distinct advantage profile from minority_cot | ~2 B200-hr | Same hook constraint as 3a |
| 6 | 4B fit check (Path C) | Qwen3-4B-Base loads at bs=128 on ≤4× B200; micro_batch tuned; one 50-step smoke for each arm | ~6 B200-hr (4× B200 × ~1.5 hr) | OOM at all micro_batch settings with FSDP offload — fall back to 1.7B |
| 7 | Logging + mid-run eval wiring | Verl run emits all §5 metrics; `log_val_generations=10` writes; AIME-25 + 2K Polaris eval scripts trigger on checkpoint | ~1 B200-hr | Skip if blocking; not gating |
| 8 | **Full 3-arm 1-epoch retrain on 4B-filtered** | All three arms complete 1 epoch; mid-run evals land | **TBD** after Stage 6 `$/step` | Wall budget / credit cap |

**Stages 3a–3b–5 are the science.** Everything else is plumbing we *believe* is standard VeRL/Modal work — but image pins, Ray on Modal, hook surfaces, and judge latency are all unvalidated until the smokes pass.

## 3. Stage 3 deep-dive: `minority_cot` (and the mock-clusters trick)

The hardest part of the migration. Split into 3a (no judge) and 3b (with judge) so we don't block the objective on judge bring-up.

**Scope rule:** only the advantage math + cluster-ID interface are custom. Rewards, rollouts, loss, and weight sync stay on VeRL — see [`verl-reference.md`](./verl-reference.md) §3–5.

**3a — mock cluster IDs.**

Goal: verify the VeRL hook surface in isolation. Cluster IDs come from a **deterministic mock** (e.g., `hash(rollout_text[:50]) % n_clusters`) — meaningless semantically but exercises every code path except the judge.

Three things have to be true after 3a:
1. **Correctness:** VeRL-hosted advantages match the **same mathematical fixture** as `main/tests/test_objective_minority.py` (algorithm only — not full trainer parity with `main/`).
2. **Hook is clean:** register custom `adv_estimator` via Hydra (`algorithm.adv_estimator: minority_cot`). Template: fork’s `@register_adv_est` entries in `verl/trainer/ppo/core_algos.py` (including the existing `maxrl` estimator — **wiring reference only**). Fallback: reward-fn-returns-advantage pattern. Do not copy `main/train/trainer.py` integration patterns.
3. **Stage 5 compatible:** `poly_epo_cot` shares the same cluster-ID input. Design the cluster-ID interface as `(batch, n_rollouts) -> cluster_ids` once, used by both objectives.

Files to write (minimal custom surface):
1. `main-verl/train/cluster_cot.py` — mock + judge-call interface (prompt ideas may reference `main/probes/group_a_rollout_judge.py`).
2. `main-verl/train/objective_minority.py` — advantage estimator (math reference: `main/train/objective.py`).
3. `main-verl/train/objective_poly_epo.py` — poly_epo branch (same reference).
4. `main-verl/tests/test_objective_minority.py` + `test_objective_poly_epo.py` — port **fixtures**, not trainer mocks.
5. VeRL glue only: Hydra registration, type adapter between VeRL tensors and our math.

**3b — swap mock for real judge.** Once Stage 4 lands, replace the mock with the async judge client. Cluster IDs now come from the judge over HTTP. The verl integration code from 3a doesn't change — only the cluster-ID source.

**Things to NOT re-implement:** `main/train/{reward, weight_sync, trainer, rollout, loss}.py` — use VeRL. If a hook is missing, escalate per §3a kill criterion; do not fall back to porting the custom trainer piecemeal.

## 4. Stage 4 deep-dive: judge service on Modal

Online RL is locked → judge HTTP service is required. From the TA notes the architecture is *directionally* pinned (details still open — see §9):

- **One judge instance**, hosted as a separate Modal function on its own B200.
- **OpenAI-compatible API** (`/v1/chat/completions`) — easiest async client.
- **Async batched calls** from the trainer: `asyncio.Semaphore(64)` (or 32 if 64 OOMs the judge) for concurrent in-flight calls.
- **Judge serves the cluster-assignment prompt**, not generic chat. One judge call per `(prompt, 8 rollouts)` group → emits cluster IDs.

**Open: judge model size.** Not TA-resolved. Candidates: Qwen2.5-7B-Instruct (cheap, fast, possibly under-capable), Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct. Decision approach: run a 50-example judge-agreement spot-check (does the judge cluster the same rollouts consistently across re-runs at the same temperature) before committing.

**Open: cluster-prompt design.** Number of clusters allowed (free-form vs forced k=2..4), how to handle "no clear cluster" rollouts (assign to a "degenerate" bucket — TA mentioned tracking `# rollouts in degenerate cluster` as a metric, so this is expected).

Modal layout: `gpu="B200:1"` for judge, lives in its own Modal app. Trainer calls it over the public Modal URL with an auth token.

## 5. Logging + mid-run eval (per TA OH)

These are now first-class requirements, not nice-to-haves. Wire in Stage 7 (or earlier if cheap).

**Metrics to log per step (additions to verl defaults):**
- `train/distinct_clusters` — number of unique cluster IDs in the batch.
- `train/prompts_unlocked` — cumulative count of unique prompts with ≥1 correct rollout across the run.
- `train/critic_mean_score` / `train/critic_mean_reward` — both names appear in verl; log whichever it emits natively, add the other as alias.
- `train/mean_response_length` — early warning for gibberish / degenerate outputs.
- `train/degenerate_cluster_rollouts` — count of rollouts the judge couldn't cluster cleanly (uses the "degenerate" bucket from §4).

**What healthy training looks like (per TA):**
- Critic mean score / reward **increasing** over time.
- Training reward increasing.
- Mean response length stable (not collapsing to 0 or exploding).
- `distinct_clusters` non-trivial (if it collapses to 1, the objective is degenerate).
- `prompts_unlocked` monotonically increasing.

**Mid-run eval:**
- `trainer.log_val_generations=10` — verl-native, dumps 10 generations per eval step for qualitative inspection.
- **AIME-25 mid-run validation** — small (n=30), cheap, runs on checkpoint.
- **2K Polaris validation** — matches our `main/` decision-grade eval slice for comparability.
- Cadence: every ~50 steps (tune after Stage 7 measures eval wall-clock).

Custom eval scripts may end up easier than verl's val loop — TA noted this as an option.

## 6. Manifest decision (paper artifact)

TA locked **Polaris-51K filtered** for training. The band-skew analysis of that filter (over-removes 2/8 and 3/8; under-removes 7/8; 0/8 proportional) stays as a **paper analysis item** — it's an honest characterization of what the filter actually did, useful for the Data section even though we're not changing the manifest.

Not a stage; just a writeup TODO. Owner: paper draft, not this plan.

## 7. GPU and credit allocation across 3 Modal accounts

Three Modal accounts (**A** = fewest credits, **B**, **C** = most). Cap of **10 B200s per account** (observed at peak). Ray cluster can't span accounts → each verl job lives in one account.

| Workload | Account | Why |
|---|---|---|
| Stages 1–3a smokes (≤4× B200) | **A** | Burn lowest-credit account on bring-up; bounded loss if something breaks. |
| Stage 4 judge service (long-lived 1× B200) | **A** or **B** | Low marginal cost; can colocate with smokes. |
| Stages 5–7 (small production) | **B** | OK for hours-long runs. |
| Stage 8 full retrain (3 arms × 1 epoch × 4B) | **C** primary | Most credits → biggest single run if needed. |
| **Parallel arms in Stage 8** | one arm per account | Verl can't batch arms — running them on three accounts simultaneously is the only way to parallelize. **This is the one real speedup from having 3 accounts.** |

Stage 8 parallelization: GRPO on **A**, minority_cot on **B**, poly_epo_cot on **C**. Each ~4× B200 = 12 B200s of concurrent burn, bounded wall-clock.

## 8. Cost priors (VeRL-measured only — do not extrapolate from `main/`)

**Do not budget from `main/` $/step or step-time splits.** Ray, multi-GPU FSDP, and MathReward change the economics entirely. The numbers below are placeholders until Stage 1–2 produce real VeRL `$/step`; discard any row that still cites `main/` after first smoke.

**Placeholders (replace entirely after Stage 2):**
- ~~1.7B GRPO step at bs=64 on 1× B200: ~$0.15/step (from `main/`)~~ — historical only, not a VeRL prior.
- 1.7B GRPO on VeRL: **TBD** after Stage 2 smoke.
- 4B on VeRL: **TBD** after Stage 6; do not scale from `main/` 1.7B numbers.
- Judge: **TBD** after Stage 4 latency test.

**Stage 8 budget estimate (order-of-magnitude only — fill in after Stage 6):**
- ~400 steps/epoch/arm at bs=128 (manifest size / batch — unchanged).
- $/step × 400 × 3 arms + judge + evals = **TBD**; prior ~~$750–1K~~ guess used `main/`-derived $/step and should not be treated as a plan anchor until VeRL smokes land.

## 9. Open decisions (still need to be made)

**Internal (we decide, no TA needed):**
- **Judge model size** (§4). Approach: run 50-example agreement spot-check with Qwen2.5-7B first; step up to 14B only if cluster IDs are unstable.
- **Cluster-prompt design** (§4): k-free vs forced k=2..4, degenerate-bucket policy. Design as we go; first version in Stage 4.
- **Whether to retain `minority_answer` as an arm.** Default: no (TA said CoT-only, answer-clustering only if time). Confirm by ripping the answer-cluster code out of the plan.

**Possibly worth a future TA ping (async, not OH-blocking):**
- Path C explicit confirmation: keep a 1.7B comparison row in the paper, or fully pivot to 4B-only? "Use 4B if it fits" suggests fully pivot, but worth a one-line confirmation before Stage 8 spend (budget TBD after Stage 6).
- Whether the **band-skew analysis of the 51K filter** belongs in the paper's Data section or in an appendix. Low-priority.

## 10. What's NOT in this plan

- Pre-milestone code cleanup. Out of scope.
- Eval harness rewrite. Verl checkpoints are FSDP-format, not `step_*.pt` — needs a small loader shim in `main/eval/passk.py`, ~30 min, add to Stage 3b close-out.
- `pip install verl` from upstream verl-project — we use the **maxrl repo’s vendored tree** only.
- Running **`algorithm.adv_estimator=maxrl`** — that is the paper’s method, not our GRPO / minority / poly-EPO arms.
- Re-running `main/` ablations on the fork (prompt format, parser rank, grader choice, data source). Settled and cited in the paper as-is.
- Re-porting `main/` trainer modules when VeRL already provides the capability (reward, rollout, sync, loss, logging defaults).
- Chasing numeric parity with `main/` reward curves or $/step — compare arms within VeRL instead.
- Multi-node verl (>8 GPUs in one job). Modal's clustered API is private beta. Parallelism comes from multiple Modal accounts (§7), not multi-node.
- `minority_answer` arm. Demoted to "only if time" per TA; not on the critical path.
