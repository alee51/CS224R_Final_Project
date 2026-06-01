# VeRL reference

**Status:** reference doc (2026-05-28) — **preliminary survey notes, not validated in our stack**  
**Context:** Working hypotheses from reading the pinned VeRL tree in [tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl), TA discussion, and upstream VeRL docs — before `main-verl/` bring-up. Treat claims below as *what we think is true* until a stage smoke proves otherwise. Custom trainer in `main/` stays frozen. **Proposed run order:** [`verl_migration_plan.md`](./verl_migration_plan.md).

**Stack (important):**
- We run **`python -m verl.trainer.main_ppo`** from the **`maxrl` repo** — a pinned, paper-era snapshot of VeRL vendored inside that repo.
- We do **not** `pip install verl` from [verl-project/verl](https://github.com/verl-project/verl) directly; upstream is reference-only when the fork’s docs/code are unclear.
- The maxRL repo also ships the **MaxRL training algorithm** (`algorithm.adv_estimator=maxrl`). **We are not using that method** — our baseline is **GRPO**; our science arms are **minority_cot** and **poly_epo_cot**. We use the repo for a cleaner fork + `@register_adv_est` / reward wiring examples (TA OH 2026-05-28).

**Integration principles (read before porting anything):**
1. **Fork VeRL first.** Use built-ins in the maxrl repo’s `verl/` tree (rollout, FSDP, weight sync, GRPO loss, built-in scorers, async reward managers, wandb, checkpointing) via config and thin hooks — do not re-implement trainer plumbing from `main/`.
2. **`main/` code = algorithm reference, not drop-in modules.** Only port what VeRL has no equivalent for (minority / poly-EPO advantage math, CoT cluster assignment). Wire through VeRL's `adv_estimator`, reward, or logging extension points; expect reshape/adapters, not copy-paste.
3. **`main/` numbers = historical context, not VeRL budgets or parity targets.** Step time, $/step, rollout%, reward curves, and microbatch ladders from the custom stack do not transfer cleanly (different grader, Ray overhead, multi-GPU layout). Re-measure on VeRL smokes; do not extrapolate or gate success on matching `grpo_s59` ±10%.

---

## 1. What VeRL is

**VeRL** (Volcano Engine Reinforcement Learning) is ByteDance’s open-source RL post-training stack for LLMs. It implements the **HybridFlow** architecture: a single lightweight **controller** process runs algorithm logic, while **Ray worker groups** handle heavy distributed work (FSDP training, vLLM/SGLang rollouts, reward scoring).

Launch pattern:

```bash
python -m verl.trainer.main_ppo  # Hydra yaml configs
```

Reference workloads: DAPO math RL, Poly-EPO (Qwen3-4B on 4× H200).

### How we used VeRL before vs now

| Phase | Approach |
| --- | --- |
| **Pre–2026-05-28** | “VeRL-flavored” custom trainer in `main/train/` — read-and-lift from `verl/trainer/ppo/core_algos.py`, not an import |
| **Post–2026-05-28** | VeRL via **[tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl)** (`pip install -e .` in that repo) — satisfies TA coding-component bar. Not the upstream `verl` PyPI package. |

Our custom stack (`main/train/trainer.py`): vLLM collocated rollout + HF backward + custom `objective.py` for minority voting. **If bring-up succeeds**, VeRL should replace most of that plumbing; we would still own the minority-voting objective and judge integration.

---

## 2. Why VeRL (not `main/`)

**Policy:** TA direction at 2026-05-28 OH — reimplement on VeRL for the coding-component bar. **Engineering:** the knob mapping below compares *what we had* vs *what we plan to use from VeRL* — not a promise of 1:1 behavior. Smokes validate VeRL runs; they do not require reproducing `main/` metrics. **Runbook:** [`verl_migration_plan.md`](./verl_migration_plan.md). **Raw notes:** [`../../main/docs/verl_move_ta_meeting.md`](../../main/docs/verl_move_ta_meeting.md).

### Knob summary vs `main/` (target state — unproven)

| Knob | `main/` today | VeRL direction (planned) |
| --- | --- | --- |
| Answer extraction | Rank-2 hybrid + mathd∨sympy (`main/` only) | **VeRL MathReward** — `math.py` / upstream `math_reward.py` via patched router ([`reward-decision.md`](./reward-decision.md)) |
| Batch size | 64 single GPU (128 OOMs) | Up to 128, VeRL splits across GPUs |
| Clustering substrate | Answer-hash | **CoT clustering** via async LLM judge (answer clustering → reward hacking) |
| Model | Qwen3-1.7B-Base | Qwen3-4B-Base if it fits |
| Judge | Sidecar vLLM (planned) | One judge, async batched (32–64 parallel calls, `asyncio.Semaphore`) |
| Logging gaps | Partial wandb panel | Add: distinct clusters, prompts unlocked, critic mean score, response length |

---

## 3. What the fork *should* handle (from maxrl repo + survey — not yet verified on Modal)

### 3.1 Train ↔ rollout pipeline (prefer fork defaults)

Do **not** re-port `main/train/{rollout,weight_sync,trainer,loss}.py`. Start from **`qwen3_experiments/run_qwen3_training.sh`** and **`examples/maxrl_data_preprocess/polaris.py`** in the maxrl repo; only override what smokes require.

| We built in `main/` | Prefer VeRL built-in |
| --- | --- |
| vLLM collocated rollout | vLLM or SGLang rollout workers |
| HF actor + weight sync | FSDP/FSDP2 actor with vLLM weight update |
| Logprob capture for GRPO/PPO | Built into rollout → actor update |
| Microbatch sizing / OOM guards | `micro_batch_size_per_gpu`, dynamic batching |
| Multi-GPU prompt batch splitting | Ray resource pools |

Our bs=128 OOM on single collocated H200/B200 is the main pain we *hope* VeRL addresses via multi-GPU layouts (Poly-EPO recipe: 128 prompts / batch 64 on 4× H200 — reference only, not our measured result).

### 3.2 Algorithms (config, not code)

- **GRPO (our baseline):** `algorithm.adv_estimator: grpo`, `actor_rollout.ref.rollout.n: 8` — **not** `maxrl` (that is the paper’s different normalization; out of scope for our arms).
- **Our custom arms:** register `minority_cot` / `poly_epo_cot` via `@register_adv_est` — follow the pattern in the fork’s `compute_maxrl_outcome_advantage` in `verl/trainer/ppo/core_algos.py` as a wiring example only.
- **PPO, DAPO, DrGRPO, …** — may exist in the fork; [verl-recipe](https://github.com/verl-project/verl-recipe) is secondary reference if the fork lacks a recipe.

### 3.3 Math rewards / answer extraction

**Locked decision:** [`reward-decision.md`](./reward-decision.md) — mentor direction is **upstream `math_reward.py`** (boxed prompt + last `\boxed{}` + Hendrycks `strip_string` + string `==`). **Not** `math_verify`, **not** `math_dapo`, **not** `main/train/reward.py`.

The fork ships the scorer as **`verl/utils/reward_score/math.py`** (same logic as upstream `math_reward.py`). At maxrl @ `7197bbb` unpatched, `polaris` wrongly routed to `math_verify`; the router fix lives on the maxrl fork as commit **`cb8160f cs224r: route polaris/math_reward to math.py reward`** (branch `cs224r-patches`).

| `data_source` | Routed scorer | Extraction + compare |
| --- | --- | --- |
| **`polaris`** (our default) | **`math.py`** | Last `\boxed{}` → `strip_string` → `==` |
| **`math_reward`** (alias) | **`math.py`** | same |
| `lighteval/MATH`, … | `math_verify.py` (fork default) | unchanged on fork — not our Polaris path |
| `math_dapo`, … (unpatched fork) | `math_verify` | **Do not use** for our stack |

**Prompt (we supply in parquet — VeRL does not auto-append):** maxrl `examples/maxrl_data_preprocess/polaris.py` suffix:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Implemented in `main-verl/data/preprocess_polaris_verl.py`.

Custom rewards: `custom_reward_function.path` + `.name` in Hydra config — **not used** for Stage 2 smoke.

### 3.4 Reward Loop (v0.7+)

- Distributed reward workers (Ray)
- **Async** custom reward fns — `async def compute_score(...)` with HTTP calls
- Reward managers: `naive`, `dapo`, `batch`, `limit` (rate-limit semaphore), `remote`
- `launch_reward_fn_async: True` — overlap reward with logprob forward
- **GenRM pattern:** VeRL runs reward-model servers + HTTP **reward router**; custom fn calls `reward_router_address`

### 3.5 Data + recipes

- Parquet: `prompt` (chat messages), `reward_model.ground_truth`, `data_source`
- Preprocess template: [verl/examples/data_preprocess/gsm8k.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py)
- Working DAPO configs: [verl-recipe/dapo](https://github.com/verl-project/verl-recipe/tree/main/dapo)

### 3.6 Logging / checkpointing

wandb integration, step metrics, checkpoint save/resume — standard RL boilerplate. Custom scalars (clusters, prompts unlocked) still need hooks.

---

## 4. Where VeRL has slowdowns / friction

### 4.1 Ray overhead (especially on Modal)

VeRL assumes a **Ray cluster**. On a single Modal container: Ray head + workers + controller in one box → startup latency, extra processes, harder debugging.

Trade: simplicity (`main/` = one process) vs scale (VeRL = multi-GPU native). On **1 GPU**, overhead may dominate; on **4–8 GPUs**, throughput wins.

### 4.2 Weight sync still costs time

VeRL handles HF↔vLLM sync but doesn’t eliminate it. Our Group B finding stands: **rollout ≈ 73%** of step time. Experimental fully-async modes exist but add complexity.

### 4.3 Config footguns

| Issue | Our setting | VeRL default | Action |
| --- | --- | --- | --- |
| KL | KL=0 everywhere | `use_kl_loss: True` | Override explicitly |
| Loss aggregation | REINFORCE-with-clip | `token-mean` (not paper’s `seq-mean-token-mean`) | Document choice |
| Qwen3-1.7B-Base | Raw string prompts | `apply_chat_template` path | No HF chat template on Base — use plain-text prompts |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **Omit** in `main-verl/infra/modal_image.py` | Set in `main/infra/modal_image.py` (fragmentation cushion) | vLLM 0.9 `CuMemAllocator` asserts on startup when VeRL colocates FSDP actor + vLLM rollout (`stage-02-log` attempt 3). Stage 1 direct `LLM()` smoke did not hit this path. If fragmentation OOM later: drop micro-batch / `gpu_memory_utilization`; optional `max_split_size_mb:128` (vLLM-safe). |

### 4.4 Reward / grader mismatch

Our train grader in `main/`: **Rank-2 hybrid → mathd∨sympy** (`main/train/reward.py`). **VeRL stack:** patched **`math.py` / MathReward only** ([`reward-decision.md`](./reward-decision.md)) — do not wrap `main/train/reward.py` for parity. Different parsers → different pass rates; expected across stacks.

### 4.5 Colocate vs standalone reward models

**Colocate (default):** all rollouts finish, then reward model runs. Fine for rule-based math (µs–ms). Bad for 4B judge on same GPU pool.

**Standalone pool:** `reward.reward_model.enable_resource_pool=True` — judge on separate GPUs, may stream with rollout. *Likely* needed for CoT arms; **you still allocate the GPUs** and need to confirm it works with our judge call pattern.

### 4.6 Version coupling

VeRL pins vLLM tightly (≥0.8.2; recent releases target 0.12). Our `main/` image uses vLLM 0.9.0 + cu128 for B200. VeRL bring-up = new image rebuild cycle.

### 4.7 No minority-voting objective

VeRL knows GRPO group baselines. It does **not** know 70 size-4 subsets, minority cluster scoring, or Poly-EPO `f(G)`. That is custom advantage logic — the coding-component work.

---

## 5. What we still wire ourselves

```
┌─────────────────────────────────────────────────────────────┐
│  VeRL likely owns (if integration works)                    │
│  • Ray orchestration, multi-GPU batching                    │
│  • vLLM rollout + FSDP actor + weight sync                  │
│  • GRPO/PPO loss, clip, (optional) KL                       │
│  • Rule-based math rewards (MathReward / math.py — patched router) │
│  • Async/batch reward scoring infrastructure                │
│  • GenRM HTTP router pattern                                │
│  • DAPO filter_groups, overlong buffer                      │
│  • Checkpointing, wandb plumbing                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  We own                                                     │
│  • Minority-voting advantage (THE coding component)         │
│  • CoT cluster assignment via LLM judge                     │
│  • Judge Modal app + OpenAI-compatible API + batching       │
│  • Second GPU allocation for judge                          │
│  • Custom wandb metrics (clusters, prompts unlocked, …)     │
│  • Polaris → parquet preprocess                             │
│  • Modal + Ray bring-up                                     │
│  • Eval harness checkpoint loading (FSDP ≠ step_*.pt)       │
│  • Small loader shim for eval only — not a trainer re-port │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 Minority-voting advantage

**Reference** the math in `main/train/objective.py` (~200 lines); **implement** as a VeRL `adv_estimator` (or documented fallback hook). Unit tests in `main/tests/test_objective_minority.py` check *algorithm correctness on fixtures* — not that VeRL integrates the same way as our custom trainer.

```
8 rollouts → rewards (VeRL built-in scorer — do not re-port reward.py)
           → cluster IDs (judge CoT, or mock in 3a)
           → minority marginal advantages  ← only custom code here
           → clipped surrogate loss (VeRL)
```

Hook: custom `adv_estimator` registered in Hydra — use the fork’s `@register_adv_est` pattern (see existing `maxrl` entry in `core_algos.py`) as a **wiring template**, not as the algorithm we train.

Arms (in order): GRPO (zero custom) → `minority_answer` → `minority_cot` → `poly_epo_answer` (stretch).

### 5.2 CoT judge — not a standard reward function

A reward fn scores one `(prompt, completion, gold)`. Our judge scores **one prompt × 8 rollouts together** (Poly-EPO cluster assignment).

| Layer | VeRL | Us |
| --- | --- | --- |
| Per-rollout correctness | patched `math.py` (MathReward) | [`reward-decision.md`](./reward-decision.md) — no `main/train/reward.py` |
| CoT cluster assignment | Nothing built-in | New code; prompt logic may reference `main/probes/group_a_rollout_judge.py` |
| Minority advantage from clusters | Nothing built-in | New `adv_estimator`; math reference `main/train/objective.py` |
| Judge hosting | GenRM / async reward infra | Prefer VeRL async reward path if it fits; else Modal HTTP |
| Batching | `limit` manager, async fn | Prefer VeRL semaphore/rate-limit hooks before custom fan-out |

TA note: “don’t know if VeRL will handle this part for us” → **Our survey says no for CoT clustering** — but we have not wired it yet; Stage 3a/4 smokes are the real test.

### 5.3 Judge on Modal — architecture options

VeRL has no Modal integration.

| Option | Layout | Pros | Cons |
| --- | --- | --- | --- |
| **A: Second Modal function** | Train container + judge container (separate GPU) | Clean isolation | 2× GPU cost for CoT arms |
| **B: Same container, 2 GPUs** | `gpu="B200:2"` — judge subprocess on GPU 1 | One job | We start judge before VeRL; VeRL doesn’t orchestrate |
| **C: VeRL standalone reward pool** | `enable_resource_pool=True`, judge as GenRM | Native streaming | Still need to provision servers |

**Planned starting point** for CoT (may change after smokes): **Option A** — 4× B200 train + 1× B200 judge as detached Modal service.

Async client pattern (from TA):

```python
async def judge_cluster_batch(tasks, semaphore=64):
    async with semaphore:
        return await httpx.post(JUDGE_URL, json=...)
```

VeRL: `async def compute_score(...)` + `reward.reward_manager: limit` or `launch_reward_fn_async: True` — but cluster logic stays ours.

### 5.4 Custom logging

TA-requested metrics not built into VeRL:

- `train/distinct_clusters` (per step)
- `train/prompts_unlocked` (cumulative unique prompts with ≥1 correct rollout)
- `train/critic_mean_score` / mean reward
- `train/mean_response_length`
- `train/mixed_reward_rate`, answer-cluster diversity (PLAN §5 C3/C4 — missing in `main/` too)

### 5.5 Data + eval

- **Data:** reuse Polaris **manifest paths** from `main/data/`; convert to parquet using **`examples/maxrl_data_preprocess/polaris.py`** in the maxrl repo as the starting template.
- **Eval:** keep `main/eval/passk.py` for scoring logic if useful; add FSDP checkpoint loader only — not a trainer re-port.

---

## 6. B200 + VeRL

### 6.1 Does it work?

**Probably, but we haven't run VeRL on B200 yet.** Upstream added a [GB200/B200 example](https://github.com/verl-project/verl/commit/3f2fd075da015579639cd2f99aa1c2811c6f48d4). We already run B200 on Modal with the *custom* stack (vLLM 0.9.0, cu128 torch, flash-attn 2.8.3 in `main/infra/modal_image.py`) — that does **not** prove VeRL + Ray + our pin set works on the same hardware.

VeRL on B200 *looks like* a software/pins + Ray bring-up problem, not a hardware blocker — Stage 1 is the check.

### 6.2 B200-specific VeRL settings

| Setting | Why |
| --- | --- |
| `actor_rollout_ref.rollout.enforce_eager=True` | Required on Blackwell |
| `actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16` | FSDP defaults fp32 → FlashAttn breaks |
| `ray_kwargs.ray_init.num_gpus=N` | Modal/Docker may not auto-detect GPUs |
| No `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | vLLM memory pool incompatible (Stage 2) — unlike `main/` image |
| SGLang: `attention_backend=flashinfer` | FA3 unsupported on SM>90 |
| CUDA 12.4+ / cuDNN 9.8+ | VeRL base images |

### 6.3 B200 vs custom trainer economics

From [`../../main/docs/reference/efficiency/b200_deep_dive_verdict_2026-05-26.md`](../../main/docs/reference/efficiency/b200_deep_dive_verdict_2026-05-26.md): B200 economics for the **custom** stack — **not** a VeRL step-cost prior. VeRL may change rollout/train split and $/step entirely; ignore these ratios until Stage 2+ reports VeRL-native timings.

---

## 7. Multi-GPU scaling on Modal

### 7.1 Modal limits

| Setup | Modal API | Max GPUs |
| --- | --- | --- |
| Standard function | `gpu="B200"`, `gpu="B200:4"`, `gpu="B200:8"` | **8 per container** (one machine) |
| Multi-node cluster | `@modal.experimental.clustered` | 32–64+ (private beta) |

**There is no `B200:10`.** “Up to 10 GPUs” in budget terms → **8 train + 1 judge** on separate functions, or multi-node beta.

Requesting >2 GPUs per container usually increases queue wait ([Modal GPU docs](https://modal.com/docs/guide/gpu)).

### 7.2 VeRL cluster config

```yaml
trainer.nnodes: 1
trainer.n_gpus_per_node: 4   # or 8
+ray_kwargs.ray_init.num_gpus: 4  # explicit on Modal
```

**Resource pools** — who shares GPUs:

| Layout | When |
| --- | --- |
| **Colocate** (default global pool) | 1.7B, tight budget |
| **Split pools** (rollout ≠ train GPUs) | 4B+, VRAM contention |
| **Standalone judge pool** | CoT clustering arm |

Poly-EPO reference: **4× H200, bs=128, Qwen3-4B**, colocated global pool.

### 7.3 Does more GPUs speed things up?

**We expect yes, but not linearly** — upstream recipes and `main/` profiling are intuition only; measure on VeRL before trusting any speedup table.

Step shape (same as our Group B):

```
[rollout ~70-90%] → [reward cheap] → [logprob fwd] → [backward]
```

| Effect | Speedup | Notes |
| --- | --- | --- |
| Larger `train_batch_size` (64→128) | ~2× fewer steps/epoch | Main VeRL win vs single GPU |
| FSDP sharded actor | 1.3–2× backward | Better at 4B than 1.7B |
| vLLM TP / dedicated rollout GPUs | 1.3–1.7× rollout | Bottleneck phase |
| Ray + NCCL overhead | −5–15% | Worse if misconfigured |
| Weight sync actor↔vLLM | Still present | Handled, not eliminated |

**Rough expectations (guesswork until Stage 2/6 smokes):**

| Config | Expected vs 1× B200 today |
| --- | --- |
| 1× B200, 1.7B | Similar; maybe bs=128 if VRAM split better |
| 4× B200, 1.7B | ~2–2.5× per epoch (not 4×) |
| 4× B200, 4B | Where VeRL pays off (Path C) |
| 8× B200, 1.7B | Diminishing returns — model too small |

**Parallel arms** (GRPO + minority + poly): VeRL does not batch these — **3 separate Modal jobs**, same as today. More GPUs per job speeds one arm; parallel jobs speed time-to-all-results.

Steps per epoch: `51139 / train_batch_size`. bs=128 → ~400 steps vs 799 at bs=64.

---

## 8. Configuration knobs (cheat sheet)

### 8.1 Cluster topology

```yaml
trainer.nnodes: 1
trainer.n_gpus_per_node: 4
+ray_kwargs.ray_init.num_gpus: 4
# resource_pool_spec + mapping for split actor/rollout/judge pools
```

### 8.2 Throughput / batch

| Knob | Role | Our starting point |
| --- | --- | --- |
| `data.train_batch_size` | Prompts per step | 64 → try 128 multi-GPU |
| `actor_rollout.ref.rollout.n` | Rollouts per prompt | 8 (locked) |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | Actor update chunks | Tune with microbatch |
| `*_micro_batch_size_per_gpu` | OOM guardrails | Re-run Group B ladder |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM VRAM share | 0.45 colocated; higher if rollout owns GPUs |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | vLLM TP | 1 @ 1.7B; 2 @ 4B maybe |

### 8.3 Algorithm / research

| Knob | Notes |
| --- | --- |
| `algorithm.adv_estimator: grpo` | Baseline |
| `actor.use_kl_loss` / `algorithm.use_kl_in_reward` | Set KL=0 if we want to mirror prior `main/` runs; otherwise accept VeRL defaults and document |
| `algorithm.norm_adv_by_std_in_grpo` | True by default |
| `algorithm.filter_groups` | DAPO dynamic sampling |
| Custom adv estimator | Minority voting — we implement |
| `actor.ppo_epochs` | 1 (REINFORCE-with-clip) |
| `actor.loss_agg_mode` | `token-mean` vs `seq-mean-token-mean` |

### 8.4 Model / engine

| Knob | Notes |
| --- | --- |
| `actor_rollout_ref.model.path` | Qwen3-1.7B-Base or 4B-Base |
| `actor_rollout_ref.rollout.name` | `vllm` or `sglang` |
| `data.max_response_length` | 4096 |
| `data.truncation` | `left` for Polaris (preserves `\boxed{}` suffix); default `error` raises mid-step on overflow. See Stage 2 attempt 7 in [`build/stage-02-log.md`](./build/stage-02-log.md). |
| `enforce_eager` | True on B200 |
| FSDP offload flags | If 4B VRAM tight |

### 8.5 Reward

| Knob | Notes |
| --- | --- |
| `data_source` in parquet | Routes to built-in scorers |
| `custom_reward_function.path` | Only if built-in MathReward path fails after patch verification — see [`reward-decision.md`](./reward-decision.md) |
| `reward.reward_manager` | `naive`, `batch`, `dapo`, `limit` |
| `launch_reward_fn_async` | Overlap slow rewards with logprob |

---

## 9. Proposed bring-up order (poster timeline — all contingent on smokes)

| Stage | GPUs | Goal |
| --- | --- | --- |
| Parity smoke | 1× B200 | VeRL + Modal + Ray runs; GRPO stable (not numeric match to `main/`) |
| Production 1.7B | 2–4× B200 | bs=128, ~2× faster epoch |
| Path C (4B filtered) | 4× B200 | Poly-EPO-scale layout |
| Path D (CoT judge) | 4× B200 train + 1× B200 judge | Second Modal function for judge |
| 8× B200 | Only if 4B still OOMs | Overkill for 1.7B |

**First milestone:** migration plan Stage 2 (GRPO bring-up smoke — stable run, not `main/` parity). Set arms: Stages 3–5. Layout: [`../README.md`](../README.md).

---

## 10. Risk summary

| Pros | Cons |
| --- | --- |
| Drops custom trainer engineering | 2–3 days bring-up before new science |
| TA coding bar satisfied | Reward/parser may differ from Rank-2 hybrid |
| 4B + bs=128 + CoT realistic | FSDP checkpoints ≠ `step_*.pt` |
| Batched FSDP may beat our per-seq logprob loop | Ray overhead on 1 GPU |
| Official B200 recipe exists | Modal + Ray + judge = three integration surfaces |

---

## 11. References

- **Our VeRL source (primary):** https://github.com/tajwarfahim/maxrl — vendored `verl/` tree; Qwen3 + Polaris scripts
- Upstream VeRL (secondary): https://github.com/verl-project/verl
- VeRL docs: https://verl.readthedocs.io/
- GRPO in VeRL: https://verl.readthedocs.io/en/latest/algo/grpo.html
- MaxRL paper site (algorithm we are **not** running): https://zanette-labs.github.io/MaxRL/
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal multi-node (beta): https://modal.com/docs/guide/multi-node-training
