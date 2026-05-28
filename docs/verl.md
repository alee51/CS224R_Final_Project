# VeRL migration guide

**Status:** planning doc (2026-05-28)  
**Context:** TA decision to reimplement set-RL arms on [VeRL](https://github.com/verl-project/verl). Custom trainer in `main/` stays frozen for paper provenance; new work lands in `main-verl/`.

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
| **Post–2026-05-28** | Actual VeRL dependency in `main-verl/` — satisfies TA coding-component bar |

Our custom stack (`main/train/trainer.py`): vLLM collocated rollout + HF backward + custom `objective.py` for minority voting. VeRL replaces that plumbing; we still own the minority-voting objective and judge integration.

---

## 2. Why move (TA decision)

From [`main/docs/verl_move_ta_meeting.md`](../main/docs/verl_move_ta_meeting.md):

- **Coding component:** “Implement set RL with minority voting objective on verl.”
- **Sanity check:** Re-run GRPO + minority on VeRL to rule out custom-trainer bugs.
- **Drop engineering tax:** Answer extraction, multi-GPU batching, FA2, weight sync — VeRL ships these.
- **Unlock experiments:** Qwen3-4B (Path C), CoT clustering (Path D), bs=128 across GPUs.

### TA-requested changes vs `main/`

| Knob | `main/` today | VeRL direction |
| --- | --- | --- |
| Answer extraction | Rank-2 hybrid + mathd∨sympy | VeRL `MathReward` / `math_dapo` built-ins |
| Batch size | 64 single GPU (128 OOMs) | Up to 128, VeRL splits across GPUs |
| Clustering substrate | Answer-hash | **CoT clustering** via async LLM judge (answer clustering → reward hacking) |
| Model | Qwen3-1.7B-Base | Qwen3-4B-Base if it fits |
| Judge | Sidecar vLLM (planned) | One judge, async batched (32–64 parallel calls, `asyncio.Semaphore`) |
| Logging gaps | Partial wandb panel | Add: distinct clusters, prompts unlocked, critic mean score, response length |

---

## 3. What VeRL does well (use directly)

### 3.1 Train ↔ rollout pipeline

| We built in `main/` | VeRL ships |
| --- | --- |
| vLLM collocated rollout | vLLM or SGLang rollout workers |
| HF actor + weight sync | FSDP/FSDP2 actor with vLLM weight update |
| Logprob capture for GRPO/PPO | Built into rollout → actor update |
| Microbatch sizing / OOM guards | `micro_batch_size_per_gpu`, dynamic batching |
| Multi-GPU prompt batch splitting | Ray resource pools |

Our bs=128 OOM on single collocated H200/B200 is the main pain VeRL addresses via normal multi-GPU layouts (Poly-EPO: 128 prompts / batch 64 on 4× H200).

### 3.2 Algorithms (config, not code)

- **GRPO:** `algorithm.adv_estimator: grpo`, `actor_rollout.ref.rollout.n: 8`
- **PPO, DAPO, DrGRPO, RLOO, REINFORCE++** — recipes in [verl-recipe](https://github.com/verl-project/verl-recipe)
- **DAPO extras:** `filter_groups` (reject all-0 / all-1 groups), overlong buffer penalty, dynamic resampling

### 3.3 Math rewards / answer extraction

VeRL routes by `data_source` in parquet ([`verl/utils/reward_score/`](../main/docs/probes/prompt_extraction_research.md)):

| `data_source` | Module | Extraction |
| --- | --- | --- |
| `math_dapo`, `aime*` | `math_dapo.py` | Last `Answer:` line + `normalize_final_answer` |
| MATH datasets | `math_reward.py` | Last `\boxed{}` + Hendrycks normalization |
| GSM8K | `gsm8k.py` | Last `####` number |

Custom rewards: `custom_reward_function.path` + `.name` in Hydra config.

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

### 4.4 Reward / grader mismatch

Our locked train grader: **Rank-2 hybrid → mathd∨sympy** (`main/train/reward.py`).

VeRL default: **`math_dapo` string normalize** or **`math_reward` boxed extract**. Different parsers → different pass rates. Parity with `main/` B200 checkpoints requires a custom reward wrapper or accepting a new contract.

### 4.5 Colocate vs standalone reward models

**Colocate (default):** all rollouts finish, then reward model runs. Fine for rule-based math (µs–ms). Bad for 4B judge on same GPU pool.

**Standalone pool:** `reward.reward_model.enable_resource_pool=True` — judge on separate GPUs, can stream with rollout. Right for CoT arms; **you allocate the GPUs**.

### 4.6 Version coupling

VeRL pins vLLM tightly (≥0.8.2; recent releases target 0.12). Our `main/` image uses vLLM 0.9.0 + cu128 for B200. VeRL bring-up = new image rebuild cycle.

### 4.7 No minority-voting objective

VeRL knows GRPO group baselines. It does **not** know 70 size-4 subsets, minority cluster scoring, or Poly-EPO `f(G)`. That is custom advantage logic — the coding-component work.

---

## 5. What we still wire ourselves

```
┌─────────────────────────────────────────────────────────────┐
│  VeRL owns                                                  │
│  • Ray orchestration, multi-GPU batching                    │
│  • vLLM rollout + FSDP actor + weight sync                  │
│  • GRPO/PPO loss, clip, (optional) KL                       │
│  • Rule-based math rewards (math_dapo, math_reward)         │
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
│  • (Optional) grader parity wrapper vs MathReward           │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 Minority-voting advantage

Port from `main/train/objective.py` (~200 lines, unit-tested in `main/tests/test_objective_minority.py`):

```
8 rollouts → rewards (VeRL or custom)
           → cluster IDs (answer-hash OR judge CoT)
           → minority marginal advantages  ← OUR CODE
           → clipped surrogate loss (VeRL)
```

Hook: custom `adv_estimator` or extend `core_algos` path.

Arms (in order): GRPO (zero custom) → `minority_answer` → `minority_cot` → `poly_epo_answer` (stretch).

### 5.2 CoT judge — not a standard reward function

A reward fn scores one `(prompt, completion, gold)`. Our judge scores **one prompt × 8 rollouts together** (Poly-EPO cluster assignment).

| Layer | VeRL | Us |
| --- | --- | --- |
| Per-rollout correctness | `math_dapo` / custom | Maybe reuse |
| CoT cluster assignment | Nothing | Port `group_a_rollout_judge.py` |
| Minority advantage from clusters | Nothing | Port `objective.py` |
| Judge hosting | GenRM router pattern only | Modal vLLM + `/v1/chat/completions` |
| Batching | `limit` manager, async fn | Semaphore fan-out 32–128 HTTP calls |

TA note: “don’t know if VeRL will handle this part for us” → **No, not for CoT clustering.**

### 5.3 Judge on Modal — architecture options

VeRL has no Modal integration.

| Option | Layout | Pros | Cons |
| --- | --- | --- | --- |
| **A: Second Modal function** | Train container + judge container (separate GPU) | Clean isolation | 2× GPU cost for CoT arms |
| **B: Same container, 2 GPUs** | `gpu="B200:2"` — judge subprocess on GPU 1 | One job | We start judge before VeRL; VeRL doesn’t orchestrate |
| **C: VeRL standalone reward pool** | `enable_resource_pool=True`, judge as GenRM | Native streaming | Still need to provision servers |

Recommended for CoT: **Option A** — 4× B200 train + 1× B200 judge as detached Modal service.

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

- **Data:** one Polaris→parquet script; reuse `main/data/polaris_train.jsonl`, preprocess pipeline.
- **Eval:** `main/eval/passk.py` stays; VeRL checkpoints are FSDP format, not `step_*.pt`.

---

## 6. B200 + VeRL

### 6.1 Does it work?

**Yes.** VeRL added an explicit [GB200/B200 example](https://github.com/verl-project/verl/commit/3f2fd075da015579639cd2f99aa1c2811c6f48d4). We already run B200 on Modal with custom stack (vLLM 0.9.0, cu128 torch, flash-attn 2.8.3 in `main/infra/modal_image.py`).

VeRL on B200 = different software pins + Ray, not a hardware blocker.

### 6.2 B200-specific VeRL settings

| Setting | Why |
| --- | --- |
| `actor_rollout_ref.rollout.enforce_eager=True` | Required on Blackwell |
| `actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16` | FSDP defaults fp32 → FlashAttn breaks |
| `ray_kwargs.ray_init.num_gpus=N` | Modal/Docker may not auto-detect GPUs |
| SGLang: `attention_backend=flashinfer` | FA3 unsupported on SM>90 |
| CUDA 12.4+ / cuDNN 9.8+ | VeRL base images |

### 6.3 B200 vs custom trainer economics

From [`main/docs/reference/efficiency/b200_deep_dive_verdict_2026-05-26.md`](../main/docs/reference/efficiency/b200_deep_dive_verdict_2026-05-26.md): B200 is ~1.38× $/s vs H200; break-even needs ≥27% wall-clock cut. Our per-seq logprob loop (`_completion_logprobs_hf`) limits raw speedup. VeRL’s batched FSDP path may improve the train phase vs our loop — another reason to migrate, independent of B200 SKU.

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

**Yes, but not linearly.**

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

**Rough expectations:**

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
| `actor.use_kl_loss` / `algorithm.use_kl_in_reward` | Set KL=0 to match `main/` |
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
| `enforce_eager` | True on B200 |
| FSDP offload flags | If 4B VRAM tight |

### 8.5 Reward

| Knob | Notes |
| --- | --- |
| `data_source` in parquet | Routes to built-in scorers |
| `custom_reward_function.path` | Wrap mathd∨sympy for parity |
| `reward.reward_manager` | `naive`, `batch`, `dapo`, `limit` |
| `launch_reward_fn_async` | Overlap slow rewards with logprob |

---

## 9. Migration sketch (`main-verl/`)

Sibling to `main/` — see [`main-verl/README.md`](../main-verl/README.md).

### 9.1 Repo layout

| Dir | Purpose |
| --- | --- |
| `configs/` | Hydra yaml per launch |
| `train/` | Custom objectives: minority_answer, poly_epo, CoT arm |
| `judge/` | Modal judge service + async HTTP client |
| `infra/` | Modal image (verl + pins), GPU class |
| `scripts/` | Wrappers around `python -m verl.trainer.main_ppo` |
| `probes/` | 50-step smoke, judge bring-up, 4B fit |
| `tests/` | Port of `main/tests/test_objective_minority.py` |
| `data/` | Manifest paths; preprocess reuses `main/data/` |

### 9.2 Bring-up order

1. **`infra/`** — Modal image with VeRL; `hello_verl.py` smoke on B200.
2. **`configs/`** — GRPO on Qwen3-1.7B + Polaris; 50-step parity gate vs `main/` wandb (`mean_reward`, `prompt_coverage`).
3. **`train/`** — Port `minority_answer`; unit tests against existing fixtures.
4. **`judge/`** — Modal OpenAI-compatible API; async client with semaphore.
5. **`configs/` (4B)** — Qwen3-4B fit check; filtered manifest (Path C).
6. **`train/`** — CoT-clustering arm (Path D) once judge is up.

### 9.3 What stays in `main/`

- All paper docs, timeline, TA notes, eval results
- Polaris preprocess + manifests
- Custom trainer + B200 run artifacts (frozen)
- Eval harness (adapt checkpoint loading for VeRL)

### 9.4 Parity gate (critical)

Before trusting VeRL science: 50-step GRPO smoke, compare to `main/` at same step. If reward/prompt pairing diverges, debug before porting arms.

Reward contract choice:

| Path | Pros | Cons |
| --- | --- | --- |
| VeRL `math_dapo` + DAPO prompt | TA-aligned, zero custom parser | Incomparable to `main/` checkpoints |
| Custom fn wrapping Rank-2 + mathd∨sympy | Comparable to B200 runs | More code, defeats “use MathReward” |

---

## 10. Practical recommendations (poster timeline)

| Stage | GPUs | Goal |
| --- | --- | --- |
| Parity smoke | 1× B200 | VeRL + Modal + Ray works; GRPO matches `main/` |
| Production 1.7B | 2–4× B200 | bs=128, ~2× faster epoch |
| Path C (4B filtered) | 4× B200 | Poly-EPO-scale layout |
| Path D (CoT judge) | 4× B200 train + 1× B200 judge | Second Modal function for judge |
| 8× B200 | Only if 4B still OOMs | Overkill for 1.7B |

**First milestone:** GRPO smoke @ 1.7B, 50 steps, same wandb metrics as `main/`. Then port `minority_answer`. Then choose 4B vs CoT based on remaining compute.

---

## 11. Risk summary

| Pros | Cons |
| --- | --- |
| Drops custom trainer engineering | 2–3 days bring-up before new science |
| TA coding bar satisfied | Reward/parser may differ from Rank-2 hybrid |
| 4B + bs=128 + CoT realistic | FSDP checkpoints ≠ `step_*.pt` |
| Batched FSDP may beat our per-seq logprob loop | Ray overhead on 1 GPU |
| Official B200 recipe exists | Modal + Ray + judge = three integration surfaces |

---

## 12. References

- VeRL repo: https://github.com/verl-project/verl
- VeRL docs: https://verl.readthedocs.io/
- GRPO in VeRL: https://verl.readthedocs.io/en/latest/algo/grpo.html
- Reward functions: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
- Reward Loop: https://verl.readthedocs.io/en/latest/advance/reward_loop.html
- Multinode: https://verl.readthedocs.io/en/latest/start/multinode.html
- B200/GB200 example commit: https://github.com/verl-project/verl/commit/3f2fd075da015579639cd2f99aa1c2811c6f48d4
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal multi-node (beta): https://modal.com/docs/guide/multi-node-training
- verl-recipe DAPO: https://github.com/verl-project/verl-recipe/tree/main/dapo
