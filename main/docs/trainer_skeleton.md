# Trainer skeleton — build doc

**Drafted:** 2026-05-25. **Purpose:** the "how to build it" doc for the lightweight GRPO trainer described in [`PLAN.md`](./PLAN.md) §5. Mirrors the pattern of [`probes/group_a_impl.md`](./probes/group_a_impl.md): take the strategic choices from PLAN, lock the implementation-level details, and hand off to an agent.

**Prerequisites to read first:**

- [`PLAN.md`](./PLAN.md) §5 (Codebase) and §7 (Sizing & cost) — architecture decisions
- [`STANDARDS.md`](./STANDARDS.md) — wandb / Modal / reproducibility / reward parser rules
- [`probes/group_a_results.md`](./probes/group_a_results.md) — what Group A confirmed and what's still open
- Poly-EPO methodology extract (`pre-milestone/pilot/docs/analysis/0519_poly_epo_methodology.md`) — loss normalization, KL=0 rationale

**Scope:** GRPO skeleton end-to-end (one full step). The set-based arms (Minority-answer, Minority-CoT, Poly-EPO-answer) plug in at `objective.py` and reuse everything else. Group B uses this skeleton for the end-to-end step probe.

**Do not duplicate PLAN §5 here** — reference section keys. This doc covers file-level structure, function signatures, config schema, and unresolved spikes.

---

## 1. Architecture: locked choices (from PLAN §5)

Pulled verbatim from PLAN §5 "Architecture (commit now)". **If any of these change, change PLAN first, then update this doc.**

| Decision | Locked value | PLAN ref |
|---|---|---|
| Rollout engine | vLLM in-process; returns per-token logprobs | §5 Proposed |
| Training model | HF transformers, separate from vLLM | §5 Proposed |
| Weight sync | HF → vLLM via vLLM `update_weights` every step | §5 Proposed + Architecture |
| Reference model | **None** — KL coef = 0 (Poly-EPO) | §5 Architecture |
| `old_logprobs` source | Reuse vLLM rollout logprobs (no 2nd HF forward) | §5 Architecture |
| Inner PPO epochs | 1 (REINFORCE-with-clip) | §5 Architecture |
| Zero-advantage prompts | Filtered before backward (mask from `objective.py`) | §5 Architecture |
| Judge | Sidecar vLLM — out of scope for skeleton | §5 Architecture |
| vLLM prefix caching | ON | §5 Architecture |
| Clip range | Asymmetric DAPO: ε_low=0.20, ε_high=0.28 | §5 Calibration table |
| Loss normalization (GRPO arm) | `T_i = |y_i|` per-sequence length | Poly-EPO methodology |
| Loss normalization (set-based arms) | `T_i = T_max` batch-max length (Dr.GRPO) | Poly-EPO methodology |
| LR | 1e-6 (sweep only if needed) | §5 Calibration table |
| Entropy coef | 0.0 | §5 Calibration table |
| Rollout temp | 1.0 | §5 Calibration table |
| Precision | bf16 everywhere | §5 Size/throughput |
| Attention | FlashAttention-2 (FA-3 on H100/H200) | §5 Size/throughput |
| Gradient checkpointing | ON | §5 Size/throughput |
| Async rollout/train overlap | **Deferred** — only if Group B shows it's binding | §5 Architecture |
| GPU class default | H100 (`modal_price_per_sec: 0.001097`) — Group A confirmed | Group A results |

---

## 2. UNDECIDED / inherited-open from Group A

These do **not** block writing the skeleton, but the skeleton must accommodate them as config knobs (not hardcoded constants). Resolution lives elsewhere; just don't lock them into code.

| Item | Why undecided | How skeleton handles it |
|---|---|---|
| **Reward parser** (`reward.py`) | Group A: only **56% parse_ok** with DAPO `Answer:` + Minerva. Rank 2 multi-path parser is pending. See `group_a_results.md`. | `compute_reward()` API stays as-is (returns dict with `reward`, `parse_ok`, …). Parser internals change; callers don't. Skeleton consumes whatever it returns. |
| **`max_response_length`** | Group A: 4096 is safe (1.25% cap hits); 3072 would truncate ~2.3% — possible savings, not urgent. | Config knob `phase1.max_response_length`, default **4096**. Easy to lower later. |
| **§2 sampling / training freeze** | Group A: band-level pass rates too noisy at N=200 to lock train subset or "drop 8/8-correct" filter. | Skeleton trains on whatever jsonl is at `train.data_path`. The freeze script (`main/data/preprocess_polaris.py` — separate work) writes that jsonl. Skeleton doesn't care about band logic. |
| **Judge VRAM / hosting** | Group A: VRAM logging bug, value=0. Judge $/call and latency known and cheap. | Out of scope for skeleton (judge is a sidecar; loaded only by CoT arms). |
| **Loss normalization for *new* arms** | Locked for GRPO and Poly-EPO-style only. Minority-answer/CoT not explicitly specified in source. | Treat Minority-* as set-based → `T_max` until told otherwise. Make it a config field per arm. |

---

## 3. Pre-flight locks (agent: do not re-litigate)

| Item | Lock |
|---|---|
| vLLM version | **0.8.5** (matches `modal_image.py`; Group A validated for Qwen3 family). Bump only after the API spike (§9) if `update_weights` is unstable. |
| HF Transformers | `>=4.55.2,<5.0.0` (pinned in image; Qwen2Tokenizer cache-path constraint) |
| Model | `Qwen/Qwen3-1.7B-Base` — plain string to vLLM, no chat template (per STANDARDS Reward §) |
| Prompt template | DAPO `Answer:` (STANDARDS verbatim block) — already in `main/train/prompts.py` |
| Reward API | `compute_reward(completion: str, gold: str) -> dict` — already in `main/train/reward.py`. Skeleton imports as-is. |
| Polaris sampling | **Not in skeleton.** Training reads frozen jsonl produced by separate `main/data/preprocess_polaris.py` (PLAN §2). For Group B's toy batch, reuse Group A's manifest (`probes/05-24/group_a/manifest.jsonl` on `main-artifacts`). |
| Seeds | STANDARDS formula: `global_seed + step * batch_size * N + prompt_idx * N + rollout_idx`. Never Python `hash()`. |
| Modal image | `from main.infra.modal_image import image` — already exists |
| Modal volumes | `main-artifacts` (mount `/vol`), `hf-cache` (mount `/root/.cache/huggingface`) |
| Modal secrets | `HUGGINGFACE`, `WANDB_API_KEY` (uppercase) |
| App name | `os.environ["CS224R_APP_NAME"]` per STANDARDS § Modal; launch wrapper sets it |
| Wandb | entity `224r-project`, project `cs224r-minority-voting` |
| Checkpoint cadence | Every ≤1 hr wall-clock per STANDARDS § Training checkpoints. Contents: weights + optim + RNG + step + wandb run ID. |

---

## 4. Files to create

```
main/
  train/
    rollout.py        # vLLM engine wrapper, generation, logprob capture, weight sync
    objective.py      # advantage computation; pluggable per arm (grpo, minority_answer, …)
    loss.py           # PPO-clipped surrogate, length-normalized, microbatched backward
    trainer.py        # main loop: rollout → reward → advantage → loss → step → sync → log
    weight_sync.py    # HF state_dict → vLLM update_weights (isolated so it can be unit-tested)
  data/
    preprocess_polaris.py   # one-shot script: PLAN §2 freeze → polaris_train.jsonl + meta.json
    dataset.py              # tiny jsonl reader + batching for trainer
  configs/
    train_grpo_05-25.yaml   # first concrete trainer config (GRPO arm)
  tests/
    test_loss.py            # shape + clip + length-norm unit tests on synthetic tensors
    test_weight_sync.py     # tiny model: change HF weight → vLLM generation reflects it
```

**Files that already exist and stay as-is** (Group A built them):

- `main/train/reward.py`, `main/train/prompts.py`
- `main/infra/modal_image.py`, `main/infra/modal_volume.py`

**Out of scope for this doc:**

- `main/probes/group_b_step_probe.py` + `main/configs/probe_b_*.yaml` — see [`probes/group_b_impl.md`](./probes/group_b_impl.md)
- `main/judge/` — sidecar judge for CoT arms; build when CoT arms are scheduled
- `main/eval/passk.py` — eval harness; separate work

---

## 5. Per-file specs

### 5.1 `main/train/rollout.py`

Wraps vLLM. Owns the engine lifecycle, generation call, logprob extraction, and the weight-sync hook.

```python
class RolloutEngine:
    def __init__(self, cfg: RolloutCfg): ...
    def generate(
        self,
        prompts: list[str],
        n_per_prompt: int,
        seeds: list[int],  # one per (prompt, rollout) — STANDARDS formula
    ) -> list[RolloutResult]: ...
    def update_weights(self, hf_state_dict: dict) -> None: ...  # delegates to weight_sync
    def shutdown(self) -> None: ...

@dataclass
class RolloutResult:
    prompt_idx: int
    rollout_idx: int
    completion_ids: list[int]      # token ids, for re-tokenization-free loss
    completion_text: str
    prompt_ids: list[int]
    old_logprobs: list[float]      # per-completion-token, from vLLM
    finish_reason: str             # verbatim from vLLM (STANDARDS-style: "stop"/"length"/…)
```

**Notes:**
- Request `logprobs=1` on the vLLM `SamplingParams`. Capture the chosen-token logprob per generated token; that becomes `old_logprobs` in `loss.py`.
- Prefix caching ON; `gpu_memory_utilization` is a **collocated** value — start at **0.45** per PLAN §5 sizing-agent note; Group B will tune.
- Engine must be **single-process** with the trainer (PLAN §5 Proposed). Do **not** use Modal-function-per-rollout — co-locate via spawn or in-process vLLM.

### 5.2 `main/train/weight_sync.py`

Isolated so it can be unit-tested with a tiny model before plugging into the full loop.

```python
def sync_hf_to_vllm(hf_model, vllm_engine) -> SyncStats:
    """Push HF state_dict into vLLM. Returns wall-clock + bytes moved."""
```

- Use vLLM's `LLM.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(...)` path. Exact call is version-dependent — see § 9 spike before depending on it.
- Synchronous (PLAN §5 default: every step). If Group B shows it's slow, batching over N steps is a config flag, not a rewrite.
- Test: load a 125M model in both HF and vLLM, randomize HF weights, sync, assert vLLM generation distribution shifts. Lives in `tests/test_weight_sync.py`.

### 5.3 `main/train/objective.py`

Compute per-rollout advantages. **Pluggable per arm.**

```python
def compute_advantages(
    arm: str,                    # "grpo" | "minority_answer" | "minority_cot" | "poly_epo_answer"
    rewards: torch.Tensor,       # shape [n_prompts, n_rollouts]
    clusters: torch.Tensor | None = None,  # arm-specific; None for grpo
) -> AdvantageOut:
    ...

@dataclass
class AdvantageOut:
    advantages: torch.Tensor     # [n_prompts, n_rollouts], per-rollout scalar
    keep_mask: torch.Tensor      # [n_prompts] bool — False = zero-advantage prompt, skip
    diagnostics: dict            # logged to wandb (e.g. fraction filtered)
```

- **GRPO:** `A_i = r_i − mean(r)` per group of N=8.
- **Minority-answer / CoT / Poly-EPO-answer:** per PLAN §3 "Objective math" — C(8,4)=70 subsets, marginal advantages. Skeleton ships **GRPO only**; other arms drop in here later. Reserve the dispatch.
- `keep_mask` is False whenever advantages within a group are all zero (collapsed prompts). `loss.py` honors it.

### 5.4 `main/train/loss.py`

PPO-clipped surrogate. Per-arm length normalization.

```python
def grpo_loss(
    new_logprobs: torch.Tensor,  # [B, T] from current HF forward
    old_logprobs: torch.Tensor,  # [B, T] from vLLM at rollout time
    advantages: torch.Tensor,    # [B] per-sequence (broadcast over T)
    mask: torch.Tensor,          # [B, T] valid-token mask (skip pad)
    keep_mask: torch.Tensor,     # [B] from objective; False rows zeroed out
    clip_low: float = 0.20,
    clip_high: float = 0.28,
    length_norm: str = "per_seq",  # "per_seq" (GRPO) | "batch_max" (set-based)
) -> torch.Tensor:
    """
    ratio = exp(new - old)
    surr1 = ratio * adv
    surr2 = clip(ratio, 1-eps_low, 1+eps_high) * adv
    per_token_loss = -min(surr1, surr2)
    Then reduce per `length_norm`:
      per_seq:   sum(per_token_loss * mask, dim=1) / sum(mask, dim=1) -> mean over seqs
      batch_max: sum(per_token_loss * mask) / T_max  -> mean over seqs (Dr.GRPO style)
    keep_mask False rows contribute 0.
    """
```

**Unit tests** (`tests/test_loss.py`):
- shape: B=4, T=16 inputs → scalar output
- clip: large ratio + positive advantage → loss equals `-clip_high * adv` (clipped branch)
- length norm: `per_seq` vs `batch_max` give different scalars for non-uniform lengths
- `keep_mask` False rows contribute exactly zero

### 5.5 `main/train/trainer.py`

The loop. Thin — orchestration only. **Export one step** for Group B instrumentation:

```python
def run_one_grpo_step(
    cfg: TrainCfg,
    rollout_engine: RolloutEngine,
    hf_model,
    opt,
    batch: StepBatch,  # prompts, golds, problem_ids for seeds
    *,
    instrument: bool = False,
) -> StepResult: ...

def train(cfg: TrainCfg) -> None:
    setup_wandb(cfg); set_seeds(cfg.global_seed); log_repro(cfg)
    rollout_engine = RolloutEngine(cfg.rollout)
    hf_model, opt = build_hf(cfg)
    dataset = JsonlPromptDataset(cfg.train.data_path)

    for step in range(cfg.train.total_steps):
        prompts, golds = dataset.next_batch(cfg.train.batch_size)
        run_one_grpo_step(cfg, rollout_engine, hf_model, opt, batch, instrument=False)
        if should_checkpoint(step, cfg): save_ckpt(...)
```

**Logged per step:** step time, mean reward, fraction filtered, loss, mean advantage, sync wall-clock, VRAM watermark, tokens/sec rollout.
**Checkpoint contents:** weights, optim state, RNG (`torch.get_rng_state()`, `numpy`, `random`, CUDA per-device), step number, wandb run ID. STANDARDS-compliant resume.

### 5.6 `main/data/preprocess_polaris.py`

One-shot script. **Not run by trainer.** PLAN §2 freeze.

- Reads Polaris from HF, applies cleaning (integer gold, non-empty problem).
- Selects bands / sizes per §2 decisions (UNDECIDED — config-driven).
- Writes `main/data/polaris_train.jsonl` + `polaris_train.meta.json` (HF revision SHA, seed, bands, row count, timestamp, cleaning filters).
- **Do not run until §2 is locked.** Group B can use Group A's manifest in the meantime.

### 5.7 `main/data/dataset.py`

```python
class JsonlPromptDataset:
    def __init__(self, path: str, seed: int): ...
    def next_batch(self, n: int) -> tuple[list[str], list[str]]: ...  # (prompts, golds)
```

Trivial. Reads jsonl, shuffles deterministically, yields batches.

---

## 6. Config schema — `main/configs/train_grpo_05-25.yaml`

Sketch (numeric values TBD by Group B):

```yaml
global_seed: 42
operator: nancy
gpu_class: H100
modal_price_per_sec: 0.001097
arm: grpo                      # grpo | minority_answer | minority_cot | poly_epo_answer

train:
  data_path: /vol/data/polaris_train.jsonl   # set by §2 freeze; toy path for Group B
  total_steps: 850             # PLAN §5 calibration; tune from probes
  batch_size: 64               # prompts per step — Group B sets
  n_rollouts: 8                # locked
  microbatch: 4                # Group B sets via OOM sweep
  grad_accum: 4                # Group B sets
  lr: 1.0e-6
  weight_decay: 0.0
  grad_clip: 1.0
  warmup_steps: 20
  checkpoint_every_steps: 50   # also bounded by 1hr wall-clock per STANDARDS
  checkpoint_dir: /vol/checkpoints/train_grpo_05-25/

rollout:
  model: Qwen/Qwen3-1.7B-Base
  max_prompt_length: 1024
  max_response_length: 4096    # UNDECIDED per Group A (3072 possible later)
  temperature: 1.0
  top_p: 1.0
  gpu_memory_utilization: 0.45 # collocated; Group B tunes
  max_model_len: 5120
  enable_prefix_caching: true
  logprobs: 1

loss:
  clip_low: 0.20
  clip_high: 0.28
  length_norm: per_seq         # grpo arm; set-based arms override to batch_max
  entropy_coef: 0.0

weight_sync:
  every_n_steps: 1             # PLAN §5 default; raise if Group B shows it's slow

artifacts:
  volume_name: main-artifacts
  volume_mount: /vol
  hf_cache_volume: hf-cache
  pointer_path: docs/artifacts/05-25_train_grpo.pointer.json

wandb:
  entity: 224r-project
  project: cs224r-minority-voting
  group: train-grpo-05-25
```

---

## 7. Modal entrypoint shape

`main/train/trainer.py` is callable both as a Modal entrypoint and locally for unit tests.

```python
app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-train-untagged"))

@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/vol": modal.Volume.from_name("main-artifacts", create_if_missing=True),
             "/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True)},
    secrets=[modal.Secret.from_name("HUGGINGFACE"), modal.Secret.from_name("WANDB_API_KEY")],
)
def train_remote(config_path: str):
    train(load_cfg(config_path))
```

Launch wrapper (`main/scripts/launch_train.sh`) sets `CS224R_APP_NAME` per STANDARDS and `modal run --detach`.

---

## 8. Build order (checklist)

Suggested order — each row is a green checkpoint to ship before moving on.

- [ ] `weight_sync.py` + `tests/test_weight_sync.py` on a 125M model (resolves the §9 spike)
- [ ] `rollout.py` with vLLM 0.8.5, logprob capture verified against a known-token completion
- [ ] `objective.py` GRPO path + tiny synthetic test (`A_i` matches hand-computed value)
- [ ] `loss.py` + `tests/test_loss.py` (shape, clip, length-norm, keep_mask)
- [ ] `dataset.py` (trivial)
- [ ] `trainer.py` skeleton: load → one step (no checkpointing yet) on a 32-prompt toy batch
- [ ] Add checkpointing + resume; smoke-test resume on the same toy batch
- [ ] Modal entrypoint + launch wrapper; smoke-run remotely
- [ ] Hand off to Group B for the timed step probe and OOM sweep

`preprocess_polaris.py` is independent — write it whenever §2 is locked. Trainer skeleton doesn't block on it.

---

## 9. Open spikes (do these *before* building the surrounding code)

| Spike | Why | Done when |
|---|---|---|
| **vLLM `update_weights` API on 0.8.5** | Signature has changed between versions; want to confirm the path documented in `weight_sync.py` before building `rollout.py` around it. | `tests/test_weight_sync.py` passes on a 125M model end-to-end (HF weight change → vLLM generation change). 30-min spike. |
| **Loss reduction confirmation** | Locked to `per_seq` (GRPO) / `batch_max` (set-based) per Poly-EPO methodology. Sanity-check against Dr.GRPO / VeRL implementation before relying on it. | Implementation matches one of those two refs; comment in `loss.py` cites which. |

---

## 10. Hand-off prompt (copy-paste to an agent)

> Implement the trainer skeleton per `main/docs/trainer_skeleton.md`. Locked architecture is in PLAN.md §5 (do not re-litigate). Files to create: `main/train/{rollout,objective,loss,trainer,weight_sync}.py`, `main/data/{preprocess_polaris,dataset}.py`, `main/configs/train_grpo_05-25.yaml`, plus `tests/test_{loss,weight_sync}.py`. Reuse existing `main/train/{reward,prompts}.py` and `main/infra/modal_{image,volume}.py` as-is. **Do the vLLM `update_weights` spike first** (§9) — get `test_weight_sync.py` passing on a 125M model before building `rollout.py`. Follow STANDARDS for wandb / Modal / seeds / checkpoint contents. The reward parser, `max_response_length`, and Polaris training freeze are UNDECIDED (Group A results); treat them as config knobs, do not hardcode. Out of scope: the Group B step probe (separate doc), the sidecar judge, the eval harness.
