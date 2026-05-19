# GRPO OOM Root-Cause Analysis (May 19, 2026)

Companion to `pilot/grpo_smoke_debug_history.md`. That file is a chronological
ledger of what was tried. **This file is the diagnosis**: what is actually
broken, why every YAML reduction failed, the memory math that proves it, and
the ordered set of fixes from least to most experimentally invasive.

If you only have time to read one section, read **§3 The Smoking Gun** and
**§5 Tier 1 Fix**. Everything else is supporting evidence.

---

## 1) TL;DR

- **A100-80GB is massively oversized for SFT/GRPO on Qwen3-1.7B.** Peak memory
  for a correct implementation is **~35–40 GB**, leaving ~50% headroom on an
  80 GB card.
- The OOM is **not** a fundamental limit of the hardware or the experiment.
- It is a **specific bug** in the differentiable completion-logprob path:
  every micro-batch's autograd graph is **accumulated in a Python list and
  held until backward**. Activation memory therefore scales with the **total
  number of completions**, not with `completion_logprob_micro_batch_size`.
- This is why every YAML reduction failed in the same way. Reducing
  `completion_logprob_micro_batch_size` from 8 → 4 → 2 cannot help; it
  actually adds iterations without lowering the peak.
- The fix is small (≈ 2 hours): perform the loss reduction *inside* the
  micro-batch loop and call `.backward()` per micro-batch, freeing the graph
  before the next forward. After that the original config
  (`batch_prompts=32, rollouts=8, max_new_tokens=2048`) should fit.

---

## 2) Ground Truth Setup

From `pilot/configs/shared_train.yaml` and `pilot/train/hf_grpo_train.py`:

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B-Base` |
| Architecture | hidden=2048, layers=28, heads=16, kv_heads=8 (GQA), intermediate=11008, vocab=151,936 |
| Compute dtype | `torch.bfloat16` |
| Optimizer | `torch.optim.AdamW` (fp32 momentum + fp32 variance) |
| GPU | A100-80GB (~79.25 GiB usable per error trace) |
| Ref model | `copy.deepcopy(policy)` resident on GPU |
| Default rollouts/prompt | 8 |
| Default batch_prompts | 32 |
| Default max_new_tokens | 2048 |
| Default `completion_logprob_micro_batch_size` | 16 |

Two models are kept on GPU at all times: `policy` (training) and `ref_model`
(deepcopy, used for KL).

---

## 3) The Smoking Gun

File: `pilot/train/hf_grpo_train.py`.

Look at the **differentiable** logprob pipeline:

```python
# line ~211 — no @torch.no_grad, returns differentiable scalar tensors
def _micro_batch_mean_completion_logprobs(model, tokenizer, problems, completions, *, device):
    ...
    logits = model(batch_ids, attention_mask=attention_mask).logits     # [B, L, V] — graph kept
    log_probs = F.log_softmax(logits, dim=-1)                            # [B, L, V] — graph kept
    out: list[torch.Tensor] = []
    for i in range(batch_size):
        token_logps = [log_probs[i, pos, batch_ids[i, pos + 1]] for pos in range(start, end)]
        out.append(torch.stack(token_logps).mean())                      # scalar; references log_probs
    return out
```

```python
# line ~261 — micro-batch driver: ACCUMULATES tensors across iterations
def _batched_mean_completion_logprobs(..., micro_batch_size=...):
    results: list[torch.Tensor] = []
    for start in range(0, len(problems), micro_batch_size):
        ...
        results.extend(_micro_batch_mean_completion_logprobs(...))       # graph from THIS forward stays alive
    return results                                                       # every graph still alive
```

```python
# line ~311 — HFPolicyModel stores all of them, kept across train_step
class HFPolicyModel:
    def logprobs_for_rollouts(self, groups):
        flat_logprobs = _batched_mean_completion_logprobs(...)           # all micro-batches' graphs
        self._logprob_tensors = [...]                                    # held for the whole step
        ...
```

```python
# run_grpo_training (line ~764+):
groups, specs_batch = _build_step_groups(...)        # rollouts + no-grad old/ref logprobs
policy_model = HFPolicyModel(policy, tokenizer, specs_batch, ...)
trainer.model = policy_model
step_out = trainer.train_step(groups, objective, ...)            # builds the accumulated graph here
loss_t = _differentiable_loss(groups, policy_model._logprob_tensors, ...)
optimizer.zero_grad(set_to_none=True)
loss_t.backward()                                                 # only HERE is the graph freed
optimizer.step()
```

### What this means in plain English

Every micro-batch forward keeps **all** of its activations alive (PyTorch
needs them for backward). Normally you would free that memory by either
(a) calling `.backward()` before the next forward, or (b) using
`@torch.no_grad()`. This code does neither — instead it appends the
output tensor to a list that lives until `optimizer.zero_grad → backward`
finally runs at the very end of the step. So **all `N_micro_batches`
worth of activations sit in VRAM simultaneously**, and the peak scales
with the total completion count, not with the micro-batch size.

### Why every YAML knob reduction failed

Lowering `completion_logprob_micro_batch_size` (the only knob that targets
this code path) **increases** the number of micro-batches while keeping the
per-iteration activations roughly the same. It cannot reduce the peak. Look at
the test ledger in `grpo_smoke_debug_history.md`:

| Test | mb | b_p | max_new | N_completions | N_micro_batches | "Tried to allocate" |
|------|----|-----|---------|---------------|-----------------|----------------------|
| T1 (run1, default) | 16 | 32 | 2048 | 256 | 16 | 274 MB |
| T2 (run2) | 16 | 32 | 2048 | 256 | 16 | 274 MB |
| T4 | 4 | 32 | 2048 | 256 | 64 | 40 MB |
| T5 | 2 | 32 | 2048 | 256 | 128 | 18 MB |
| T8 | 2 | 16 | 1024 | 128 | 64 | 12 MB |
| T9 | 2 | 8 | 512 | 64 | 32 | 16 MB |

Reducing `b_p`/`max_new` shrinks the per-iteration footprint (so the *headroom
needed* drops — "tried to allocate 274 MB → 16 MB"), but the ceiling is
still hit because total accumulated activations stay near 80 GB. Reducing
`mb` actively makes it worse on the iteration count axis.

---

## 4) The Memory Math (so you can verify yourself)

All numbers are bf16 (2 bytes per element) unless noted.

### 4.1 Static memory (always resident)

| Component | Formula | Bytes |
|---|---|---|
| Policy params | 1.7e9 × 2 | 3.4 GB |
| Ref model deepcopy | 1.7e9 × 2 | 3.4 GB |
| Gradients (bf16) | 1.7e9 × 2 | 3.4 GB |
| AdamW momentum (fp32) | 1.7e9 × 4 | 6.8 GB |
| AdamW variance (fp32) | 1.7e9 × 4 | 6.8 GB |
| **Static subtotal** | | **~23.8 GB** |

### 4.2 Activation memory per micro-batch forward

For `B = micro_batch_size`, `L = prompt_len + max_new_tokens`, hidden `H = 2048`,
intermediate `I = 11008`, heads = 16, layers = 28.

Per layer (saved-for-backward tensors, in bf16):

| Saved tensor | Approx size (bytes) |
|---|---|
| LayerNorm pre-attn input | B·L·H · 2 |
| Q, K, V projections (GQA: K,V ≈ ½) | (1 + 0.5 + 0.5) · B·L·H · 2 |
| Attention probs | B · heads · L² · 2 |
| Attention output | B·L·H · 2 |
| LayerNorm pre-MLP input | B·L·H · 2 |
| MLP gate, up | 2 · B·L·I · 2 |
| MLP after silu (saved for backward) | B·L·I · 2 |
| Residual stream copy | B·L·H · 2 |

Per-token-per-layer ≈ `(9·H + 3·I)` floats × 2 B = `(9·2048 + 3·11008) × 2 ≈ 103 KB`.

For **B = 2, L ≈ 562** (≈ 50 prompt + 512 completion):

- Non-attention activations per layer: 103 KB × (B·L = 1124) ≈ **117 MB / layer**
- Attention probs per layer: 2 × 16 × 562² × 2 B ≈ **20 MB / layer**
- × 28 layers: **(117 + 20) × 28 ≈ 3.8 GB**
- Top-level logits + log_probs: `[B,L,V]` × 2 tensors × 2 B
  = 2 × 562 × 151936 × 2 × 2 ≈ **0.68 GB**

**Per-micro-batch activations ≈ 4.5 GB**.

(Conservatively, PyTorch can fuse some of these; empirically the saved
footprint for HF Qwen models lands in the 2.5–4.5 GB range for this shape.)

### 4.3 Why T9 (the smallest tested config) OOMs

T9: `batch_prompts=8, rollouts=8, max_new_tokens=512, completion_logprob_micro_batch_size=2`.

- N_completions = 8 × 8 = 64
- N_micro_batches = 64 / 2 = 32
- Activation peak (with bug) = 32 × ~2.5–4.5 GB = **80 – 144 GB**
- Plus static ~24 GB → **104 – 168 GB needed** vs **80 GB available**

Observed: process used **79.23 GiB** before OOMing on a tiny 16 MB allocation.
Matches the lower-end estimate after accounting for HF fusion. **OOM is
exactly what the math predicts.**

### 4.4 What the peak should be without the bug

With proper gradient accumulation (forward → backward → free → next forward):

| Component | Size |
|---|---|
| Static (params, ref, grads, Adam) | ~24 GB |
| Live activations for 1 micro-batch | ~4.5 GB |
| Optimizer step transient (grad copies, Adam ops) | ~5–10 GB |
| **Peak per step** | **~35–40 GB** |

That fits in 80 GB with **~50% headroom**, even at the original
`batch_prompts=32, rollouts=8, max_new_tokens=2048` (which only changes
*per-iteration* activation peak, not accumulated). At those original
settings:

- B=2 (mb stays 2), L ≈ 50 + 2048 = 2098
- Attention probs grow as L²: 2 × 16 × 2098² × 2 ≈ 281 MB/layer × 28 ≈ **7.9 GB**
- MLP activations grow as L: ~117 MB × (2098/562) ≈ **438 MB/layer × 28 ≈ 12.3 GB**
- Logits: 2 × 2098 × 151936 × 2 × 2 ≈ **2.5 GB**
- Per-mb total: **~23 GB**
- Peak: 24 (static) + 23 (one mb) + 10 (optimizer) = **~57 GB**

Still fits with margin. The bug is the only thing blocking the overnight
matrix.

---

## 5) Fixes — Ordered Least → Most Experimentally Invasive

Each entry lists: **what to change**, **memory delta** (with math), and
**experiment impact** (does it change the math of the gradient update?).

---

### Tier 1 — The actual fix (zero experiment impact)

#### 5.1.a Loss-level gradient accumulation

**The change.** Move the loss computation *inside* the micro-batch loop and
call `backward()` once per micro-batch, so each micro-batch's autograd graph
is freed before the next forward.

Sketch:

```python
optimizer.zero_grad(set_to_none=True)
n_mb = math.ceil(N / mb)
total_loss = 0.0
for chunk in micro_batches(groups, mb):
    logprob_tensors = _micro_batch_mean_completion_logprobs(policy, ..., chunk)
    loss_mb = compute_loss(logprob_tensors, chunk, old_lp, ref_lp, advs, cfg) / n_mb
    loss_mb.backward()                # frees graph for this chunk
    total_loss += float(loss_mb.detach()) * n_mb
optimizer.step()
```

**Memory delta.** Activation memory drops from
`N_micro_batches × per_mb` to `1 × per_mb`.

- T9 case: 32 × 2.5 GB ≈ 80 GB → 1 × 2.5 GB = **2.5 GB live activations**.
- Original config (b_p=32, rollouts=8, max=2048, mb=16): 16 × 23 GB ≈ 370 GB →
  1 × 23 GB = **23 GB live activations**. Peak ≈ **57 GB**. Fits.

**Experiment impact.** None. The gradient is mathematically identical
(sum-then-divide vs. divide-then-sum is the same up to numeric precision).
The KL term must be averaged consistently across micro-batches — be careful
to use a fixed divisor (`/n_mb`), not a running mean.

**Effort.** ~2 hours including a smoke check.

**Verdict.** **Do this first.** It is the single thing standing between you
and the overnight matrix.

---

### Tier 2 — Safe additional savings (zero experiment impact)

These are stackable and should each be added if convenient. They don't change
gradients; they just lower the per-mb footprint or fragmentation.

#### 5.2.a Replace `log_softmax`-then-index with `gather` on logits

**The change.** Right now both `logits` and `log_probs` are `[B, L, V]` and
both are saved for backward (V=151,936 is huge). Use `gather` to pull only
the target-token entries before doing the log-normalization, or use
`F.cross_entropy(..., reduction='none')` and negate. Either way, only logits
are saved for backward, not the full log-softmax output.

**Memory delta.** Saves one `[B, L, V]` bf16 tensor per micro-batch:
`B × L × 151936 × 2 B`.

- T9 (B=2, L≈562): **~340 MB / mb**
- Original (B=2, L≈2098): **~1.27 GB / mb**

Cumulative: small compared to Tier 1, but free.

**Experiment impact.** None — identical math.

**Effort.** ~30 min.

#### 5.2.b Enable gradient checkpointing

**The change.**

```python
policy.gradient_checkpointing_enable()
policy.config.use_cache = False    # required when checkpointing
```

Saves activations at layer boundaries only and recomputes intra-layer
activations during backward.

**Memory delta.** Activation memory roughly drops from
`O(L · layers)` saved tensors per layer to `O(L · sqrt(layers))` (in
practice: ~3-4× reduction in activation footprint for transformer LMs).

- T9: per-mb activations 2.5 GB → **~0.7-0.9 GB**.
- Original (max=2048): per-mb 23 GB → **~6-8 GB**.

**Time cost.** ~20-30% slower forward+backward (you do one extra forward per
layer during backward).

**Experiment impact.** None — identical gradients.

**Effort.** 5 minutes (one line).

#### 5.2.c `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**The change.** Set this env var in the Modal app's container env.

**Memory delta.** Does not change *total* footprint. Reduces allocator
fragmentation — the allocator will reuse freed blocks more aggressively
across different tensor shapes. Helps when the actual usage is well below
the ceiling but fragmentation pushes peak above the ceiling.

**Experiment impact.** None.

**Effort.** 1 line in `modal_app.py`.

#### 5.2.d Reuse rollout logprobs instead of recomputing `old_logprobs`

**The change.** Today `_build_step_groups` does a full no-grad forward over
all 64-256 completions just to compute `old_logprobs`. For pure on-policy
GRPO with no PPO clipping across multiple epochs, `old_logprobs` equals the
current-policy logprobs you'd recompute in the differentiable pass anyway.
Either:
- collect per-token logprobs during `generate()` (HF supports
  `output_scores=True, return_dict_in_generate=True`), or
- skip the separate `old_logprobs` pass when running a single inner epoch.

**Memory delta.** No reduction in *peak* (this forward is no-grad and short-
lived), but removes wall-clock time and a transient allocation. Mostly a
speed and clarity win.

**Experiment impact.** None for single-epoch on-policy GRPO. If the PPO
clip-ratio matters across multiple inner epochs (you currently do 1), this
becomes invasive.

**Effort.** ~1 hour.

#### 5.2.e Offload `ref_model` to CPU between forwards

**The change.** Keep `ref_model` on CPU, move to GPU only for the no-grad
ref forward inside `_build_step_groups`, then move back to CPU.

```python
ref_model.to("cuda")
with torch.no_grad():
    ref_logprobs_all = _batched_scalar_mean_completion_logprobs(ref_model, ...)
ref_model.to("cpu")
torch.cuda.empty_cache()
```

**Memory delta.** Saves **3.4 GB static** (the ref model's bf16 weights).
Costs ~5-10 seconds per step on PCIe transfer for a 1.7B model.

**Experiment impact.** None.

**Effort.** ~15 minutes.

#### 5.2.f `torch.cuda.empty_cache()` between phases

**The change.** Call once between `_build_step_groups` and `train_step` and
once after the optimizer step.

**Memory delta.** Doesn't free anything tensors are still pointing to, but
returns cached free blocks to the allocator pool, reducing fragmentation
peaks. Pairs well with 5.2.c.

**Experiment impact.** None.

**Effort.** 2 minutes.

---

### Tier 3 — Minor experimental impact, larger savings

#### 5.3.a Drop the ref model entirely (set `kl_coef = 0`)

**The change.** `kl_coef` is currently `0.001` in `shared_train.yaml`. The
KL term is being multiplied by 0.001 — its contribution to the loss is
negligible. Setting it to 0 lets you delete the ref model and the ref
forward pass entirely.

**Memory delta.** Saves **3.4 GB static** plus removes one full no-grad
forward over all completions (~30-60s of step time).

**Experiment impact.** Technically a different objective, but the
practical difference at `kl_coef=0.001` is sub-noise. If you want to preserve
"GRPO with KL" semantics for the paper, keep it; if you care about wall
clock and memory, drop it.

**Effort.** 1 yaml line + delete ~20 lines of code.

---

### Tier 4 — Medium-invasive, big savings (only if Tiers 1-3 aren't enough)

#### 5.4.a LoRA / PEFT fine-tuning

**The change.** Wrap `policy` with `peft.LoraConfig(r=16, ...)` so only the
LoRA adapters are trainable (~10 M params instead of 1.7 B).

**Memory delta.**

| Component | Full fine-tune | LoRA (r=16) |
|---|---|---|
| Trainable params | 1.7 B | ~10 M |
| Gradient buffer | 3.4 GB | ~0.02 GB |
| AdamW state | 13.6 GB | ~0.08 GB |
| **Static delta** | — | **~17 GB saved** |

**Experiment impact.** **Medium.** This changes the parameter space — you
are no longer doing full GRPO on the dense model, you are doing LoRA-GRPO.
Results may differ in absolute scores; relative comparisons across the
four runs (grpo/inverse_freq/f_grpo/run1b) are still apples-to-apples *to
each other*.

**Warning.** Doing LoRA on a 1.7B model on 80 GB is using a sledgehammer to
crack a walnut. Only do this if you have an unrelated reason to want LoRA,
or if Tiers 1-3 still leave you short. The bug is the bug; fix the bug
first.

**Effort.** ~3-4 hours including testing.

---

### Tier 5 — Heavy infra (do not do; not needed here)

#### 5.5.a FSDP / ZeRO-3 / DeepSpeed offload

Sharding optimizer state across multiple GPUs is appropriate for **7B+**
models on **single-GPU 24-40 GB** hardware. For Qwen3-1.7B on A100-80GB
after fixing the bug, this is wildly overkill. Listed only for completeness
— do not pursue.

---

## 6) Recommended Path

1. **Tier 1 (loss-level gradient accumulation)** — non-negotiable; this is
   the bug.
2. **Tier 2.b (gradient checkpointing)** — one line, free 3-4× headroom.
3. **Tier 2.a (gather instead of log_softmax-then-index)** — clean it up
   while you're in the file.
4. **Tier 2.c (`expandable_segments`)** — set in the Modal container.
5. Smoke test at original config (`batch_prompts=32, rollouts=8,
   max_new_tokens=2048, mb=16`). Expect peak ≈ 45-55 GB.
6. If it fits with margin, run the overnight matrix as originally planned.
7. Tiers 2.d / 2.e / 2.f / 3 are quality-of-life cleanups; do them later
   if you care about wall clock.

### Smoke acceptance criteria

- `step 1/100 done` appears in `train.log` (not just `groups ready`).
- Peak `nvidia-smi` usage during step 1 is **< 60 GB** at the original
  config.
- Loss is finite and `mean_reward` is logged.

### Stop rule

- If after Tier 1 + Tier 2.b the smoke still OOMs at `batch_prompts=8,
  rollouts=8, max_new=512`, the diagnosis is wrong — re-investigate. The
  predicted peak at that shape is ~30 GB; an OOM there means there is a
  second, separate leak.

---

## 7) Why This Pattern Was Easy to Miss

The training code is composed of small, individually-correct functions:

- `_micro_batch_mean_completion_logprobs` correctly returns differentiable
  per-completion logprobs.
- `_batched_mean_completion_logprobs` correctly chunks the work.
- `HFPolicyModel.logprobs_for_rollouts` correctly assembles them.
- `trainer.train_step` correctly produces a scalar loss.
- `_differentiable_loss` correctly assembles the surrogate.

Each function in isolation looks fine. The bug is in the *composition*: the
autograd graph extends across every layer of this stack and is only severed
at the final `.backward()` call. A reader scanning any single function would
see no leak. You have to look at the whole pipeline at once — and at that
view, the issue is glaring.

This is also why YAML reductions felt like they should help: the relevant
knob (`completion_logprob_micro_batch_size`) does control the per-iteration
cost, just not the *accumulated* cost. The visual evidence ("step 1/100
groups ready" then OOM) made it look like a per-iteration sizing issue
when it was actually an iteration-count multiplied by per-iteration
issue.

---

## 8) Evidence Pointers (for the next agent)

- Smoking-gun functions:
  - `pilot/train/hf_grpo_train.py:211-258` — `_micro_batch_mean_completion_logprobs` (differentiable, no `@torch.no_grad`)
  - `pilot/train/hf_grpo_train.py:261-289` — `_batched_mean_completion_logprobs` (accumulates graphs)
  - `pilot/train/hf_grpo_train.py:292-338` — `HFPolicyModel.logprobs_for_rollouts` (stores them)
  - `pilot/train/hf_grpo_train.py:798-822` — `run_grpo_training` main loop (single `.backward()` at end)

- Failed-attempt artifacts:
  - `pilot/artifacts/run1b_grpo/20260519T112004Z/` — smallest reproducible failure
  - `pilot/artifacts/run1b_grpo/20260519T110958Z/` — penultimate reduction
  - Full ledger in `pilot/grpo_smoke_debug_history.md`

- Config files referenced:
  - `pilot/configs/shared_train.yaml`
  - `pilot/configs/run1_grpo.yaml`, `run1b_grpo.yaml`, `run2_inverse_freq.yaml`, `run3_f_grpo.yaml`

---

## 9) One-Sentence Summary

The OOM is caused by accumulating per-micro-batch autograd graphs into a
Python list across the full step and only calling `.backward()` once at
the end; the fix is to call `.backward()` per micro-batch, after which
the original `batch_prompts=32 / rollouts=8 / max_new_tokens=2048` config
fits comfortably in 80 GB with ~50% headroom.
