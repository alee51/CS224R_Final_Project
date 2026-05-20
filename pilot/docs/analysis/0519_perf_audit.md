# GRPO A100 performance audit (2026-05-19)

**Baseline:** ~99 min/step (`build_seconds` ~26 min + `train_seconds` ~73 min; 32×8=256 completions, 2048 `max_new_tokens`, Qwen3-1.7B, A100-80GB). **Goal:** <30 min/step.

---

## 1. Rollout generation — **fixable (blocking for <30 min alone)**

**Verdict:** HuggingFace `model.generate()`, not vLLM/SGLang.

```97:97:pilot/train/rollout_engine.py
                    out = model.generate(**inputs, **gen_kw)
```

GRPO uses per-prompt seeds (`step_seed + i`); with `allow_seeded_prompt_batching=false` (default), each prompt in a micro-batch decodes **serially**:

```175:197:pilot/train/rollout_engine.py
            else:
                # Per-prompt seeds (run0: seed+i, GRPO: step_seed+i) keep strict legacy RNG semantics.
                for problem, row_seed in zip(chunk_probs, chunk_seeds):
                    ...
                        out = model.generate(
```

`build_seconds` (~1500–1656s) is all of `_build_step_groups` (rollout + policy/ref logprobs); rollout dominates.


| Change                                                                    | Effort | Rollout speedup                           |
| ------------------------------------------------------------------------- | ------ | ----------------------------------------- |
| `allow_seeded_prompt_batching: true` (batched rows + per-row `Generator`) | ~1–2 h | **~4–6×** on decode (~20 min → ~4–5 min)  |
| vLLM engine + weight sync each step                                       | 1–2 d  | **~3–5×** vs HF batched (~2–4 min decode) |


**vLLM risks:** per-prompt seeded sampling must match HF semantics; weight reload each GRPO step; image size + version pins (`modal_app.py` has no `vllm`).

---

## 2. Attention backend — **fixable**

**Verdict:** No FlashAttention-2 / `attn_implementation` set; default SDPA/eager path.

```841:845:pilot/train/hf_grpo_train.py
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
```

`modal_app.py:63-71` does not install `flash-attn`. Long padded forwards (logprob + train) leave decode throughput on the table. **~15–25%** on 2k-token sequences if `attn_implementation="flash_attention_2"` (or `sdpa` + env) after adding dep.

---

## 3. Precision — **OK**

```825:825:pilot/train/hf_grpo_train.py
    dtype = torch.bfloat16
```

Policy/ref load bf16; no `autocast` wrapper (weights already bf16). `F.log_softmax` runs in activation dtype — no obvious fp32 layernorm override. **Not a primary lever.**

---

## 4. Gradient checkpointing — **fixable (high train impact)**

**Verdict:** **ON** during training.

```846:847:pilot/train/hf_grpo_train.py
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
```

At 256×~~2k tokens, 80GB may allow checkpointing **off** + larger logprob micro-batch → **~~35–45% faster backward** (~73 min → ~40–48 min). Verify with one step; OOM → raise `completion_logprob_micro_batch_size` only.

---

## 5. Optimizer — **fixable (small)**

```862:862:pilot/train/hf_grpo_train.py
    optimizer = AdamW(policy.parameters(), lr=lr)
```

Vanilla AdamW. `torch.optim.AdamW(..., fused=True)` or Apex FusedAdam: **~10–15%** on optimizer step (smaller vs forward/backward).

---

## 6. Synchronization stalls — **OK (minor)**

`.item()` in logprob paths (`hf_grpo_train.py:129,185,343,463`) — once per completion/micro-batch, not per token. `objectives.py` is pure Python on 8 rewards — negligible. Logging is per-step only (`hf_grpo_train.py:890-959`); `PYTHONUNBUFFERED=1` is fine.

---

## 7. Dataloader / tokenization — **fixable (medium)**

Prompts loaded once (`hf_grpo_train.py:815`). **No cache** for `(problem, completion)` — `_encode_prompt_completion` re-tokenizes on every logprob pass (build: policy+ref; train: backward) — **3× redundant tokenization** per completion.

---

## 8. Decode batching — **fixable (major)**

Default `allow_seeded_prompt_batching: false` (`hf_grpo_train.py:803-805`). With 32 prompts, `micro_batch_size=8`: **32 serial `generate` calls** vs **4 batched** (same-seed) or **32 batched** (`allow_seeded` = 4 chunks × 8 decode rounds).

Debug scaling (`run1b` logs): 8 prompts ×512 tok → 145s build; 32×2048 → ~~1500s (~~10× prompts ×4× tokens). Serial path ≈ **~70–80% of build** → **~18–22 min** wasted vs batched-seeded path.

---

## Top 3 by ROI (speedup × effort)


| Rank  | Change                                                                                | Est. step time          | Effort                 |
| ----- | ------------------------------------------------------------------------------------- | ----------------------- | ---------------------- |
| **1** | Turn off grad checkpointing + `completion_logprob_micro_batch_size: 32` (if OOM-safe) | train **73→~38–45 min** | Low (config + 1 smoke) |
| **2** | `allow_seeded_prompt_batching: true`                                                  | build **26→~10–14 min** | Low (config flag)      |
| **3** | FlashAttention-2 in Modal image + `attn_implementation` on load                       | **~10–15%** both phases | Medium (image + load)  |


**All three applied (optimistic):** build ~~10 min + train ~32–38 min → **~~42–48 min/step** (still above 30).

**To reach <30 min** add **vLLM rollouts** (rank 4) or cut tokens/completions (science change).

### Pseudocode sketches

**#1 — `hf_grpo_train.py:846-847`, config**

```python
# if smoke step fits in 80GB:
# policy.gradient_checkpointing_enable()  # remove
completion_logprob_micro_batch_size: 32  # yaml
```

**#2 — run yaml / `hf_grpo_train.py:803`**

```python
allow_seeded_prompt_batching: true
# rollout_engine.py:107-147 already implements batched+Generator path
```

**#3 — `modal_app.py:63` + `hf_grpo_train.py:841`**

```python
.pip_install("flash-attn", ...)  # pin for torch/CUDA
AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")
```

---

## Estimated step time (top 3 only)


| Phase     | Baseline    | After top 3    |
| --------- | ----------- | -------------- |
| Build     | ~26 min     | ~10–12 min     |
| Train     | ~73 min     | ~32–38 min     |
| **Total** | **~99 min** | **~42–50 min** |


**+ vLLM (#4):** total **~28–35 min** (borderline <30).

---

## Risks

- `**allow_seeded_prompt_batching`:** HF `generator=` batched path may diverge from legacy per-prompt seeds → validate `mean_reward` on fixed slice before main matrix.
- **No grad checkpointing:** OOM mid-step; revert or reduce batch.
- **vLLM:** seeded multi-sample parity, weight sync latency, larger container cold start.
- **FlashAttention:** build fragility on Modal; Qwen3 compatibility.

**Not audited:** tier-1 eval after training, duplicate end-of-run checkpoint save (`hf_grpo_train.py:962-996`).