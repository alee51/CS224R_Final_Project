# B200 GC-Off VRAM Explainer (minority_answer, n_kept~512)

## Executive verdict

Primary failure mode is **true CUDA OOM during HF train forward** when `gradient_checkpointing=false` is combined with long completions at this workload scale.  
The later `Trying to free a pointer not allocated here` / allocator errors are **secondary crash fallout** after the process is already in a bad allocator state from the OOM or cuMem wake failure.

For the failing `sleep_gc75` run (`76inco1f`), the stack shows OOM at `trainer.py::_completion_logprobs_hf -> model(...)` with only ~0.9 GiB free and a failed +1.14 GiB allocation.

---

## What code path does (relevant to memory)

- `main/train/trainer.py`
  - `build_hf`: enables checkpointing only when `train.gradient_checkpointing=true`.
  - `_train_step_microbatched`: packs by `token_budget`, then runs differentiable logprob forward + backward per chunk.
  - `_completion_logprobs_hf`: creates full logits for each prompt+completion sequence; this is where the failing run OOMs.
- `main/train/rollout.py`
  - `sleep_for_train`: vLLM sleep frees KV memory before HF train.
  - `wake_weights_only` and `wake_for_rollout`: staged wake path after training.
- configs
  - `train_real.yaml`: baseline `token_budget=105000`, `gradient_checkpointing=true`.
  - `train_real_b200_ablate_gc_off.yaml`: gc off only.
  - `train_real_b200_ablate_sleep_gc_off_75k.yaml`: gc off + `token_budget=75000` (with sleep enabled by launcher label).

---

## Artifact readout

- `sfp0xwag` (`gc_off.log`): OOM at step 0 with process at ~178.31/178.35 GiB.
- `76inco1f` (`sleep_gc75.log`): sleep freed ~83.84 GiB, then OOM in HF forward (`_completion_logprobs_hf`) trying to allocate 1.14 GiB; allocator/free-pointer crash follows.
- `wdl3fczm` (baseline from efficiency docs): stable around `vram_peak ~148-153 GB`, step ~266s.
- `yqlmvnw0` (`sleep.log`, gc-on): includes `use_cache=True is incompatible with gradient checkpointing...` and later cuMem wake crash path; not a gc-off run.

---

## Quantified VRAM estimate (explicit assumptions)

Assumptions used for first-order estimate:

- B200 usable cap from logs: `V_cap ~= 178.35 GiB`.
- Baseline (gc-on, no-sleep, budget 105k): `V_peak_base ~= 150 GiB` (148-153 range).
- vLLM reserve at rollout settings (`gpu_memory_utilization=0.45`): `V_vllm_awake ~= 80 GiB`.
- Sleep frees about `~84 GiB` and leaves `~4 GiB` vLLM resident during HF train.
- With this workload (`n_kept~512`, long completions), turning gc off adds about `Delta_gc ~= +30 GiB` HF-train activation pressure (from prior probe notes and gc_off failures).

Memory model:

- Collocated baseline: `V_peak_base ~= V_hf_gc_on + V_vllm_awake`
  - so `V_hf_gc_on ~= 150 - 80 = 70 GiB`.
- Estimated HF with gc off: `V_hf_gc_off ~= 70 + 30 = 100 GiB`.
- Sleep+gc_off train-phase estimate: `V_train_sleep_gc_off ~= V_hf_gc_off + V_vllm_sleep_resident ~= 100 + 4 = 104 GiB` before accounting for long-tail chunk spikes/fragmentation.

That explains why sleep should create large headroom *in principle*, but not guarantee safety at extreme token tails: the OOM is triggered by peak chunk/sequence allocations (lm-head/logits + activation graph) and allocator fragmentation, not just average phase footprint.

| Scenario | Approx peak formula | Estimated peak | Outcome |
|---|---:|---:|---|
| Baseline gc-on, no sleep (`wdl3fczm`) | `70 + 80` | `~150 GiB` | Stable |
| gc-off, no sleep (`sfp0xwag`) | `(70+30) + 80` | `~180 GiB` | OOM (observed) |
| sleep + gc-on (`yqlmvnw0`) | `70 + 4` | `~74 GiB` train phase | Train fits; wake path crashed |
| sleep + gc-off + 75k (`76inco1f`) | `(70+30)+4 + tail/frag overhead` | nominal `~104 GiB`, but large spikes | OOM + allocator cascade (observed) |

Why 75k can still OOM under sleep+gc_off:

1. `token_budget` constrains **sum completion tokens per chunk**, but a single very long sequence can still form a high-peak chunk.
2. `_completion_logprobs_hf` still builds large logits tensors for prompt+completion; gc-off keeps more graph state live.
3. Long-completion heavy minority batches increase variance; worst-case chunk, not mean chunk, determines failure.
4. Post-sleep allocator state + many large allocations can fragment memory; final failure was a +1.14 GiB request with <1 GiB free.

---

## Warning line interpretation

Warning:

`use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False.`

Meaning:

- This is expected **when checkpointing is ON** in HF training forward.
- It is emitted by Transformers because KV cache (`use_cache=True`) conflicts with gradient checkpointing in training.

Does it affect gc-off run?

- For a true gc-off path (`train.gradient_checkpointing=false`), this warning should generally **not** appear from the HF train model.
- In the artifacts here, that warning appears in `sleep.log` (`yqlmvnw0`, gc-on sleep test), not in `sleep_gc75.log` (`76inco1f`).
- So it is not evidence that gc-off failed to apply for `76inco1f`; it likely came from a different smoke or config path where gc remained enabled.

---

## Why observed failures happen (primary vs follow-on)

1. **Primary failure (`76inco1f`)**: CUDA OOM in `modeling_qwen3.py` lm-head path during `_completion_logprobs_hf` forward.
2. **Secondary failures**: NCCL/process-group warning and `Trying to free a pointer not allocated here` are downstream cleanup failures after allocator/runtime is already corrupted by the primary exception path.
3. Similar secondary signature also appears after cuMem wake errors in sleep-only run (`yqlmvnw0`), reinforcing that allocator/free errors are not root cause.

---

## Next 2 experiments (maximize information gain)

### 1) Controlled gc-off memory slope without sleep confound

- Config: clone `train_real_b200_ablate_gc_off.yaml` into two smokes with:
  - `token_budget: 60000` and `token_budget: 80000`
  - keep `logprob_seq_batch=1`, `n_rollouts=8`, same arm and data.
- Goal: identify lowest stable gc-off budget and quantify `vram_peak` slope under identical non-sleep allocator behavior.
- Success criteria: complete 10 steps each; collect per-step `n_kept`, `num_chunks`, `vram_peak_gb_step`.

### 2) Sleep+gc-off with stricter anti-tail cap

- Config: new sleep+gc-off smoke:
  - `gradient_checkpointing: false`
  - `token_budget: 50000` (more conservative than 75k)
  - keep `logprob_seq_batch=1`.
- Goal: test whether `76inco1f` is driven by long-tail chunk peaks vs general incompatibility.
- Success criteria: if this survives 10 steps, raise budget in +10k increments (50k -> 60k -> 70k) to find safe envelope.

These two runs separate (a) pure gc-off activation cost from (b) sleep allocator/tail interaction, which should reduce confusion quickly.
