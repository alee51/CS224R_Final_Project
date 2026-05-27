# B200 sleep + `gc_off` bring-up — stop status (2026-05-27)

## Executive decision

We are **stopping further iteration** on the **`vllm_sleep=1` + `gradient_checkpointing=false`** bundle for poster runs **unless a single additional smoke with reduced vLLM KV reservation (`rollout.gpu_memory_utilization`) goes green**.

Rationale: repeated **true CUDA OOMs** at the device cap (~178.35 GiB) on **sleep+gc_off** smokes, plus allocator instability after OOM, mean this path is no longer a predictable time sink.

We will proceed with **B200, `vllm_sleep=0`, `gradient_checkpointing=true`**, and optionally the safe **`token_budget=130k`** tweak (already measured) if we want a small step-time win.

---

## What we were trying to achieve

Goal: use B200 headroom to speed up `minority_answer` by:

- **sleep**: evict vLLM KV cache from GPU during HF train to free tens of GB VRAM
- **gc_off**: spend that freed VRAM on storing activations to reduce backward recompute time
- **token_budget**: tune chunking to use remaining headroom safely

Expected big win: `gc_off` attacks the slow pole (HF backward), which is ~200s of a ~266s step on B200 minority.

---

## Ground-truth baselines (measured)

### Baseline B200 minority (no sleep, gc on)

Run: wandb `wdl3fczm` (10-step smoke).

- **Step time**: ~266s (rollout ~59s, train ~209s)
- **Train breakdown**: logprob_fwd ~11s, **backward ~197s**
- **PyTorch VRAM peak** (`torch.cuda.max_memory_allocated`): ~149–153 GB
- **Chunks**: 5 at `token_budget=105k`

### Safe small win: token budget 130k (no sleep, gc on)

Run: wandb `au96bwh1` (10-step smoke).

- **Step time**: ~244s (≈ **−22s** vs baseline)
- **PyTorch VRAM peak**: ~161 GB (≈ **+13 GB** vs baseline)
- **Chunks**: 4

Interpretation: `token_budget=130k` is a **safe-ish**, modest speed/VRAM trade with no semantic change.

---

## Sleep: what worked and what it does *not* do

### What worked

In the `sleep_only` run (Modal app `ap-9FLdEfszTX8vrw6x22baYx`), vLLM logs show:

- vLLM awake reservation at `gpu_memory_utilization=0.45`:
  - total_gpu_memory 178.35 GiB
  - vLLM pool: 80.26 GiB
  - KV cache reserved: 76.17 GiB
- **Sleep freed ~82–85 GiB** each step (example log: “Sleep mode freed 84.87 GiB …”).

We also fixed wake sequencing to use `wake_up(tags=['kv_cache'])` (instead of untagged full wake), which allowed the sleep run to log steps without the prior `cumem_allocator` crash.

### What sleep does *not* do

Sleep does **not** reduce the **awake** peak. When vLLM wakes for rollout it re-reserves its full KV pool, so total GPU memory can still spike near the device cap in the awake phase.

This explains why the Modal GPU graph can show ~178+ GB peaks even when sleep “works”: the peaks occur during **rollout / wake**, not during the **HF train** window.

---

## `gc_off`: the hard blocker

### Minimum additional VRAM requirement (measured)

Comparing baseline vs `gc_off`-only OOM:

- baseline PyTorch peak: ~148.5 GB
- `gc_off` OOM: ~178.31 / 178.35 GiB in use (at failure)

So `gc_off` needs **at least ~30 GB** additional headroom *in this workload shape* (and likely more for fragmentation / long-tail sequences).

### Sleep + gc_off still OOM’d (multiple times)

Even with sleep freeing ~84 GiB, both `sleep_gc75` and `sleep_gc40` smokes hit true OOMs at the cap:

- `sleep_gc75` (wandb `76inco1f`): OOM during HF forward; followed by allocator abort (“free pointer not allocated here”).
- `sleep_gc40` (wandb `qfb917u2`): OOM trying to allocate 24 MiB with GPU at ~178.30 GiB in use; allocator abort after.

Interpretation: in practice, with this single-process collocation stack (PyTorch HF + vLLM cuMem), `gc_off` can expand HF’s peak into the freed space and still reach the device cap. After OOM, allocator interactions produce noisy secondary errors.

---

## Why the “valleys ~70 GB” and “peaks ~178 GB” can both be true

Two different measurements:

- **wandb `train/vram_peak_gb_*`**: tracks **PyTorch allocated** VRAM only.
- **Modal GPU memory plot**: tracks **total device memory**, including vLLM non-torch / cuMem / KV reservations.

So a run can have:

- **train-phase valleys** (vLLM asleep) while still
- hitting **awake-phase peaks** (vLLM wakes and reserves KV) near the full device.

---

## “Give up” decision criteria

We stop here unless the next single experiment succeeds:

- **sleep + gc_off + low token_budget + reduced vLLM KV** (`rollout.gpu_memory_utilization <= 0.35`)
- 10 steps complete
- no OOM and no allocator abort
- and step-time win is meaningful (≥15–20%) vs baseline

If that fails, we proceed with:

- B200 production runs with `vllm_sleep=0`
- keep `gradient_checkpointing=true`
- optionally adopt `token_budget=130k` (measured ~8% win) if we can spare ~+13 GB VRAM margin

---

## Notes / miscellany

- Warning `use_cache=True is incompatible with gradient checkpointing...` is expected on gc-on runs and not itself an error.
- After OOM, errors like “Trying to free a pointer not allocated here” are treated as **post-OOM allocator fallout**, not the primary failure.

