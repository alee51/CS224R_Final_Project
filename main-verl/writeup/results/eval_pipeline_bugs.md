# Eval pipeline bugs hit + fixed (2026-06-02 → 2026-06-04)

Bugs discovered and fixed while running the locked eval spec (`eval.md`).
Recorded here so future sessions don't re-hit them.

For higher-level pipeline audit (which scripts work, schema changes) see
`eval_pipeline_verification.md`. For the current results matrix see
`INDEX.md`.

## Bug 1 — `kl_from_base.py` stale `parents[2]` after `posthoc/` reorg

**Symptom:** Modal app `ap-Q6ajet56a2yK5qZGOKWL1x` (2026-06-03 10:21 PDT) failed
at import with:

```
File "/root/kl_from_base.py", line 60, in <module>
    _MAIN_VERL_ROOT = Path(__file__).resolve().parents[2]
IndexError: 2
```

**Root cause:** The file originally lived at
`main-verl/eval/analysis/kl_from_base.py` (`parents[2]` = `main-verl/`).
Commit `d3adcc5` reorganized analysis into `posthoc/`, so the new path is
`main-verl/eval/analysis/posthoc/kl_from_base.py` (depth +1). On Modal-side
the file is mounted flat at `/root/kl_from_base.py`, where `parents[2]`
doesn't exist at all.

**Fix:** Bumped to `parents[3]`, wrapped in `try/except IndexError: pass`
for the Modal-flat-mount case. `main-verl/eval/analysis/posthoc/kl_from_base.py:58-65`.

**Lesson:** Any time analysis scripts get reorganized into a subdirectory,
audit every `Path(__file__).resolve().parents[N]` reference. Same pattern
existed in `analysis_io.py:111` and was patched independently by the
parallel session.

## Bug 2 — `kl_from_base.py` vLLM OOM at default `max_num_seqs`

**Symptom:** Modal app `ap-dvz549H9hytfCryAIwdN7C` (2026-06-03 10:22 PDT)
crashed mid-batch with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.70 GiB.
GPU 0 has a total capacity of 178.35 GiB of which 3.79 GiB is free.
Process 1 has 174.54 GiB memory in use.
```

Crash at `vllm/model_executor/layers/sampler.py:_get_ranks` after processing
13 of 240 input prompts (logs at `13/240` `est. speed output: 1.19 toks/s`).

**Root cause:** The KL pass uses `SamplingParams(prompt_logprobs=20)` to get
the base distribution at every input position. Per-prompt logprob storage
is roughly `len(prompt+rollout) × 20 × sizeof(float)` — far heavier than
generation. The default `max_num_seqs=256` let vLLM try to batch 240 inputs
concurrently with up to 5120 tokens each. Combined with `gpu_memory_utilization=0.85`
(151 GB reserved), there was no headroom for the logprob tensors.

**Fix:** Added `max_num_seqs=16` and lowered `gpu_memory_utilization=0.70` in
`main-verl/eval/analysis/posthoc/kl_from_base.py:198-203`. This caps
concurrency without changing semantics. Re-fire (`ap-8u9GPVGGm9HQCd1IzPpZnE`)
completed all 5 GRPO + 5 Minority cells without OOM.

**Lesson:** Teacher-forcing workloads with `prompt_logprobs=N` are
qualitatively different from generation workloads. Defaults tuned for
generation will OOM. Cap `max_num_seqs` aggressively, lower
`gpu_memory_utilization` to leave headroom for the logprob tensors.

## Bug 3 — `kl_from_base.py` `max_model_len=5120` too small for polyepo

**Symptom:** After Bug 2's fix, the KL pass on polyepo cells crashed:

```
ValueError: The decoder prompt (length 5898) is longer than the maximum
model length of 5120.
```

**Root cause:** Teacher-forcing concatenates `rendered_prompt + rollout_text`
and feeds it to vLLM. Polyepo rollouts run consistently longer than GRPO/Minority
(observed during MATH-500 generation: polyepo's output throughput was ~800 toks/s
vs GRPO's ~2800 toks/s, consistent with longer rollouts at `temperature=1.0` and
`max_tokens=4096`). The combined length exceeded the 5120-token model-length cap.

**Fix:** Raised `max_model_len` from 5120 to 8192 in
`main-verl/eval/analysis/posthoc/kl_from_base.py:200`. The B200 has plenty of
HBM headroom at `gpu_memory_utilization=0.70` and `max_num_seqs=16` even with
the bigger context window. Re-fire on polyepo cells (`ap-ge2TggKflVz9x3C96bmnkK`)
completed all 5 cells cleanly.

**Lesson:** When deciding `max_model_len` for teacher-forcing, you need the
worst-case `prompt + completion` length, not just `max_tokens`. Polyepo's
high-entropy policy can use most of the 4096 generation budget AND get a
~1500-token prompt (long chat-templated math problems).

## Bug 4 — `run_eval.py` `max_num_seqs=4096` KV-cache preemption thrash

**Symptom:** Poly-EPO math500 GEN (first attempt, `ap-CDXJXhgCP6lMe1VKRH4ZYV`,
2026-06-03 10:20 PDT) ran for 6 hours and only reached 22144/32000 (69%) before
hitting Modal's 6-hour task timeout. Output rate at completion was 1204 toks/s
vs GRPO/Minority's ~2800 toks/s under identical config. Logs were dominated by
`Sequence group N_parallel_sample_M is preempted by PreemptionMode.RECOMPUTE
mode because there is not enough KV cache space` — `total_num_cumulative_preemption`
crossed 17,000.

**Root cause:** Commit `4289d0c` had set `max_num_seqs=4096` (from vLLM default 256)
"to fill the B200's batch capacity". For 4096 concurrent sequences each up to 5120
tokens at `max_tokens=4096`, the worst-case KV-cache budget needed is far more
than the ~150 GB available (Qwen3-4B GQA per-seq KV ≈ 752 MB worst-case ×
4096 = 3 TB). vLLM admitted way more sequences than could fit, then thrashed
between preempting and recomputing them. **GPU compute utilization fell to ~35%
because compute units sat idle waiting for KV space to free.** GRPO/Minority/Base
math500 also ran under this config; their shorter rollouts hid the issue.

**Fix:** Lowered `max_num_seqs` to 256 first (`main-verl/eval/run_eval.py:167`),
then to 128 after observation showed 256 was still preempting at the tail.
Bumped `gpu_memory_utilization` from 0.95 to 0.98 to claw back a bit of headroom.
Per-token KV math: at 128 concurrent seqs × 752 MB worst-case = 96 GB, well under
the ~166 GB budget at 0.98 utilization. The retry (`ap-h8zHYGx8IuvDhiPOfYtITd`)
ran at ~2900–3050 toks/s steady, ETA ~2 hr.

**Lesson 1:** `max_num_seqs` is NOT auto-tuned by vLLM. It accepts whatever you
set and preempts when oversubscribed. The actual ceiling depends on `max_tokens`,
`max_model_len`, and per-token KV size — calculate before setting.

**Lesson 2:** Lower concurrency can be faster than higher concurrency for
small-model decode on big GPUs. Decode is HBM-bandwidth-bound; once enough
concurrent sequences saturate bandwidth, additional sequences just compete for
KV cache and cause preemption thrash. The throughput peak is at the bandwidth-
saturation point, not the GPU-memory-capacity point.

**Lesson 3:** A "successful" run that completes under timeout can still be
running at <50% optimal throughput. GRPO/Minority math500 completed under the
old config but were ~2× slower than they would have been at `max_num_seqs=128`.
We didn't catch this because they fit in the budget; polyepo's longer rollouts
made it visible.

## Bug 5 — `run_eval.py` json.dump hung on polyepo math500 post-generation

**Symptom:** Modal app `ap-h8zHYGx8IuvDhiPOfYtITd` (2026-06-03 21:48 PDT)
generated all 32000 math500 rollouts successfully (log: `math500: generated
in 12597.0s`), then hung in the post-generation phase. After 50+ minutes
the on-volume JSON was stuck at 85.3 MiB (vs expected ~50 GB based on
GRPO 44 GB / Minority 38 GB / Base 23 GB) with mtime frozen at 01:40 PDT.
No `wrote per-dataset` log line ever appeared. Modal app task remained
in `1 task` state with no further stdout. Eventually `modal app stop -y`'d
at ~02:30 PDT.

Recovered data: 2 of 500 prompts written (from the partial 85 MB). Pass@k
on those 2 prompts: pass@1=0.023, pass@64=0.500 — meaningless given the
tiny sample. Effectively unrecoverable.

**Root cause (hypothesized):** The post-generation phase scores 32000
rollouts then `json.dump(..., indent=2)` of the full results dict to the
mounted Modal volume. For polyepo (longer rollouts than GRPO/Minority at
`temperature=1.0`), the serialized payload would be ~50 GB. The
combination of (a) `indent=2` adding per-token formatting overhead, (b)
nested logprob dicts (32000 rollouts × ~5000 tokens × 20 top-K entries),
and (c) Modal volume write throughput appears to have stalled the process.
Base/GRPO/Minority math500 completed under identical config — only
polyepo's longer rollouts triggered the failure.

**Fix:** None applied in v1. The 23 other cells are sufficient for the
poster story (3/4 arms × math500 + 4 arms × 5 smallood). Re-fire options
for v2:
- `json.dump(..., indent=None)` to skip pretty-printing — should reduce
  output size by 30–50% and write much faster.
- Stream-write via `json.dump` chunks (per-prompt) instead of one giant
  dict serialization.
- Skip `logprobs` in the per-rollout dict when `max_tokens > N` to bound
  the serialization size.

**Lesson:** A Modal function with `gpu="B200:1"` that runs for 3.5 hours
generating + scoring is expensive (~$15). Failing in the final 5-minute
JSON-write step wastes the entire run. Always have a "first save the
small bits then attach the heavy bits" pattern: write pass@k + n_correct
+ preds first (tiny), THEN write rollouts + logprobs (heavy). Crashing
mid-heavy still leaves the headline number safe.

## Cross-cutting takeaway

All four bugs share a pattern: **a configuration value that was "fine"
under one workload silently broke a slightly different workload**.

- Bug 1: a path index that worked for the old directory location.
- Bug 2: a `max_num_seqs` default that worked for generation but not for
  teacher-forcing.
- Bug 3: a `max_model_len` that worked for GRPO/Minority's shorter rollouts
  but not for polyepo's longer ones.
- Bug 4: a `max_num_seqs=4096` that "worked" for shorter-rollout arms but
  was throttling them ~2× without anyone noticing.

For future eval re-runs, the right defensive moves are:
1. Per-arm-or-per-dataset throughput watchdog (alert if toks/s drops >50%
   from the median).
2. KV-budget calculator at startup that logs `worst_case_kv_needed_GB` vs
   `kv_budget_GB` and refuses to start if oversubscribed.
3. Length-overflow guard in `kl_from_base.py` that warns when sliced
   `prompt + rollout` exceeds `max_model_len * 0.9`.
