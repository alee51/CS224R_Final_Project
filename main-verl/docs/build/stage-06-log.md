# Stage 6 log — 4B fit check

**Stage ID:** `stage-06`
**Plan:** [stage-06-agent-plan.md](./stage-06-agent-plan.md)
**Modal profile:** `chicken602`

---

## Dispatch log


| Section                       | Executor   | Audit   | Verdict                                                                   |
| ----------------------------- | ---------- | ------- | ------------------------------------------------------------------------- |
| S6.1 grpo_smoke_4b.yaml       | 2026-05-31 | pending | ladder 1b yaml kept for Stage 8 fork; fit probe uses judge config instead |
| S6.2 probes                   | 2026-05-31 | pending | `minority_cot_judge_smoke_4b` = single combined probe                     |
| S6.3 4B fit + step-time       | pending    | pending | one run: minority_cot + judge + ladder 1b (not GRPO)                      |
| S6.0 judge sanity (3 prompts) | partial    | pending | 1.7B training trace v3 only                                               |


---

## S6.3 — 4B combined probe (minority_cot + judge + ladder 1b)


| Attempt    | Config                                                              | App ID                      | Verdict                                                    | Notes                                                                    |
| ---------- | ------------------------------------------------------------------- | --------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1          | micro=2, log_prob=2, gpu_mem=0.40, GRPO                             | `ap-S2R6SHVIejJEvR37sRc5UH` | **CANCELLED**                                              | Reached step-0 update_policy ~34% then cancellation ~04:52 UTC (not OOM) |
| 1b         | micro=4, log_prob=4, gpu_mem=0.45, **minority_cot+judge**, 10 steps | `ap-QJytg857yN6exqsxUsttAq` | **CANCELLED**                                              | Modal cancellation signal ~04:52 UTC (not OOM); was loading 4B           |
| 1b retry   | same                                                                | `ap-3XZ8D0l3muUrnDQl9yhMXn` | **STOPPED**                                                | Fabric Manager GPU error at FSDP init; manually stopped                  |
| 1b retry 2 | same                                                                | `ap-hEnPrmJY7C9X3pxT3cJT3b` | **STOPPED**                                                | step1 timing_s/adv=505s, step=777s; killed for ladder1d                  |
| 1d         | gpu_mem=0.65, batched judge (yaml batch 16)                         | `ap-sE4d5Nyot7kXCaS9krVRRw` | **CANCELLED**                                              | killed by `head -35` pipe during launch (not OOM)                        |
| 1d retry   | same                                                                | `ap-VLCE5m8Spangt8O7NjYywP` | **STOPPED @ step 5**                                       | CLI stop (not OOM); steps 1–4 PASS — see metrics below                   |
| 1c         | gpu_mem=0.50, fast smoke (3 steps)                                  | superseded                  | use 1d instead                                             |                                                                          |


---

## S6.3 ladder 1d results (ap-VLCE5m8Spangt8O7NjYywP)

Batched judge + `gpu_memory_utilization: 0.65` — **4 steps completed before manual stop.**

| Metric | 1b baseline | 1d step 4 |
|--------|-------------|-----------|
| `timing_s/adv` (judge) | 505s | **~96s** |
| `[clusters_judge] wall_s` | ~442s | **~94–118s** |
| `timing_s/step` | 777s | **~249s** |
| `timing_s/gen` | 164s | **~54s** |
| Peak GPU mem | ~115 GB/rank | **~154 GB/rank** (no OOM) |
| Batch client | 128 HTTP calls | **`n_http_posts=8`** (batch 16) |
| `parse_ok_rate` | ~0.98 | **~0.95** |

**Verdict:** ladder 1d knobs look **lockable** for Stage 8 (fit + judge speedup confirmed).

---

Plan: after GRPO 4B loads, run `judge_cluster_trace_fast` with `CS224R_TRACE_ACTOR_MODEL=Qwen/Qwen3-4B-Base` on prompt indices 0, 5, 100 and compare degenerate rate vs 1.7B trace.

```bash
export MODAL_PROFILE=chicken602
export CS224R_TRACE_ACTOR_MODEL=Qwen/Qwen3-4B-Base
export JUDGE_BASE_URL=https://chicken602--v1-chat-completions.modal.run
for idx in 0 5 100; do
  CS224R_JUDGE_TRACE_PROMPT_IDX=$idx ./main-verl/scripts/launch_judge_cluster_trace_fast.sh
done
```

