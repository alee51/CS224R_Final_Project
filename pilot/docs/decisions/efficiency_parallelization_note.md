> **Superseded note (2026-05-19):** This document is kept for historical context.  
> Use `./training_parallelization_plan.md` as the current execution plan,
> and `../operations/MAIN_RUNS_PLAYBOOK.md` for main-run guidance.

# Efficiency & parallelization (agent spec)

**Do not change:** `preflight_lock.json` metrics/seeds/data, run configs’ scientific knobs, artifact schema.

## Problem

GPU ~20% on A100: **1.7B + serial `generate` per prompt + serial logprob forwards**. `batch_prompts` = GRPO step size, not parallel decode.

## Two layers

| Layer | Fix | How |
|-------|-----|-----|
| **Across runs** | 4 GPUs | `./pilot/scripts/launch_pilot_matrix.sh` (already exists). Never `--run-ids` for matrix. |
| **Inside a run** | Batch decode + batch logprobs | Code below |

## Inside-run implementation (priority order)

### P0 — Batched logprobs (run1–3)

**File:** `pilot/train/hf_grpo_train.py`

- Replace per-completion `_scalar_mean_completion_logprob` loops in `_build_step_groups` with a micro-batch forward: pad `(prompt+completion)` `input_ids`, mask prompt positions, mean logprob over completion tokens.
- Keep same math/seed behavior; batch size tunable (e.g. 8–32 completions).
- **Verify:** `debug_max_prompts=2`, same seed → rewards match pre-change within float tolerance.

### P1 — Batched rollouts (run0 + run1–3)

**Files:** `pilot/train/rollout_engine.py`, `pilot/infra/execute.py`, `pilot/train/hf_grpo_train.py` (`_sample_rollouts`)

- Add `sample_rollouts_batch(prompts: list[str], n: int, seeds: list[int]) -> list[list[str]]`.
- HF path: tokenize with `padding=True`, batch `generate` where possible; else micro-batch K prompts (start K=4–8).
- Run0: replace per-prompt loop in `run0_proxy` with micro-batches.
- GRPO: in `_build_step_groups`, call batch API instead of inner `for i, row`.
- **Verify:** run0 `debug_max_prompts=5` — `minority_correct_prompt_rate` unchanged vs baseline commit.
- **Seeds:** per-prompt `seed+i` uses serial `generate` inside micro-batch; true multi-prompt batch only when seeds match (rare). Cross-prompt GPU batch → P3 vLLM.

### P2 — run0 Modal shard (optional)

**Files:** `modal_app.py` CLI (`--run0-slice-start/end`), merge script `pilot/scripts/merge_run0_shards.py`

- 4 shards × 125 prompts, 4× `modal run`, merge jsonl → recompute metrics.
- Only if P0+P1 insufficient.

### P3 — vLLM rollouts (optional, post-pilot)

- New `pilot/train/vllm_rollout_engine.py`; same interface as `HFRolloutEngine`; add `vllm` to `modal_app.py` image.
- Sync weights to vLLM each GRPO step. Pin versions; gate on 50-prompt slice.

## Key code locations

- Run0 loop: `pilot/infra/execute.py` → `run0_proxy`
- HF rollouts: `pilot/train/rollout_engine.py`
- GRPO: `pilot/train/hf_grpo_train.py` → `_sample_rollouts`, `_build_step_groups`, `run_grpo_training`
- Modal: `pilot/infra/modal_app.py` (`gpu="A100-80GB"`)

## Done when

- Modal GPU util materially higher during decode-heavy phase OR `cost.json` `gpu_seconds` drops at same `debug_max_prompts`.
- Preflight + existing tests pass; frozen metrics unchanged on fixed seed slice.
