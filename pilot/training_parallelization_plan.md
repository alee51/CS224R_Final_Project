# Training Parallelization Plan (Current-State Grounded)

Last updated: 2026-05-19  
Scope: docs-only execution plan for improving GPU utilization without changing scientific comparability.

## 1) Current state snapshot

### Across independent runs (already parallelizable today)
- `pilot/scripts/launch_pilot_matrix.sh` launches `run1_grpo`, `run1b_grpo`, `run2_inverse_freq`, `run3_f_grpo` as four independent `modal run` jobs in parallel by default.
- `pilot/infra/modal_app.py` also supports `--run-ids`, but this path is explicitly sequential in one process (`main()` loops each run id).
- `run0_proxy` is intentionally outside the matrix and launched separately.

### Within a single run (already parallelized today)
- Rollout API supports prompt micro-batching via `batch_generate_rollouts()` in `pilot/train/rollout_engine.py`.
- GRPO rollout construction calls `_sample_rollouts_batch()` from `_build_step_groups()` in `pilot/train/hf_grpo_train.py`.
- Completion logprobs are already batched in `_batched_scalar_mean_completion_logprobs()` and `_batched_mean_completion_logprobs()` (micro-batched forwards, not per-sample forwards).
- Run0 now chunks prompts in `run0_proxy()` (`pilot/infra/execute.py`) and calls `engine.sample_rollouts_batch(...)`.

### Where execution is still effectively serial
- In both run0 and GRPO, seeds are set per prompt (`seed + i`), and config default is `allow_seeded_prompt_batching: false`.
- Under that default, `batch_generate_rollouts()` falls back to per-prompt `model.generate(...)` for mixed seeds (serial inner loop, even though prompts are chunked).
- `run_training_with_eval()` is train-then-eval in sequence for a run; no overlap.

## 2) Gap analysis: likely GPU utilization bottlenecks

1. **Decode remains mostly serial under strict seed semantics**
   - The highest-cost part (`generate`) often runs one prompt at a time because seeds differ per row and seed-batching is disabled.
2. **Prompt-length variance causes padding waste when batching is enabled**
   - If seeded batching is turned on, uneven prompt lengths can reduce effective tokens/sec.
3. **Training+eval lifecycle has idle opportunities**
   - Each run does train then tier-1 eval serially; no scheduling optimization yet.
4. **Operational retries are still manual**
   - Overnight failures require manual diagnosis/relaunch; no minimal 2am playbook was previously codified.

## 3) Non-negotiable scientific guardrails

Do not relax these guardrails while pursuing throughput:
- Keep `pilot/preflight_lock.json` fixed for data hashes, frozen metrics, gate thresholds, and bootstrap settings.
- Do not change canonical pilot seeds unless explicitly running a labeled perf-only experiment branch.
- Keep run IDs/objectives intact (`run1*`, `run2_inverse_freq`, `run3_f_grpo`) so comparisons remain apples-to-apples.
- Do not change artifact schema (`metrics*.json`, `raw_predictions.jsonl`, run directory layout).
- Any throughput experiment that changes RNG behavior must be labeled and never overwrite baseline `latest` comparisons silently.

## 4) Tiered implementation plan

Priority convention:
- **P0**: emergency fixes for tonight (fast rollback, low blast radius)
- **P1**: next-day low-risk throughput gains
- **P2**: medium-risk, higher-impact changes
- **P3**: post-pilot architecture

### P0-1: Add explicit serial-fallback telemetry
- **Objective:** Make hidden serial decode visible in logs before touching behavior.
- **Files/functions:**  
  - `pilot/train/rollout_engine.py` -> `batch_generate_rollouts()`  
  - `pilot/infra/execute.py` -> `run0_proxy()` progress logs
- **Change:** Log per chunk whether path was batched, seeded-batched, or per-prompt fallback; include chunk size and `allow_seeded_prompt_batching`.
- **Acceptance criteria:**  
  - `train.log` clearly reports decode mode distribution over a run.  
  - No metric drift on fixed-seed smoke slice.
- **Rollback:** Remove added logging lines only.

### P0-2: Codify emergency debug/restart wrappers
- **Objective:** Ensure failed overnight runs can be restarted in <10 minutes with consistent commands.
- **Files/functions:**  
  - `pilot/scripts/launch_pilot_matrix.sh`  
  - `pilot/infra/modal_app.py` local entrypoint usage block/docstring
- **Change:** Add documented restart modes (single failed run, debug subset) without modifying training math.
- **Acceptance criteria:**  
  - One-command relaunch exists for a single failed run id.  
  - Debug command path (`--debug-max-prompts`) documented and tested manually.
- **Rollback:** Revert script/doc-only edits.

### P1-1: Safe seeded prompt batching trial flag
- **Objective:** Use existing code path to increase decode parallelism with explicit opt-in.
- **Files/functions:**  
  - `pilot/configs/run*.yaml` (opt-in on selected run only)  
  - `pilot/train/rollout_engine.py` -> `batch_generate_rollouts()` seeded batching branch
- **Change:** Enable `allow_seeded_prompt_batching: true` for one labeled trial run; keep baseline configs unchanged.
- **Acceptance criteria:**  
  - Throughput increases on same prompt slice (lower wall-clock or lower `gpu_seconds`).  
  - Gate metrics remain within `preflight_lock.json` tolerances versus baseline.
- **Rollback:** Set flag back to `false`; rerun baseline config.

### P1-2: Tune micro-batch sizes using bounded sweep
- **Objective:** Find best stable `rollout_micro_batch_size` and `completion_logprob_micro_batch_size` for A100-80GB.
- **Files/functions:**  
  - `pilot/configs/run3_f_grpo.yaml` (or dedicated perf config)  
  - `pilot/train/hf_grpo_train.py` (`_build_step_groups`, `HFPolicyModel`)
- **Change:** Small sweep (e.g., 8/12/16 rollout MB; 16/24/32 logprob MB) on fixed debug slice.
- **Acceptance criteria:**  
  - No OOM/retry storms.  
  - Best config shows measurable wall-clock gain and stable metrics on fixed slice.
- **Rollback:** Restore previous micro-batch values.

### P2-1: Prompt bucketing by length before generate
- **Objective:** Reduce padding overhead when true batching is active.
- **Files/functions:**  
  - `pilot/train/rollout_engine.py` -> `batch_generate_rollouts()` preprocessing
- **Change:** Bucket/sort chunk prompts by tokenized length, then restore original order after decode.
- **Acceptance criteria:**  
  - Higher tokens/sec or lower chunk latency when batching path is used.  
  - Output ordering and artifact semantics unchanged.
- **Rollback:** Remove bucketing and keep existing chunk order.

### P2-2: Optional overlap of train/eval orchestration
- **Objective:** Reduce end-to-end wall time for matrix completion.
- **Files/functions:**  
  - `pilot/infra/execute.py` -> `run_training_with_eval()`  
  - potentially `pilot/infra/modal_app.py` for split execution modes
- **Change:** Add optional mode to defer tier-1 eval to separate job so training GPUs start next runs sooner.
- **Acceptance criteria:**  
  - Same eval outputs produced, just scheduled differently.  
  - No missing artifacts in run `latest`.
- **Rollback:** Use existing train-then-eval path.

### P3-1: vLLM rollout backend (post-pilot)
- **Objective:** Achieve high-throughput multi-prompt decode with stronger serving primitives.
- **Files/functions:**  
  - new `pilot/train/vllm_rollout_engine.py` (interface-compatible)  
  - `pilot/infra/execute.py` / `pilot/infra/modal_app.py` backend selection  
  - image deps in `pilot/infra/modal_app.py`
- **Change:** Add vLLM backend behind explicit config switch; preserve HF as default.
- **Acceptance criteria:**  
  - 50-prompt gate slice parity on correctness metrics.  
  - Material throughput gain on same hardware/caps.
- **Rollback:** Switch config back to HF engine; keep vLLM code path disabled.

## 5) Smoke/validation matrix

Use this matrix before promoting any throughput tweak:

- **S0: Wiring smoke (5 prompts)**  
  - Command:  
    - `modal run pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 5`  
  - Check: artifacts pulled, no exceptions, `metrics.json` exists.

- **S1: Run0 parity smoke (50 prompts)**  
  - Command:  
    - `modal run pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 50`  
  - Check: `minority_correct_prompt_rate` not materially worse than baseline slice.

- **S2: GRPO short smoke (training run with small prompt cap)**  
  - Command:  
    - `modal run pilot/infra/modal_app.py --run-id run1_grpo --debug-max-prompts 64`  
  - Check: completes multiple steps, writes checkpoint + `metrics_train.json` + eval outputs.

- **S3: Matrix launch smoke**  
  - Command:  
    - `./pilot/scripts/launch_pilot_matrix.sh --dry-run`  
  - Check: budget caps print correctly; commands match intended run ids.

## 6) Launch decision gates

Promote a parallelization change only if all are true:
- No change to frozen data/metrics/gates in `preflight_lock.json`.
- No new artifact schema deltas.
- No gate-regression signal on pilot eval splits beyond locked tolerances.
- Throughput gain is real (lower wall-clock and/or lower `gpu_seconds`) on comparable slices.
- Rollback path verified (single config revert or single-file revert).

## 7) “If overnight fails at 2am” playbook

### Step A: Triage quickly (2-5 min)
```bash
cd /Users/nancybao/Desktop/dev/cs224r_finalproject
ls pilot/artifacts/matrix_logs
rg -n "FAILED|ERROR|Traceback|RuntimeError" pilot/artifacts/matrix_logs/*.log
```

### Step B: Identify which run(s) failed
```bash
for r in run1_grpo run1b_grpo run2_inverse_freq run3_f_grpo; do
  echo "=== $r ==="
  ls "pilot/artifacts/$r" || true
done
```

### Step C: Relaunch only failed run(s)
```bash
cd /Users/nancybao/Desktop/dev/cs224r_finalproject
modal run pilot/infra/modal_app.py --run-id run2_inverse_freq
# repeat per failed run id
```

### Step D: Fast diagnostic rerun (if failure repeats)
```bash
cd /Users/nancybao/Desktop/dev/cs224r_finalproject
modal run pilot/infra/modal_app.py --run-id run2_inverse_freq --debug-max-prompts 32
```

### Step E: Run0 proxy fallback check (if training failures look decode-related)
```bash
cd /Users/nancybao/Desktop/dev/cs224r_finalproject
modal run pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 50
```

### Step F: Resume full matrix when stable
```bash
cd /Users/nancybao/Desktop/dev/cs224r_finalproject
./pilot/scripts/launch_pilot_matrix.sh
```

## 8) Execution order recommendation

1. Tonight: do only **P0** items if failures occur (telemetry + restart hygiene).
2. Next day: run **P1** bounded experiments on fixed slices, keep baselines untouched.
3. After pilot stability: pursue **P2**, then evaluate **P3** only behind strict parity gates.

This ordering preserves scientific validity while still giving a practical path to recover overnight runs and improve utilization.
