# GRPO Smoke + Reliability Debug History (May 19, 2026)

This document is a handoff log of the tests we ran while trying to make the overnight pilot (`run0`, `run1`, `run1b`, `run2`, `run3`) launch-ready.

It focuses on:
- execution reliability (detach/cancellation, artifact persistence/pulls),
- GRPO OOM behavior on A100-80GB,
- iterative config reductions and whether they were actually honored.

---

## 1) Scope and Ground Truth

- Primary objective: get all pilot runs to complete overnight with reliable artifact recovery.
- Main blocker: GRPO training runs (`run1`, `run1b`, `run2`, `run3`) repeatedly OOM in training step 1.
- Key confirmation from latest runs: aggressive reduction settings were applied in runtime logs for later attempts, but OOM still occurred.

Representative OOM trace pattern:
- `torch.OutOfMemoryError: CUDA out of memory... GPU 0 has a total capacity of 79.25 GiB ... Process 1 has ~79.23 GiB memory in use`.

---

## 2) Reliability Tests (Execution + Artifact Handling)

### R1. `run0_proxy` baseline smoke succeeded (5 prompts)
- Command context: `modal run pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 5`
- Evidence: terminal `5.txt` shows `Run0 done` and valid metrics output.
- What this proved: base inference path can complete and write artifacts remotely.

### R2. Volume mount path conflict
- Error: `cannot mount volume on non-empty path: "/root/pilot/artifacts"`
- Evidence: terminal `5.txt`.
- Outcome: path/mount handling in Modal app needed hardening.

### R3. Artifact pull overwrite failure
- Error: `modal volume get ... Output path ... already exists. Use --force`
- Evidence: terminal `5.txt`.
- Outcome: pull path handling was updated to support overwrite-safe pulls (and avoid local-path collision failures).

### R4. Final `run0_proxy` smoke passed end-to-end (including pull)
- Evidence: terminal `5.txt` shows successful run completion plus successful local artifact pull.
- What this proved: for `run0`, local+remote artifact path is now functioning in normal smoke flow.

### R5. Local entrypoint + detach cancellation behavior
- Commands (examples):
  - `modal run --detach -n "smoke-run1-grpo-mb8-detached" ...`
  - `modal run --detach -n "smoke-run1b-grpo-mb4-detached" ...`
- Evidence:
  - terminal `238330.txt`: `RemoteError: Function call was cancelled by user or a failure.`
  - terminal `974214.txt`: same cancellation class.
- What this proved: `--detach` with this local entrypoint pattern was not reliable for these smokes.

---

## 3) GRPO Functional Blocker Tests

### F1. Initial GRPO dispatch bug
- Failure: `NameError: name 'bootstrap_run_artifacts' is not defined`
- Evidence: terminal `23.txt`.
- Outcome: import/dispatch path in execution code was patched so GRPO runs could proceed far enough to hit real training behavior.

---

## 4) OOM Test Ledger (Chronological)

All tests below were run against Modal A100-80GB with smoke mode (`--debug-max-prompts 2`) for GRPO runs.

## A) Early GRPO/Objective smokes (run1/run2/run3)

### T1. `run1_grpo` verbose smoke
- Command label: `smoke-run1-grpo-verbose-logging`
- Evidence: terminal `936354.txt`
- Runtime log evidence:
  - `step 1/100 start: ... batch_prompts=32 ... max_new_tokens=2048`
  - `step 1/100 groups ready: prompts=32 completions=256`
- Failure: OOM (`Tried to allocate 274.00 MiB`)
- Conclusion: baseline GRPO config fails during/after first group build into train step.

### T2. `run2_inverse_freq` rerun smoke
- Command label: `smoke-run2-inversefreq-rerun`
- Evidence: terminal `112522.txt`
- Failure: OOM (`Tried to allocate 274.00 MiB`)
- Conclusion: not specific to `run1` objective; same memory regime appears in run2.

### T3. `run3_f_grpo` smoke
- Command: `modal run ... --run-id run3_f_grpo --debug-max-prompts 2`
- Evidence: terminal `479165.txt`
- Failure: OOM (`Tried to allocate 274.00 MiB`)
- Conclusion: same issue generalizes to run3 objective variant.

## B) `run1b_grpo` targeted memory-reduction sequence

### T4. `smoke-run1b-grpo-mb4-direct`
- Artifact dir: `pilot/artifacts/run1b_grpo/20260519T100045Z`
- Snapshot: `completion_logprob_micro_batch_size: 4`, `batch_prompts: 32`, `max_new_tokens: 2048`
- Evidence: terminal `194986.txt`
- Runtime:
  - `step 1/100 groups ready: prompts=32 completions=256`
- Failure: OOM (`Tried to allocate 40.00 MiB`)
- Result: reducing completion-logprob micro-batch from default to 4 was insufficient.

### T5. `smoke-run1b-grpo-mb2-direct`
- Artifact dir: `pilot/artifacts/run1b_grpo/20260519T102104Z`
- Snapshot: `completion_logprob_micro_batch_size: 2`, `batch_prompts: 32`, `max_new_tokens: 2048`
- Evidence: terminal `942428.txt`
- Runtime:
  - `step 1/100 groups ready: prompts=32 completions=256`
- Failure: OOM (`Tried to allocate 18.00 MiB`)
- Result: mb=2 alone still insufficient.

### T6. `smoke-run1b-grpo-mb2-max1536`
- Artifact dir: `pilot/artifacts/run1b_grpo/20260519T104548Z`
- Snapshot: `completion_logprob_micro_batch_size: 2`, `max_new_tokens: 1536`, `batch_prompts: 32`
- Evidence: snapshot + terminal `554664.txt`
- Runtime anomaly:
  - runtime still logged `max_new_tokens=2048` at `step 1/100 start`
- Failure: OOM (`Tried to allocate 18.00 MiB`)
- Result: this attempt suggested some overrides were not consistently taking effect at that stage.

### T7. Config precedence bug fix applied
- Code path updated so run-specific config overrides for
  - `rollouts_per_prompt`,
  - `batch_prompts`,
  - `max_new_tokens`
  are read from run config before shared defaults.
- Purpose: ensure later reduction attempts are genuinely tested.

### T8. `smoke-run1b-mb2-max1024-b16`
- Artifact dir: `pilot/artifacts/run1b_grpo/20260519T110958Z`
- Snapshot: `completion_logprob_micro_batch_size: 2`, `max_new_tokens: 1024`, `batch_prompts: 16`
- Evidence: terminal `284688.txt`
- Runtime:
  - `step 1/100 start: ... batch_prompts=16 ... max_new_tokens=1024`
  - `step 1/100 groups ready: prompts=16 completions=128`
- Failure: OOM (`Tried to allocate 12.00 MiB`)
- Result: reductions were honored but still OOM.

### T9. `smoke-run1b-mb2-max512-b8` (latest critical repro)
- Artifact dir: `pilot/artifacts/run1b_grpo/20260519T112004Z`
- Snapshot: `completion_logprob_micro_batch_size: 2`, `max_new_tokens: 512`, `batch_prompts: 8`
- Evidence: terminal `965518.txt`
- Runtime:
  - `step 1/100 start: ... batch_prompts=8 ... max_new_tokens=512`
  - `step 1/100 groups ready: prompts=8 completions=64`
  - stack trace shows failure in differentiable completion-logprob path:
    `trainer.train_step -> model.logprobs_for_rollouts -> _batched_mean_completion_logprobs -> _micro_batch_mean_completion_logprobs -> logits = model(...).logits`
- Failure: OOM (`Tried to allocate 16.00 MiB`, process ~79.23 GiB used)
- Artifact recovery behavior:
  - pull attempted successfully after failure,
  - expectedly missing `metrics.json`/`cost.json` because run did not complete.
- Result: confirms genuine memory ceiling issue even after aggressive knob reductions.

---

## 5) What We Learned from the Test Sequence

1. OOM is genuine, not just a launcher artifact.
- Multiple direct (non-detach) runs reach the same OOM class with near-full VRAM.

2. Early cancellation noise existed, but is not the root cause.
- Some earlier tests died with `Function call was cancelled...` under detach/local-entrypoint interactions.
- Later direct runs still OOM reproducibly.

3. Config reductions eventually were applied and visible in logs.
- Latest attempts clearly show runtime honoring reduced `batch_prompts` and `max_new_tokens`.
- OOM persisted anyway.

4. Failure point is consistent.
- Run advances through rollout/group-building.
- OOM occurs when entering differentiable logprob/train-step forward path.

5. Artifact reliability improved.
- Pull-on-failure path now recovers partial artifacts for postmortem.
- Missing final outputs (`metrics.json`, `cost.json`) on failure is expected because training does not finish.

---

## 6) Snapshot of Final Tested `run1b` Settings

Latest tested failing config (`20260519T112004Z/config.snapshot.yaml`):
- `batch_prompts: 8`
- `rollouts_per_prompt: 8`
- `max_new_tokens: 512`
- `completion_logprob_micro_batch_size: 2`
- `debug_max_prompts: 2`
- GPU: `A100-80GB`

Even here, run OOMs in step 1.

---

## 7) Useful Evidence Pointers

- Latest failing `run1b`:
  - `pilot/artifacts/run1b_grpo/20260519T112004Z/config.snapshot.yaml`
  - `pilot/artifacts/run1b_grpo/20260519T112004Z/train.log`
  - terminal trace: `965518.txt`

- Prior reduction attempts:
  - `pilot/artifacts/run1b_grpo/20260519T110958Z/config.snapshot.yaml`
  - `pilot/artifacts/run1b_grpo/20260519T104548Z/config.snapshot.yaml`
  - terminals: `284688.txt`, `554664.txt`, `942428.txt`, `194986.txt`

- Other objective smokes showing same memory regime:
  - terminals: `936354.txt` (`run1_grpo`), `112522.txt` (`run2_inverse_freq`), `479165.txt` (`run3_f_grpo`)

- Reliability/cancellation/path issues:
  - detach cancellation: `238330.txt`, `974214.txt`, `876531.txt`, `480359.txt`
  - run0 artifact/mount issues + successful retest: `5.txt`
  - initial NameError: `23.txt`

---

## 8) Practical Hand-off Notes for Next Agent

- Treat this as two separate tracks:
  1) reliability path (largely improved),
  2) GRPO train-step memory path (still blocking full overnight matrix).
- For memory work, focus specifically on differentiable completion-logprob implementation and train-step tensor lifetime.
- Do not assume further small YAML-only tweaks will fix the issue; they already hit very aggressive values and still OOM.

