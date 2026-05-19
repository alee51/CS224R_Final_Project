# Incident: Main-matrix GRPO lost ~1h/step on Modal preemption (no resume, preds wiped)

**Documented:** 2026-05-19 14:15 PDT  
**Status:** Open  
**Affects:** Detached main matrix (`run1_grpo` confirmed; any GRPO run can hit this)  
**Does not affect:** Runs that finish a step before preemption and are not restarted yet (step-1 preds on volume for run1b/run2/run3)

---

## Summary

The overnight main matrix launched five detached Modal jobs (~12:01 PDT). Training is **~100 minutes per GRPO step** (~26 min rollouts + ~73 min backward). The stack **does not resume** after Modal **worker preemption**: the function restarts from scratch, **`raw_predictions.jsonl` is truncated**, and **model weights are never checkpointed mid-run**. On `run1_grpo`, preemption at ~13:56 PDT after a completed step 1 (~99 GPU-min) forced a full redo of step 1 and **deleted** the 256 saved completions (~600 KB on volume). That is real money (~**$4+** per lost step at `modal_price_per_sec: 0.000694`) with no training benefit.

This is a **launch/resilience gap**, not a training bug. Parallel detached apps are fine; missing per-step durability and preemption handling is not.

---

## Symptoms

- Modal UI shows containers **draining** under an app while work continues or restarts.
- `modal app logs` for one app shows **two** `GRPO run1_grpo …` boot sequences and `step 1/100 start` twice.
- `run1_grpo/raw_predictions.jsonl` on volume drops to **0 bytes** after preemption while `train.log` still shows `step 1/100 done` from the first attempt.
- Operators believe “step 1 finished” from logs but volume preds are empty.
- `run0_proxy` shows **no** progress lines for hours despite GPU use (separate issue: `0519-13_progress-log-milestone-misfire.md`).

---

## Root cause

### 1. Modal worker preemption (platform)

Log line on `ap-CpcEIWjwiNMb8MvGCZFpAT` (`run1_grpo`):

```text
Runner interrupted due to worker preemption. Your Function will be restarted with the same input.
```

Modal reclaimed the GPU worker; the **same function invocation** is retried on a **new container**. Old container enters **draining** state in the dashboard.

### 2. No application-level resume

`run_pilot_remote` → `execute_run` → `run_grpo_training()` has **no** checkpoint load, step cursor, or idempotency. On restart the training loop starts at **step 0** again.

### 3. Predictions file explicitly wiped on every process start

`pilot/train/hf_grpo_train.py` (~865):

```python
pred_path = out_dir / "raw_predictions.jsonl"
pred_path.write_text("")
```

Any new Python process (including preemption retry) **deletes** prior step completions on the shared volume path `pilot-artifacts/<run_id>/raw_predictions.jsonl`.

Append-after-step (`_append_predictions`) only helps **completed** steps on a **single uninterrupted** process.

### 4. Model weights not saved between steps

Checkpoint `policy.save_pretrained(out_dir / "checkpoint")` runs only after the **full** training loop (or budget break), not per step. Preemption loses all in-GPU optimizer state.

### 5. Volume `commit()` only at function exit — **bad for long runs**

`pilot/infra/modal_app.py` calls `artifacts_volume.commit()` once in a `finally` block when `run_pilot_remote` **ends**. That is insufficient for ~100 min/step jobs:

- **Preemption recovery:** Uncommitted or process-local state may not survive worker replacement the way operators expect; we need explicit **`artifacts_volume.commit()` after each completed step** (and on a **~30 minute** wall-clock cadence during long rollout/train phases if a step can exceed that).
- **Manual inspection:** Operators must be able to `modal volume get` / `pull_run_artifacts.py` and see **durable** step outputs without waiting for the full 8–9 budget-limited steps to finish.

Mid-run `volume get` often works on an active container, but that is **not** a substitute for a defined durability contract. Combined with **no resume** and **pred wipe on restart**, end-only commit is a launch bug, not a minor optimization.

### 6. Run0 has no incremental artifact write

`run0_proxy` only calls `write_run0_artifacts()` **after all 500 prompts**. A preemption or crash loses the entire rollout phase.

---

## Evidence

### Launch manifest (2026-05-19T19:01:58Z)

| Run | App ID | Container ID (active @ 14:10 PDT) | Container start (PDT) |
|-----|--------|-----------------------------------|------------------------|
| run0_proxy | `ap-Zk6zAIs9tWpGerJHufSud1` | `ta-01KS0SVMY535DZG98J582C9X5E` | 12:02:06 (original) |
| run1_grpo | `ap-CpcEIWjwiNMb8MvGCZFpAT` | `ta-01KS10CZW6AVARRHJBW8XPYYTY` | **13:56:25 (post-preempt)** |
| run1b_grpo | `ap-EWhmIPbGpflmnM2IcrKp77` | `ta-01KS0SVMMXCBNKV9VAN58VGDM1` | 12:02:07 |
| run2_inverse_freq | `ap-aAYroxfDF3TuZY5NJ1pbOP` | `ta-01KS0SVMMSQDG1PZSXNE297SSF` | 12:02:05 |
| run3_f_grpo | `ap-MO0JD72gMTybU9Sv7VCSrn` | `ta-01KS0SVMHPDZVFNZMXEER9367B` | 12:02:05 |

See `pilot/artifacts/matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md`.

### Why the dashboard shows “multiple containers” per app

| Source of “extra” containers | What it is |
|------------------------------|------------|
| **Draining worker after preemption** | Old GPU box shutting down while Modal starts a new one for the **same** function call (`run1_grpo` only, this launch). |
| **`modal app logs` history** | One log stream per **app**, aggregating every container that ever served that app — not one line per live box. |
| **Shared `train.log` on volume** | **Append-only across all launches** for that `run_id` (smokes + main). Example: `run1b_grpo/train.log` has **9** `GRPO run1b_grpo` boot lines from T1–T9 smokes **plus** the 12:02 main run — looks like “many containers” but is **many past jobs**, same volume path. |
| **`modal container list`** | Only **currently running** containers. @ 20260519T211846Z: **exactly one per app** (five total). |

**This main matrix:** one detached spawn → one app ID → one active task. Only `run1_grpo` has had a **second** physical container (preemption @ 13:56 PDT). The other four apps are still on their original container from 12:02 PDT.

### run1_grpo timeline (first container, preempted)

| UTC | Event |
|-----|--------|
| 19:02:25 | `step 1/100 start` |
| 19:28:31 | `groups ready` (build_seconds=1565.7) |
| 20:41:43 | `step 1/100 done` (total_step_seconds=5958.0 ≈ 99.3 min) |
| 20:41:43 | `step 2/100 start` |
| ~20:56 | **Preemption** → new container |
| 20:56:45 | `step 1/100 start` again (redo) |

### Volume snapshot (@ 14:10 PDT)

| Run | `raw_predictions.jsonl` | Interpretation |
|-----|-------------------------|----------------|
| run1_grpo | 0 lines | Wiped by restart |
| run1b_grpo | 256 lines (~600 KB) | Step 1 safe |
| run2_inverse_freq | 256 lines | Step 1 safe |
| run3_f_grpo | 256 lines | Step 1 safe |
| run0_proxy | 40 lines | Stale from earlier short run, not current 500-prompt job |

### Cost order-of-magnitude (one lost GRPO step)

- GPU time: ~5958 s × $0.000694/s ≈ **$4.13** (config: `shared_train.yaml` `modal_price_per_sec`)
- Plus duplicate rollouts on retry (second ~26 min rollout phase before preemption hits again)

---

## What is NOT the cause

| Suspected cause | Verdict |
|-----------------|--------|
| Detached launch disabled persistence | **No** — volume writes work; failure is wipe-on-restart + no resume |
| `modal volume commit` never ran mid-run | **Partially wrong in earlier triage** — preds append per completed step in-process, but **end-only `commit()` is still bad** for preemption + mid-run pulls; loss on run1 was primarily **restart + `write_text("")`** |
| Multiple containers = duplicate billing for same run | **No** — one **active** container per app; draining container is exiting; only run1 got a replacement container |
| Wrong run_id on volume | **No** — paths are per `run_id`; shared `train.log` append across launches is confusing but separate |

---

## Impact on this launch

| Issue | Severity |
|-------|----------|
| run1 step 1 completions + backward pass discarded | **High** (~$4+ and ~1.7 h wall time) |
| Any future preemption on 4 other GRPO runs | **High** (same code path) |
| run0 500-prompt proxy with no incremental save | **High** if preempted before end |
| ~100 min/step × 100 steps vs $36 `budget_cap_usd` | **Plan** — only ~8–9 steps fit budget; preemption makes effective $/step worse |
| Misleading `train.log` after restart | **Medium** — shows old `step 1 done` while preds empty |

---

## Recommended fix (ordered)

### Durability contract (target)

| When | What to persist to `pilot-artifacts/<run_id>/` | `volume.commit()` |
|------|-----------------------------------------------|-------------------|
| **Each completed GRPO step** | Append preds, `training_state.json` (`steps_done`, timestamps), `checkpoint/step_N/` | **Yes** |
| **Every ~30 min wall-clock** during a single step (rollout or train phase) | Flush partial preds or `step_K_partial.jsonl`, update `train.log`, heartbeat in `training_state.json` | **Yes** |
| **Run0** | Append preds every micro-batch or every N prompts | **Yes** on same cadence |
| **Function exit** | Final metrics, cost, eval artifacts | **Yes** (keep) |

### Code changes

1. **Stop wiping preds on startup** — only create if missing, or use `run_id/<attempt-utc>/` per container generation.
2. **After each completed step:** `policy.save_pretrained(checkpoint/step_N/)` + `training_state.json` + `_append_predictions` + **`artifacts_volume.commit()`**.
3. **Wall-clock checkpoint hook** (~30 min): commit volume + optional lightweight heartbeat file even if step not done (so preemption during the 73 min train phase loses at most ~30 min, not the whole step).
4. **On startup:** read `training_state.json`; resume from `steps_done` and load latest `checkpoint/step_N/`; **do not** call `write_text("")` if resuming.
5. **Modal:** review [preemption docs](https://modal.com/docs/guide/preemption); consider reserved GPUs if preemption rate is high.
6. **Ops (today):** mid-run pull via `modal volume get` (see below) — does not require job to finish.

---

## Workaround (current run)

### Mid-run pull (no waiting for job end)

From repo root with venv active:

```bash
# Single file
.venv/bin/modal volume get --force pilot-artifacts run1b_grpo/train.log /tmp/train.log
.venv/bin/modal volume get --force pilot-artifacts run1b_grpo/raw_predictions.jsonl /tmp/preds.jsonl

# Or whole run prefix (creates staging/<run_id>/…)
.venv/bin/modal volume get --force pilot-artifacts run1b_grpo/ ./tmp-staging/
```

`pull_run_artifacts.py` expects **finished** runs (`metrics.json` required) — use **`modal volume get`** per file mid-run, or pull into `pilot/artifacts/<run_id>/<UTC>_midrun_pull/` manually.

**Pulled @ 20260519T211846Z** into `pilot/artifacts/<run_id>/20260519T211846Z_midrun_pull/` (agent run).

### Containers vs dashboard

- `modal container list` shows **only running** containers (one per app right now except run1’s replacement worker).
- Modal **dashboard** may list **draining + running** under one app after preemption — that is **two workers for one function retry**, not parallel duplicate training.
- `modal app logs <app-id>` is the union of **all attempts** on that app; use `modal container logs <ta-...>` for a single worker.

### Operator checks

- Treat run1 `train.log` “step 1 done” as **historical** until preds repopulate.
- New `Start Time` on same `App ID` in `modal container list --json` = preemption/restart.
- Draining container ≠ run dead; look for the newer container.

---

## References

- `pilot/infra/modal_app.py` — `run_pilot_remote`, `artifacts_volume.commit()`
- `pilot/train/hf_grpo_train.py` — `pred_path.write_text("")`, `_append_predictions`, checkpoint at end only
- `pilot/infra/execute.py` — `run0_proxy` end-only `write_run0_artifacts`
- `pilot/scripts/launch_pilot_matrix.sh` — one detached spawn per run → one app per run
- Related: `0519-13_progress-log-milestone-misfire.md` (run0 logging)
