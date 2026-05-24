# Issue: Run0 `completed 25/500` progress logs never fire with `rollout_micro_batch_size: 8`

**Documented:** 2026-05-19 13:53:23 PDT  
**Status:** Fixed in repo (redeploy required); see also `0519-21_run0-silent-rollout-progress-investigation.md`  
**Affects:** `run0_proxy` full runs (500 prompts) launched after micro-batch refactor  
**Does not affect:** Detached vs non-detached Modal launch (remote logging still works)

---

## Summary

Run0 progress logging uses `if done % 25 == 0`, but `done` advances in steps of **`rollout_micro_batch_size` (8)**. With `mb=8`, `done` takes values `8, 16, 24, 32, …` and **never equals 25, 50, 75, …**. The first log line is **`completed 200/500`**, not `completed 25/500`.

Operators watching for `25/500` (based on earlier runs or runbook expectations) will see **long silence + GPU activity** and may think the job is stuck or that logging was disabled by `modal run --detach`. Neither is true.

---

## Symptoms

- Modal GPU utilization is non-zero (e.g. steady ~30%) for hours.
- `modal app logs` and volume `run0_proxy/train.log` show no new lines after model load.
- No `completed 25/500 prompts` line appears even after **5×+** the time it took on an older run to reach that line (~14 minutes on 2026-05-19 ~07:49 UTC).
- Local spawn log (`pilot/artifacts/matrix_logs/*_run0_proxy.log`) never updates during training (expected — spawn metadata only).

---

## Root cause

### Old loop (logged at 25, 50, 75, …)

Per-prompt iteration (`pilot/infra/execute.py` before commit `468c99c`):

```python
for i, row in enumerate(prompts):
    ...
    if (i + 1) % 25 == 0:
        logger.info("completed %s/%s prompts", i + 1, len(prompts))
```

### Current loop (micro-batch chunks of 8)

After OOM refactor (`468c99c`, 2026-05-19 ~12:15 PDT):

```python
mb = max(1, int(config.get("rollout_micro_batch_size", ROLLOUT_MICRO_BATCH_SIZE)))  # default 8
for mb_start in range(0, len(prompts), mb):
    ...
    done = mb_start + len(chunk)
    if done % 25 == 0 or done == len(prompts):
        logger.info("completed %s/%s prompts", done, len(prompts))
```

### Which `done` values trigger a log (`mb=8`, 500 prompts)

| `done` | `done % 25 == 0`? | Logs? |
|--------|-------------------|-------|
| 8, 16, 24 | No | — |
| 32, 40, …, 192 | No | — |
| **200** | **Yes** | **`completed 200/500`** |
| 400 | Yes | `completed 400/500` |
| 500 | Yes (end) | `completed 500/500` |

**`25/500` never appears** with current config (`run0_proxy.yaml`: `rollout_micro_batch_size: 8`).

---

## Evidence

### Historical log that showed `25/500`

From shared volume `run0_proxy/train.log` (append-only across launches):

```text
2026-05-19 07:49:16,376 INFO Run0: 500 prompts, N=8, model=Qwen/Qwen3-1.7B-Base
...
2026-05-19 07:49:27,659 INFO ... custom_generate/generate.py "HTTP/1.1 404 Not Found"
2026-05-19 08:03:00,208 INFO completed 25/500 prompts
```

~13.5 minutes from model-ready to first progress line — **before** micro-batch Run0 loop landed.

### Detached launch (2026-05-19 ~12:01 PDT)

- App: `ap-Zk6zAIs9tWpGerJHufSud1`
- `Run0: 500 prompts` at 19:02:16 UTC; model ready ~19:02:31 UTC
- No `completed 25/500` (or any progress line) in `modal app logs` for **2+ hours** afterward (**29 lines total** in stream — see `0519-21`)
- Logging handlers unchanged: `FileHandler(train.log)` + `StreamHandler()` in `_setup_run_logging`
- **Evening check:** `logger.info("completed …")` had **not run** on the worker (not a Modal shipping bug). GPU active → rollout in progress or stuck **inside** `sample_rollouts_batch`, not disabled logging.

### Timing expectation if per-prompt rate unchanged

| Milestone | Old code | Current code (`mb=8`) |
|-----------|----------|------------------------|
| First progress log | ~14 min @ **25** prompts | ~**110 min** @ **200** prompts (8× prompts) |
| Full 500 (linear extrap.) | ~4–6 h | ~4–6 h (same total work) |

---

## What is NOT the cause

| Suspected cause | Verdict |
|-----------------|--------|
| `modal run --detach` disabled logging | **No** — remote handlers unchanged; HF/load lines appear |
| Concatenated `train.log` on volume | **Confusing but separate** — all runs append to `pilot-artifacts/run0_proxy/train.log`; does not block new lines |
| No intermediate logging | **By design** — only milestone logs; issue is **wrong milestones** for `mb=8` |
| Job not doing work | **Unlikely** if GPU non-zero; may still be slow/stuck, but absence of `25/500` alone is explained |

---

## Related issues

1. **Volume `train.log` is append-only** across all Run0/smoke launches on `pilot-artifacts` — debug smokes and full runs share one file; use `modal app logs <app-id>` for a single attempt.
2. **Run0 has no in-loop USD budget check** — only post-run `record_cost`; GRPO runs check `budget_cap_usd` between steps.
3. **`budget_cap_gpu_hours` in YAML is not enforced in Python.**

---

## Fix (applied in repo HEAD)

**`pilot/infra/execute.py`**

- Log **before** each chunk: `run0 chunk 1-8/500 (rollouts=N)`.
- Log **after** each chunk: `completed 8/500`, `16/500`, … (every micro-batch, not `done % 25`).

**`pilot/infra/modal_app.py`**

- `PYTHONUNBUFFERED=1` on the image.

**Still open (see `0519-21`):** per-run volume log path; mid-run `artifacts_volume.commit()`; remote `git_sha` on volume.

Requires **new Modal deploy** — running containers keep old code.

---

## Workaround (containers still on `468c99c` without redeploy)

Watch for:

```text
completed 200/500 prompts
```

not `25/500`. Monitor:

```bash
source .venv/bin/activate
modal app logs ap-Zk6zAIs9tWpGerJHufSud1 -f
```

Do **not** treat mid-run `modal volume get …/train.log` as live progress (volume `commit()` is end-of-run only).

---

## References

- `pilot/infra/execute.py` — `run0_proxy()`, lines ~165–205
- `pilot/configs/run0_proxy.yaml` — `rollout_micro_batch_size: 8`
- Introduced: git commit `468c99c` ("fixed OOM issues with pilot runs")
- Launch manifest: `pilot/artifacts/matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md`
