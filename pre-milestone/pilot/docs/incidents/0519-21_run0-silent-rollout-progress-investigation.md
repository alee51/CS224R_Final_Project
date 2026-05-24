# Investigation: Run0 “logging completely broken” (GPU active, zero lines after model load)

**Documented:** 2026-05-19 21:30 PDT  
**Status:** Mitigated in repo (redeploy required); original overnight container unaffected  
**Related:** `0519-13_progress-log-milestone-misfire.md`, `0519-14_main-run-preemption-no-resume.md`  
**App (example):** `ap-Zk6zAIs9tWpGerJHufSud1` (`run0_proxy`, detached matrix launch 2026-05-19T19:01:58Z)

---

## Summary

During the overnight `run0_proxy` full run, operators saw **steady GPU utilization** (similar to an earlier run that logged `completed 25/500` at ~14 minutes) but **`modal app logs` froze after model load** for **2+ hours** with no `completed X/500` lines. The natural conclusion was that **Modal logging was disabled or broken**.

Investigation showed the opposite: **startup logging works** (same handlers emit `Run0: 500 prompts` and HuggingFace/httpx lines). After model ready, **no Python log statement runs until a full micro-batch chunk of rollouts returns** from `sample_rollouts_batch()`. Combined with the **`done % 25` milestone bug** (`0519-13`), the first `completed` line on deployed `468c99c` code is at **`200/500`**, not `25/500`. **Similar GPU % does not prove 200 prompts finished** — one slow chunk is only **8 prompts** of forward progress.

This is **silent rollout**, not a broken logging pipeline.

---

## Symptoms

- `modal app logs <app-id>` shows **~29 lines** ending at `custom_generate/generate.py` **404** (~model ready), then nothing for hours.
- `modal app logs` grep finds **no** `completed`, `error`, or `Run0 done`.
- GPU dashboard shows **non-zero, steady** utilization (e.g. ~30%), comparable to the 07:49 UTC run that reached `25/500`.
- `modal volume get …/run0_proxy/train.log` may show the current run **stopping mid–HF load** (e.g. 19:02:18) while the stream has later lines (19:02:31) — looks like “file stopped logging” but is **volume visibility**, not handler failure.
- Local `pilot/artifacts/matrix_logs/*_run0_proxy.log` never updates during training (**expected** — spawn metadata only).
- Operator belief: “**200 prompts must have passed**” because elapsed time ≫ old `25/500` cadence.

---

## Root cause (layered)

### Layer 1 — Milestone misfire (`0519-13`)

On `468c99c` code, progress logs used `if done % 25 == 0` while `done` advances by `rollout_micro_batch_size` (8). First milestone: **`completed 200/500`**, not `25/500`.

### Layer 2 — No logs inside `model.generate`

`run0_proxy()` only logs **before** and **after** `engine.sample_rollouts_batch()`:

```python
logger.info("run0 chunk …")          # only after fix in HEAD
texts_batch = engine.sample_rollouts_batch(...)  # silent inside
logger.info("completed %s/%s …")     # only when batch returns
```

`batch_generate_rollouts()` (`pilot/train/rollout_engine.py`) has **no** progress logging during generation. With `allow_seeded_prompt_batching: false` and per-prompt seeds (`seed + mb_start + j`), mixed seeds force the **sequential per-prompt `generate` path** (up to **8× `model.generate` per 8-prompt chunk**). GPU can be busy the entire time with **zero new log lines**.

### Layer 3 — Observability traps

| Trap | What happens |
|------|----------------|
| **`modal volume get` mid-run** | `artifacts_volume.commit()` runs **only at end** of `run_pilot_remote` (`modal_app.py` `finally`). Volume copy can lag or truncate vs in-container file. |
| **Flat `run0_proxy/train.log`** | Append-only across **all** smokes and full runs; old `completed 25/500` lines are from **earlier** attempts. |
| **`git_sha.txt` on local pull dir** | Written at **local bootstrap**; Modal ships code via `add_local_dir(pilot/)` from **working tree at deploy**, which can differ from recorded SHA. |
| **`modal app logs` without `-f`** | Misses **future** lines; for this incident the stream had only **29 entries total** — not truncation hiding older `completed` lines. |

---

## Evidence (live, 2026-05-19 evening)

### `modal app logs ap-Zk6zAIs9tWpGerJHufSud1`

- **29 lines** total (including weight-loading progress bars).
- Last line: **19:02:31 UTC** (`custom_generate/generate.py` 404).
- **No** `completed` anywhere in retained history after **~2h 10m** from model-ready (~21:12 UTC check).

**Implication:** `logger.info("completed …")` **never executed** on that worker (not “logging failed to ship a line that was emitted”).

### Volume `run0_proxy/train.log` (via `modal volume get`)

- Current attempt: `Run0: 500 prompts` at **19:02:16 UTC**.
- File on volume ended **19:02:18** (mid–HF HEAD) — **behind** Modal stream (19:02:31).
- **No** `completed` for the 19:02 run; historical **`completed 25/500` at 08:03** is from the **07:49 per-prompt** run.

### Baseline timing (07:49 UTC per-prompt run)

| Event | Time |
|-------|------|
| Model ready (`custom_generate` 404) | 07:49:27 UTC |
| `completed 25/500` | 08:03:00 UTC |
| **Δ** | **~13.5 min** (~32.5 s/prompt) |

Extrapolation for **`468c99c` milestone-only logging:** first line at **200 prompts** → **~108 min** if rate unchanged. At **~130 min** with still **zero** lines, either (a) throughput slower than baseline, (b) not yet at 200 prompts, or (c) stuck inside an early chunk.

### Deployed code vs `git_sha.txt`

- Local launch dir `20260519T190202Z/git_sha.txt` = **`b936ee88`** (per-prompt loop, logs at 25).
- Same dir `config.snapshot.yaml` already has **`rollout_micro_batch_size: 8`** (field added in **`468c99c`**, committed **12:15 PDT**).
- Matrix spawn **12:01 PDT** — working tree likely had micro-batch **code** before that commit landed; **`git_sha` understates** baked behavior.

**Do not infer loop semantics from `git_sha.txt` alone** — use `config.snapshot.yaml`, `git show`, and log cadence.

---

## What is NOT the cause

| Suspected cause | Verdict |
|-----------------|--------|
| `modal run --detach` killed logging | **No** — HF/httpx/`Run0:` lines appear |
| `logging.basicConfig` / `StreamHandler` broken | **No** — same root logger as httpx |
| Filter blocking `pilot.infra.execute` only | **No** — no `setLevel` on that logger; propagation default |
| `PYTHONUNBUFFERED` unset | Affects `print`/some C buffers; **`logging.StreamHandler` flushes per record** — minor factor |
| Concatenated `train.log` “blocking” writes | **No** — confuses readers; does not suppress emit |
| GPU idle / job dead | **Contradicted** by operator GPU observation — job likely in `generate` |
| “200 prompts done” from GPU time alone | **Unreliable** — see Layer 2; could be **one chunk (8 prompts)** for hours |

---

## Misread: “logging is completely failing”

| Observation | Correct interpretation |
|-------------|------------------------|
| No lines after model load | **No log sites reached** in the hot path, not disabled handlers |
| Same GPU % as 25/500 run | Utilization during **`generate`**; not comparable prompt throughput |
| 120+ min, no `25/500` | Expected on **`468c99c`** (watch **`200/500`**); on **`b936ee88`** would be **anomaly** |
| 90% sure 200 prompts passed | Without `completed 200/500` in Modal stream, **not proven** — only **25 chunk returns** needed for that line on `468c99c` |

---

## Fix applied (repo HEAD; requires new Modal deploy)

**`pilot/infra/execute.py` — `run0_proxy()`**

1. Log **before** each chunk: `run0 chunk 1-8/500 (rollouts=8)`.
2. Log **after** each chunk: `completed 8/500`, `16/500`, … (every **8** prompts, not only multiples of 25).

**`pilot/infra/modal_app.py`**

- `PYTHONUNBUFFERED=1` on the image env.

**Does not fix** a container already running — only a **relaunch** picks up `add_local_dir` changes.

---

## Operator playbook

### Live progress (preferred)

```bash
source .venv/bin/activate
modal app logs ap-Zk6zAIs9tWpGerJHufSud1 -f
```

After redeploy with fix, expect **`run0 chunk …`** before long generates and **`completed 8/500`** every ~few minutes (rate-dependent).

### Do not use as live progress

- `modal volume get pilot-artifacts run0_proxy/train.log` mid-run (stale until `commit()`).
- Local timestamped dir `pilot/artifacts/run0_proxy/20260519T*/` during run (bootstrap only; remote writes flat `…/run0_proxy/train.log`).
- `pilot/artifacts/matrix_logs/*_run0_proxy.log`.

### Stuck vs slow

| Signal | Likely |
|--------|--------|
| GPU active, redeployed code, `run0 chunk` lines advancing | Healthy |
| GPU active, **no** chunk lines after fix + ≫1h | Stuck in `generate` or old code still running |
| GPU active, old `468c99c`, no line by **~3h** after model-ready | Investigate hang; don’t wait for `25/500` |

### Inferring code version on a running job

```bash
# Milestone cadence in logs (if any lines appear):
#   25, 50, 75 …  → per-prompt loop (b936ee88-style)
#   200, 400, 500 only → 468c99c milestone bug
#   8, 16, 24 …     → fixed HEAD (per-chunk logging)

git show 468c99c:pilot/infra/execute.py   # deployed bug
git show b936ee88:pilot/infra/execute.py  # per-prompt loop
```

---

## Recommended follow-ups (not all done)

- [x] Per-chunk + pre-chunk logging in `run0_proxy` (HEAD).
- [x] `PYTHONUNBUFFERED=1` on Modal image.
- [ ] `artifacts_volume.commit()` every N prompts during Run0 (pairs with `0519-14`).
- [ ] Per-run volume path `run0_proxy/<UTC>/train.log` instead of flat append-only file.
- [ ] Remote `bootstrap_run_artifacts` at start of `run_pilot_remote` so volume gets matching `git_sha.txt`.
- [ ] Optional heartbeat inside `batch_generate_rollouts` for very long single-chunk runs.

---

## References

- `pilot/infra/execute.py` — `run0_proxy()`, `_setup_run_logging()`
- `pilot/train/rollout_engine.py` — `batch_generate_rollouts()` (sequential path ~175–197)
- `pilot/infra/modal_app.py` — `add_local_dir`, `artifacts_volume.commit()`, image env
- `pilot/infra/artifacts.py` — flat `artifact_dir(run_id)` vs timestamped local dirs
- `pilot/artifacts/matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md`
- `pilot/docs/incidents/0519-13_progress-log-milestone-misfire.md` — milestone math
- `pilot/docs/incidents/0519-14_main-run-preemption-no-resume.md` — volume commit / Run0 end-only artifacts
