# Run0 first full launch aborted by blocking client (no detach)

**Documented:** 2026-05-19 23:05 PDT  
**Status:** Closed (superseded by detached matrix launch)  
**Affects:** Long runs started with blocking `modal run` / `.remote()` without `--detach`  
**Does not affect:** Matrix launch after 2026-05-19T19:01:58Z (`modal run --detach` + spawn)

---

## Summary

The first **full** `run0_proxy` on Modal (~18:13 PDT) used a **blocking** launch path tied to the local machine. After ~35 minutes the operator stopped the client; remote work did not produce rollout artifacts. A **relaunch** without fully stopping the first remote invocation **cancelled** the in-flight function. The fix adopted for production is **`modal run --detach`** + default **spawn** in `modal_app.py` (see `pilot/infra/README.md`).

---

## Symptoms

- App `ap-PQzPrP3ZGZLGvHxOSR3znT` (example first attempt).
- Local dir `pilot/artifacts/run0_proxy/20260519T181343Z/` contains **bootstrap only** (config snapshot, `git_sha.txt`) — **no** `metrics.json`, no meaningful `raw_predictions.jsonl`.
- No `completed 25/500` (or any progress) in logs for that attempt.
- Second launch ~18:49 PDT reported cancellation of the first remote input.

---

## Root cause

### 1. Blocking Modal client

`modal run pilot/infra/modal_app.py --run-id run0_proxy` **without** `--detach` keeps the local entrypoint connected. Laptop sleep, terminal close, or Ctrl+C can **abort the client** while Modal behavior for in-flight work is easy to misread.

`launch_run.py` uses blocking `.remote()` and is explicitly **not** safe for unattended runs (`pilot/infra/README.md`).

### 2. Spawn vs wait modes

Current production path (`_spawn_only_one`):

- Requires `modal run --detach` so the ephemeral app stays up after local exit.
- Uses `run_pilot_remote.spawn(...)` — remote work is **not** tied to laptop.

Interactive path (`--wait`) uses `.remote()` and auto-pull; laptop must stay on.

### 3. Relaunch without `modal app stop`

Starting a new invocation while the old app is still active can yield **`Function call was cancelled by user or a failure`** on the earlier call (same class of failure as smoke detach tests in `0519-11` §R5).

---

## What is NOT the cause

| Theory | Verdict |
|--------|---------|
| Run0 training logic failed immediately | **Unproven** — no rollout artifacts to inspect |
| Detached logging disabled | **N/A** — this attempt was not detached |
| Volume mount broken | **No** — later detached runs wrote to volume |

---

## Workaround / fix (adopted)

```bash
# Production — survives laptop off
modal run --detach pilot/infra/modal_app.py --run-id run0_proxy

# Overnight matrix
./pilot/scripts/launch_pilot_matrix.sh
```

Before relaunching a stuck run:

```bash
modal app list
modal app stop -y <old-app-id>
```

Manifest for successful detached matrix: `pilot/artifacts/matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md` (app `ap-Zk6zAIs9tWpGerJHufSud1` for run0).

---

## References

- `pilot/infra/modal_app.py` — `_spawn_only_one`, `--wait`, docstring
- `pilot/infra/README.md` — detach vs interactive
- `pilot/docs/incidents/0519-11_grpo-smoke-debug-history.md` — §R5 detach cancellation
- `pilot/docs/incidents/0519-22_main-matrix-operator-notes.md` — “App completed” vs remote still running
