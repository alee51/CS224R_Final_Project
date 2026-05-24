# Main matrix operator notes (2026-05-19 detached launch)

**Documented:** 2026-05-19 22:30 PDT  
**Status:** Reference (live launch in progress)  
**Related:** `0519-14_main-run-preemption-no-resume.md`, `0519-13_progress-log-milestone-misfire.md`

This file captures **operations and triage knowledge** from the first detached main matrix that is not fully spelled out in the runbook or earlier incidents. Use it when monitoring, pulling artifacts mid-run, or interpreting the Modal dashboard.

---

## Launch topology

| Run | Modal app ID | Spawned (PDT) | Config seed |
|-----|--------------|---------------|-------------|
| run0_proxy | `ap-Zk6zAIs9tWpGerJHufSud1` | ~12:01 | — |
| run1_grpo | `ap-CpcEIWjwiNMb8MvGCZFpAT` | ~12:01 | 42 |
| run1b_grpo | `ap-EWhmIPbGpflmnM2IcrKp77` | ~12:01 | 43 |
| run2_inverse_freq | `ap-aAYroxfDF3TuZY5NJ1pbOP` | ~12:01 | 42 |
| run3_f_grpo | `ap-MO0JD72gMTybU9Sv7VCSrn` | ~12:01 | 42 |

Manifest: `pilot/artifacts/matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md`

- **One** `modal run --detach` spawn → **one** ephemeral app → **one** `run_pilot_remote` function call → normally **one** live GPU container.
- All apps display the same name **`cs224r-pilot`** in the UI; distinguish runs by **app ID**, not name.
- GRPO matrix: `pilot/scripts/launch_pilot_matrix.sh` (run1–run3). Run0 launched separately.

---

## GRPO step timing (measured, main matrix)

Config: `batch_prompts=32`, `rollouts_per_prompt=8`, `max_new_tokens=2048` → **256 completions per step**.

| Phase | Typical wall time | Log line |
|-------|-------------------|----------|
| Rollouts / “build groups” | **~25–28 min** | `groups ready: … build_seconds=1500–1656` |
| Backward / “train” | **~73–77 min** | `step N/100 done: … train_seconds=4392–4643` |
| **Total per step** | **~98–105 min (~1.7 h)** | `total_step_seconds=5893–6300` |

Per-run step 1 (UTC, main matrix):

| Run | build_seconds | train_seconds | total_step_seconds |
|-----|---------------|---------------|---------------------|
| run1_grpo | 1566 | 4392 | 5958 |
| run1b_grpo | 1656 | 4643 | 6300 |
| run2_inverse_freq | 1501 | 4392 | 5893 |
| run3_f_grpo | 1536 | 4394 | 5930 |

**Budget implication:** `budget_cap_usd: 36` at `modal_price_per_sec: 0.000694` ≈ **14.4 GPU-hours** ≈ **~8–9 steps** before `run_grpo_training` stops (checked between steps). **100 steps × ~1.7 h** is not reachable under cap; preemption wastes additional spend.

**Log lines to watch:**

```text
step N/100 start: building groups …
step N/100 groups ready: prompts=32 completions=256 build_seconds=…
step N/100 done: … train_seconds=… total_step_seconds=…
```

---

## What persists when (GRPO vs Run0)

### After each **completed** GRPO step (single uninterrupted process)

| Artifact | On volume during run? | Survives preemption? |
|----------|----------------------|----------------------|
| `raw_predictions.jsonl` (+256 lines) | Yes, append | **No** if process restarts (`write_text("")` on boot) |
| `train.log` (append) | Yes | **Yes** (history kept; can mislead) |
| `config.snapshot.yaml`, `git_sha.txt` | Yes (bootstrap) | Yes |
| Model weights | **No** (RAM only until end) | **No** |
| `checkpoint/`, `metrics_train.json` | End of training only | **No** mid-run |

**Mid-step:** Rollouts for step N are **not** in `raw_predictions.jsonl` until step N **fully** finishes (append runs after backward pass). If killed during the 73 min train phase, you keep step N−1 preds only.

### Run0 (`run0_proxy`)

| Artifact | When written |
|----------|----------------|
| `raw_predictions.jsonl`, `prompt_inputs.jsonl`, `metrics.json` | **End only** (`write_run0_artifacts`) |
| Progress logs | Milestones only; with `mb=8` first line is `completed 200/500` — see `0519-13` |

### Durability gap (fix target)

See durability contract in `0519-14`: per-step + ~30 min `volume.commit()`, checkpoint, resume, no pred wipe.

---

## Mid-run artifact pull

### Use `modal volume get` (not `pull_run_artifacts.py`)

`python pilot/scripts/pull_run_artifacts.py` requires **`metrics.json`** (`REQUIRED_ARTIFACTS`) and is for **finished** runs. Mid-run, pull from the volume directly.

**Single file:**

```bash
cd /path/to/cs224r_finalproject
source .venv/bin/activate
.venv/bin/modal volume get --force pilot-artifacts run1b_grpo/train.log ./local/train.log
.venv/bin/modal volume get --force pilot-artifacts run1b_grpo/raw_predictions.jsonl ./local/preds.jsonl
```

**Full run prefix into a timestamped folder (do not overwrite old pulls):**

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN=run1b_grpo
DEST="pilot/artifacts/${RUN}/${TS}_midrun_pull"
mkdir -p "$DEST"
TMP=$(mktemp -d)
.venv/bin/modal volume get --force pilot-artifacts "${RUN}/" "$TMP"
cp -R "$TMP/${RUN}/." "$DEST/"
rm -rf "$TMP"
wc -l "$DEST/train.log" "$DEST/raw_predictions.jsonl" 2>/dev/null
```

**Agent pulls on 2026-05-19 (kept separate on purpose):**

| Timestamp folder | Notes |
|------------------|--------|
| `pilot/artifacts/<run_id>/20260519T211846Z_midrun_pull/` | First mid-run snapshot |
| `pilot/artifacts/<run_id>/20260519T212216Z_midrun_pull/` | Second snapshot |

Volume paths are flat: `pilot-artifacts/<run_id>/…` — not per-attempt subdirs (another fix item).

---

## Containers: what “multiple containers” means

### CLI vs dashboard

| Tool | What it shows |
|------|----------------|
| `modal container list --json` | **Only currently running** containers |
| `modal app logs <app-id>` | **All attempts** on that app (every container ever) |
| `modal container logs <ta-id>` | **One** container only |
| Modal dashboard | Often **running + draining** during handoff |

### Main matrix (this launch)

| Run | Second training container? | Notes |
|-----|----------------------------|--------|
| **run1_grpo** | **Yes** | Preemption ~20:56 UTC; replacement started 13:56 PDT |
| run1b, run2, run3, run0 | **No** (one boot in app logs) | One live container each from 12:02 PDT |

### Why you see **short** second containers on *some* runs

| Cause | Applies to main matrix? |
|-------|-------------------------|
| **Draining worker after preemption** | **run1 only** — old box exits (~15 min into step 2); looks “short” in UI |
| **Smoke / debug apps (same app name)** | **Separate app IDs**, stopped early — e.g. `ap-U5KNv40LGvxP4ootXQP1hS` (run2 smoke), `ap-tl9o0I5MMUpfcyXBjMxXmr` (run1b smoke). See `pilot/artifacts/smoke_logs/20260519T182837Z/`. |
| **Shared `train.log`** | Many `GRPO run…` boots in **one file** = many **past jobs**, not many live containers |
| **Local “App completed” @ 12:02:12** | **Misleading** — see below |
| **Billing ~1.07 GPU in an hour** | Brief overlap possible; see `pilot/artifacts/matrix_logs/20260519_modal_stats.json` |
| **Per-app GPU chart spike 1→2** | Preemption handoff on **run1** only; see `0519-23_per-app-gpu-chart-spike.md` |

### Detached spawn trap: “App completed”

Every matrix spawn log contains (~12:02:12 PDT):

```text
[modal-client] Timed out waiting for final app logs.
✓ App completed. View run at https://modal.com/apps/.../ap-...
```

That is the **local `modal run --detach` client** exiting, **not** the remote GPU job finishing. Remote training continues on the same app. Do not treat this as a short-lived training container.

Modal also prints at spawn:

```text
Note that running a local entrypoint in detached mode only keeps the last triggered Modal function alive …
```

---

## Monitoring cheat sheet

```bash
.venv/bin/modal app list
.venv/bin/modal container list --json
.venv/bin/modal container list --app-id ap-EWhmIPbGpflmnM2IcrKp77 --json

# Progress (freshest step lines)
.venv/bin/modal app logs ap-EWhmIPbGpflmnM2IcrKp77 2>&1 | grep -E "step [0-9]+/[0-9]+ (start|groups ready|done)"

# Single worker
.venv/bin/modal container logs ta-01KS0SVMMXCBNKV9VAN58VGDM1 2>&1 | tail -20

# Preemption?
.venv/bin/modal app logs ap-CpcEIWjwiNMb8MvGCZFpAT 2>&1 | grep -i preempt
```

**run0:** Do not wait for `completed 25/500`; first milestone with `mb=8` is **`completed 200/500`** (~110 min after model ready if per-prompt rate unchanged). See `0519-13`.

---

## What is NOT wrong

| Observation | Verdict |
|-------------|---------|
| Five parallel apps = 5× duplicate training on one run | **No** — one run per app |
| `pull_run_artifacts.py` failed mid-run | **Expected** — use `volume get` |
| run1b `train.log` has 9+ boot lines | **Old smokes** on shared volume path |
| Draining container = job failed | **No** — often handoff; check for newer container |

---

## References

- `pilot/infra/modal_app.py` — `_spawn_only_one`, `run_pilot_remote`, end-only `commit()`
- `pilot/train/hf_grpo_train.py` — step loop, `_append_predictions`, `write_text("")`
- `pilot/infra/execute.py` — `run0_proxy`, end-only artifacts
- `pilot/infra/artifacts.py` — `REQUIRED_ARTIFACTS` (blocks mid-run pull script)
- `pilot/artifacts/matrix_logs/20260519_modal_stats.json` — hourly cost / est GPU count
- `pilot/scripts/pull_modal_stats.py` — regenerate billing snapshot
- `pilot/docs/incidents/0519-23_per-app-gpu-chart-spike.md` — per-run “GPUs” dashboard
- `pilot/docs/incidents/0519-24_modal-observability-budget-gaps.md` — CLI/API limits, budget YAML gaps
- `pilot/docs/incidents/0519-25_blocking-launch-client-abort.md` — blocking launch before matrix
- `pilot/configs/shared_train.yaml` — steps, batch, budget caps
