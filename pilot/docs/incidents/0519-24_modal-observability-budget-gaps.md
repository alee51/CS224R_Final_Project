# Modal observability + pilot budget enforcement gaps

**Documented:** 2026-05-19 23:00 PDT  
**Status:** Open (documentation / tooling; some fixes in other incidents)  
**Affects:** Operators monitoring spend and progress; cost attribution  
**Does not affect:** Remote training when launched correctly with `--detach`

---

## Summary

Modal’s **public CLI/API** exposes **billing cost** and **container list**, not the per-app **GPU allocation time series** shown in the web UI. The pilot has **no Weights & Biases** integration. YAML **`budget_cap_gpu_hours`** is not enforced in Python; **Run0** has no in-loop USD cap (GRPO checks `budget_cap_usd` between steps only).

This incident records **platform + repo gaps** called out in chat but easy to miss in runbooks.

---

## What Modal documents vs what we can pull

| Capability | Modal docs | Pilot CLI/API |
|------------|------------|----------------|
| Per-app / workspace **cost** over time | [`modal billing report`](https://modal.com/docs/reference/cli/billing), [`modal.billing.workspace_billing_report`](https://modal.com/docs/reference/modal.billing) | `python pilot/scripts/pull_modal_stats.py` |
| **GPU utilization %**, memory, power | [GPU Metrics guide](https://modal.com/docs/guide/gpu-metrics) | **No export** in CLI; use dashboard |
| Dashboard **“GPUs”** (allocation count) | UI only (per app) | **No API** found in client v1.x |
| Running containers | `modal container list --json` | Used in `pull_modal_stats.py` |
| App logs (all attempts) | `modal app logs <app-id>` | Operator monitoring |
| Workspace budget cap | [Billing guide](https://modal.com/docs/guide/billing) — Settings → Usage & Billing | Manual; separate from per-run YAML caps |

**Billing report rows** (verified 2026-05-19): `object_id`, `description`, `environment_name`, `interval_start`, `cost`, `tags` — **no GPU-second field**. We estimate GPU-hours as `cost / ~$2.50` per A100-80GB-hr (see `A100_80GB_USD_PER_HR` in `pull_modal_stats.py`).

Modal notes billing reports may be **Team/Enterprise** in docs; `modal billing report --json` worked on profile `chicken602` for this pilot.

---

## Pilot monitoring gaps (repo)

| Gap | Detail | Where tracked |
|-----|--------|----------------|
| **No wandb** | Metrics live in volume artifacts + `train.log` only | This file |
| **`budget_cap_gpu_hours`** in YAML | Not read in `execute.py` / trainers | `0519-13` related issues |
| **Run0 USD cap** | `preflight_lock.json` / YAML caps; **post-run** `record_cost` only | `pilot/infra/execute.py` `run0_proxy` |
| **GRPO USD cap** | Checked **between steps** in `hf_grpo_train.py` | `0519-22` budget table |
| **Mid-run pull** | `pull_run_artifacts.py` needs `metrics.json` | `0519-22` — use `modal volume get` |
| **Volume `train.log`** | Flat append per `run_id` across smokes + mains | `0519-13`, `0519-21` |
| **End-only `volume.commit()`** | Mid-run volume copies can lag | `0519-14`, `0519-21` |

---

## Recommended operator workflow

```bash
source .venv/bin/activate

# Cost + est. GPU-hours for matrix apps (edit MATRIX_APPS in script if IDs change)
python pilot/scripts/pull_modal_stats.py --start 2026-05-19 --hours 24 \
  --out pilot/artifacts/matrix_logs/modal_stats.json

# Live training signals
modal app logs <app-id> 2>&1 | grep -E "step [0-9]+/[0-9]+ (start|groups ready|done)|completed|preempt"

# Mid-run files
modal volume get --force pilot-artifacts <run_id>/raw_predictions.jsonl /tmp/preds.jsonl
```

For **GPU allocation spikes** on a single app, see `0519-23_per-app-gpu-chart-spike.md` (preemption overlap).

---

## Platform facts (from Modal docs, relevant to pilot)

1. **All Functions are preemptible by default**; likelihood grows with duration. [Preemption guide](https://modal.com/docs/guide/preemption)
2. **`nonpreemptible=True` is not supported for GPU Functions** — cannot buy non-preemptible A100 via decorator.
3. **Exit handlers** (`@modal.exit`) run on preemption with **~30s** grace — we do not use `@app.cls` hooks today; no cleanup before kill.
4. **Restart uses same input** — our single-shot `run_pilot_remote(config_json)` restarts the **whole** training loop unless we add checkpoints.
5. **Detached local client** exiting ≠ remote job done — see `0519-22` (“App completed” trap).

---

## Recommended fixes (ordered)

1. Keep **`pull_modal_stats.py`** updated with current matrix `app_id`s after each launch.
2. Enforce or remove **`budget_cap_gpu_hours`** from YAML (implement or delete to avoid false confidence).
3. Add Run0 **between-chunk** budget check if USD cap must be hard mid-run.
4. Implement durability contract in `0519-14` (per-step checkpoint + `commit()`).
5. Optional: `@modal.exit` on a cls-based wrapper to flush partial state within 30s on preemption.
6. Optional: wandb only if orchestrator requests; not in pilot scope today.

---

## References

- `pilot/scripts/pull_modal_stats.py`
- `pilot/artifacts/matrix_logs/20260519_modal_stats.json`
- `pilot/configs/shared_train.yaml` — `budget_cap_usd`, `budget_cap_gpu_hours`, `modal_price_per_sec`
- `pilot/preflight_lock.json` — per-run USD caps
- `pilot/train/hf_grpo_train.py` — budget break between steps
- `pilot/infra/execute.py` — Run0 post-run cost only
