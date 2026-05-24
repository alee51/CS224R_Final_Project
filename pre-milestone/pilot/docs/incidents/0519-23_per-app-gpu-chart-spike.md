# Per-app Modal GPU chart: steady 1.0, brief spike to 2.0

**Documented:** 2026-05-19 23:00 PDT  
**Status:** Closed (explained)  
**Affects:** Interpreting the **per-run** “GPUs” chart in the Modal app dashboard  
**Does not affect:** Training correctness, GPU allocation per container (`gpu="A100-80GB"`)

---

## Summary

Operators saw a **per-run** dashboard graph (not workspace-wide) sit at **1.0 × A100-80GB** for ~1h, then spike to **2.0 GPUs (mean)** around **01:51 PM PDT**, then return to **1.0**. This is **not** one job using two GPUs internally, and **not** evidence that five matrix runs share one GPU.

For a **single app**, **1.0** means one allocated GPU container is billing. **2.0** means two GPU containers for that **same app** overlapped in the chart’s time bucket—almost always a **preemption restart handoff** (old worker draining + new worker starting).

---

## Symptoms

- App-level “GPUs” metric flat at **1.0** during long GRPO step 1.
- Tooltip at spike: **“A100-80GB 2.0 GPUs (mean)”**.
- Confusion: “We launched five runs in parallel—shouldn’t the chart show 5?” → **Wrong scope** (this chart is **one app**, not the workspace).

---

## Root cause

### 1. Chart scope is **one Modal app**, not the workspace

Each matrix launch is `modal run --detach` → **one ephemeral app ID** → **one** `run_pilot_remote` invocation → normally **one** live GPU container.

Workspace total concurrent GPUs = sum across apps (up to **five** for the 2026-05-19 matrix). The per-app chart never shows that sum.

### 2. Code requests exactly one GPU per container

`pilot/infra/modal_app.py` sets `gpu="A100-80GB"` (no `:2`). Policy + reference models in GRPO share **one** `cuda:0` device (`pilot/train/hf_grpo_train.py`).

### 3. Spike on `run1_grpo` = preemption overlap

Modal [preemption](https://modal.com/docs/guide/preemption): workers can be reclaimed; the function **restarts on the same input**. During handoff, the dashboard may list **running + draining** containers under one app.

Evidence for `ap-CpcEIWjwiNMb8MvGCZFpAT`:

| Signal | Value |
|--------|--------|
| Log | `Runner interrupted due to worker preemption. Your Function will be restarted with the same input.` |
| New container | `ta-01KS10CZW6AVARRHJBW8XPYYTY` @ **13:56:25 PDT** (`modal container list --json`) |
| Training | `step 1/100 start` again @ **20:56:45 UTC** after `step 2/100 start` @ 20:41 UTC |
| Billing hour 20 UTC | ~**1.07** estimated GPU-hours (`pilot/artifacts/matrix_logs/20260519_modal_stats.json`) |

Only **run1_grpo** preempted in this matrix; run1b/run2/run3/run0 had no preemption line in logs.

### 4. Modal GPU **metrics** docs ≠ this chart

[GPU Metrics](https://modal.com/docs/guide/gpu-metrics) documents **utilization %**, **power**, **temperature**, and **memory used**—not a billed “GPU count” series. The dashboard **“GPUs”** line is **allocation / concurrent containers** for the scoped app, not FLOPS or utilization.

---

## What is NOT the cause

| Theory | Verdict |
|--------|---------|
| GRPO uses 2 GPUs (policy on GPU0, ref on GPU1) | **No** — both on same device |
| All five runs share one GPU | **No** — separate apps; parallel step-1 timelines on four GRPO runs |
| Workspace chart would show 1 GPU while 5 run | **Misread** — per-app chart shows **that app only** |
| Spike means pilots 1/2/3 all doubled GPUs | **No** — only run1 app showed second container; chart was per-run |

---

## Operator playbook

```bash
# Per-app containers (running only)
modal container list --app-id ap-CpcEIWjwiNMb8MvGCZFpAT --json

# Preemption in app log stream (all attempts)
modal app logs ap-CpcEIWjwiNMb8MvGCZFpAT 2>&1 | grep -i preempt

# Workspace spend / est. GPU-hours (not the dashboard GPU line)
python pilot/scripts/pull_modal_stats.py --start 2026-05-19 --hours 24 \
  --out pilot/artifacts/matrix_logs/modal_stats.json
```

**GPU Functions cannot be `nonpreemptible=True`** per Modal docs (only CPU/memory; 3× price). Long training must checkpoint/resume in application code — see `0519-14_main-run-preemption-no-resume.md`.

---

## References

- Modal: [Preemption](https://modal.com/docs/guide/preemption), [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
- `pilot/docs/incidents/0519-14_main-run-preemption-no-resume.md` — preds wipe, no resume
- `pilot/docs/incidents/0519-22_main-matrix-operator-notes.md` — containers vs dashboard
- `pilot/scripts/pull_modal_stats.py` — billing-backed est. GPU-hours
- `pilot/artifacts/matrix_logs/20260519_modal_stats.json`
