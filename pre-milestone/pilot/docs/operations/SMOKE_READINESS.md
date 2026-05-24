# Smoke readiness — Stage 1 gate

Operator checklist before matrix. Spec: `PILOT_REDESIGN.md` §6.

**Orchestrator status (2026-05-20):** Local preflight passes. Image build uses prebuilt `flash-attn` wheel (`modal_app.py`). A detached smoke job may already be running — check `modal app list` for `cs224r-pilot` with 1 task. Qwen3 does not accept batched `generator=`; rollout falls back to per-prompt decode (slower, B1 parity criterion 9 may need separate waiver).

## Local (no GPU)

```bash
source .venv/bin/activate
python pilot/scripts/smoke_preflight.py
python -c "from pilot.infra.modal_launch import launch_run; launch_run('smoke', dry_run=True)"
```

## Launch (detached)

```bash
./pilot/scripts/modal_run_pilot.sh --run-id smoke
modal app list
modal app logs <app-id>
```

First launch after image changes may take **15–45+ min** while Modal builds the image (torch + flash-attn wheel). Later launches reuse the cached image.

## Preempt test (criterion 4)

1. Wait until step 2 rollout is in progress (logs / `step_diagnostics.jsonl`).
2. `modal app stop -y <app-id>`
3. Re-run: `./pilot/scripts/modal_run_pilot.sh --run-id smoke`
4. Verify resume at step 2, `raw_predictions.jsonl` has 768 lines after 3 steps.

## Pull + verify

```bash
python pilot/scripts/pull_run_artifacts.py --run-id smoke --local-dir pilot/artifacts/smoke/<UTC-timestamp>
```

```bash
python pilot/scripts/smoke_verify_artifacts.py pilot/artifacts/smoke/<UTC-timestamp>
wandb sync pilot/artifacts/smoke/<UTC-timestamp>/wandb
```

## B6 eval (optional, criterion 10)

After smoke training completes, 4-prompt eval with `max_new_tokens=1536` in logs — run separately (not in default smoke path).

## Pass → matrix

Three detached GRPO runs only (`run1_grpo`, `run2_inverse_freq`, `run3_f_grpo`). No `run0_proxy` in matrix — see `pilot/docs/decisions/20260519_skip_run0_stage1_redesign.md`.

```bash
./pilot/scripts/launch_pilot_matrix.sh --dry-run
./pilot/scripts/launch_pilot_matrix.sh
```
