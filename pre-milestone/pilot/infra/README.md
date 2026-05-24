# Pilot infra — Modal launch and budget guards

**Start here for HF + weights:** [`MODEL_DATA_SETUP.md`](./MODEL_DATA_SETUP.md)

Orchestrator uses `pilot/scripts/launch_run.py` (dry-run / local) or **`pilot/scripts/modal_run_pilot.sh`** (GPU) with per-run USD caps from `pilot/preflight_lock.json`.

## Modal authentication

1. Activate venv and install deps:

   ```bash
   source .venv/bin/activate
   pip install -r pilot/requirements.txt
   ```

2. Create a Modal account at [https://modal.com](https://modal.com) and create an API token.

3. Authenticate (one-time per machine):

   ```bash
   modal token new
   ```

4. Verify:

   ```bash
   modal profile current
   ```

   Stage 1 uses each operator's **personal** profile. Secrets and volumes live on that profile. See
   `pilot/docs/operations/PERSONAL_WORKSPACE_COLLAB.md` and `nancy_explore/narrative/decisions.md`.

5. Secrets on **your** profile (once per machine/profile):

   ```bash
   modal secret create huggingface HF_TOKEN=...
   modal secret create wandb-api-key WANDB_API_KEY=...
   ```

## Launching runs (detached by default)

Use the wrapper — it always passes `modal run --detach` unless you pass **`--wait`**:

```bash
chmod +x pilot/scripts/modal_run_pilot.sh   # once

# Smoke gate (before matrix)
./pilot/scripts/modal_run_pilot.sh --run-id smoke

# Single training run
./pilot/scripts/modal_run_pilot.sh --run-id run1_grpo

# Quick debug on Modal (5 prompts)
./pilot/scripts/modal_run_pilot.sh --run-id run0_proxy --debug-max-prompts 5
```

Wait for `Spawned function call id:` — then safe to close the laptop.

**Interactive** (blocking client + auto-pull when using `--wait` on `modal_app.py`):

```bash
./pilot/scripts/modal_run_pilot.sh --run-id run0_proxy --wait
```

**Debug only** (attached client, no detach):

```bash
PILOT_MODAL_ATTACH=1 ./pilot/scripts/modal_run_pilot.sh --run-id run1_grpo --debug-max-prompts 2
```

Dry-run resolved config (no Modal call):

```bash
python -c "from pilot.infra.modal_launch import launch_run; launch_run('smoke', dry_run=True)"
```

After a detached run completes, pull artifacts:

```bash
python pilot/scripts/pull_run_artifacts.py --run-id run1_grpo \
  --local-dir pilot/artifacts/run1_grpo/<UTC-timestamp>
```

Do **not** use raw `modal run pilot/infra/modal_app.py` without `--detach` for long jobs (see `pilot/docs/incidents/0519-25_blocking-launch-client-abort.md`).

Modal outputs persist on Volume `pilot-artifacts`. Weights cache on Volume `hf-cache`.

Local GPU only:

```bash
python pilot/scripts/launch_run.py --run-id run0_proxy --no-modal
```

## Stage-1 matrix (three GRPO runs)

`run1_grpo` + `run2_inverse_freq` + `run3_f_grpo` (see `PILOT_REDESIGN.md`). Run 0 is waived — pre-redesign artifacts only (`pilot/docs/decisions/20260519_skip_run0_stage1_redesign.md`).

```bash
./pilot/scripts/launch_pilot_matrix.sh --dry-run
./pilot/scripts/launch_pilot_matrix.sh
```

Old matrix script (included `run1b_grpo`, excluded `run0`): `pilot/scripts/archive/launch_pilot_matrix_pre_redesign.sh`.

## Budget guards

Caps from `pilot/preflight_lock.json` (Stage 1): `smoke` $10; matrix runs $50 each (3-run nominal $150); `pilot_total` $200 ceiling. `run0_proxy` cap retained for optional manual runs, not in matrix launcher.

`launch_run(..., dry_run=True)` prints resolved yaml and cap for any `run_id` in `config_resolver.RUN_CONFIG_FILES`.
