# Pilot infra — Modal launch and budget guards

**Start here for HF + weights:** [`MODEL_DATA_SETUP.md`](./MODEL_DATA_SETUP.md)

Orchestrator uses `pilot/scripts/launch_run.py` to schedule Run0–3 with per-run USD caps from `pilot/preflight_lock.json`.

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

   Or set environment variables from the Modal dashboard:

   ```bash
   export MODAL_TOKEN_ID=...
   export MODAL_TOKEN_SECRET=...
   ```

4. Verify:

   ```bash
   modal profile current
   ```

## Launching runs

Dry-run (resolved config + budget cap, no files written):

```bash
python pilot/scripts/launch_run.py --run-id run1_grpo --dry-run
```

### Run0 on Modal GPU (recommended)

From repo root, after `modal setup` and `modal secret create huggingface HF_TOKEN=...`:

```bash
# Full Run0: 500 prompts × 8 rollouts, ~$4 cap
modal run pilot/infra/modal_app.py --run-id run0_proxy

# Quick smoke (5 prompts) before full run:
modal run pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 5
```

Or via launch wrapper (same Modal job):

```bash
python pilot/scripts/launch_run.py --run-id run0_proxy
```

Writes `pilot/artifacts/run0_proxy/metrics.json` with `minority_correct_prompt_rate`.

Modal runs persist outputs on Volume `pilot-artifacts` and auto-pull to a new
timestamped folder `pilot/artifacts/<run_id>/<UTC-timestamp>/` each time;
`pilot/artifacts/<run_id>/latest` points at the newest run. Gate scripts use
`resolve_latest_run_dir()`. Weights cache on Volume `hf-cache`.

Local GPU only (CUDA machine):

```bash
python pilot/scripts/launch_run.py --run-id run0_proxy --no-modal
```

GPU training requires registering a trainer before launch:

```python
from pathlib import Path

from pilot.infra.modal_launch import register_train_fn

def train_fn(config: dict) -> Path:
    artifact_dir = Path(f"pilot/artifacts/{config['run_id']}")
    # ... vLLM / GRPO loop ...
    return artifact_dir

register_train_fn(train_fn)
```

## Overnight matrix (Run1–Run3)

Runs **run1_grpo**, **run1b_grpo**, **run2_inverse_freq**, and **run3_f_grpo** are independent (different objectives/seeds/configs). They do not share training state and **can run in parallel** on separate Modal GPUs to finish overnight.

**run0_proxy is separate** — proxy rollouts only, not part of the matrix. Launch it before or after the matrix as needed:

```bash
modal run pilot/infra/modal_app.py --run-id run0_proxy
```

### Shell launcher (recommended)

From repo root with venv active (script auto-activates `.venv` if present):

```bash
chmod +x pilot/scripts/launch_pilot_matrix.sh   # once
./pilot/scripts/launch_pilot_matrix.sh --dry-run   # caps + commands only
./pilot/scripts/launch_pilot_matrix.sh             # 4 parallel Modal jobs
./pilot/scripts/launch_pilot_matrix.sh --sequential  # one process, all run ids
```

The script checks venv + `modal` CLI, prints per-run caps from `pilot/preflight_lock.json`, and writes parallel logs under `pilot/artifacts/matrix_logs/`.

### Manual / single process

One run:

```bash
modal run pilot/infra/modal_app.py --run-id run1_grpo
```

Sequential matrix in one local Modal entrypoint (remote jobs still one GPU each, back-to-back):

```bash
modal run pilot/infra/modal_app.py \
  --run-ids run1_grpo,run1b_grpo,run2_inverse_freq,run3_f_grpo
```

Parallel matrix (four terminals or background `&`):

```bash
modal run pilot/infra/modal_app.py --run-id run1_grpo &
modal run pilot/infra/modal_app.py --run-id run1b_grpo &
modal run pilot/infra/modal_app.py --run-id run2_inverse_freq &
modal run pilot/infra/modal_app.py --run-id run3_f_grpo &
wait
```

Each invocation bootstraps a new **timestamped** local dir `pilot/artifacts/<run_id>/<UTC-timestamp>/`, runs on Modal, pulls from Volume `pilot-artifacts`, and updates `pilot/artifacts/<run_id>/latest`.

## Budget guards

| Run ID | Cap (USD) | Hard abort (1.5×) |
|--------|-----------|-------------------|
| `run0_proxy` | 4 | 6 |
| `run1_grpo`, `run1b_grpo`, `run2_inverse_freq`, `run3_f_grpo` | 12 each | 18 |

Pricing default: A100-80GB @ `$0.000694`/sec (`shared_train.yaml`). `budget_guard.check_cost` raises if estimated spend exceeds 1.5× the run cap.

## Artifact layout

Each run directory under `pilot/artifacts/<run_id>/` must contain:

- `config.snapshot.yaml` — written at launch start
- `git_sha.txt` — written at launch start
- `raw_predictions.jsonl`, `metrics.json`, `train.log`, `cost.json` — trainer/eval
