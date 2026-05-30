# CS224R — agent instructions

Minority-voting / set-based RL on VeRL (maxrl fork). Active work is in `main-verl/`; `main/` is a frozen custom-trainer archive.

## Repo layout

| Path | Role |
|------|------|
| `main-verl/` | Active VeRL training, Modal infra, probes, configs |
| `main-verl/docs/STATUS.md` | Stage checklist and current blocker |
| `main-verl/docs/build/stage-*-agent-plan.md` | Orchestrator dispatch: executor + audit sections |
| `main-verl/docs/build/stage-*-log.md` | Run records — append verdicts here |
| `main-verl/docs/verl_migration_plan.md` | Stage gates, kill criteria, credit allocation |
| `main-verl/docs/reward-decision.md` | **LOCKED** MathReward stack |
| `main/` | Frozen 1.7B trainer + eval history — read only, no new training features |
| `pre-milestone/` | Archive — do not edit |

## Cursor Cloud specific instructions

Cloud agents run on a CPU VM with no local GPU. Verification splits into **local pytest** (cheap) and **Modal smokes** (B200, costs money). Only launch Modal when the task explicitly requires it or a stage plan section says to run S*.5.

### Environment setup

Dependencies install via `.cursor/environment.json` into repo-root `.venv`:

```bash
source .venv/bin/activate
```

Modal auth uses Cursor secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (from the Modal workspace that owns profile `chicken602`). HF and W&B credentials for GPU jobs live as **Modal secrets** (`HUGGINGFACE`, `WANDB_API_KEY`) on that profile — not in this repo.

Optional: `export MODAL_PROFILE=chicken602` if your token config uses named profiles.

### Local tests (no GPU)

From repo root with venv active:

```bash
pytest main/tests/ -v
```

When touching shared Polaris/preprocess logic:

```bash
PYTHONPATH=main-verl:main python3 -m data.preprocess_polaris_verl --out-dir main-verl/data
```

Parquet outputs are committed under `main-verl/data/`; source manifest is `main/data/polaris_train.jsonl`.

### Modal smokes (GPU on Modal)

Always run from **repository root**. Image pins and maxrl SHA: `main-verl/infra/modal_image.py`.

Stage 1 hello (import verl + one rollout):

```bash
export CS224R_APP_NAME=cs224r-verl-stage01
./main-verl/scripts/launch_hello_verl.sh
```

Stage 2 GRPO smoke (50-step; current blocker: `max_prompt_length` — see `main-verl/docs/build/stage-02-log.md`):

```bash
export CS224R_APP_NAME=cs224r-verl-stage02
./main-verl/scripts/launch_grpo_smoke.sh
```

Upload parquet to Modal artifacts volume:

```bash
PYTHONPATH=main-verl:main python3 -m data.preprocess_polaris_verl --out-dir main-verl/data --upload
```

### Scope rules

- Do **not** edit `pre-milestone/` or add training features to `main/`.
- Do **not** use upstream `pip install verl` or `algorithm.adv_estimator=maxrl` unless a stage plan says otherwise.
- Do **not** wrap `main/train/reward.py` for VeRL parity — use patched MathReward per `reward-decision.md`.
- Image rebuild budget: ≤3 per stage; document rebuild count in the stage log.
- Changing `main-verl/infra/modal_image.py` or anything under `main-verl/` snapshotted into the image requires a **Modal image rebuild** on next run.

### Orchestrator workflow

For multi-step bring-up, follow the section DAG in `main-verl/docs/build/stage-*-agent-plan.md`:

1. Dispatch executor on one section's **Executor brief**.
2. Dispatch auditor on the same section's **Audit brief**.
3. Append results to the matching `stage-*-log.md`.
4. Do not advance until audit is **PASS** or **PASS WITH NOTES** (non-blocking notes only).

Current priority: Stage 2 S2.5 — fix `max_prompt_length` in `main-verl/configs/grpo_smoke_1p7b.yaml` and re-run GRPO smoke.
