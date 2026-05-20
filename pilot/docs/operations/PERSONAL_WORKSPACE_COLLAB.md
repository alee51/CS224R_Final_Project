# Personal Modal workspace — operator cheat sheet

**Decision:** Stage 1 pilot runs on each operator's **personal** Modal profile, not a shared team workspace. See `nancy_explore/decisions.md` (2026-05-19).

---

## Before launch

```bash
source .venv/bin/activate
modal profile current          # expect your GitHub-username profile (e.g. chicken602)
modal secret list              # huggingface + wandb-api-key on *this* profile
```

One-time per machine/profile: `modal token new`, `modal secret create huggingface HF_TOKEN=...`, `modal secret create wandb-api-key WANDB_API_KEY=...`.

---

## Detached run

```bash
modal run --detach pilot/infra/modal_app.py --run-id run1_grpo
# wait for "Spawned function call id:" — safe to close laptop
```

Matrix (four parallel): `./pilot/scripts/launch_pilot_matrix.sh` or repeat `--detach` per `run_id`.

---

## Pull artifacts (your workspace volume → local repo)

```bash
python pilot/scripts/pull_run_artifacts.py \
  --run-id run1_grpo \
  --local-dir pilot/artifacts/run1_grpo/<UTC-timestamp>
```

Use the timestamp dir printed at launch. Mid-run pull: see `../incidents/0519-22_main-matrix-operator-notes.md`.

---

## wandb (cross-team metrics)

Runs log to project **`cs224r-minority-voting`**. If Modal blocks outbound wandb, artifacts include offline runs under the run dir; sync after pull:

```bash
wandb sync pilot/artifacts/run1_grpo/<UTC-timestamp>/wandb
```

Use run names that include **operator + run_id** so dashboards are distinguishable.

---

## Sharing checkpoints with teammates

Modal volumes are **not** shared across profiles. After pull, publish via:

- HuggingFace Hub (preferred for weights), or
- Shared drive / file share, or
- git LFS (watch repo size limits)

Do not assume a teammate can `volume get` from your workspace.
