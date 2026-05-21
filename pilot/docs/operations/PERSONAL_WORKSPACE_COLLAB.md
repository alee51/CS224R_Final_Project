# Personal Modal workspace — operator cheat sheet

**Decision:** Stage 1 pilot runs on each operator's **personal** Modal profile, not a shared team workspace. See `nancy_explore/narrative/decisions.md` (2026-05-19).

---

## Before launch

```bash
source .venv/bin/activate
modal profile current          # expect your GitHub-username profile (e.g. chicken602)
modal secret list              # huggingface + wandb-api-key on *this* profile
```

One-time per machine/profile: `modal token new`, `modal secret create huggingface HF_TOKEN=...`, `modal secret create wandb-api-key WANDB_API_KEY=...`.

---

## Config dry-run (before smoke / matrix)

```bash
python -c "from pilot.infra.modal_launch import launch_run; launch_run('smoke', dry_run=True)"
```

## Smoke gate (§6 — run before matrix)

```bash
./pilot/scripts/modal_run_pilot.sh --run-id smoke
```

Config: `pilot/configs/smoke.yaml` (3 steps, 32 prompts, `$10` cap). Preempt/resume test: kill app mid step 2, then re-run same command.

## Detached run (default)

```bash
./pilot/scripts/modal_run_pilot.sh --run-id run1_grpo
# wait for "Spawned function call id:" — safe to close laptop
```

Interactive debug (laptop must stay on): add `--wait`.

Matrix: `./pilot/scripts/launch_pilot_matrix.sh` (three GRPO runs; no `run0_proxy` — see `pilot/docs/decisions/20260519_skip_run0_stage1_redesign.md`).

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
