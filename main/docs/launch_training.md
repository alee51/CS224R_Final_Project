# How to launch training (agents + operators)

**Canonical doc for “how do I launch a run?”** — copy commands from here; do not invent flags.

Run everything from **repo root** (`cs224r_finalproject/`), not from `main/`.

---

## Production full runs (B200, current default)

**2026-05-27 B200 dual launch (fresh GRPO + minority):** see [`handoff/b200_production_launch_2026-05-27.md`](./handoff/b200_production_launch_2026-05-27.md) for configs, settings, Modal/W&B links, and replay commands. Monitor: `bash main/scripts/monitor_b200_prod.sh`.

### Fresh B200 runs (parallel to H200 GRPO — isolated checkpoint dirs)

Use when H200 GRPO is still writing `/vol/checkpoints/train_real/` and you want a **new** B200 line that does not resume or collide:

```bash
# Fresh B200 GRPO → /vol/checkpoints/train_real_b200/
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm grpo \
  --config main/configs/train_real_b200_fresh_grpo.yaml --no-resume --fresh-wandb

# Fresh B200 minority_answer → /vol/checkpoints/train_minority_answer_b200/
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm minority_answer \
  --config main/configs/train_real_b200_fresh_minority.yaml --no-resume --fresh-wandb
```

Kill the H200 GRPO app once B200 GRPO has finished a few steps and checkpoints land under `train_real_b200/`.

### Default B200 (resume from yaml checkpoint_dir)

```bash
# GRPO (arm 1) — resumes latest in /vol/checkpoints/train_real/
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm grpo

# minority_answer (arm 2) — resumes latest in /vol/checkpoints/train_minority_answer/
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm minority_answer
```

**Do not pass `--config` unless you know why.** Defaults:

| `--gpu-class` | Config auto-selected |
|---------------|----------------------|
| `b200` | `main/configs/train_real_b200.yaml` |
| `h200` | `main/configs/train_real.yaml` |

**Wrong (common agent mistake):**

```bash
# BAD: B200 hardware + H200 yaml → wrong wandb tags / pricing metadata
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --config main/configs/train_real.yaml
```

**H200 fallback:**

```bash
bash main/scripts/launch_train.sh --mode full --gpu-class h200 --arm grpo
```

---

## Smoke runs (10 steps, no resume)

```bash
bash main/scripts/launch_train.sh --mode smoke --gpu-class b200 --arm grpo
bash main/scripts/launch_train.sh --mode smoke --gpu-class b200 --arm minority_answer
```

Shorter smoke (e.g. 5 steps for a quick gate):

```bash
bash main/scripts/launch_train.sh --mode smoke --gpu-class b200 --arm grpo --steps 5
```

---

## Preflight

`launch_train.sh` runs `preflight_train_launch.py` before Modal. It fails fast if:

- config file missing
- `gpu_class` in yaml ≠ `--gpu-class`
- required `train` / `rollout` / `weight_sync` keys missing

---

## What each flag does

| Flag | Meaning |
|------|---------|
| `--mode smoke` | 10 steps (or `--steps N`), `CS224R_TOTAL_STEPS` override, no resume |
| `--mode full` | `train.total_steps` from yaml (799 for prod) |
| `--gpu-class b200` | Modal fn `train_remote_b200` + default `train_real_b200.yaml` |
| `--gpu-class h200` | Modal fn `train_remote_h200` + default `train_real.yaml` |
| `--arm NAME` | Overrides yaml arm (`grpo`, `minority_answer`, `poly_epo_answer`) |
| `--fresh-wandb` | New wandb run on resume (handoff / broken wandb history) |
| `--no-resume` | Start at step 0 (no checkpoint load); smoke sets this by default |
| `--checkpoint-dir PATH` | Override `train.checkpoint_dir` (e.g. `/vol/checkpoints/foo`) |
| `--steps N` | Smoke only: override step count (default 10) |

---

## Weight sync (HF → vLLM)

Training is **collocated**: HF does backward; vLLM does rollouts. After each train step, HF weights are pushed into vLLM (`weight_sync.every_n_steps: 1` in `train_real.yaml`).

**Isolated smoke (no full train loop):**

```bash
bash main/scripts/launch_smoke_weight_sync.sh --gpu-class b200
```

**In-loop check (production path):** 5-step GRPO smoke on B200; confirm W&B logs `train/weight_sync_s` every step and no sync errors in Modal logs.

---

## Other arms / eval

- `poly_epo_answer`: `--arm poly_epo_answer` (same launcher)
- Checkpoint eval: `main/docs/efficiency/B200_build_notes.md`
- Resume handoff: `main/docs/handoff/resume_grpo_training.md`

---

## W&B / Modal

- Project: `224r-project/cs224r-minority-voting`
- App name pattern: `cs224r-train-<arm>-<smoke|full>-<operator>-<MM-DD-HHMM>`
- Monitor: Modal dashboard + wandb run linked in launch output

---

## All-filtered batch (no gradient signal)

If every prompt in a batch is filtered (`n_kept = 0`), the trainer **skips** backward and weight sync for that step, logs `train/skipped_no_kept=1` on W&B, and continues. The run does **not** crash. Rollout cost for that step is still spent.

---

## Decisions (do not “fix” without explicit ask)

- **No `n_kept` cap** on set arms — full ~512 kept sequences; accept longer steps on B200.
- **Checkpoint `weights_only`** hardening deferred (see timeline 2026-05-27 audit triage).
