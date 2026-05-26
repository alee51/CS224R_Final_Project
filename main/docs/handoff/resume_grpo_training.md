# GRPO training handoff — resume from step 139

You are continuing a CS 224R final project run that Nancy started. Training stopped cleanly at step 141 (out of 850) when the Modal function hit its 8-hour wall-clock timeout. This doc walks you through resuming on **your** Modal credits.

---

## What you'll receive from Nancy

1. **`step_000139.pt`** — checkpoint, ~13.7 GB. Contains: model weights, optimizer state, RNG, dataset cursor.
2. **`polaris_train.jsonl`** — filtered training data, ~50 MB, **51,139 rows**.
3. A pointer to the repo + git SHA: **`9b9e104`** (or newer — confirm with Nancy).

> The checkpoint will have its `wandb_run_id` stripped, so your launch starts a fresh wandb run under your account. Step numbers stay continuous (140 onward); plotting can stitch the runs in post.

---

## One-time setup on your machine

```bash
# 1. Clone repo + check out the right commit
git clone <repo URL>
cd cs224r_finalproject
git checkout 9b9e104

# 2. Install Python deps (venv at main/.venv)
python3.11 -m venv main/.venv
main/.venv/bin/pip install -r main/requirements.txt
# (Modal will rebuild its own image — local venv is just for the CLI/scripts)

# 3. Modal auth (your account)
main/.venv/bin/modal token new

# 4. Modal secrets — create these in YOUR Modal workspace
main/.venv/bin/modal secret create HUGGINGFACE HF_TOKEN=hf_xxx       # your HF token
main/.venv/bin/modal secret create WANDB_API_KEY WANDB_API_KEY=xxx   # your wandb key
```

If you don't have a wandb account, make one at https://wandb.ai/. The run will log to project `cs224r-minority-voting` under entity `224r-project` — ask Nancy to add you to the team workspace so the run lands in the right place (otherwise it'll log to your personal workspace, which is fine for a one-shot run).

---

## Upload Nancy's artifacts to your Modal volume

```bash
# From the directory where you saved step_000139.pt and polaris_train.jsonl:
main/.venv/bin/modal volume create main-artifacts          # one-time, idempotent
main/.venv/bin/modal volume put main-artifacts step_000139.pt checkpoints/train_real/step_000139.pt
main/.venv/bin/modal volume put main-artifacts polaris_train.jsonl data/polaris_train.jsonl
```

Sanity check:
```bash
main/.venv/bin/modal volume ls main-artifacts checkpoints/train_real/
# should show: step_000139.pt
main/.venv/bin/modal volume ls main-artifacts data/
# should show: polaris_train.jsonl
```

---

## Configure the run (your name on the wandb tags)

Edit `main/configs/train_real.yaml`:

```yaml
operator: emma           # or anastasia — your name
```

Leave everything else alone. The yaml already points at `/vol/data/polaris_train.jsonl` and `/vol/checkpoints/train_real/`, and `resume: auto` will auto-load step 139.

---

## Launch

```bash
bash main/scripts/launch_train.sh --mode full
```

You should see (within ~30 seconds):
```
Launching train mode=full ... app=cs224r-train-grpo-full-<you>-<MMDDHHMM>
✓ Initialized. View run at https://modal.com/apps/<workspace>/main/ap-XXXXXXXX
```

Within ~2 minutes you'll see:
```
INFO Resuming from checkpoint /vol/checkpoints/train_real/step_000139.pt
wandb: View run at https://wandb.ai/.../runs/<your_run_id>
```

Training picks up at **step 140**. Checkpoints land at step 149, 159, 169, … on the volume.

---

## What to watch on wandb during your leg

Most important panels (build these as line plots — wandb's "Add panel" UI):

| Panel | Key | What to look for |
|---|---|---|
| Reward trend | `train/mean_reward` (rolling 20-step mean) | Should hover at 0.08–0.10; sustained drop below 0.07 = collapse |
| Policy mass | `train/frac_prompts_0_correct`, `..._4_correct`, `..._7_correct` | `0_correct` should shrink, mid/high should grow |
| Length watch | `train/mean_completion_tokens` | Flat 800–900 = good; monotonic up = length blowup brewing |
| Stop vs length | `train/frac_finish_stop`, `train/frac_finish_length` | `frac_finish_length` rising means model is rambling — bad sign |
| Importance ratio | `train/ratio_max`, `train/clipped_high_frac` | `ratio_max` > 10 or `clipped_*` > 0.10 sustained = unstable updates |
| Entropy proxy | `train/mean_neg_logprob` | Should be stable ~2–4; sharp drop = mode collapse |
| Grad norm | `train/grad_norm_preclip` | Should hover around 1–5; spike to 100+ = instability |
| VRAM | `train/vram_peak_gb_step` | Should stay 115–130 GB; if it touches 140 = OOM next step |
| Step time | timestamp delta between successive `_step` rows | Should be 165–230s; sustained 250+ = something off |
| Sample completions | `sample/completion_0/1/2` (logged every 50 steps) | Read 2–3 to confirm model is producing real math, not garbage |

If anything looks weird, ping Nancy with the wandb URL before stopping the run.

---

## When Modal times out (or you want to stop)

Just let it timeout naturally, or `modal app stop <app-id>` to stop early. Either way: **a checkpoint at the most recent step divisible by 10 is on volume**. To resume — yours or hand back to Nancy — just relaunch:

```bash
bash main/scripts/launch_train.sh --mode full   # auto-resumes from latest .pt
```

The wandb run will be a different ID each time, but step numbers stay continuous.

---

## Handing back to Nancy

If you want Nancy to continue at the end of your leg:

```bash
# Download your latest checkpoint:
main/.venv/bin/modal volume ls main-artifacts checkpoints/train_real/   # find latest step_XXXXXX.pt
main/.venv/bin/modal volume get main-artifacts checkpoints/train_real/step_000XXX.pt ./step_000XXX.pt
# Send it to Nancy (gdrive / scp / s3 — it's ~13.7 GB)
```

Don't strip her wandb_run_id when sending back — she'll handle it on her side.

---

## Troubleshooting

**"Resuming from checkpoint" doesn't appear in logs**
→ Check `modal volume ls main-artifacts checkpoints/train_real/`. If empty, your upload failed; re-run the `modal volume put` step.

**Modal function errors with "wandb run already exists"**
→ Nancy's `wandb_run_id` wasn't stripped. Either ask her to strip it, or strip locally:
```bash
main/.venv/bin/python -c "
import torch
c = torch.load('step_000139.pt', map_location='cpu', weights_only=False)
c.pop('wandb_run_id', None)
torch.save(c, 'step_000139.pt')
"
# then re-upload to volume
```

**OOM on resume step**
→ VRAM headroom should be ~15 GB. If it OOMs, lower `train.token_budget` in yaml from 90000 to 75000 and relaunch. Should not happen with the current config but possible if your H200 has a slightly smaller usable VRAM (Modal sometimes provisions trimmed cards).

**Step time much higher than ~200s**
→ Check `train/num_chunks` — if it's 3+ every step, the token budget is too low for current n_kept. Conversely if VRAM is comfortable (peak < 120 GB), bump `token_budget` to 105000 for fewer chunks → ~25% faster steps. Restart needed; resume picks up from latest checkpoint.

---

## Quick reference — what completes the run

| Item | Status when you're done |
|---|---|
| Step count | 799 (one epoch on filtered Polaris) — yaml says 850 but the dataset has only 799 unique batches |
| Final checkpoint | `step_000799.pt` (or whichever step you stopped on) |
| Time to ship back | When Modal timeout hits OR you've done your share OR you've finished the epoch |

Ping Nancy when you're done or if anything looks off. Thanks for the credits 🙏
