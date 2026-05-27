# GRPO training handoff — resume from latest checkpoint

You are continuing a CS 224R final project GRPO run that Nancy started. As of the 2026-05-26 restart, the trainer **auto-relaunches itself** across Modal's 24h timeout, so once you launch, you can walk away — chained legs handle themselves until `total_steps=799` is reached.

**Step count:** 159 done out of 799 (`total_steps=799` in yaml = one epoch on filtered Polaris: 51,139 rows / 64-prompt batch ≈ 799 unique batches). The handoff checkpoint is **`step_000159.pt`**.

**Key improvements landed 2026-05-26** (see [`efficiency_wins_2026-05-26.md`](../efficiency_wins_2026-05-26.md) for full reasoning):

- **`token_budget` bumped to 105000** in `train_real.yaml` (drops most steps from 2 chunks → 1, ~25% step time savings when it triggers).
- **Self-spawn auto-relaunch**: `train_remote` chains itself before Modal's 24h cap, no manual intervention needed.
- **`--fresh-wandb` flag**: starts a new wandb run on resume when the live run has logged past the resume checkpoint. Useful for your first launch.
- **FlashAttention-2 attempted then reverted** — first relaunch died at ~90s during HF model load (no stack trace recovered). See `efficiency_wins_2026-05-26.md` §2. The `flash-attn` wheel is still pinned in the Modal image so re-enabling later is a one-line trainer change with no image rebuild.

---

## What you'll receive from Nancy

1. **`step_000159.pt`** (~9.6 GB). Contains model weights, optimizer state, RNG, dataset cursor. Launch with `--fresh-wandb` to start a new wandb run (Nancy's wandb run `pcas3emd` is past the checkpoint step, so the silent-log-drop would otherwise bite).
2. **`polaris_train.jsonl`** — filtered Polaris training data, ~29 MB, **51,139 rows**. **Skip this if you already have it from a previous Nancy handoff** — the file hasn't changed since 2026-05-26.

That's it. Repo + code are public on GitHub.

---

## One-time setup on your machine

```bash
# 1. Clone Nancy's repo (or git pull if you already have it)
git clone https://github.com/alee51/CS224R_Final_Project.git
cd CS224R_Final_Project
# IMPORTANT: must include the 2026-05-26 efficiency changes (token_budget=105k + self-spawn + --fresh-wandb).
# Check by grep:
grep "token_budget: 105000" main/configs/train_real.yaml && echo "OK: 105k budget" || echo "STALE: pull latest main"
grep "leg_budget_s" main/train/trainer.py && echo "OK: self-spawn wired" || echo "STALE: pull latest main"

# 2. Install Python deps (venv at main/.venv)
python3.11 -m venv main/.venv
main/.venv/bin/pip install -r main/requirements.txt
# Modal rebuilds its own image — local venv is only for the CLI + scripts.

# 3. Modal auth (your account)
main/.venv/bin/modal token new

# 4. Modal secrets in YOUR workspace
main/.venv/bin/modal secret create HUGGINGFACE HF_TOKEN=hf_xxx       # your HF token
main/.venv/bin/modal secret create WANDB_API_KEY WANDB_API_KEY=xxx   # your wandb key
```

If you don't have a wandb account, make one at https://wandb.ai/. The run will log to project `cs224r-minority-voting` under entity `224r-project`. Ask Nancy to add you to the team workspace so the run lands with hers; otherwise it logs to your personal workspace which is fine for a one-shot run.

---

## Upload Nancy's artifacts to your Modal volume

```bash
main/.venv/bin/modal volume create main-artifacts          # one-time, idempotent
main/.venv/bin/modal volume put main-artifacts step_000159.pt checkpoints/train_real/step_000159.pt
# Skip the next line if you already have polaris_train.jsonl on your volume from a prior handoff:
main/.venv/bin/modal volume put main-artifacts polaris_train.jsonl data/polaris_train.jsonl
```

Sanity check:
```bash
main/.venv/bin/modal volume ls main-artifacts checkpoints/train_real/
# should show: step_000159.pt
main/.venv/bin/modal volume ls main-artifacts data/
# should show: polaris_train.jsonl
```

---

## Configure the run (your name on the wandb tags)

Edit `main/configs/train_real.yaml`:

```yaml
operator: emma           # or anastasia — your name
```

Leave everything else alone. The yaml already points at `/vol/data/polaris_train.jsonl` and `/vol/checkpoints/train_real/`, and `resume: auto` will auto-load the latest checkpoint on the volume. The Modal function timeout is 24h; the trainer self-spawns a successor leg at `CS224R_LEG_HOURS=23` (default) before the timeout. **You launch once; chained legs run themselves until `total_steps=799` is reached.** At post-2026-05-26 step times (~150–180s/step expected with `token_budget=105k`), a single 23h leg covers ~460–550 steps.

---

## Launch

```bash
# First launch: --fresh-wandb starts a new wandb run (the checkpoint's old wandb_run_id
# may point at a finished run; this avoids the "current step > log step" silent drop).
bash main/scripts/launch_train.sh --mode full --fresh-wandb
```

After the **first** leg lands cleanly, subsequent legs (which spawn themselves) do NOT need `--fresh-wandb` — they inherit the new wandb_run_id from your checkpoint and chain onto it.

You should see (within ~30 seconds):
```
Launching train mode=full ... app=cs224r-train-grpo-full-<you>-<MMDDHHMM>
✓ Initialized. View run at https://modal.com/apps/<workspace>/main/ap-XXXXXXXX
```

The first launch in your workspace triggers a **~3–5 min Modal image rebuild** (one-time; includes the flash-attn wheel even though the trainer doesn't currently load it). Subsequent legs reuse the cached image and start in ~30s.

Within ~2–5 minutes you'll see:
```
INFO Resuming from checkpoint /vol/checkpoints/train_real/step_000159.pt
INFO CS224R_FRESH_WANDB set; starting fresh wandb run
wandb: View run at https://wandb.ai/.../runs/<your_new_run_id>
```

Training picks up at **step 160**. Checkpoints land every 10 steps on the volume.

After ~23h the leg will log:
```
Leg budget 23.00h reached at step N; spawning successor and exiting
Spawning leg 2 (last completed step=N, config=...)
```
and a new Modal app appears with `leg_number=2` in its wandb tag. Same wandb run continues.

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
| Step time | timestamp delta between successive `_step` rows | Should be 150–200s (105k token_budget); sustained 230+ = something off |
| Sample completions | `sample/completion_0/1/2` (logged every 50 steps) | Read 2–3 to confirm model is producing real math, not garbage |

If anything looks weird, ping Nancy with the wandb URL before stopping the run.

---

## When Modal times out (or you want to stop)

**You shouldn't need to do anything for normal timeouts** — the trainer self-spawns a successor leg at 23h elapsed. Just let it run.

**To stop the chain manually:** `modal app stop <app-id>` on whichever leg is currently running. External stop does NOT trigger a successor — one `modal app stop` = full halt. A checkpoint at the most recent step divisible by 10 is on the volume.

**To resume after a manual stop or crash:**
```bash
bash main/scripts/launch_train.sh --mode full   # auto-resumes from latest .pt; no --fresh-wandb needed
```
The same wandb run continues (no fresh flag) because the latest checkpoint has the active run_id.

---

## Handing back to Nancy

If you want Nancy to continue at the end of your leg:

```bash
# Download your latest checkpoint:
main/.venv/bin/modal volume ls main-artifacts checkpoints/train_real/   # find latest step_XXXXXX.pt
main/.venv/bin/modal volume get main-artifacts checkpoints/train_real/step_000XXX.pt ./step_000XXX.pt
# Send it to Nancy (gdrive / scp / s3 — it's ~13.7 GB)
```

No wandb_run_id stripping needed — Nancy launches with `--fresh-wandb` on her side if her live run has logged past your checkpoint.

---

## Troubleshooting

**"Resuming from checkpoint" doesn't appear in logs**
→ Check `modal volume ls main-artifacts checkpoints/train_real/`. If empty, your upload failed; re-run the `modal volume put` step.

**Modal function errors with "wandb run already exists"**
→ The checkpoint's wandb_run_id points at Nancy's old run. Relaunch with `--fresh-wandb`:
```bash
bash main/scripts/launch_train.sh --mode full --fresh-wandb
```
The legacy "strip wandb_run_id from the checkpoint" workaround is no longer needed.

**Wandb says "current step is X" and silently drops your logs after resume**
→ The live wandb run logged past the resume checkpoint. Same fix: `--fresh-wandb` on next launch.

**Self-spawn didn't fire at 23h (leg just exited)**
→ Check `modal app list` — if no leg 2 appeared, the spawn call failed silently. Manually relaunch with `bash main/scripts/launch_train.sh --mode full` (no `--fresh-wandb` — the latest ckpt's wandb_run_id is fine). Ping Nancy with the modal app logs.

**OOM on resume step**
→ VRAM headroom should be ~15 GB. If it OOMs, lower `train.token_budget` in yaml from 90000 to 75000 and relaunch. Should not happen with the current config but possible if your H200 has a slightly smaller usable VRAM (Modal sometimes provisions trimmed cards).

**Step time much higher than ~200s**
→ Check `train/num_chunks` — if it's 3+ every step, the token budget is too low for current n_kept. Conversely if VRAM is comfortable (peak < 120 GB), bump `token_budget` to 105000 for fewer chunks → ~25% faster steps. Restart needed; resume picks up from latest checkpoint.

---

## Quick reference — what completes the run

| Item | Status when you're done |
|---|---|
| Step count | 799 (one epoch on filtered Polaris: 51,139 rows / 64-prompt batch) |
| Final checkpoint | `step_000799.pt` (or whichever step you stopped on) |
| Time to ship back | When Modal timeout hits OR you've done your share OR you've finished the epoch |

Ping Nancy when you're done or if anything looks off. Thanks for the credits 🙏
