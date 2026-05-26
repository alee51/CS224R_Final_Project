# GRPO training efficiency wins — 2026-05-26

**Context.** GRPO training on Qwen3-1.7B + H200 (single GPU, collocated vLLM + HF). Current step time **3–4 min/step at `batch_size=64`**. Two epochs over filtered Polaris ≈ **1600 steps** ≈ ~107h wall-clock at current pace. Modal hard-caps function timeout at **24h**, so the run requires ~5 manual relaunches per branch. PLAN §3 has up to **4 arms** (GRPO, Minority-answer, Minority-CoT, Poly-EPO-answer) — every per-step saving compounds across however many arms we end up training.

Only ~12h is sunk on the current GRPO branch, so a one-time restart cost is cheap relative to the savings across the remaining budget.

Step time decomposition from Group B probe (H200, bs=64): **rollout ≈ 73%**, **backward ≈ 25%**, weight sync + advantage + scoring ≈ 2%.

---

## Wins ranked by ROI

### 1. Bump `token_budget` 90000 → 105000  — **[DONE 2026-05-26]**

**Win.** Up to ~25% faster steps when chunks drop from 2 → 1.

**Why it works.** `_train_step_microbatched` greedy-packs sequences so each chunk's completion tokens ≤ `token_budget`. With a higher budget, more steps fit in a single fwd+bwd chunk instead of two — eliminating one extra HF forward pass + activation buildup per step.

**Why it's safe.** Handoff doc (`docs/handoff/resume_grpo_training.md:166`) explicitly recommends 105k if VRAM peak < 120 GB. Probe data shows ~115–130 GB peak at 90k → ~10–15 GB headroom to absorb the bump.

**Action.** Edit `main/configs/train_real.yaml` → `train.token_budget: 105000`. Restart needed (resume auto-loads from latest ckpt).

**Verify after launch.** Watch `train/vram_peak_gb_step` for ~10 steps. If it touches 140 GB, dial back to 95–100k. If it stays < 130, consider 110k.

### 2. FlashAttention-2 on the HF model  — **[DONE 2026-05-26]**

**Win.** ~20–30% faster on the HF forward+backward block, which is ~27% of step → **~5–8% off step time**. Also cuts activation memory by ~30% on the attention path, which buys more `token_budget` headroom (compounds with #1).

**Why it was missing.** PLAN §5 specifies "FlashAttention-2 (FA-3 if H100/H200)" as a baseline knob, but `build_hf` at `main/train/trainer.py:526` never set `attn_implementation`. HF transformers defaults to SDPA which is slower than FA2 on Hopper.

**Action.**
- Add `attn_implementation="flash_attention_2"` to `AutoModelForCausalLM.from_pretrained(...)` in `build_hf`.
- Install `flash-attn` in `main/infra/modal_image.py` (after vLLM, since vLLM owns the torch pin). Use `--no-build-isolation` so it links against the already-installed torch.

**Risk.** Low. Qwen3 is a standard decoder; FA2 is well-supported in `transformers>=4.55`. Modal image build adds ~3–5 min (one-time).

**Verify after launch.** Compare `train/t_logprob_fwd_s` and `train/t_backward_s` against the wandb baseline (current run pre-FA2). Expect both ~20–25% lower.

### 3. DAPO-style dynamic sampling  — **deferred**

**Win.** Group A: ~65% of arm-C prompts are all-wrong → those rollouts get filtered to zero gradient. Effective gradient-yielding batch is ~35% of nominal. Replacing filtered prompts with resampled ones could push effective utilization toward ~80% → **~2x effective throughput per dollar**.

**Why deferred (your call, agreed).** Changes training dynamics: prompt distribution per step shifts toward mid/hard prompts. Would muddy comparison against PLAN's existing arm-C readouts and Poly-EPO Fig. 2 parity. Worth a separate ablation, not a mid-run swap.

**If reconsidered.** Cleanest path: wrap `rollout_engine.generate` with a "resample-on-empty-group" loop bounded by a step rollout budget; flag in yaml so we can A/B against current behavior.

### 4. 8-bit AdamW (bitsandbytes)  — **deferred**

**Win.** Drops optimizer state from ~4× model size to ~1× → **~10 GB freed**, lets `token_budget` climb past 105k, compounds with #1.

**Why deferred.** Requires restart and a new optimizer state — checkpoint compatibility breaks. Worth it for a fresh branch (e.g. Minority-answer when we kick that off), not mid-GRPO.

### 5. Checkpoint cadence loosening  — **deferred**

**Win.** Saving 13.7 GB every 10 steps + `_artifacts_vol.commit()` blocks the train loop for ~10–20s. Moving to every 20–30 steps + keeping the hourly wall-clock backstop saves ~5–10% wall-clock with negligible risk (resume still works; we just rewind at most 30 steps on crash).

**Why deferred.** Trivial change but low marginal value vs. #1 + #2. Bundle into the next code update.

### 6. Async rollout / train overlap  — **skip for this project**

**Win.** Rollout is 73% of step; full overlap saves ~25–30% wall-clock.

**Why skip.** Complex to implement (needs double-buffered vLLM engine + non-blocking weight sync), high risk of subtle race bugs, and PLAN §5 already deferred it explicitly. 4-day runway is not enough to debug if it goes sideways.

---

## Compounding math across branches

| Branches trained | Hours saved by #1+#2 (~12% off step time) | Total Modal $ saved (~$0.001261/s × 64-prompt step × steps) |
|---|---|---|
| 1 (just GRPO finishing) | ~13h | ~$60 |
| 2 (GRPO + Minority-answer) | ~26h | ~$120 |
| 4 (all arms) | ~52h | ~$240 |

#1+#2 together are ~30 min of work. ROI is excellent even at 1 branch; at 4 branches they pay for the entire eval phase.

---

## Modal auto-relaunch

**No native option.** 24h is a hard per-function timeout. Modal's `retries=` only retries on exceptions/crashes — natural timeouts don't auto-retry.

### Recommended: self-spawn from inside the train loop

In `train()` loop, track elapsed wall-clock since start. When `elapsed > 23.5h`:
1. Force a checkpoint.
2. Call `train_remote.spawn(config_path=...)` to detach a fresh container (Modal spawn doesn't wait on the caller).
3. `return` cleanly.

The new container's `resume: auto` picks up the latest `.pt`. Legs chain themselves; no babysitting. ~30 lines added in `trainer.py`. Optionally guard with a max-leg-count env var so a bug can't infinite-loop.

### Alternatives (lower priority)

- **Modal cron watchdog** — separate `@app.function(schedule=modal.Cron(...))` that runs every 23h, checks if latest ckpt step < total_steps, and launches if so. Race-prone (need a lockfile on volume).
- **Local while-loop** — `while true; do modal run --detach ...; sleep 86400; done`. Brittle (laptop sleep kills it).

**Implementation status:** not yet done. Suggested as the next code change once #1 + #2 land and we've confirmed they speed up steps as expected.

---

## What landed in this commit

- `main/configs/train_real.yaml`: `token_budget: 90000` → `105000` (+ updated inline comment).
- `main/train/trainer.py`: `build_hf` now passes `attn_implementation="flash_attention_2"`.
- `main/infra/modal_image.py`: `flash-attn` added to image (post-vllm, `--no-build-isolation`).

Modal will rebuild the image on next launch (~3–5 min one-time cost). Resume picks up at step 140 (or wherever the latest ckpt is).
