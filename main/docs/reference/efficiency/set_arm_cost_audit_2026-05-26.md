Here's the audit. The big finding upfront: at full n_kept=512, set arms cannot fit a 799-step epoch in 48h even with every "cheaper without objective change" knob stacked. The
  math forces an objective-side decision.

  ---
  1. Cost-model recalibration (read before the table)

  Reading _train_step_microbatched + _completion_logprobs_hf (main/train/trainer.py:296,608): inside a chunk, sequences are forwarded one at a time in a Python loop, then a single
   chunk_loss.backward() is called. So:

  - Dominant cost = n_kept × per-sequence (fwd + bwd) time.
  - num_chunks only changes (a) activation-graph size held in VRAM and (b) per-chunk overhead (synchronize, alloc). Going 4→2 chunks at fixed n_kept is maybe 5–10% — not the
  headline win the doc implies.
  - Set-arm pain is fundamentally n_kept going from ~150 → ~512 (3.4×), which matches the observed ~3× backward ratio. Token-budget tweaks alone won't close that gap.

  This reframes the deferred items: most attack per-seq cost (FA2, gc-off, fused/8-bit Adam) or VRAM headroom (8-bit Adam) — they help linearly, not multiplicatively. The only
  multiplicative levers are n_kept itself and async overlap (skipped per PLAN).

  ---
  2. Prioritized list — set arms (where the schedule risk lives)

  ROI assumes set-arm baseline ~380 s/step, n_kept ≈ 512, num_chunks 4–5, modal price 0.001261 $/s.

  #: S1
  Lever: Cap/subsample n_kept to 256 (random per-step, after keep_mask)
  Δ s/step: −170s → ~210s
  Δ $/epoch: −$170/ep
  Files: trainer.py:817–836 (10-line insert before _train_step_microbatched)
  Restart?: New checkpoint OK; resumes fine
  GRPO?: N/A
  Risk: Objective change — smaller effective batch (~σ↑); flag in PLAN/decisions.md
  ────────────────────────────────────────
  #: S2
  Lever: gradient_checkpointing: false A/B on set arms
  Δ s/step: −80–110s → ~270s
  Δ $/epoch: −$80–100/ep
  Files: train_real.yaml:37 + smoke first
  Restart?: Restart on config flip
  GRPO?: Yes (likely best single GRPO win too)
  Risk: VRAM — must keep token_budget ≤ ~70k or chunks balloon; need probe
  ────────────────────────────────────────
  #: S3
  Lever: token_budget 105k → 140–170k
  Δ s/step: −20–40s (chunk overhead only)
  Δ $/epoch: −$25–50/ep
  Files: train_real.yaml:31
  Restart?: Live (no restart)
  GRPO?: Yes
  Risk: Watch vram_peak_gb_step; smoke 5 steps
  ────────────────────────────────────────
  #: S4
  Lever: 8-bit AdamW (bitsandbytes)
  Δ s/step: Indirect: frees ~10GB → enables higher budget / gc-off
  Δ $/epoch: up to −$60/ep stacked
  Files: trainer.py:643 build_hf (swap AdamW → bnb.optim.AdamW8bit), infra/modal_image.py add bitsandbytes
  Restart?: Breaks optimizer state in ckpt → fresh branch only
  GRPO?: Yes (only at fresh-branch start)
  Risk: Low — well-tested at this scale; verify ckpt resume path
  ────────────────────────────────────────
  #: S5
  Lever: Fused AdamW (torch.optim.AdamW(..., fused=True))
  Δ s/step: −2–5s (optim is small)
  Δ $/epoch: −$5/ep
  Files: trainer.py:661 add fused=True
  Restart?: Live; same ckpt
  GRPO?: Yes
  Risk: Trivial; bf16 params supported in PT 2.6
  ────────────────────────────────────────
  #: S6
  Lever: Checkpoint cadence 10 → 25 steps
  Δ s/step: −5–8s/step (commit blocks loop)
  Δ $/epoch: −$10/ep
  Files: train_real.yaml:38
  Restart?: Live
  GRPO?: Yes
  Risk: Lose ≤25 steps on crash; hourly backstop in should_checkpoint already exists (trainer.py:1041)
  ────────────────────────────────────────
  #: S7
  Lever: DAPO dynamic sampling
  Δ s/step: ~0 on set arms
  Δ $/epoch: ~0
  Files: rollout loop
  Restart?: Restart
  GRPO?: GRPO only (~65% all-wrong)
  Risk: Doesn't help set arms — they aren't filtering at zero
  ────────────────────────────────────────
  #: S8
  Lever: vLLM sleep enabled (it isn't by default — launcher passes no flag)
  Δ s/step: Frees ~30–40 GB during fwd/bwd → unlocks gc-off + bigger budget
  Δ $/epoch: enables S2+S3
  Files: launch_train.sh: add --vllm-sleep 1
  Restart?: Live; ckpt safe
  GRPO?: Yes
  Risk: Already coded (rollout.py:180); just turn it on for smoke first
  ────────────────────────────────────────
  #: S9
  Lever: B200 migration
  Δ s/step: Unknown until smoke; need ≥37% wall-clock cut to beat $/step
  Δ $/epoch: unknown
  Files: infra/modal_image.py (vLLM≥0.9.x, cu128)
  Restart?: Major
  GRPO?: Yes
  Risk: High setup risk; skip unless S1+S2+S6 underperform
  ────────────────────────────────────────
  #: S10
  Lever: Async rollout∥train overlap
  Δ s/step: −25–30% wall-clock
  Δ $/epoch: −$70+/ep
  Files: Major rewrite
  Restart?: Restart
  GRPO?: Yes
  Risk: Skip per PLAN — 4-day runway

  Items not separately listed but worth knowing they're already on: FA2 (trainer.py:655), prefix caching (train_real.yaml:50), vLLM logprobs:1 reuse, per-chunk interleaved
  fwd+bwd, self-spawn legs.

  ---
  3. Make-cheaper vs change-what-we-train

  Without objective change (S2 + S3 + S4 + S5 + S6 + S8): best-case set-arm step time ≈ 240–270 s (stacked, optimistic).
  - Math: 799 × 250 ≈ 55 h/epoch ≈ 2.3 days. Still doesn't fit 48 h with buffer, and assumes S2 works without OOM (not yet verified).

  With objective change (add S1 on top): step time ≈ 180–210 s.
  - Math: 799 × 200 = 44 h/epoch ≈ 1.8 days. Fits with slack.

  So the deliverable answers your "X and Y → Z" question as:

  ▎ If we implement S1 (n_kept cap to 256) + S2 (gc=False on a smoke-verified VRAM envelope) + S6 (ckpt every 25) before the minority launch, set-arm epoch goes from ~84h →
  ▎ ~40–45h, at ~$180/epoch saved, with scientific risk = batch-noise increase from random subsampling + needing one paragraph in decisions.md justifying it. Without S1, you
  ▎ cannot fit one epoch in 3 days even with every memory/perf knob on.

  ---
  4. Minimum viable plan to get set arms under 48h

  Order matters; smoke between each.

  1. Today (30 min eng): turn on vllm_sleep=1 in launcher (S8). Re-run minority smoke → confirm vram_peak_gb_step drops during fwd/bwd phase by ~30 GB. This is the prereq for S2.
  2. Today (1 hr eng + 10-step smoke): gradient_checkpointing: false (S2) with token_budget dialed down to 75k initially. Read train/vram_peak_gb_step and train/t_backward_s. If
  peak < 130 GB and backward drops to ~150s, keep. If OOM, fall back to gc=true and skip S2.
  3. Today (15 min): ship S5 (fused=True), S6 (ckpt every 25). Both live, no restart needed for the live GRPO run — apply on next launch only.
  4. Tomorrow (1–2 hr eng + smoke): S1 (n_kept cap). Implement as a config knob train.max_kept_prompts: 32 so it's per-arm-tunable. Diff is ~10 lines in run_one_grpo_step
  (trainer.py:817): after building kept_rollouts/kept_adv, if len(kept_prompts) > max_kept_prompts, sample without replacement using the step's RNG and keep all 8 rollouts per
  sampled prompt. Log a train/n_subsampled_dropped metric. Document in decisions.md as objective change (batch noise tradeoff).
  5. Defer: S4 (8-bit AdamW) only worth it if there's still a budget gap and you're on a fresh checkpoint. Set-arm runs are fresh, so it's compatible — bundle into the minority
  full launch but don't gate the launch on it.
  6. Skip for poster: S9 (B200), S10 (async). Both too risky for the 2026-06-03 deadline.

  ---
  5. Caveats / things to verify before you trust the numbers above

  - I did not pull live wandb panels — numbers above use your prompt's reported phase splits. If gc=False smoke shows t_backward drops less than ~30%, the math shifts and S1
  becomes even more necessary.
  - Per-seq forward in _completion_logprobs_hf is single-batch (input_ids shape [1, L]). A separate efficiency idea worth a future probe (not for poster): batch sequences inside a
   chunk via padded forward + masked loss. That's the real "fewer chunks = wins" path — but it's an algorithmic change, not a config flip, so out of scope.
  - S2 risk: H200 has 141 GB but FA2 + grad activations + AdamW state + bf16 model can spike past 130 GB without gc at high n_kept. The 5-step smoke is non-negotiable.
  - S1's "random subsample" should use cfg.global_seed + step for reproducibility; not seeding makes the choice unrecoverable on resume.

⏺ Bottom line: the schedule risk is real and only an n_kept cap (objective change) reliably gets one set-arm epoch under 48h; the memory/perf stack alone tops out around
  ~55h/epoch optimistically. Want me to draft the S1 patch + a smoke yaml for the gc=False probe?