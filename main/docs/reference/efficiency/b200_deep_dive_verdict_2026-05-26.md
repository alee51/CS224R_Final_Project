B200 deep-dive: revised verdict

  Headline: skip for poster. The economics changed since the 2026-05-26 probe (you're already on H200, not H100), and the per-seq-forward kernel here doesn't benefit from
  Blackwell as much as a batched workload would.

  1. Numbers updated to current state (vs probe)

  The probe doc compares B200 against H100. You're on H200. That cuts the break-even bar significantly:

  ┌─────────────┬─────────────────────────────┬──────────────────────────────────────────────┐
  │    Path     │             $/s             │                Break-even bar                │
  ├─────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
  │ H100 → B200 │ 0.001097 → 0.001736 (×1.58) │ B200 ≤ 63% H100 wall-clock (≥37% faster)     │
  ├─────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
  │ H200 → B200 │ 0.001261 → 0.001736 (×1.38) │ B200 ≤ 72.6% H200 wall-clock (≥27.4% faster) │
  └─────────────┴─────────────────────────────┴──────────────────────────────────────────────┘

  So the question is: does B200 actually deliver ≥27% wall-clock cut on set-arm steps?

  2. Per-phase expected gains (best honest read)

  Set-arm baseline: rollout ~90s + train ~280s + sync/overhead ~10s = ~380s.

  ┌────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────┐
  │     Phase      │                                                             Why it might gain                                                             │  Realistic   │
  │                │                                                                                                                                           │    factor    │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┤
  │ Rollout (vLLM  │ Memory-bandwidth bound. B200 HBM3e ~8 TB/s vs H200 ~4.8 TB/s (~67% more).                                                                 │ 1.5–1.7× →   │
  │ decode)        │                                                                                                                                           │ ~55s         │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┤
  │ HF fwd+bwd     │ _completion_logprobs_hf runs one sequence at a time in a Python loop (trainer.py:378-381). Kernel launch + Python overhead becomes a      │ ~180–215s    │
  │                │ larger fraction on faster HW → diminishing returns. Marketing FLOPs say 2.5–4×; realistic for this kernel pattern is 1.3–1.6×.            │              │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┤
  │ Weight sync /  │ I/O-bound; no GPU help.                                                                                                                   │ ~10s         │
  │ overhead       │                                                                                                                                           │              │
  └────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────┘

  Optimistic B200 set-arm step: ~245s = 64% H200 → meets break-even, ~21% cheaper $/epoch.
  Realistic B200 set-arm step: ~280s = 74% H200 → just misses break-even, ~2% MORE $/epoch.

  The per-seq forward loop is the reason B200 underperforms its raw spec on this workload. A batched-padded forward would change this analysis — but that's its own refactor.

  3. Bring-up risk inventory (delta since the probe)

  What the probe flagged as work is mostly still true, plus two new items:

  ┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────┐
  │         Risk         │                                                                Status                                                                │ Time cost   │
  │                      │                                                                                                                                      │ if it bites │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ FA2 wheel SM100      │ Current wheel is cu12torch2.6 — Hopper-only kernels. Loading on Blackwell either fails or silently falls back to SDPA. Need FA-3     │ +0.5–1 day  │
  │ incompat ⚠️ NEW      │ wheel + matching transformers attn impl, or source-build FA2 with nvcc (debian_slim has no nvcc → need image rebase).                │             │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ vLLM 0.8.5 → 0.9.x   │ Confirmed pinned (modal_image.py:15). weight_sync.py reaches into                                                                    │ +0.5–2 days │
  │ bump                 │ llm_engine.model_executor.driver_worker.model_runner.model.load_weights — internal API; 0.9.x re-shuffled. Spike required.           │             │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ VLLM_USE_V1=0        │ Hard-set in modal_image.py:27 to work around Modal fork issue. May or may not be needed on 0.9.x; either way needs A/B smoke.        │ +0.5 day    │
  │ revalidation         │                                                                                                                                      │             │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ Modal B200 capacity  │ B200 has longer queue times than H200 on Modal; with self-spawn legs (train_remote.spawn), each leg pays the queue. Hard to          │ +1–3 h/leg  │
  │ ⚠️ NEW               │ estimate; could add 30–60 min/leg.                                                                                                   │             │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ CUDA 12.8 image      │ debian_slim likely fine with cu128 pip wheels; if not, swap to nvidia/cuda base image.                                               │ +0.5 day    │
  │ rebase               │                                                                                                                                      │             │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ Weight-sync          │                                                                                                                                      │             │
  │ correctness on new   │ tests/test_weight_sync.py is the spike but requires Modal — test_weight_sync.py doesn't run locally.                                 │ +0.5 day    │
  │ vLLM                 │                                                                                                                                      │             │
  └──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────┘

  Optimistic total: 1.5 days. Pessimistic: 4 days. Poster is 7 days out.

  4. Stacking analysis (what gets you the most actual win)

  ┌────────────────────────────────┬────────┬─────────┬─────────┬──────────┬────────────────────────────────────┐
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ H200 baseline (today, set arm) │ 380    │ 84      │ $380    │ 0        │ —                                  │
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ H200 + S1 (n_kept cap 256)     │ ~210   │ 47      │ $190    │ 0.25     │ Low (objective change, documented) │
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ H200 + S1 + S2 (gc off)        │ ~180   │ 40      │ $165    │ 0.5      │ Med (VRAM smoke required)          │
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ B200 alone (realistic)         │ ~280   │ 62      │ $390    │ 1.5–4    │ High                               │
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ B200 alone (optimistic)        │ ~245   │ 54      │ $340    │ 1.5–4    │ High                               │
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ B200 + S1 (realistic)          │ ~155   │ 34      │ $215    │ 1.5–4    │ High                               │
  ├────────────────────────────────┼────────┼─────────┼─────────┼──────────┼────────────────────────────────────┤
  │ B200 + S1 + S2 (optimistic)    │ ~115   │ 26      │ $160    │ 2–5      │ High                               │
  └────────────────────────────────┴────────┴─────────┴─────────┴──────────┴────────────────────────────────────┘

  Key observation: even the optimistic B200+S1+S2 stack ($160/epoch, 26h) only beats H200+S1+S2 ($165/epoch, 40h) on wall-clock, not on cost. And it costs 2–5 days of eng risk to
  find out. If the realistic case holds, you're spending eng days and 30% more $/epoch for a marginal time win.

  5. When B200 would be worth it

  It'd flip if any of these were true:
  - Pre-loaded image already validated. If someone in your group had already booted B200 + vLLM 0.9.x + Qwen3 + FA-3 successfully, eng cost drops to ~2h and it's a clear win.
  - Workload were batched. If _completion_logprobs_hf did padded multi-seq forward, the HF-train factor would be closer to 2× and B200 alone meets break-even comfortably.
  - More runway. Two weeks out, the eng risk is fine. Seven days with two set arms + GRPO + eval + retries — the spare-day budget is gone.
  - The full-report (not poster) deadline. Different math.

  6. Recommendation

  Don't bring up B200 for the poster. Do these instead, in order:

  1. Ship S1 (n_kept cap) today. Locks in 40h/epoch deterministically.
  2. Smoke S2 (gc off) on H200 in parallel. If VRAM is OK, get to ~40h.
  3. Park a B200 bring-up doc as a 2-day post-poster task — it's the natural next-up for the final report runs (more epochs, more arms), where the schedule slack exists.

  If you want, I can add a one-page "B200 deferred — do for final report" stub to docs/probes/ so the work isn't lost. Otherwise the existing probe doc plus this conversation are
  the trail.