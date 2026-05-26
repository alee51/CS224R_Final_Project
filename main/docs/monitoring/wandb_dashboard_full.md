# GRPO training — wandb dashboard guide

Panels to build for live monitoring of `cs224r-minority-voting` training runs. Organized by what each catches. Build top-to-bottom; the first section is the must-have set.

All keys are under the `train/` prefix unless noted.

---

## §1. Learning-signal panels (look at first)

These tell you whether the policy is actually improving.

### 1.1 Mean reward — rolling

- **Panel:** line plot of `train/mean_reward`, with a 20-step rolling mean overlay.
- **Why:** raw per-step reward is noisy (sd ≈ 0.02 at this batch size); the trend is what matters.
- **Healthy range:** 0.08–0.18 on filtered Polaris with arm C + Rank-2 grader; starts near 0.085 at step 0.
- **Yellow flag:** sustained drop below pre-warmup baseline (0.07) for 50+ steps with shrinking variance.
- **Red flag:** monotonic drop with reward < 0.05 — likely policy collapse.

### 1.2 Pass@k histogram over time

- **Panel:** stacked area chart of `train/frac_prompts_0_correct` through `train/frac_prompts_8_correct` (9 series).
- **Why:** the cleanest learning-signal plot. `frac_prompts_0_correct` should shrink and the mid-counts (4–7) should grow over time. Far more informative than `mean_reward`.
- **Healthy trajectory:** `0_correct` declines from ~0.78 toward 0.5; `4-7_correct` grow; `8_correct` may stay near zero (means the policy isn't simply memorizing easy prompts).

### 1.3 Reward by extract path

- **Panel:** line plot of `train/mean_reward_extract_hybrid`, `..._boxed`, `..._answer_line`, `..._none`.
- **Why:** separates "learning math" from "learning format compliance." If overall reward rises but it's all driven by `extract_path_boxed` going up while math correctness stays flat, the model is just learning to wrap answers in `\boxed{}` rather than solving problems.
- **Healthy:** all paths' rewards trend up together; `hybrid` (DAPO-style fallback chain) should dominate volume.

### 1.4 Extract-path mix

- **Panel:** stacked area of `train/extract_path_hybrid`, `..._boxed`, `..._answer_line`, `..._none` (4 series, summing to ~1.0).
- **Why:** if `none` rate stays high (>20%), the parser is failing to extract anything from many completions — model isn't producing parseable answers.
- **Healthy:** `hybrid` + `boxed` together >80%; `none` <15%.

---

## §2. Stability panels (early-warning signals)

These catch trouble before it shows up as reward collapse.

### 2.1 Importance ratio — multi-line

- **Panel:** line plot of `train/ratio_mean`, `train/ratio_max`, `train/ratio_p95` (3 series, log scale on y).
- **Why:** DAPO PPO clips at `clip_low=0.20, clip_high=0.28`. The ratio measures how far the current policy has drifted from `old_policy` since the rollouts were generated. If ratios blow up, importance sampling is broken.
- **Healthy:** `ratio_mean` ≈ 1.0; `ratio_max` < 5; `ratio_p95` < 1.5.
- **Yellow flag:** `ratio_max` > 10 sustained — policy is moving too fast.
- **Red flag:** `ratio_max` > 100 — gradient updates are nonsense.

### 2.2 Clipped fraction

- **Panel:** line plot of `train/clipped_low_frac`, `train/clipped_high_frac`.
- **Why:** fraction of tokens where the importance ratio fell outside the trust region and got clamped. High clipping means gradients are constantly being attenuated → learning stalls.
- **Healthy:** both < 0.05 (less than 5% of tokens clipped).
- **Yellow flag:** either > 0.10 sustained — consider lowering LR or widening clip range.
- **Red flag:** either > 0.30 — training is effectively making no progress on those tokens.

### 2.3 Gradient norm

- **Panel:** line plot of `train/grad_norm_preclip` (log scale).
- **Why:** the L2 norm of the gradient *before* `clip_grad_norm_(grad_clip=1.0)`. If it's constantly above 1.0, gradients are being attenuated. Spikes signal instability.
- **Healthy:** 0.5–5.0, slowly trending down as policy converges.
- **Yellow flag:** spikes to >50 single-step — might be a bad-batch artifact; one isolated spike OK.
- **Red flag:** sustained >100 or NaN — training is diverging.

### 2.4 Entropy proxy (mean neg-logprob)

- **Panel:** line plot of `train/mean_neg_logprob`.
- **Why:** mean negative log-probability of generated tokens under the current policy. Lower = policy more confident about its outputs. Sharp drops = mode collapse (policy specializing on a narrow output distribution before learning).
- **Healthy:** stable 2.0–4.0, slowly trending down as policy converges.
- **Yellow flag:** drop of >1.0 in <50 steps — policy is specializing fast, watch reward.
- **Red flag:** drop below 0.5 — likely degenerate (single-token completions, repetition).

### 2.5 Finish-reason distribution

- **Panel:** stacked area of `train/frac_finish_stop`, `train/frac_finish_length`, `train/frac_finish_other`.
- **Why:** if most rollouts terminate by hitting `max_response_length` (4096) instead of EOS, the model is rambling — early sign of length blowup, repetition collapse, or degraded reasoning.
- **Healthy:** `frac_finish_stop` > 0.85, `frac_finish_length` < 0.15.
- **Yellow flag:** `frac_finish_length` > 0.25 — model is timing out on too many problems.
- **Red flag:** `frac_finish_length` > 0.50 — likely infinite-loop / repetition pathology.

---

## §3. Completion-shape panels

These catch reward hacking or output degradation that doesn't show up in reward.

### 3.1 Completion length over time

- **Panel:** line plot of `train/mean_completion_tokens` + `train/p95_completion_tokens` (two series).
- **Why:** monotonic length growth is a known RL pathology — the model learns that longer chains-of-thought get rewarded (sometimes spuriously). Especially watch p95.
- **Healthy:** mean stable 700–950 tokens; p95 stable 1700–2500.
- **Yellow flag:** mean rising >20% over 100 steps with no reward gain.
- **Red flag:** p95 approaching `max_response_length=4096` — completions are getting truncated, learning signal is corrupted.

### 3.2 Sample completions (every 50 steps)

- **Panel:** wandb "Media" panel pointing at keys `sample/completion_0`, `sample/completion_1`, `sample/completion_2`.
- **Why:** the *only* way to catch reward hacking ("model always outputs `\boxed{0}`"), garbage tokens, repetition loops, or mode collapse that the scalar metrics miss. **Read 2–3 of these every time you check the dashboard.**
- **Healthy:** completions look like real chain-of-thought math, ending with `\boxed{answer}`.
- **Red flag:** repetitive nonsense, empty completions, all-identical outputs, prompt-injection-style hijacks of the model.

---

## §4. Performance / cost panels

These don't affect learning quality but tell you whether the run is wasting money or about to crash.

### 4.1 VRAM peak with headroom

- **Panel:** line plot of `train/vram_peak_gb_step` with a horizontal line at 140 GB (H200 max).
- **Why:** if peak ever touches 140, the next step OOMs.
- **Healthy:** 115–130 GB peak (~10–25 GB headroom).
- **Yellow flag:** > 132 GB sustained — reduce `train.token_budget` in yaml (currently 90000) for the next launch.

### 4.2 Step time decomposition

- **Panel:** stacked area of `train/t_rollout_s` + `train/t_train_fwd_bwd_s` + `train/t_weight_sync_s`.
- **Why:** see where time is going. Rollout typically ~110s, train ~70–130s, weight sync ~0.01s.
- **Yellow flag:** sustained step time > 230s — costing more than expected. Common causes: high `n_kept` (look at §4.3) or completion-length growth (look at §3.1).

### 4.3 n_kept and chunk count

- **Panel:** two-line plot of `train/n_kept_sequences` (left axis) + `train/num_chunks` (right axis).
- **Why:** `n_kept` drives train cost; `num_chunks` is how it interacts with the token-budget. More chunks = more gradient_checkpointing recompute = slower step.
- **Healthy:** n_kept 100–220 (mean ~160); num_chunks 1–3.
- **Useful tuning data:** if num_chunks is mostly 1 with VRAM headroom > 20 GB, you can raise `token_budget` in yaml. If num_chunks is often 3+ with VRAM near 130, lower it.

### 4.4 Wall-clock cost

- **Panel:** custom — `step` × wall-clock-per-step × `modal_price_per_sec=0.001261` (H200).
- **Why:** running tally of how much the run has cost. Useful for setting expectations.
- **Approximate:** at ~200s/step that's ~$0.25/step or ~$15/hour.

---

## §5. Diagnostic panels (only look when something's wrong)

Not for routine monitoring — only consult these when a §1–§4 panel flagged.

| Panel | Key(s) | When to look |
|---|---|---|
| Per-phase VRAM | `train/vram_peak_gb_t_rollout`, `..._t_train_fwd_bwd` | If §4.1 spikes — narrow down which phase grew |
| Effective microbatch / max chunk size | `train/effective_microbatch`, `train/max_chunk_size` | If §4.3 looks wrong — verify token-budget packing is working |
| Parse OK rate | `train/parse_ok_rate` | If §1.3 shows weird path mix |
| Prompt coverage | `train/prompt_coverage` | If §1.2 stalls — share of prompts with ≥1 correct |
| Mixed reward rate | `train/mixed_reward_rate` | Share of prompts with both correct & incorrect rollouts; low = no learning signal |
| Mean advantage | `train/mean_advantage` | Should be ≈ 0 (GRPO group-relative). Non-zero = bug. |
| Fraction filtered | `train/fraction_filtered` | Same info as `prompt_coverage` inverted; cross-check |
| Weight sync time | `train/t_weight_sync_s` | If it's not ~0.01s, something's wrong with vLLM sync |

---

## §6. Setting up the dashboard in wandb

1. Open the run at https://wandb.ai/224r-project/cs224r-minority-voting (or whichever workspace).
2. Top of run page → click **"Edit panels"** (or `Add panel` if dashboard is empty).
3. For each line plot above: `+ Add panel` → `Line plot` → pick the key(s) from the dropdown.
4. For stacked-area: same flow, change chart type from `Line` to `Area` and select multiple keys.
5. For the "sample completions" panel: `+ Add panel` → `Markdown` / `Media`, point at `sample/completion_0` etc.
6. **Save** as a workspace view named "GRPO training health" — both you and your collaborators can load it.

For comparing multiple runs (yours + handoff partner's), use wandb's **"Group runs"** feature with `group=train-real` (already set in `train_real.yaml`).

---

## §7. What to actually do during a check-in

A 90-second routine check:

1. Open §1.1 (mean_reward) and §1.2 (pass@k stacked) — is reward stable or moving up? Are histogram counts shifting in the right direction?
2. Glance at §2.1 (ratio_max) and §2.5 (finish_reason) — any red flag?
3. Glance at §3.1 (completion length) — flat or growing?
4. Read one sample completion from §3.2 — does it look like math?
5. Glance at §4.1 (VRAM) — comfortable headroom?

If any of those are red, drop into §5 to diagnose. Otherwise close the tab and come back in a few hours.
