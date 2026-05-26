# GRPO training — wandb dashboard quickstart

8 panels for routine monitoring of `cs224r-minority-voting` runs. If something looks off, drop into [`wandb_dashboard_full.md`](./wandb_dashboard_full.md) for the diagnostic panels.

A 60-second check: scan top-to-bottom. If panels 1–6 look normal and a sample completion in 8 looks like real math, close the tab.

The companion script `scripts/setup_wandb_quickstart_view.py` creates this exact layout as a saved view in wandb — run it once per project and pick "GRPO quickstart" from the workspace dropdown.

---

## The 8 panels

| # | Panel | Key(s) | Healthy | Worry when |
|---|---|---|---|---|
| 1 | **Mean reward (rolling 20)** | `train/mean_reward` | 0.08–0.18, slow uptrend | < 0.07 for 50+ steps |
| 2 | **Pass@k stacked** | `train/frac_prompts_0_correct` … `_8_correct` | `0_correct` shrinking, mid (4–7) growing | `0_correct` rising |
| 3 | **Importance ratio + clip frac** | `train/ratio_max`, `train/clipped_high_frac` | ratio_max < 5, clip < 0.05 | ratio_max > 10 or clip > 0.10 |
| 4 | **Grad norm (pre-clip)** | `train/grad_norm_preclip` | 0.3–5, slowly drifting | sustained > 50, or NaN |
| 5 | **Completion length** | `train/mean_completion_tokens`, `train/p95_completion_tokens` | mean 700–950, p95 1700–2500 | p95 approaches 4096 |
| 6 | **Finish reason** | `train/frac_finish_stop`, `..._length`, `..._other` | stop > 0.85 | length > 0.25 |
| 7 | **VRAM peak** | `train/vram_peak_gb_step` | 115–130 GB | touches 140 |
| 8 | **Sample completions** | `sample/completion_0`, `_1`, `_2` (every 50 steps) | looks like real chain-of-thought math | repetition, empty, `\boxed{0}` spam |

---

## Routine check (60 sec)

1. Reward trending sideways or up? (panel 1)
2. Histogram mass moving from `0_correct` toward mid bins? (panel 2)
3. Ratio stable, clip frac low, grad norm not spiking? (3, 4)
4. Completion length flat, finish_stop dominant? (5, 6)
5. VRAM comfortable headroom? (7)
6. One sample completion looks sane? (8)

If any of those look wrong → open `wandb_dashboard_full.md` and find the diagnostic that drills into the issue.
