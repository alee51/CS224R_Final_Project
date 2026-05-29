# Project status

**Updated:** 2026-05-29

**Also read:** [`human-remaining-work.md`](human-remaining-work.md) · [`verl_migration_plan.md`](verl_migration_plan.md) · [`stage-01-agent-plan.md`](stage-01-agent-plan.md)

## Deadlines

| When | Milestone |
|------|-----------|
| **Fri 2026-05-29 EOD** | Training **launched** (all 3 arms) |
| **Mon 2026-06-01 11:59 PM** | Training **done** |
| **Tue 2026-06-02 4:00 PM** | Poster to print |
| **Wed 2026-06-03 9:00 AM** | Poster due |

If training does not start Fri EOD, the Mon finish line still holds — less wall time for the run.

## Active path

| What | Where |
|------|--------|
| Code + VeRL docs | [`../`](../) (this tree) |
| Frozen 1.7B archive | [`../../main/`](../../main/) — no new training features |

**Bring-up:** `main-verl/` skeleton only; Stage 1 not started. Stack: [tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl) (pinned VeRL fork — not upstream package; not MaxRL algorithm).

**Training:** Qwen3-4B-Base · Polaris-51K filtered · CoT judge · `GRPO` / `minority_cot` / `poly_epo_cot` · **1 epoch** (2 only if time — see migration plan, once).

## Poster framing (not locked)

Depends on VeRL results after evals.

| Outcome | Direction |
|---------|-----------|
| **minority_cot beats GRPO** | Lead with separation at 4B + CoT |
| **minority_cot flat or loses** | Analysis: why minority voting fails; collapse diagnostics |
| **poly_epo_cot weak vs paper / 1.7B** | Inconclusive at 1 epoch; stronger concern if we had run 2 |

Background: 1.7B table ([`../../main/docs/checkpoint_eval_morning_2026-05-28.md`](../../main/docs/checkpoint_eval_morning_2026-05-28.md)) — GRPO wins all slices at convergence.

## Stage checklist

| # | Stage | Done? |
|---|--------|-------|
| 1 | Modal image + verl + Ray | ☐ |
| 2 | GRPO bring-up smoke (1.7B) | ☐ |
| 3a | `minority_cot` + mock clusters | ☐ |
| 4 | Judge on Modal | ☐ |
| 3b | Real judge wired | ☐ |
| 5 | `poly_epo_cot` | ☐ |
| 6 | 4B fit check | ☐ |
| 7 | Logging + mid-run eval wiring | ☐ |
| 8 | Full 3-arm 1-epoch retrain | ☐ |

## What not to do

- New training in `main/`
- Second epoch unless Stage 8 finishes early Mon and metrics look under-trained
- `pre-milestone/` cleanup
