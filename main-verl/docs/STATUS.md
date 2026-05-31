# Project status

**Updated:** 2026-05-30

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

**Bring-up:** Stage 1 complete. **Stage 2 PASS** (attempt 7, externally cancelled at step 38/50 — bash background ceiling killed CLI; treated as PASS because plumbing question was settled by step 27). **Stage 3a PASS** 2026-05-30 (mock clusters, 10-step smoke, ckpt at `global_step_10`). **Stage 4 PASS w/ notes** (Qwen2.5-7B variant, superseded by 3b's Qwen3-4B redeploy). **Stage 3b PASS** 2026-05-30: real judge wired end-to-end on Qwen3-4B-Instruct-2507 + chicken602; ray_trainer expose-data patch + clusters_judge.py + prompt-decode path all validated; 10/10 training steps with entropy stable in 1.04–1.33 band (matches Stage 2/3a); checkpoint persisted at `/vol/checkpoints/main-verl/minority_cot_smoke_judge_1p7b/global_step_10/`. Judge steady-state latency 1.6s/call (enforce_eager=False CUDA graphs), S4.5 v2 revalidation 100% parse + 100% agreement. **NEW Stage-8 prereq:** explicit `finish_reason="length"` wiring — Stage 2 hits `response_length/max=4096` on 100% of observed steps (well over the ≥10/50 trigger).

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
| 1 | Modal image + verl + Ray | ☑ |
| 2 | GRPO bring-up smoke (1.7B) | ☑ |
| 3a | `minority_cot` + mock clusters | ☑ |
| 4 | Judge on Modal | ☑ (Qwen3-4B-Instruct-2507, S4.5 v2 100/100) |
| 3b | Real judge wired | ☑ |
| 5 | `poly_epo_cot` | ☐ |
| 6 | 4B fit check | ☐ |
| 7 | Logging + mid-run eval wiring | ☐ (`finish_reason` is hard Stage-8 prereq) |
| 8 | Full 3-arm 1-epoch retrain | ☐ |

## What not to do

- New training in `main/`
- Second epoch unless Stage 8 finishes early Mon and metrics look under-trained
- `pre-milestone/` cleanup
