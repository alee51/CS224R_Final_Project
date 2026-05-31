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

**Bring-up status (one-liner):** Stages 1–5 ☑ (GRPO bring-up, minority_cot mock + real judge, judge service, poly_epo_cot registered + unit-tested). **2026-05-31:** fixed silent mock fallback (`config.algorithm` nesting); 1-step training trace v3 proves real judge in adv path (parse 98%, distinct clusters ~2.6/step). Prior 10-step “judge” smoke was plumbing-only until fix — see `build/stage-03b-log.md`. Past stage logs in `docs/build/stage-0*-log.md`.

**What's left (compressed):**

1. **Judge sanity gate (NEW, blocks Stage 8 dispatch)** — eyeball-validate judge I/O on real Polaris prompts via `probes/judge_cluster_trace_fast.py` AND by enabling `CS224R_JUDGE_TRACE=1` in the Stage 8 production yamls (artifact lands on `/vol/`, pull with `modal volume get` mid-run). See Stage 6 plan S6.0.
2. **Stage 6 (slim)** — GRPO 4B OOM ladder only (S6.1–S6.3). S6.4/S6.5 mock-cluster 4B smokes **dropped** (mock path doesn't exercise judge; minority/poly 4B fit is verified by the first ~5 steps of their Stage 8 runs).
3. **Stage 7 prereq** — `finish_reason="length"` wiring. Stage 2 hit `response_length/max=4096` on 100% of observed steps; required for eval story.
4. **Stage 8** — fork locked 4B GRPO yaml three ways (`cluster_source: judge` on set arms), enable judge trace artifacts, launch.

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
| 5 | `poly_epo_cot` | ☑ (override: S5.5 mock smoke skipped; S5.1–S5.4 + unit tests PASS) |
| **5.5** | **Judge sanity gate (eyeball real I/O)** | ☐ **NEW — blocks Stage 8** |
| 6 | 4B GRPO fit check (S6.1–S6.3 only; mock set-arm smokes dropped) | ☐ |
| 7 | `finish_reason="length"` wiring (hard Stage-8 prereq) | ☐ |
| 8 | Full 3-arm 1-epoch retrain (with live judge trace artifacts) | ☐ |

## What not to do

- New training in `main/`
- Second epoch unless Stage 8 finishes early Mon and metrics look under-trained
- `pre-milestone/` cleanup
