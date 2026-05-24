# Pilot status (2026-05-20)

Single entry point for **where we are** — operational and research.

---

## Research status: **uncertain — do not treat redesign matrix as locked**

The team is **not confident** the current Stage 1 objective (`inverse_freq`) is the right instantiation of the mentor’s minority-voting pitch. See:

- [`nancy_explore/narrative/briefs/pilot_strategy_20260520.md`](../../nancy_explore/narrative/briefs/pilot_strategy_20260520.md) — critique and Q I vs Q II split
- [`nancy_explore/narrative/briefs/ta_office_hours_20260521.md`](../../nancy_explore/narrative/briefs/ta_office_hours_20260521.md) — questions for mentor
- [`nancy_explore/narrative/timeline.md`](../../nancy_explore/narrative/timeline.md) — full chronology

**`operations/PILOT_REDESIGN.md`** remains the **implementation spec** for infra (caps, smoke, checkpointing) but the **3-run objective lineup may change** after mentor alignment.

---

## Operational status

| Item | State |
|------|--------|
| **Run 0** (`20260519T190202Z`) | **Done** — canonical proxy artifacts + handoff |
| **First matrix** (run1–run3) | **Failed** — no complete training runs |
| **Redesign code** | In repo; smoke launches 2026-05-20 |
| **Smoke gate** | **Not verified locally** (config snapshots only; pull Modal volume to confirm) |
| **Stage 1 matrix** | **Not launched** (blocked on smoke + research decision) |
| **Pilot gate token** | `PENDING` (`decisions/decision_memo.md`) |

---

## What to run next (operator)

Only after team/mentor confirms objective **or** explicit decision to smoke-test infra only:

1. [`operations/SMOKE_READINESS.md`](operations/SMOKE_READINESS.md)
2. `pilot/scripts/smoke_preflight.py` → detached smoke → `pull_run_artifacts.py` → `smoke_verify_artifacts.py`
3. If pass and objective unchanged: `launch_pilot_matrix.sh` per [`PILOT_REDESIGN.md`](operations/PILOT_REDESIGN.md)

---

## Doc map

| Need | Doc |
|------|-----|
| Project narrative | [`nancy_explore/narrative/context.md`](../../nancy_explore/narrative/context.md) |
| Timeline | [`nancy_explore/narrative/timeline.md`](../../nancy_explore/narrative/timeline.md) |
| Run next (when cleared) | [`PILOT_REDESIGN.md`](operations/PILOT_REDESIGN.md) |
| First matrix postmortem | [`MAIN_RUNS_PLAYBOOK.md`](operations/MAIN_RUNS_PLAYBOOK.md) |
| Incidents | [`README.md`](README.md) |
