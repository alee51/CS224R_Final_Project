# pilot/docs/

Documentation for the small-compute RL pilot. Organized by *kind*, not chronology.

## Layout

- `incidents/` — postmortems and debug logs from things that went wrong. One file per incident, named `MMDD-HH_short-slug.md` (month-day-hour, since the pilot spans only a few days).
- `operations/` — how to run the pilot: `RUNBOOK.md` (frozen scope) and `MAIN_RUNS_PLAYBOOK.md` (pre-launch checklist + main-run migration plan).
- `decisions/` — decision memos and execution plans. Things we chose and why.

## Current contents

### Incidents
- `incidents/0519-11_grpo-smoke-debug-history.md` — T1–T9 OOM test ledger
- `incidents/0519-12_grpo-oom-root-cause.md` — memory math + tiered fix list (the diagnosis)
- `incidents/0519-13_progress-log-milestone-misfire.md` — `done % 25 == 0` never fires with `rollout_micro_batch_size=8`
- `incidents/0519-14_main-run-preemption-no-resume.md` — Modal preemption + no resume; preds wiped (~1h/step lost on run1)

### Operations
- `operations/RUNBOOK.md` — frozen pilot scope (orchestrator-owned)
- `operations/MAIN_RUNS_PLAYBOOK.md` — what-to-do-differently for main runs

### Decisions
- `decisions/decision_memo.md` — pilot decision token (currently `PENDING`)
- `decisions/training_parallelization_plan.md` — P0–P3 throughput plan
- `decisions/efficiency_parallelization_note.md` *(superseded; kept for context)*

## Writing a new incident postmortem

Filename: `MMDD-HH_<short-kebab-slug>.md` (e.g. `0520-09_volume-mount-conflict.md`). The hour is when the incident was *documented*, not when it started.

Reasonable section structure (adapt as needed; `incidents/0519-13_*` is a good template):

- Summary — one paragraph: what broke, what it affects, what it doesn't
- Symptoms — what an operator would observe
- Root cause — the actual bug or mechanism
- Evidence — terminal IDs, artifact paths, line numbers, log excerpts
- What is NOT the cause — kill plausible-but-wrong theories
- Recommended fix / Workaround
- References — code paths, commits

## Lessons (TBD)

A `LESSONS.md` distillation will land here — either appended as incidents close, or written at pilot end. Each entry: one paragraph of takeaway + link to the underlying incident. Goal: the real-implementation phase reads `LESSONS.md` alone and doesn't have to re-read every postmortem.
