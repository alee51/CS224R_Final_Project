# pilot/docs/

**Read first:** [`STATUS.md`](STATUS.md) — operational + research status (2026-05-20).

**Research:** Stage 1 objective lineup is **pending mentor alignment**; do not assume the `inverse_freq` 3-run matrix is final. See [`../../nancy_explore/narrative/briefs/ta_office_hours_20260521.md`](../../nancy_explore/narrative/briefs/ta_office_hours_20260521.md).

**History:** First matrix (2026-05-19) failed structurally; Run 0 complete. Timeline: [`../../nancy_explore/narrative/timeline.md`](../../nancy_explore/narrative/timeline.md).

**Implementation (when cleared):** [`operations/PILOT_REDESIGN.md`](operations/PILOT_REDESIGN.md) — infra spec (caps, smoke, checkpointing); objective may change.

---

Documentation for the small-compute RL pilot. Organized by *kind*, not chronology.

## Layout

- `incidents/` — postmortems and debug logs from things that went wrong. One file per incident, named `MMDD-HH_short-slug.md` (month-day-hour, since the pilot spans only a few days).
- `operations/` — how to run the pilot: `RUNBOOK.md` (frozen scope), `MAIN_RUNS_PLAYBOOK.md` (first-matrix post-mortem), `PERSONAL_WORKSPACE_COLLAB.md` (personal Modal profile ops).
- `decisions/` — decision memos and execution plans. Things we chose and why.

## Current contents

### Incidents
- `incidents/0519-11_grpo-smoke-debug-history.md` — T1–T9 OOM test ledger + detach/cancellation (R1–R5)
- `incidents/0519-12_grpo-oom-root-cause.md` — memory math + tiered fix list (the diagnosis)
- `incidents/0519-13_progress-log-milestone-misfire.md` — `done % 25 == 0` never fires with `rollout_micro_batch_size=8`
- `incidents/0519-14_main-run-preemption-no-resume.md` — Modal preemption + no resume; preds wiped (~1h/step lost on run1)
- `incidents/0519-21_run0-silent-rollout-progress-investigation.md` — GPU active but no logs (silent `generate`, not broken handlers)
- `incidents/0519-22_main-matrix-operator-notes.md` — detached launch ops: timing, mid-run pull, containers, “App completed” trap
- `incidents/0519-23_per-app-gpu-chart-spike.md` — per-run dashboard 1→2 GPUs = preemption overlap (not 2× in one job)
- `incidents/0519-24_modal-observability-budget-gaps.md` — billing API vs UI metrics, no wandb, YAML budget gaps
- `incidents/0519-25_blocking-launch-client-abort.md` — first Run0 killed by blocking client; use `--detach` + spawn

### Operations
- **`operations/PILOT_REDESIGN.md`** — **infra / runbook** — caps, smoke, checkpointing (objectives may change; see `STATUS.md`)
- `operations/PERSONAL_WORKSPACE_COLLAB.md` — personal Modal profile: launch, pull, wandb sync, sharing weights
- `operations/RUNBOOK.md` — *historical* — original frozen pilot scope (orchestrator-owned)
- `operations/MAIN_RUNS_PLAYBOOK.md` — *post-mortem* — first matrix lessons (workspace §2 superseded 2026-05-19)

### Decisions
- `decisions/20260519_skip_run0_stage1_redesign.md` — Stage 1 matrix excludes `run0_proxy`; use `20260519T190202Z` artifacts
- `nancy_explore/narrative/decisions.md` — canonical decision log (incl. 2026-05-19 personal Modal workspace)
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
