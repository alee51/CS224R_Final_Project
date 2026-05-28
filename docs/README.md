# Project docs (repo root)

Cross-cutting documentation that spans `main/` (custom trainer) and `main-verl/` (VeRL reimplementation).

| Doc | Contents |
| --- | --- |
| [`verl.md`](./verl.md) | VeRL overview: what it is, built-ins, slowdowns, custom wiring (judge), B200, multi-GPU knobs, Modal limits, migration sketch |
| [`verl_migration_plan.md`](./verl_migration_plan.md) | Staged migration plan: gates, parity smoke, minority-objective port (Stage 3 deep-dive), Polaris-53K vs 51K decision, Modal credit allocation across 3 accounts |

Related docs elsewhere in the repo:

| Path | Contents |
| --- | --- |
| [`main/docs/verl_move_ta_meeting.md`](../main/docs/verl_move_ta_meeting.md) | Raw TA meeting notes (2026-05-28) |
| [`main/docs/ta_discussion.md`](../main/docs/ta_discussion.md) | Poster framing, Path A/C/D, eval results |
| [`main/docs/probes/prompt_extraction_research.md`](../main/docs/probes/prompt_extraction_research.md) | VeRL prompt ↔ parser ↔ reward pairing research |
| [`main-verl/README.md`](../main-verl/README.md) | Planned layout for the VeRL codebase |
| [`main/docs/PLAN.md`](../main/docs/PLAN.md) | Original experiment plan (custom trainer architecture) |
