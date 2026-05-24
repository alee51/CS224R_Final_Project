# Context

Pointer file for agents and teammates. Read this first, then `PLAN.md`.

---

## Reading list (authoritative for what)


| Read                                                          | For                                                                                 |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `pre-milestone/main.md`                                       | Submitted milestone — current research framing, objective math, preliminary results |
| `pre-milestone/nancy_explore/narrative/decisions.md`          | Prior decisions (objectives, Modal workspace policy, answer grading)                |
| `pre-milestone/nancy_explore/narrative/timeline.md`           | Chronology through milestone                                                        |
| `pre-milestone/pilot/docs/analysis/0519_perf_consolidated.md` | Pilot performance findings, what's been applied, what's still open                  |
| `pre-milestone/pilot/docs/incidents/`                         | Pilot incident postmortems                                                          |
| `main/docs/PLAN.md`                                           | This project's working plan                                                         |
| `main/docs/LESSONS_FROM_PILOT.md`                             | Curated pilot lessons (to be written)                                               |


---

## Post-milestone facts (not yet in pre-milestone docs)

- **Budget confirmed:** $1,600 Modal credits. Per-person split: Nancy $650, Anastasia $475, Emma $475. Possible +$400 stretch.
- **Deadlines:** poster due 2026-06-03; internal target for experiments done 2026-05-31.
- **Rollout engine:** vLLM, in-process. Lightweight VeRL-flavored trainer, not the full VeRL stack.
- **Code ownership:** Nancy owns all `.py` for the main experiment.
- **Arm priority:** GRPO baseline (must), minority-answer (headline), minority-CoT (in scope, first cut if compute tight), Poly-EPO (stretch only).
- **Hardware:** single A100-80GB per run baseline; GPU class flexible — see PLAN.md §7.
- **Dropped from scope** (vs. earlier pilot directions): `inverse_freq`, F-GRPO, Cover@τ as training objective. 

