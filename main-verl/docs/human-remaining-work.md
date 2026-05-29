# Remaining work

Due dates, what must exist at each, and who owns it. No day-by-day schedule — see [`STATUS.md`](STATUS.md) for stage checklist and [`verl_migration_plan.md`](verl_migration_plan.md) for bring-up gates.

---

## Fri 2026-05-29 EOD — training launched

| Role | Deliverable |
|------|-------------|
| **Code** | Stages 1–7 done enough to detach **Stage 8**: three 1-epoch runs on Modal (GRPO / minority_cot / poly_epo_cot), one arm per account; judge service up; W&B streaming. Handoff with relaunch commands for Sat. |
| **Poster** | Slide skeleton; sections **not blocked by VeRL results** (intro, 1.7B background table, diagnosis, method draft for CoT + 4B). |
| **Eval / ops** | Eval yaml templates ready to fill once ckpt paths exist; Modal credits checked; W&B project names set. Mid-training eval: **nice to have** if ckpts exist; not required Fri. |

---

## Mon 2026-06-01 11:59 PM — training done

| Role | Deliverable |
|------|-------------|
| **Code** | All three arms finished 1 epoch; final checkpoint paths documented. |
| **Poster** | Same skeleton; training-curve placeholders from W&B if available. |
| **Eval / ops** | **Evals ready to launch** as soon as final ckpts land (Polaris 2k, DAPO 2k, MATH-500 @ n=16 — same panel as May 28). Mid-training evals: done if cheap; not blocking. W&B health during run (migration plan §5). Judge 50-example agreement if not done earlier. |

---

## Tue 2026-06-02 4:00 PM — print

| Role | Deliverable |
|------|-------------|
| **Code** | Support eval/debug only; no new scope. |
| **Poster** | **Finished poster** with team — mostly filled **after** training + evals; 1.7B + method sections already done. Narrative locked from eval outcomes (see STATUS poster table). |
| **Eval / ops** | **Evals finished**; tables/figures to poster; initial results summary (arm ranking, 1–2 bullet headline, collapse notes if minority loses). 51K pass-rate histogram if merge completed. |

---

## Wed 2026-06-03 9:00 AM — poster due

Submit. No new experiments.

---

## Role reference (steady-state)

**Code:** all `main-verl/` implementation; stage gates; Stage 8 one arm per Modal account (see migration plan §7); FSDP→eval shim in `main/eval/` if needed.

**Poster:** writing + figures; early work on non-blocked sections; **final content after evals materialize** (whole team Mon–Tue).

**Eval / ops:** launch and finish checkpoint evals; W&B monitoring; judge QA; optional mid-run evals; failure/collapse analysis if minority does not beat GRPO (`main/scripts/analyze_aime_rollouts.py` patterns).
