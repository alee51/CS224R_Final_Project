# Remaining work

Due dates, what must exist at each, and who owns it. No day-by-day schedule — see [`STATUS.md`](STATUS.md) for stage checklist and [`verl_migration_plan.md`](verl_migration_plan.md) for bring-up gates.

---

## Current state (Mon 2026-06-01)

All three Stage 8 arms are launched and training on Modal. Live W&B workspace:
<https://wandb.ai/224r-project/cs224r-minority-voting/workspace?nw=vqymgsruo5>

| Arm | Run ID | Avg step time¹ | ETA (400 steps) |
|-----|--------|----------------|------------------|
| GRPO | `rof8t8kf` | 157 s/step | **Mon 6/1 ~6:15 PM PT** |
| minority_cot | `yfpxs7wo` | 213 s/step | **Tue 6/2 ~12:30 AM PT** |
| poly_epo_cot | `m29o33k1` | 214 s/step | **Tue 6/2 ~12:45 AM PT** |

¹ Average over completed steps excluding step 1 (cold start: model load + compile, ~5–9 min).

**Slack to print deadline:** ~15–16 h between slowest training finish and Tue 4 PM print. Tight for full eval panel + poster fill — evals must launch as soon as each final ckpt lands, not batched.

---

## Mon 2026-06-01 evening → Tue 2026-06-02 ~1 AM PT — training finishes

| Role | Deliverable |
|------|-------------|
| **Code** | Monitor W&B: watch for crash / `ppo_kl` blow-up / pass@8 stall on the v3 arms. Relaunch only if an arm dies before step ~350; otherwise let them run to 400. Document final ckpt paths per arm as they land. |
| **Poster** | Non-blocked sections finalized (intro, 1.7B background table, diagnosis, method draft for CoT + 4B). Drop in W&B training curves (reward, response length, ppo_kl) as soon as they look representative. |
| **Eval / ops** | Eval yamls pre-filled with ckpt paths the moment each arm finishes — **do not wait for all three**. Same panel as 5/28: Polaris 2k, DAPO 2k, MATH-500 @ n=16. Modal credits / W&B project confirmed. Judge 50-example agreement done or queued. |

---

## Tue 2026-06-02 4:00 PM — print

| Role | Deliverable |
|------|-------------|
| **Code** | Eval/debug support only; no new scope. FSDP→eval shim in `main/eval/` only if a ckpt fails to load. |
| **Poster** | **Finished poster.** Narrative locked from eval outcomes (see STATUS poster table). Collapse / minority-loses framing pre-drafted so it can be dropped in fast. |
| **Eval / ops** | **Evals finished.** Tables/figures into poster. Initial results summary: arm ranking, 1–2 bullet headline, collapse notes if minority loses GRPO. 51K pass-rate histogram if merge completed. |

---

## Wed 2026-06-03 9:00 AM — poster due

Submit. No new experiments.

---

## Role reference (steady-state)

**Code:** all `main-verl/` implementation; stage gates; Stage 8 one arm per Modal account (see migration plan §7); FSDP→eval shim in `main/eval/` if needed. **Past launch, code's job is babysitting W&B and pinning ckpt paths — the critical path is now eval + poster.**

**Poster:** writing + figures; non-blocked sections done now; final content fills in **as evals land** Tue morning (whole team Mon evening → Tue).

**Eval / ops:** launch and finish checkpoint evals the moment each arm hits its final ckpt; W&B monitoring; judge QA; optional mid-run evals; failure/collapse analysis if minority does not beat GRPO (`main/scripts/analyze_aime_rollouts.py` patterns).
