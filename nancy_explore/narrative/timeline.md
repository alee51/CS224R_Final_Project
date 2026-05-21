# Project timeline — minority voting (CS224R Spring 2026)

Canonical chronology for the team, agents, and TA discussions.  
**Last updated:** 2026-05-20.

For narrative context see [`context.md`](context.md). For tomorrow’s office-hours brief see [`briefs/ta_office_hours_20260521.md`](briefs/ta_office_hours_20260521.md). For strategic uncertainty see [`briefs/pilot_strategy_20260520.md`](briefs/pilot_strategy_20260520.md).

---

## Phase 0 — Proposal and mentor alignment

| Date | Event | Doc / artifact |
|------|--------|----------------|
| Early Spring | Team selects mentor pitch: majority vs minority voting set-RL; generalization on hard reasoning sets | [`context.md`](context.md) (verbatim pitch) |
| Pre-05-07 | Submit **QC-GRPO** proposal (quantile-conditioned GRPO) | [`../reference/proposal_qc_grpo_v1.txt`](../reference/proposal_qc_grpo_v1.txt) |
| **2026-05-07** | Meeting with Ifdita: Qwen-1.7B-Base, DaPO ~17k, ~400 steps, lightweight VeRL, Pass@k train, Cover@τ eval | [`../reference/mentor_meeting_20260507.md`](../reference/mentor_meeting_20260507.md) |

---

## Phase 1 — Exploration (Poly-EPO lane → pivot)

| Date | Event | Doc / artifact |
|------|--------|----------------|
| ~05-18 | **Kill QC-GRPO** — binary RLVR collapses quantile knob | [`../archive/poly_epo/findings.md`](../archive/poly_epo/findings.md) D1 |
| **2026-05-18** | **Stop Poly-EPO scaling/schedule** as project center; simulations complete (B+ ceiling, TA overlap) | [`decisions.md`](decisions.md), [`../archive/poly_epo/why_stop.md`](../archive/poly_epo/why_stop.md), [`../archive/poly_epo/simulation_results.md`](../archive/poly_epo/simulation_results.md) |
| Exploration | Agent design-space + 7 depth directions; lock **Tier 1 pilot matrix**: Run0 + GRPO + `inverse_freq` + F-GRPO | [`../agents/outputs/final_decision.md`](../agents/outputs/final_decision.md) |
| **2026-05-18** | Pilot scope frozen **v1.1.0** | `pilot/preflight_lock.json`, `pilot/docs/decisions/decision_memo.md` |

**Direction chosen in exploration:** “Kill the LM judge” — cheap exact-match clustering instead of Poly-EPO’s Qwen judge; test `inverse_freq` vs GRPO vs F-GRPO on DaPO 3k.

---

## Phase 2 — First pilot (Modal, 5-run matrix)

| Date (UTC unless noted) | Event | Outcome |
|-------------------------|--------|---------|
| **2026-05-19 AM** | GRPO OOM debug, smoke attempts | Fixed micro-batch backward; incidents `0519-11`–`12` |
| **2026-05-19 ~19:02** | Detached launch: `run0_proxy` + `run1` + `run1b` + `run2` + `run3` (100 steps each) | [`matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md`](../pilot/artifacts/matrix_logs/20260519T190158Z_LAUNCH_MANIFEST.md) |
| **2026-05-19 19:02 → 2026-05-20 01:23** | **Run 0 completes** — 500 prompts × 8 rollouts, proxy only | [`pilot/artifacts/run0_proxy/20260519T190202Z/`](../pilot/artifacts/run0_proxy/20260519T190202Z/) |
| **2026-05-19 PM** | `run1_grpo`: **one** full training step (~99 min); preemption into step 2; preds wiped on restart | `pilot/docs/incidents/0519-14_*` |
| **2026-05-19 PM** | Matrix **terminated** by operators (~21:48 PDT) | [`MAIN_RUNS_PLAYBOOK.md`](../pilot/docs/operations/MAIN_RUNS_PLAYBOOK.md) |
| **2026-05-19** | `run1b`/`run2`/`run3`: at most step-1 rollout build; **no completed training comparisons** | [`0519_perf_consolidated.md`](../pilot/docs/analysis/0519_perf_consolidated.md) |

**Run 0 scientific headline:** `minority_correct_prompt_rate = 0` under answer-only clustering; strong wrong-answer diversity; parser/canonicalization issues. Handoff: [`RUN0_HANDOFF_FOR_REVIEW.md`](../pilot/artifacts/run0_proxy/20260519T190202Z/RUN0_HANDOFF_FOR_REVIEW.md).

**Infra headline:** ~$1,275 projected vs ~$210 intent; no resume/checkpointing; logging gaps; `canonicalize_answer` bugs.

---

## Phase 3 — Redesign (infra + Stage 1 spec)

| Date | Event | Doc |
|------|--------|-----|
| **2026-05-19** | **Stage 1 redesign** drafted: 3 runs (`run1`–`run3`), $50/run, ~25 steps, smoke gate, checkpointing, wandb | [`PILOT_REDESIGN.md`](../pilot/docs/operations/PILOT_REDESIGN.md) |
| **2026-05-19** | **Skip Run 0** in redesign matrix; reuse `20260519T190202Z` | [`20260519_skip_run0_stage1_redesign.md`](../pilot/docs/decisions/20260519_skip_run0_stage1_redesign.md) |
| **2026-05-19** | Personal Modal workspaces (no shared team profile) | [`decisions.md`](decisions.md) |
| **2026-05-20 ~05:22 UTC** | Four **smoke** detached launches (config snapshots locally; training artifacts **not pulled**) | `pilot/artifacts/smoke/20260520T052*_Z/` |

**Status as of 2026-05-20:** Infra redesign is implemented on paper and in code; **smoke gate not verified locally**; **3-run matrix not launched** under redesign.

---

## Phase 4 — Strategic pause (open research questions)

| Date | Event | Doc |
|------|--------|-----|
| **2026-05-20** | Critique: mentor pitch admits multiple formalizations; **`inverse_freq` should not run as written**; split **Q I** (LM judge ablation) vs **Q II** (minority vs majority set-RL); recommend short mentor sync before more GPU spend | [`briefs/pilot_strategy_20260520.md`](briefs/pilot_strategy_20260520.md) |

**Current uncertainty (not yet resolved in ops docs):**

- Headline experiment: cheap-substrate Poly-EPO replication (**Q I**) vs minority-voting set objective (**Q II**)?
- If Q II: which `f_minority` — `worst_subset`, smallest-cluster reward, or other?
- Is Run 0’s 0% minority-correct gate a **dead proxy** or a **broken metric**?

`PILOT_REDESIGN.md` still describes the `inverse_freq` 3-run matrix; treat it as **implementation-ready but research-pending** until the team or mentor confirms the objective.

---

## What we have vs what we need

| Have | Need |
|------|------|
| Complete Run 0 proxy + analysis | Chosen formal objective aligned with mentor pitch |
| Failed/partial matrix training (≤1 GRPO step) | Smoke pass + matrix **or** smaller pre-registered comparison |
| Redesign spec + code path | Explicit decision: run redesign as-is vs pivot objective |
| Simulations + exploration depth docs | Milestone narrative tying experiment → mentor pitch |

---

## Quick links by role

| Role | Read first |
|------|------------|
| TA / mentor | [`briefs/ta_office_hours_20260521.md`](briefs/ta_office_hours_20260521.md) |
| Operator | [`pilot/docs/STATUS.md`](../pilot/docs/STATUS.md), [`SMOKE_READINESS.md`](../pilot/docs/operations/SMOKE_READINESS.md) |
| New agent | [`context.md`](context.md) → this file → [`briefs/pilot_strategy_20260520.md`](briefs/pilot_strategy_20260520.md) |
| Incident history | [`pilot/docs/README.md`](../pilot/docs/README.md) (`incidents/0519-*`) |
