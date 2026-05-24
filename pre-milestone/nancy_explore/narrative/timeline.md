# Project timeline — minority voting (CS224R Spring 2026)

Canonical chronology for the team, agents, and TA discussions.  
**Last updated:** 2026-05-21.

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

**Run 0 scientific headline:** `minority_correct_prompt_rate = 0` under answer-only clustering; ~14.5% under LLM reasoning clusters (cleaned labels). Handoff (historical): [`pilot/docs/archive/RUN0_HANDOFF_FOR_REVIEW.md`](../../pilot/docs/archive/RUN0_HANDOFF_FOR_REVIEW.md). Analysis writeup: [`../run0_analysis/run0_exec_plan.md`](../run0_analysis/run0_exec_plan.md).

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
| **2026-05-21** | Human labels promoted to `cleaned_answers.parquet`; v2-based A/B/C/D archived. | [`../run0_analysis/README.md`](../run0_analysis/README.md) |
| **2026-05-21** | **Run 0 offline analysis complete** (E1 set-score sim on cleaned labels; eval floor Pass@1 9.03%, Pass@8 34.4%). **Training direction:** test **minority-voting set-RL** (reward the rarest mode in each 4-rollout subset → marginal advantages), not the redesign `inverse_freq` / Poly-EPO diversity matrix as the headline. Tie-break: random. **Defer to training:** answer-hash vs LLM-CoT minority (`ans-rand` vs `cot-rand`). | [`../run0_analysis/run0_exec_plan.md`](../run0_analysis/run0_exec_plan.md), [`../run0_analysis/analysis_c/set_score_simulation.md`](../run0_analysis/analysis_c/set_score_simulation.md) |

### Open issue (2026-05-21): human labels from 0a never wired into A/B/C/D

The 4000-row blind-A/B + dispute-resolved human labels from §0a were referenced only by an audit script (now archived at `nancy_explore/run0_analysis/archive/2026-05-21_pre_human_label_audit/prereq_0b_reparse/audit_1024_token_labels.py`). Analyses A/B/C/D all read `data/predictions_reparsed.jsonl` — pure v2-parser output, no human cross-check. Treat all current Analysis A/B/C/D numbers as **provisional pending parser validation**.

**Update (later 2026-05-21):** v2-vs-human comparison showed only **78% presence agreement** (3123/4000), with `last_line` extraction path at 0.28% agreement (effectively useless). Both parsers (v1, v2) were judged not good enough; human labels are now canonical. New artifact: `nancy_explore/run0_analysis/data/cleaned_answers.parquet` (4000 rows, schema documented in `run0_analysis/README.md`). Old analyses archived to `archive/2026-05-21_pre_human_label_audit/`. Headline numbers under cleaned labels: Pass@1 **9.03%**, Pass@8 **34.40%** (vs v2's 8.25% / 33.00%).

Concrete gaps:

1. **No v2-parser-vs-human agreement number exists.** Every Analysis A/B/C/D headline depends on v2 being roughly right, but its accuracy has never been measured against the human labels.
2. **Analysis B's substrate table is missing a `human_tail` row.** Per design doc framing, the human labels were *the* substrate that captures "did the model state an answer? vs runon." Without it, B compares parser-vs-LLM and feature-vs-LLM but skips human-vs-LLM — the strongest cheap substrate available.
3. **A.7.3 degenerate sanity used pattern-counts (9.2–28.5% qual tags) instead of the human "runon"/"no_answer" labels** that would directly anchor the 16.95% LLM-degenerate rate.

Fix plan (three small jobs, none requires re-running the LLM judge or re-embedding): parser-vs-human validation report, add `human_tail` as substrate in B and re-run aggregates, LLM-degen × human-runon cross-tab for A.7.3.

**Not needed as separate prereq work (2026-05-21):** The three fix-plan bullets above (parser-vs-human report, `human_tail` substrate row, LLM-degen × runon cross-tab) are **not** blocking — human labels are already canonical via `cleaned_answers.parquet`. Phase 2B/C/D in [`../run0_analysis/run0_analysis_plan.md`](../run0_analysis/run0_analysis_plan.md) subsumes the substrate and metric refresh on cleaned labels.

---

**Resolved (2026-05-21):** Headline objective is **minority-voting set-RL** (`f(G)` = reward of rarest mode in subset G). Run 0 analysis settled tie-break (random), distinct from Poly-EPO/GRPO, and enough per-rollout signal (~34%). Cheap substrates do not replace LLM CoT clusters.

**Still open at training time:** `ans-rand` vs `cot-rand` (same answer vs LLM reasoning bucket for “minority”). `PILOT_REDESIGN.md` `inverse_freq` matrix is legacy infra spec, not the chosen research arm.

---

## What we have vs what we need

| Have | Need |
|------|------|
| Complete Run 0 proxy + offline analysis (human labels) | Training runs with minority set-RL |
| E1 objective comparison + eval floor metrics | Compare `ans-rand` vs `cot-rand` in training |
| Failed/partial matrix training (≤1 GRPO step) | Smoke pass + training launch |
| Redesign spec + code path | Wire minority `f(G)` into train loop |

---

## Quick links by role

| Role | Read first |
|------|------------|
| TA / mentor | [`briefs/ta_office_hours_20260521.md`](briefs/ta_office_hours_20260521.md) |
| Operator | [`pilot/docs/STATUS.md`](../pilot/docs/STATUS.md), [`SMOKE_READINESS.md`](../pilot/docs/operations/SMOKE_READINESS.md) |
| New agent | [`context.md`](context.md) → this file → [`briefs/pilot_strategy_20260520.md`](briefs/pilot_strategy_20260520.md) |
| Incident history | [`pilot/docs/README.md`](../pilot/docs/README.md) (`incidents/0519-*`) |
