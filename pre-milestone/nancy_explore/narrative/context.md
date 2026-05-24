# Project Context — Minority Voting Optimization for LLM Reasoning

## Team and class
- Three-person team: Nancy Bao, Anastasia Lee, Emma Gao.
- Class: CS 224R (Deep Reinforcement Learning), Stanford, Spring 2026.
- Officially assigned project mentor: Ifdita Hasan Orney. She is both the
  TA who proposed the project (selected from a list of TA-pitched ideas)
  and an author of the Poly-EPO paper, which is the closest related work.
  She is the sole grader of the milestone and final report.

## The project, as originally posted by the mentor

Verbatim text of the project pitch the team selected:

> Majority voting is a powerful test-time compute method in RLVR. At a
> high-level, the model first generates a set of solutions per problem and
> then selects the final answer that majority of the generations outputted.
> One can train a model to optimize this objective using set RL; however,
> the objective is lenient in the sense that it rewards the model highly as
> long as majority of its answers is correct. In contrast, one can consider
> minority voting optimization where we instead reward the answer that some
> randomly chosen minority voted for. Intuitively, such a training objective
> optimizes the model to improve its worst case performance. The goal of
> this project would be to, first, instantiate training algorithms for both
> of these methods and, then, systematically study its effect on model
> generalization harder test sets for reasoning.

This is the load-bearing constraint on direction. Any research question the
project pursues should be a recognizable instantiation of "instantiate
training algorithms for majority and minority voting optimization, and
study generalization to harder reasoning test sets."

## Where the project stands
- A v1 proposal was submitted as **QC-GRPO** (quantile-conditioned GRPO).
  It has since been killed: under binary RLVR rewards the quantile knob
  collapses to a small number of discrete cases that reduce to existing
  methods. Analysis in `nancy_explore/archive/poly_epo/../archive/poly_epo/findings.md`.
- The team met with Ifdita on 2026-05-07. Her concrete steer
  (`nancy_explore/reference/mentor_meeting_20260507.md`): stay on minority voting,
  work at a lightweight scale — Qwen-1.7B-Base, DaPO ~17k, ~400 training
  steps / 1 epoch, custom lightweight VeRL rather than her full stack,
  Pass@k for training, Cover@tau for evaluation on AIME-25, AIME-26,
  Beyond-AIME, HMMT.
- A separate deep dive into Poly-EPO scaling/scheduling concluded the team
  should **not** try to replicate or extend Poly-EPO directly
  (`nancy_explore/archive/poly_epo/why_stop.md`, `nancy_explore/narrative/decisions.md`).
  Poly-EPO remains the closest related work and primary baseline, but it
  is not the project. The project is minority voting.
- **Working hypothesis (2026-05-19): "kill the LM-judge."** Keep set-RL
  structure; replace Poly-EPO's LM clustering judge with a cheap substrate
  (exact-match `\boxed{}` integers in Stage 1). Stage 1 was coded as
  GRPO vs `inverse_freq` vs F-GRPO.
- **Current uncertainty (2026-05-20):** We are **not** confident this is
  the right experiment without mentor alignment. `inverse_freq` may not
  instantiate minority voting cleanly; we may be conflating (I) judge/substrate
  ablation vs (II) minority vs majority set-RL. See
  [`narrative/briefs/pilot_strategy_20260520.md`](narrative/briefs/pilot_strategy_20260520.md) and
  [`narrative/briefs/ta_office_hours_20260521.md`](narrative/briefs/ta_office_hours_20260521.md). Full chronology:
  [`narrative/timeline.md`](narrative/timeline.md). Ops entry point:
  `pilot/docs/STATUS.md`.
- Implementation state: a first pilot was built and launched on
  2026-05-19 (5-run attempt: `run0_proxy` + four GRPO variants on Qwen-1.7B-Base
  + DaPO 3k subset). It failed structurally — cost mismatch (~$1,275 projected
  vs ~$210 budgeted), no mid-run checkpointing/resume, missing logging during
  long rollouts, broken `canonicalize_answer` parser, milestone-log math bug —
  and was killed mid-run. Only `run0_proxy` completed (`pilot/artifacts/run0_proxy/20260519T190202Z/`).
  Root causes consolidated in `pilot/docs/analysis/0519_perf_consolidated.md`.
  Redesigned **Stage 1** (`pilot/docs/operations/PILOT_REDESIGN.md`): **3-run GRPO
  matrix** (`run1`–`run3`), ~$150 matrix burst + smoke, ~25 steps/run, $50/run cap;
  Run 0 **not** re-run (`pilot/docs/decisions/20260519_skip_run0_stage1_redesign.md`).
  Time-gated checkpointing, wandb + diagnostics, mechanism + outcome decision rules.
  Smoke launched 2026-05-20; artifacts not verified locally. **Matrix
  launch paused** pending objective clarity (office hours 2026-05-21).
  Stage 1 gates a **Stage 2**
  headline run at the mentor-prescribed scale (400 steps × DaPO 17k ×
  1 epoch, 64-sample Pass@k eval on AIME-25/26 + Beyond-AIME + HMMT +
  Minerva).
- Anastasia has begun exploratory tooling around Cover@tau evaluation;
  treat this as a candidate primitive, not a locked-in dependency.
- Roles are flexible and not locked in. The proposal sketched
  Emma → theory, Anastasia → implementation, Nancy → evaluation, but in
  practice work is being shared based on what each person is currently
  picking up. Agents should not assume any specific person owns any
  specific workstream.
- The team has skimmed the obvious adjacent literature (Poly-EPO, PKPO,
  F-GRPO, SetPO, DAPO, distributional RL for LLMs, self-consistency,
  test-time-RL). The agent should not assume zero prior knowledge but
  should still evaluate research directions on their own merits rather
  than relying on what the team has or has not read.

## What "settled" looks like
The team needs a sharper, defensible research question inside the
mentor-defined umbrella. Open questions for an exploring agent to help
resolve:
- What concrete **training objective** instantiates "minority voting
  optimization"? Multiple plausible formulations exist; the team has not
  picked one and the choice is non-trivial.
- What is the strongest comparison story against **majority voting**
  (same data, same model, same compute — only objective differs)?
- What is the generalization claim being tested, on which evaluation
  sets, with which metrics, and what would falsify it?
- What is the strongest **contribution shape** at this compute scale —
  theoretical derivation, mechanistic study, curriculum/schedule,
  CoT-diversity analysis, ablation, or some combination?
- What does the project look like if the minority-voting objective does
  *not* improve worst-case generalization? i.e., what is the consolation
  contribution?

## Scope and constraints
- **Compute:** ~$1400 of Modal credits, combined across the team. The
  agent should look up Modal's current GPU pricing and translate this
  into a realistic GPU-hour budget for the appropriate GPU class, then
  pressure-test whether any proposed experimental plan fits.
- **Model scale:** Qwen-1.7B-Base is the working reference, set by the
  mentor as what's tractable at this compute. Larger models only with
  strong compute-justified motivation.
- **Data:** DaPO ~17k as the working training set. Substitutions allowed
  if motivated.
- **Evaluation:** AIME-25, AIME-26, Beyond-AIME, HMMT as the
  generalization targets. Additional reasoning evals fine to add.
- **Framework:** custom lightweight VeRL-style trainer, built from
  scratch. Not the full VeRL stack.
- **Direction lock:** the project must be a recognizable instantiation of
  the mentor's pitch above. Adjacent papers (PKPO, F-GRPO, SetPO, etc.)
  are fair game as related work, baselines, or sources of mechanism, but
  not as the project center.

## Team capability and working style
- Comfortable with AI-assisted coding, ML, and RL. Engineering throughput
  is not the bottleneck — building a custom trainer, eval pipeline, or
  ablation harness is fast. Long GPU runs still take real wall-clock
  time; code does not.
- Optimize advice for "strongest research thesis defensible at this
  scale," not for calendar pressure.
- Prefer depth in one direction over breadth across many. The exploration
  phase is intentionally short; the team is converging, not diverging.

## Existing artifacts (read in order before generating new ideas)
0. [`narrative/timeline.md`](narrative/timeline.md) — canonical project chronology.
1. [`reference/mentor_meeting_20260507.md`](reference/mentor_meeting_20260507.md) — the mentor's actual
   steer. Highest priority context.
2. [`reference/proposal_qc_grpo_v1.txt`](reference/proposal_qc_grpo_v1.txt) — the killed QC-GRPO proposal.
3. [`archive/poly_epo/findings.md`](archive/poly_epo/findings.md) — direction verdicts (archive).
4. [`archive/poly_epo/why_stop.md`](archive/poly_epo/why_stop.md) — why we are not extending Poly-EPO directly.
5. [`archive/poly_epo/simulation_results.md`](archive/poly_epo/simulation_results.md) — set-RL gradient mechanics (background).
6. [`reference/poly_epo_paper.pdf`](reference/poly_epo_paper.pdf) — the mentor's paper.
7. `pilot/docs/operations/PILOT_REDESIGN.md` — Stage 1 **infra** spec
   (2026-05-19). Research objectives may change; see `pilot/docs/STATUS.md`.
8. `pilot/docs/analysis/0519_perf_consolidated.md` — synthesis of why
   the first pilot failed. Read before proposing any change to the
   training rig.
9. `pilot/docs/analysis/0519_poly_epo_methodology.md` — extracted
   methodology from the Poly-EPO paper; documents where our pilot
   intentionally diverges from the paper.
