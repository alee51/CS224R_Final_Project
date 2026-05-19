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
  methods. Analysis in `nancy_explore/findings.md`.
- The team met with Ifdita on 2026-05-07. Her concrete steer
  (`nancy_explore/ifdita_meeting_transcript.md`): stay on minority voting,
  work at a lightweight scale — Qwen-1.7B-Base, DaPO ~17k, ~400 training
  steps / 1 epoch, custom lightweight VeRL rather than her full stack,
  Pass@k for training, Cover@tau for evaluation on AIME-25, AIME-26,
  Beyond-AIME, HMMT.
- A separate deep dive into Poly-EPO scaling/scheduling concluded the team
  should **not** try to replicate or extend Poly-EPO directly
  (`nancy_explore/why_stop_poly_epo.md`, `nancy_explore/decisions.md`).
  Poly-EPO remains the closest related work and primary baseline, but it
  is not the project. The project is minority voting.
- Implementation state: nothing built. No training loop, no weights
  pulled, no eval pipeline, no VeRL-style trainer. Anastasia has begun
  exploratory tooling around Cover@tau evaluation; treat this as a
  candidate primitive, not a locked-in dependency. The team is in the
  "decide and build" phase.
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
1. `nancy_explore/ifdita_meeting_transcript.md` — the mentor's actual
   steer. Highest priority context.
2. `nancy_explore/proposal.txt` — the killed QC-GRPO proposal. Tells you
   what's already been ruled out.
3. `nancy_explore/findings.md` — synthesis of 12 candidate research
   directions with explicit verdicts. Do not redo this analysis; build
   on it.
4. `nancy_explore/why_stop_poly_epo.md` — why we are not extending
   Poly-EPO directly. Read so you don't re-propose that.
5. `nancy_explore/simulation_results.md` — gradient mechanics for the
   Poly-EPO objective; useful set-RL background.
6. `nancy_explore/poly-epo paper.pdf` — the mentor's paper. The
   mentor will mentally compare any proposal to this; the agent should
   too.
