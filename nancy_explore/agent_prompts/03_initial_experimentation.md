# Stage 3: Initial Experimentation

## How to use this file
Fill in `{DIRECTION_NAME}` and `{DIRECTION_SLUG}` below. Spawn one
agent per direction that survived stage 2 with a positive verdict
(typically 1-3 directions). Each produces a runnable code artifact and
an empirical report.

---

## Mission

You are running a minimal experiment to validate one minority-voting
direction empirically. The direction has already passed depth evaluation
in stage 2 (see
`nancy_explore/agent_outputs/02_depth_{DIRECTION_SLUG}.md`). Read that
file, the design-space map at
`nancy_explore/agent_outputs/01_design_space.md`, and
`nancy_explore/context.md` before anything else.

Your job is to produce running code, a tiny training run, and an honest
empirical signal about whether this direction is implementable and
behaves as theorized. You are NOT trying to produce the project's final
result. You are trying to discover, cheaply, whether the next two weeks
of work on this direction would hit a wall.

## The direction to experiment on

**Name:** {DIRECTION_NAME}
**Slug:** {DIRECTION_SLUG}
**Depth eval:** `nancy_explore/agent_outputs/02_depth_{DIRECTION_SLUG}.md`

## What to produce

### Code artifacts (in `nancy_explore/experiments/{DIRECTION_SLUG}/`)
- `loss.py` — standalone PyTorch implementation of the proposed
  training objective from section 4 of the depth eval. Should be
  runnable, unit-tested with a fake batch, and ~< 200 lines. No VeRL
  dependency.
- `run.py` — minimal training loop. Tiny model (Qwen-0.5B or even
  smaller, e.g. Qwen2.5-0.5B-Instruct, or a HuggingFace tiny LM if no
  GPU is available), 50-200 training steps, DaPO ~100-prompt slice (or
  synthetic prompts if DaPO not pulled yet). Use whatever is cheapest
  to get signal.
- `sanity.py` — checks: gradient is finite, loss decreases (or
  doesn't, with a documented reason why), policy entropy doesn't
  collapse to 0, within-group cluster count over training, comparison
  against vanilla GRPO loss on the same fake batch.
- `README.md` — how to run, what to expect, what each script does, how
  much it cost.

### Empirical report (`nancy_explore/agent_outputs/03_experiment_{DIRECTION_SLUG}.md`)
- **What was actually run**: model, data slice, steps, GPU class (if
  any), wall-clock time, total cost in $.
- **Headline plot**: one figure (loss curve, or diversity-over-training,
  or within-group reward distribution shift). Save as PNG in the
  experiment folder, reference inline.
- **Signal observed**: does the proposed objective do what it's
  supposed to do at this scale? Make specific quantitative claims, not
  vibes.
- **Surprises**: anything unexpected. This is the most important
  section. Be honest.
- **Implementation gotchas**: anything that took >30 minutes to figure
  out and is worth documenting for the team's full implementation.
- **Updated verdict**: does the empirical signal confirm, weaken, or
  contradict the stage 2 verdict? One paragraph. Be explicit if your
  empirical result changes the recommendation.

## Constraints
- **Compute budget**: <$30 of Modal credit total per direction. If you
  cannot get useful signal under that cap, write that explicitly and
  stop — do not silently exceed.
- **Wall-clock**: aim for <2 hours.
- Do not hyperparameter-tune. Pick defaults from the closest
  literature, run once, report what happened.
- Negative results are valuable. If the loss diverges, the objective is
  degenerate, or gradients are zero, that is the answer — write it up
  cleanly.
- Do not pretend the experiment validates more than it does. A
  200-step run on a 0.5B model with 100 prompts is not evidence about
  Qwen-1.7B on AIME. State the inferential gap explicitly.
- The training pipeline you build here is throwaway scaffolding. It
  does NOT need to be the team's eventual lightweight VeRL trainer.
  Optimize for "get signal in 2 hours," not for code reuse.

## Tone
Empirical, terse, no spin. The reader is using this to decide whether
to commit a week of project time to building this direction out fully.
