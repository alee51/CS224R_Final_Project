# Current Final Take

Candid assessment of where the Poly-EPO direction stands as a CS224R quarter
project, after the analysis in [`../archive/poly_epo/findings.md`](../archive/poly_epo/findings.md) and the simulations in
[`../archive/poly_epo/simulation_results.md`](../archive/poly_epo/simulation_results.md).

## Verdict: the current direction is mid

The annealed-schedule project (D11 in `../archive/poly_epo/findings.md`) is real research, not
nothing — but it's probably a B+ project, not an A. The reasons are concrete:

1. **It's a hyperparameter-study at heart.** "Anneal $n$ over training" is a
   schedule on an existing knob. That's a real contribution, but the kind that
   ends up in an ablation table of someone else's paper, not the headline of
   yours.

2. **The TA is one of the paper's authors.** Ifdita Hasan Orney is on both
   the Poly-EPO paper and the proposal as project mentor. This raises the
   bar significantly. Anything obvious — annealing $n$, annealing $N$,
   comparing $f_{poly}$ variants — they have either already tried, considered
   and rejected, or have on a list. The "scaling laws are open" line in the
   conclusion is a real invitation, but they're inviting *interesting*
   answers, not "we ran the obvious sweep." First TA question on a D11
   pitch is going to be "did you try X?" where X is something they thought
   of in week 2 of writing the paper.

3. **The validation will be on toy domains.** Multi-digit multiplication and
   polynomial solving are constructed to make Poly-EPO look good. A schedule
   that works on those doesn't tell you much. The real benchmarks (AIME,
   HMMT, BeyondAIME) require Qwen3-4B-Base + 850 training steps + 4×H200s.
   Almost certainly out of reach for a quarter project.

4. **Sweep 2 is interesting but small.** "$n/N$ isn't scale-invariant
   because of LLN" is real, but it's the kind of thing that gets a paragraph
   in someone's appendix, not a paper. Anyone who stares at $\bar r \cdot |U|/n$
   for ten minutes can predict it.

5. **D12 (the diversity-preservation hypothesis) is the actual interesting
   question** — but the experimental answer is essentially "train Poly-EPO,
   switch to GRPO, watch a metric." That's an experiment, not a research
   project.

## What would actually be interesting

The pattern to look for: a contribution that **changes a load-bearing
assumption**, not one that tunes around it. In rough order of how compelling
they'd be:

1. **Kill the LM judge.** Poly-EPO requires Qwen3-4B-Instruct or Gemini-Flash
   as a clustering judge — huge dependency, expensive at training time,
   brittle, and the paper's own limitations section calls out reward-hacking
   risk. Replace it with: sentence embeddings + clustering, final-answer
   matching for verifiable tasks, token-level n-gram fingerprints, or a
   self-judge from the training model itself. If you can show comparable
   Pass@k gains without the judge, that's a genuine contribution. "We made
   Poly-EPO 5× cheaper without losing the gains" is a real abstract.

2. **Investigate the LM-judge as a failure mode.** The paper itself calls
   out reward-hacking via judge manipulation. Demonstrate this empirically —
   find prompts where the policy learns to game the judge — then propose a
   fix. "This method has a bug, here's how to plug it" papers always read
   well.

3. **Propose a different set objective.** $f_{poly} = \bar r \cdot d$ is one
   composition out of many. Why not max, geometric mean over corrects,
   or a quantile? Each induces a different marginal advantage. This is
   closer to D5 in `../archive/poly_epo/findings.md` but framed as a *principled* search for
   the right objective ("the objective should give zero advantage to
   redundant-correct exactly when..."), not just an ablation.

4. **Extend Poly-EPO to a new domain where diversity matters in the wild.**
   Code generation (multiple valid implementations), theorem proving
   (multiple proof tactics), creative writing with constraints. The
   synthetic domains validate the *method*; a real domain validates the
   method's *claim of generality*.

5. **Switch papers entirely.** PKPO (Pass@k Policy Optimization), F-GRPO,
   SetPO, ProRL, the count-based exploration paper — all recent, all have
   visible open questions, none have your TA as an author. Lower bar to
   clear, more degrees of freedom, less risk of the TA already having
   thought of your contribution.

## The honest recommendation

If you stay on Poly-EPO, push toward **(1) or (2)** — a contribution that
fixes a real weakness, not one that tunes a knob. The annealing schedule
sketched in `../archive/poly_epo/findings.md` should become an *appendix experiment* inside that
bigger story, not the main contribution. The "kill the LM judge" angle is
particularly attractive because:

- It addresses an actual practical pain point (compute, latency, dependency).
- The paper authors flag it as a known limitation, so it's a sanctioned
  direction rather than an obvious follow-up they'd have already tried.
- It produces a concrete, marketable result if it works ("Poly-EPO without
  the judge").
- It produces a useful negative result if it doesn't ("the judge is
  load-bearing in a non-obvious way").

If you're not committed to Poly-EPO, **seriously consider switching papers.**
The TA-is-an-author dynamic is a real handicap. PKPO is the most natural
target — clean open questions (fixed $k$, non-adaptive schedules, no
covariance analysis), no Stanford-faculty authors to compete with for ideas,
and the original QC-GRPO proposal was already pointed in that direction
before we shot it down for binary-reward reasons (which a non-quantile angle
on PKPO would sidestep).

## What to decide next

Concrete forks in the road, in priority order:

1. **Stay on Poly-EPO, schedule project (current direction).** Floor: B+
   project, real but unmemorable. Ceiling: B+ project, real but
   unmemorable. Low risk, low ceiling.

2. **Stay on Poly-EPO, "kill the judge" project.** Floor: workshop-quality
   negative result if it fails. Ceiling: a clean engineering contribution
   that gets cited. Medium risk, medium-high ceiling.

3. **Switch to PKPO or another adjacent paper.** Floor: depends on what
   open question we identify. Ceiling: higher than Poly-EPO because the
   competitive landscape is friendlier. Medium risk, higher ceiling, but
   throws away the analysis we've already done.

The tiebreaker is: **what would you most enjoy working on for ten weeks?**
The schedule project gets you to a B+ paper without much pain. The "kill
the judge" project is harder but actually has a shot at being interesting.
A pivot is the boldest move and might be the right one given everything
above.
