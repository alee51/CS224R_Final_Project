# Stage 2: Depth Evaluator

## How to use this file
Fill in `{DIRECTION_NAME}`, `{DIRECTION_SLUG}`, and `{DIRECTION_SKETCH}`
below — these come from a specific entry in
`nancy_explore/agent_outputs/01_design_space.md`. Spawn one agent per
direction. Each agent produces an artifact in the same schema so
outputs are directly comparable.

---

## Mission

You are evaluating whether the following direction is a strong research
direction for the CS224R final project described in
`nancy_explore/context.md`. Read that file in full before anything else.
Then read the artifacts it points to, in the order it specifies. Also
read `nancy_explore/agent_outputs/01_design_space.md` for context on
which other directions are being evaluated in parallel.

## The direction to evaluate

**Name:** {DIRECTION_NAME}

**Sketch:** {DIRECTION_SKETCH}

## Required output

Produce a single markdown file at
`nancy_explore/agent_outputs/02_depth_{DIRECTION_SLUG}.md` with the
following sections, in this exact order, with these exact headings:

### 1. Direction restatement (<=150 words)
Restate the direction in your own words, framed against the mentor's
original pitch (majority/minority voting optimization, generalization to
harder reasoning test sets). If you cannot restate it as a recognizable
instantiation of that pitch, STOP, write `OFF-DIRECTION`, and explain
why.

### 2. Related work scan (6-12 papers, last ~18 months)
For each: arXiv ID, one-line claim, one-line delta vs this direction.
Prioritize work the mentor has likely already considered: Poly-EPO,
PKPO, F-GRPO, SetPO, distributional RL for LLMs, diversity-preserving
RLVR, self-consistency variants, test-time-RL. Do not invent citations;
verify arXiv IDs with web search.

### 3. Novelty check
- The specific scientific claim this direction would make.
- The closest existing work and how this direction is different.
- Steelman the mentor's pushback: write the 3-sentence objection Ifdita
  (Poly-EPO author) would give at the next meeting.
- Novelty rating: {high / medium / low / already-done}.

### 4. Concrete training objective
Write the actual training loss in math notation. Be specific about the
per-sample advantage, the baseline, the per-batch aggregation, and how
gradients flow. If you cannot write the loss precisely in <20 lines of
math, the direction is underspecified — say so explicitly.

### 5. Experimental plan
- The minimal experiment that produces the headline figure.
- Baselines: at minimum {standard GRPO, majority-voting-trained model}.
  What else?
- Ablations needed to make the claim defensible.
- Compute estimate in Modal GPU-hours: look up current Modal pricing,
  justify the GPU class, multiply through. Does the plan fit in $1400?
- Which evaluation produces the headline number (Pass@k, Cover@tau,
  worst-case-subset accuracy, CoT-diversity score, something new).

### 6. Failure modes and consolation result
- Three concrete ways this direction can fail empirically. For each,
  what specifically would you observe?
- If the main hypothesis is null, the **minimum publishable result**
  that still produces a defensible final paper. Be specific — name the
  figure or table.

### 7. Killer experiment
The single result that, if positive, makes the project worth writing as
a final paper. One sentence and one sentence describing the expected
figure.

### 8. Overall verdict
- Rating: {strong / promising / weak / off-direction}.
- "Worth running stage 3 (initial experimentation) on?" {yes / no /
  conditional on X}.
- One-paragraph honest summary of why.
- Be calibrated — a boring tractable direction is worse here than a
  harder higher-ceiling one. Don't recommend just because it's safe.

## Constraints and anti-patterns

- DO NOT re-derive analysis already in `nancy_explore/findings.md`.
  Cite it and move on.
- DO NOT propose Poly-EPO scaling laws, $(N, n)$ schedules, or
  annealing schedules over Poly-EPO knobs — explicitly ruled out in
  `nancy_explore/why_stop_poly_epo.md`.
- DO NOT propose anything that is not a recognizable instantiation of
  the mentor's pitch. PKPO-as-the-project, F-GRPO-as-the-project, or
  "kill the LM judge in Poly-EPO as a repair-the-paper project" are all
  off-direction. Note the distinction: a minority-voting project whose
  contribution happens to include a cheaper clustering substrate (e.g.
  embeddings or final-answer matching in place of an LM judge) IS
  in-direction, because the project center is still the minority-voting
  algorithm. The off-direction case is when "fix Poly-EPO's judge
  dependency" is the project, not a sub-component.
- DO NOT invent citations. Verify arXiv IDs by web search.
- DO NOT pad with caveats about timeline or compute being "tight."
  Assume the team has built the trainer and can run reasonable
  experiments. Your job is research-direction quality, not project
  management.
- DO NOT recommend a direction simply because it is tractable.
- If you find yourself writing more than ~3 pages, you are padding.
  Stop.

## Tone
Skeptical, terse, evidence-grounded. The team has done its own analysis
(see `findings.md`); they want pressure-testing, not validation.
