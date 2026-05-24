# Stage 1: Design-Space Mapper

## Mission
You are mapping the design space of "minority voting optimization"
instantiations for the CS224R final project described in
`nancy_explore/narrative/context.md`. Read that file fully before anything else,
then read the artifacts it points to in the order it specifies.

Your job is NOT to evaluate or recommend any direction. It is to produce
a systematic, well-classified enumeration of plausible instantiations so
the team can pick which 4-5 to send to depth evaluation in stage 2.

## Required output

Produce a single markdown file at
`nancy_explore/agents/outputs/design_space.md` with the following
sections, in this exact order, with these exact headings:

### 1. Axis structure (1 paragraph + table)
Identify the 3-5 design axes that meaningfully distinguish
minority-voting instantiations from each other. Examples (not
prescriptive): mechanism of action (loss modification vs sampling
modification vs regularizer vs two-stage training); training signal
source (correctness vs cluster-frequency vs CoT-diversity vs
worst-case-subset); when minority voting enters (training-time vs
inference-then-distill vs both); granularity (per-sample vs per-set vs
per-cluster); relationship to GRPO (replacement vs add-on). Use whatever
axes actually carve the space cleanly, not these specific ones.
Summarize as a table at the end of this section.

### 2. Enumerated instantiations (>= 12)
For each instantiation:
- **Name** (short, memorable, <6 words)
- **Sketch** (3-5 sentences: what the algorithm actually does,
  mechanically — not what it tries to achieve in vibes)
- **Axis coordinates** (where it sits on each axis from section 1)
- **Closest neighbor in literature** (one paper, arXiv ID, one-line gap
  description)
- **Why it's a recognizable instantiation of the mentor's pitch** (one
  sentence — if you cannot answer this, drop the entry)

Cover the space, not your preferences. Include some uncomfortable-looking
options. Explicitly avoid the easy bias toward "modify the GRPO loss" —
at least 3 of your 12+ entries must be non-loss-modification approaches
(sampling, two-stage training, inference-then-distill,
exploration-bonus, curriculum, etc.).

### 3. Cluster map
Group the >=12 entries into 3-6 clusters of similar instantiations. For
each cluster: name a representative entry and note in one sentence what
makes the cluster coherent.

### 4. Explicit non-entries (3-5)
Three to five directions you considered but pruned, with a one-sentence
reason for each. This demonstrates you actively narrowed the space
rather than just brain-dumping.

## Constraints
- DO NOT rank, evaluate, or recommend. That is stage 2's job. The
  reader will pick from your enumeration.
- DO NOT propose Poly-EPO scaling laws, $(N, n)$ schedules,
  kill-the-LM-judge variants, or "switch to PKPO" as the project
  center. Off-direction per `nancy_explore/archive/poly_epo/why_stop.md` and
  `context.md`.
- DO NOT invent citations. Verify arXiv IDs by web search.
- DO NOT exceed ~2 pages.
- Bias toward coverage over depth. Section 2 entries should be 5-8
  lines each, not full pages.

## Tone
Catalogue-style. Terse, structured, low-emotion. The reader will scan
this and pick 4-5 entries to send to stage 2.
