# Stage 2 Spawn: Per-Token Uncertainty Minority

Follow the instructions in `nancy_explore/agents/prompts/stage2_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Per-Token Uncertainty Minority

**Slug:** token_uncertainty

**Sketch:** Redefine "minority" via the model's own per-token policy
distribution rather than via empirical answer frequency. A trajectory
is minority-like to the degree that its tokens were sampled from
low-probability regions of the model's distribution at generation
time. Compute a per-trajectory self-surprise score (e.g., mean log-prob
under the generating policy, or fraction of tokens below a per-position
percentile), then upweight high-self-surprise correct trajectories in
the GRPO advantage. Distinct from the rest of the menu because it
requires no clustering, no LM judge, and no answer-frequency estimation.
The minority-voting framing: "minority" = trajectories that beat the
model's own confident prior, which is a token-level analog of beating
the within-set majority answer.

Output your evaluation to
`nancy_explore/agents/outputs/depth/02_depth_token_uncertainty.md`.
