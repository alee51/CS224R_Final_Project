# Stage 2 Spawn: Cover-at-Tau as Training Objective

Follow the instructions in `nancy_explore/agents/prompts/stage2_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Cover-at-Tau as Training Objective

**Slug:** cover_at_tau

**Sketch:** Take Cover@τ — which measures, across a set of N rollouts,
the fraction of distinct correct-answer clusters whose support exceeds
threshold τ — and use it directly as the training reward. Construct
the policy-gradient signal via REINFORCE / GRPO over the per-prompt
scalar Cover@τ score, with τ either fixed or scheduled. The
minority-voting framing: rewarding the model for ensuring multiple
distinct correct answer modes each have sufficient probability mass is
functionally an upweighting of minority-correct trajectories at the
answer-cluster level. Position against PKPO (which trains on Pass@k,
a binarized version of Cover@τ) and Poly-EPO (which uses reward x
diversity rather than a coverage threshold). Headline framing: close
the train-test gap by training on the metric we evaluate on.

Output your evaluation to
`nancy_explore/agents/outputs/depth/02_depth_cover_at_tau.md`.
