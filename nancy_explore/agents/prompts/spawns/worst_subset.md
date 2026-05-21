# Stage 2 Spawn: Worst-Subset Set Objective

Follow the instructions in `nancy_explore/agents/prompts/stage2_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Worst-Subset Set Objective

**Slug:** worst_subset

**Sketch:** Build all (or sampled) size-n subsets from each prompt's N
rollouts, score each subset by subset-level correctness (e.g.
fraction-correct or minority-vote correctness), and use the minimum (or
lower-tail average) subset score as the training target. Backpropagate
marginal set advantages to member trajectories via the same machinery
as Poly-EPO. Conceptually: minority voting at the set level. Comparison
story: same set-RL framework as Poly-EPO, but swap the
diversity-weighted aggregator for a worst-subset aggregator.

Output your evaluation to
`nancy_explore/agents/outputs/depth/02_depth_worst_subset.md`.
