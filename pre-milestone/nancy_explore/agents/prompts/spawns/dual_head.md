# Stage 2 Spawn: Dual-Head Majority-Minority Policy

Follow the instructions in `nancy_explore/agents/prompts/stage2_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Dual-Head Majority-Minority Policy

**Slug:** dual_head

**Sketch:** Share one policy trunk with two lightweight output heads:
a majority-optimized head trained against the standard mean-baseline
GRPO objective, and a minority-optimized head trained against a
minority-voting objective (e.g. inverse-frequency or worst-subset).
Each batch routes loss to one head depending on mode; the trunk
receives both gradients. At evaluation, either head, a mixture, or a
continuum between them can be used without retraining. Directly
instantiates the mentor's "both methods" framing in a single training
setup.

Output your evaluation to
`nancy_explore/agents/outputs/depth/02_depth_dual_head.md`.
