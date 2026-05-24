# Stage 2 Spawn: Prompt Minority Curriculum

Follow the instructions in `nancy_explore/agents/prompts/stage2_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Prompt Minority Curriculum

**Slug:** prompt_curriculum

**Sketch:** Maintain per-prompt statistics of answer concentration
(e.g. entropy of within-prompt answer histogram, or majority-vote
margin). Prioritize prompts where minority answers are underrepresented
or unstable; downsample prompts already dominated by one answer mode.
Train with standard RL loss but non-uniform prompt sampling. The
curriculum updates online as concentration metrics shift. The
"minority voting optimization" lives at the data-selection layer
rather than in the loss.

Output your evaluation to
`nancy_explore/agents/outputs/depth/02_depth_prompt_curriculum.md`.
