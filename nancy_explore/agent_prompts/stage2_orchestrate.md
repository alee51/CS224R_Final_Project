# Stage 2 Orchestration

You are the orchestrator. Your job is to spawn stage-2 depth-evaluator
agents in parallel and synthesize their outputs.

## Step 1: Spawn agents

List every `*.md` file in `nancy_explore/agent_prompts/stage2_spawns/`.
There should be 7. For each file:

1. Read the file. Its contents are the complete initial prompt for the
   spawned agent.
2. Spawn an agent (background, parallel) using that prompt verbatim.
3. The agent will read its prompt, read `nancy_explore/context.md` and
   the artifacts that context.md points to, read
   `nancy_explore/agent_prompts/02_depth_evaluator.md` for the full
   schema, and produce an output file at
   `nancy_explore/agent_outputs/02_depth_<slug>.md`.

All 7 spawns are independent. Launch them concurrently, not
sequentially.

## Step 2: Wait for all spawns to complete

Do not start synthesis until all 7 output files exist at
`nancy_explore/agent_outputs/02_depth_*.md`. If any spawn fails or
returns an empty file, note it in the synthesis rather than retrying
silently.

## Step 3: Produce synthesis

Read all `nancy_explore/agent_outputs/02_depth_*.md` files. Produce a
single file at `nancy_explore/agent_outputs/stage2_synthesis.md`
containing, in this order:

### Section 1: Comparison table
One row per direction, with these columns:
- Name
- Slug
- Novelty rating (from section 3 of each depth eval)
- Verdict (from section 8: strong / promising / weak / off-direction)
- Worth stage 3? (yes / no / conditional)
- Killer experiment (one line, distilled from section 7)
- Top failure mode (one line, distilled from section 6)

### Section 2: Cross-cutting observations (1-2 paragraphs)
Identify any of the following:
- Directions that overlap substantially in mechanism, despite differing
  on the surface.
- Directions that depend on shared sub-components (e.g. multiple
  directions need an answer-clustering substrate — does
  `embedder_clustering`'s contribution apply to them all?).
- Compute estimates that look suspicious (way too high or way too low
  relative to the $1400 envelope).
- Citations that multiple depth evals share, especially if any look
  hallucinated (verify suspicious arXiv IDs by web search).

### Section 3: OFF-DIRECTION list
Any direction that came back marked OFF-DIRECTION, with the agent's
exact reasoning. Do not editorialize — pass through verbatim.

### Section 4: Notes for the human reviewer
A short bulleted list of things the human (Nancy) should pay attention
to when reading the depth evals. Examples: "the dual_head depth eval
flags an implementation concern about gradient interference that the
team should verify", or "two depth evals propose nearly identical
killer experiments — only one should survive to stage 3."

## Constraints
- Do not editorialize beyond what's in the source files. The
  comparison table and OFF-DIRECTION list should be pass-through. Only
  sections 2 and 4 are your synthesis.
- Do not rank or recommend a winning direction. The human will do
  that.
- Verify suspicious arXiv citations with web search if any look
  hallucinated, but otherwise trust the depth evals' content.
- Notify the user when the synthesis is ready.
