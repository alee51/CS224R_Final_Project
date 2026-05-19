# Stage 2 Spawn: Inverse-Frequency Reweighting

Follow the instructions in `nancy_explore/agent_prompts/02_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Inverse-Frequency Reweighting

**Slug:** inverse_freq

**Sketch:** Keep base correctness reward; multiply each rollout's
advantage by inverse answer-frequency (or inverse cluster-frequency)
estimated within the prompt's sampled N-rollout set, normalized per
prompt. Rare-but-correct trajectories receive larger gradient mass;
common-correct ones are damped. Core GRPO machinery (clipping, KL) is
unchanged. The "minority" signal is operationalized as low
answer-cluster frequency, not as low reward quantile, so this is
distinct from QC-GRPO / PKPO-style tail objectives.

Output your evaluation to
`nancy_explore/agent_outputs/02_depth_inverse_freq.md`.
