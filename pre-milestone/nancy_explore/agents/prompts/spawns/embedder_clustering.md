# Stage 2 Spawn: Embedding-Based Minority Clustering

Follow the instructions in `nancy_explore/agents/prompts/stage2_depth_evaluator.md`
in full. Your assigned direction:

**Name:** Embedding-Based Minority Clustering

**Slug:** embedder_clustering

**Sketch:** The project is a minority-voting training algorithm. The
contribution is the clustering substrate. Minority voting requires
defining "minority" over some answer- or reasoning-equivalence
structure; Poly-EPO defaults to an LM judge (Qwen3-4B-Instruct or
Gemini-Flash), which is expensive at training time and a known
reward-hacking surface. This direction trains a minority-voting
objective — pick whichever the team's depth evals find strongest among
inverse-frequency reweighting or worst-subset — with the equivalence
structure provided by a cheap substrate: final-answer matching for
verifiable tasks, sentence-embedding similarity (e.g. all-MiniLM-L6-v2
or jina-embeddings), reasoning-token-level n-gram fingerprints, or
self-judge from the training model itself. The experimental
contribution is a substrate ablation: train the same minority-voting
objective under N clustering substrates, measure (a) compute saved vs
LM-judge baseline, (b) downstream Cover@τ / Pass@k preserved or lost.
Headline claim: minority voting works without the LM judge, at
substantial compute savings.

**IMPORTANT framing constraint:** this is NOT a Poly-EPO repair
project. The project center is the minority-voting algorithm itself;
the substrate question is a sub-component that happens to be the most
interesting empirical knob. Read the updated anti-pattern in
`02_depth_evaluator.md` clarifying this distinction before deciding on
OFF-DIRECTION.

Output your evaluation to
`nancy_explore/agents/outputs/depth/02_depth_embedder_clustering.md`.
