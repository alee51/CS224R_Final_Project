### 1. Axis structure (1 paragraph + table)
Minority-voting optimization variants separate cleanly along five axes: where the optimization acts (loss, sampling, set construction, or two-stage distillation), how the minority signal is defined (tail correctness, rarity, disagreement, or under-covered clusters), when that signal is injected (online training vs inference-to-distill vs hybrid), what unit receives credit (trajectory, set, or cluster), and how tightly the method is coupled to GRPO (replace advantage, reweight updates, or wrap GRPO with outer-loop selection). These axes capture meaningful algorithmic differences without presuming one "correct" minority objective. They also make non-loss approaches first-class citizens rather than treating them as afterthoughts.

| Axis | Values used in this map |
|---|---|
| Optimization locus | Advantage/loss shaping; rollout sampling; set-construction objective; two-stage distillation |
| Minority signal source | Low-quantile correctness; low-frequency answer/cluster; disagreement minority vote; worst-subset score |
| Injection stage | Training-time online; inference-then-distill; hybrid (inference mining + online RL) |
| Credit granularity | Per-trajectory; per-set; per-cluster |
| Relation to GRPO | Direct replacement; multiplicative/additive add-on; outer-loop data/teacher wrapper |

### 2. Enumerated instantiations (>= 12)

**1) Name:** Low-Quantile Advantage  
**Sketch:** For each prompt, sample `N` rollouts and compute binary rewards. Replace GRPO's mean baseline with a low quantile (for example `q=0.1`) of group rewards, so trajectories that beat the low tail get positive advantage. This directly biases updates toward reducing worst-case failures inside the rollout set. PPO/GRPO clipping, KL regularization, and batch construction are unchanged.  
**Axis coordinates:** Locus=advantage/loss; Signal=low-quantile correctness; Stage=training-time online; Granularity=trajectory; GRPO relation=direct replacement.  
**Closest neighbor in literature:** PKPO, arXiv:2505.15201 — PKPO targets Pass@k directly, while this uses quantile-tail baselines as the minority proxy.  
**Why it's a recognizable instantiation of the mentor's pitch:** It explicitly trains on signals from low-frequency/low-tail outcomes within multi-sample voting sets.

**2) Name:** Minority-Winner Indicator  
**Sketch:** Generate `N` answers, count final-answer frequencies, and designate a minority vote winner by sampling one low-frequency answer class uniformly. Assign positive reward only to rollouts in that sampled minority class (and zero/negative otherwise), then run standard policy-gradient updates. Repeat per prompt so the "minority winner" changes stochastically across batches. This turns minority voting into a direct stochastic supervision target.  
**Axis coordinates:** Locus=advantage/loss; Signal=disagreement minority vote; Stage=training-time online; Granularity=trajectory; GRPO relation=direct replacement.  
**Closest neighbor in literature:** Self-Consistency, arXiv:2203.11171 — self-consistency chooses the majority answer at inference, while this trains against sampled minority winners.  
**Why it's a recognizable instantiation of the mentor's pitch:** The training reward is defined by minority voting over generated sets.

**3) Name:** Inverse-Frequency Reweighting  
**Sketch:** Keep base correctness reward, but multiply each rollout's advantage by inverse answer-frequency (or inverse cluster-frequency) estimated within the prompt's sampled set. Rare-but-correct trajectories receive larger gradient mass; common-correct ones are damped. Frequency weights are normalized per prompt to avoid exploding updates. Core GRPO machinery remains intact.  
**Axis coordinates:** Locus=advantage/loss; Signal=low-frequency answer/cluster; Stage=training-time online; Granularity=trajectory/cluster; GRPO relation=multiplicative add-on.  
**Closest neighbor in literature:** F-GRPO, arXiv:2602.06717 — both reweight updates to protect rare solutions; this reweighting is explicitly minority-frequency based.  
**Why it's a recognizable instantiation of the mentor's pitch:** It operationalizes minority voting as "upweight trajectories from minority answer groups."

**4) Name:** Worst-Subset Set Objective  
**Sketch:** Build multiple subsets from each prompt's `N` rollouts and score each subset by subset-level correctness. Use the minimum (or lower-tail average) subset score as the training target, then backpropagate marginal set advantages to member trajectories. This pushes learning toward improving the worst-performing subsets under voting. The implementation follows set-RL credit assignment rather than per-trajectory only.  
**Axis coordinates:** Locus=set-construction objective; Signal=worst-subset score; Stage=training-time online; Granularity=set then trajectory; GRPO relation=direct replacement via set objective.  
**Closest neighbor in literature:** Poly-EPO, arXiv:2604.17654 — Poly-EPO couples reward and diversity; this variant swaps in a worst-subset minority criterion.  
**Why it's a recognizable instantiation of the mentor's pitch:** Minority performance is optimized at the set level, matching "minority voting optimization" directly.

**5) Name:** Minority-CVaR Policy Gradient  
**Sketch:** Treat per-prompt rollout outcomes as an empirical return distribution and optimize CVaR at level `alpha` over that distribution. Only the bottom `alpha` fraction of outcomes contributes to the policy gradient each update. This is a risk-sensitive objective that systematically targets failure-prone regions. It can be implemented as a filtered-advantage estimator on top of GRPO sampling.  
**Axis coordinates:** Locus=advantage/loss; Signal=tail correctness; Stage=training-time online; Granularity=trajectory; GRPO relation=direct replacement.  
**Closest neighbor in literature:** Distributional RL with Quantile Regression, arXiv:1710.10044 — QR-DQN models return quantiles; this applies a CVaR-style tail objective to rollout correctness.  
**Why it's a recognizable instantiation of the mentor's pitch:** Minority voting is interpreted as optimizing the low-performance tail rather than the average vote.

**6) Name:** Minority Replay Buffer  
**Sketch:** During rollout generation, keep a buffer of minority-class trajectories (rare answers, rare clusters, or low-consensus correct traces). Construct each update mini-batch with a fixed minority quota plus fresh on-policy samples. Policy updates still use standard loss, but data composition forces consistent minority exposure. Buffer refresh rules prevent stale trajectory dominance.  
**Axis coordinates:** Locus=rollout sampling; Signal=low-frequency answer/cluster; Stage=training-time online; Granularity=trajectory/cluster; GRPO relation=outer-loop data wrapper.  
**Closest neighbor in literature:** DAPO, arXiv:2503.14476 — DAPO changes sample selection to avoid dead zones; this samples by minority-status quotas.  
**Why it's a recognizable instantiation of the mentor's pitch:** It optimizes minority voting behavior through minority-focused training data selection.

**7) Name:** Prompt Minority Curriculum  
**Sketch:** Maintain per-prompt statistics of answer concentration (for example, entropy of answer histogram). Prioritize prompts where minority answers are underrepresented or unstable, and downsample prompts already dominated by one answer mode. Train with unchanged RL loss but non-uniform prompt sampling. The curriculum updates online as concentration metrics shift.  
**Axis coordinates:** Locus=rollout sampling; Signal=disagreement/undercoverage; Stage=training-time online; Granularity=prompt-level set stats; GRPO relation=outer-loop data wrapper.  
**Closest neighbor in literature:** Scaling test-time compute, arXiv:2408.03314 — both allocate compute by difficulty/structure, while this allocates training probability by minority-coverage signals.  
**Why it's a recognizable instantiation of the mentor's pitch:** The algorithm explicitly spends training budget where minority voting behavior is weakest.

**8) Name:** Minority Teacher Distillation  
**Sketch:** At inference, sample many CoTs per prompt and pick supervision traces from minority-vote-correct groups (rather than majority winners). Distill these selected traces into the base model using supervised fine-tuning, then optionally run short RL polishing. The core minority signal is injected in trace selection, not in RL loss engineering. Teacher generation and student distillation are decoupled stages.  
**Axis coordinates:** Locus=two-stage distillation; Signal=low-frequency correct groups; Stage=inference-then-distill; Granularity=trajectory group; GRPO relation=outer-loop wrapper.  
**Closest neighbor in literature:** Self-Consistency, arXiv:2203.11171 — self-consistency keeps majority answers at inference; this distills minority-correct traces into training targets.  
**Why it's a recognizable instantiation of the mentor's pitch:** Training is explicitly derived from minority-vote outcomes over generated sets.

**9) Name:** Hybrid Minority Bootstrapping  
**Sketch:** Alternate two phases: (A) inference mining that labels minority-correct trajectories, and (B) on-policy RL updates using those labels as auxiliary rewards. The mined minority labels are refreshed periodically to track policy drift. This hybrid keeps the minority signal explicit while retaining online adaptation. It avoids committing fully to pure RL-loss or pure distillation.  
**Axis coordinates:** Locus=hybrid (distill + loss); Signal=minority-correct pseudo-labels; Stage=hybrid; Granularity=trajectory; GRPO relation=add-on auxiliary signal.  
**Closest neighbor in literature:** Test-Time Reinforcement Learning, arXiv:2512.15146 — TTRL bootstraps from consensus labels; this specifically bootstraps from minority-labeled correctness.  
**Why it's a recognizable instantiation of the mentor's pitch:** It instantiates minority voting as the pseudo-label source for ongoing training.

**10) Name:** Cluster-Coverage Bonus  
**Sketch:** Cluster reasoning traces for each prompt and maintain a moving estimate of cluster-level correctness coverage. Add a bonus when a rollout improves correctness in low-coverage clusters; apply no bonus to already-covered clusters. Update bonus tables online and decay stale counts. Base correctness reward remains the anchor term.  
**Axis coordinates:** Locus=advantage/loss; Signal=under-covered clusters; Stage=training-time online; Granularity=cluster; GRPO relation=additive regularizer.  
**Closest neighbor in literature:** SetPO, arXiv:2602.01062 — SetPO rewards set diversity globally; this rewards minority cluster correctness coverage explicitly.  
**Why it's a recognizable instantiation of the mentor's pitch:** Minority voting is captured as targeted improvement of low-support reasoning clusters.

**11) Name:** Dual-Head Majority/Minority Policy  
**Sketch:** Share one policy trunk with two lightweight output heads: a majority-optimized head and a minority-optimized head. During training, each batch routes to one head depending on majority-vote or minority-vote supervision mode, while the trunk receives both gradients. At evaluation, either head or a mixture can be used without retraining the trunk. This creates an explicit objective split rather than a single blended scalar.  
**Axis coordinates:** Locus=architecture + loss routing; Signal=majority vs minority vote modes; Stage=training-time online; Granularity=trajectory; GRPO relation=parallel objective heads.  
**Closest neighbor in literature:** DeepSeekMath/GRPO, arXiv:2402.03300 — GRPO uses a single objective stream; this forks majority and minority streams with shared representation.  
**Why it's a recognizable instantiation of the mentor's pitch:** It literally instantiates both majority and minority voting optimization in one training setup.

**12) Name:** Minority-Constrained Decoding Distill  
**Sketch:** Use constrained decoding at teacher time to suppress already-dominant answer clusters (for example, via penalties on high-frequency clusters), forcing generation into minority regions. Filter for correct constrained samples and distill them into the student. RL is optional and secondary; the core optimization is through constrained minority sample harvesting. This is mechanically distinct from changing policy-gradient loss.  
**Axis coordinates:** Locus=inference-time sampling + distill; Signal=low-frequency cluster targeting; Stage=inference-then-distill; Granularity=cluster; GRPO relation=outer-loop wrapper.  
**Closest neighbor in literature:** Scaling test-time compute, arXiv:2408.03314 — both manipulate inference-time sampling budget/strategy; this specifically constrains toward minority regions before distillation.  
**Why it's a recognizable instantiation of the mentor's pitch:** Minority-vote regions are made the source of supervised improvement.

### 3. Cluster map
- **Cluster A — Tail-risk objective replacements** (Representative: *Low-Quantile Advantage*): Methods that redefine advantage/return to optimize low-tail outcomes directly (Entries 1, 5).  
- **Cluster B — Frequency-aware credit shaping** (Representative: *Inverse-Frequency Reweighting*): Methods that keep core RL but reweight gradients toward minority answer/cluster events (Entries 3, 10).  
- **Cluster C — Set-level minority operators** (Representative: *Worst-Subset Set Objective*): Methods where minority optimization is defined on subsets/sets before marginalizing credit (Entry 4).  
- **Cluster D — Data and sampling wrappers** (Representative: *Minority Replay Buffer*): Methods that alter which prompts/rollouts are trained on, leaving core loss mostly unchanged (Entries 6, 7).  
- **Cluster E — Two-stage or hybrid minority supervision** (Representative: *Minority Teacher Distillation*): Methods that create minority-based supervision via inference and then train via distillation or hybrid loops (Entries 8, 9, 12).  
- **Cluster F — Explicit dual-objective architectures** (Representative: *Dual-Head Majority/Minority Policy*): Methods that jointly train majority and minority objectives in separate but coupled model pathways (Entry 11).

### 4. Explicit non-entries (3-5)
- **Poly-EPO `(N, n)` schedule/scaling project** — pruned because the project center must be minority-voting instantiations, not Poly-EPO scaling-law studies.  
- **"Kill the LM-judge" variant as project core** — pruned because this targets Poly-EPO infrastructure dependence rather than minority-voting objective design.  
- **"Switch to PKPO as the project center"** — pruned because PKPO is a related baseline/mechanism source, but direction lock requires a minority-voting project identity.  
- **General reward-model/LM-judge robustness audit** — pruned because it is a meta-evaluation problem and does not directly instantiate minority voting optimization.  
