### 1. Direction restatement (<=150 words)
Train a minority-voting RL objective as the project center, and treat "how minority groups are defined" as the key empirical knob. Concretely: hold model/data/training recipe fixed, choose one minority objective (best candidate: inverse-frequency reweighting over rollout groups), and swap the clustering substrate that defines group membership: LM-judge baseline vs cheap alternatives (exact final-answer matching, embedding similarity, n-gram fingerprints, self-judge). Then measure whether minority-voting gains on harder reasoning sets survive when the expensive judge is removed, and how much training compute drops. This is in-direction: it is still "instantiate minority-voting training + test generalization to hard sets." The substrate study is a mechanism/efficiency ablation inside that objective, not a "repair Poly-EPO" project.

### 2. Related work scan (6-12 papers, last ~18 months)
- **2604.17654 (Poly-EPO)**: Set-RL objective coupling reward and diversity improves exploration/generalization in reasoning. **Delta**: uses LM-judge-centric clustering; this direction asks whether minority-voting still works when cluster substrate is cheap.
- **2505.15201 (PKPO)**: Optimizes Pass@k directly with unbiased estimators and annealing over k. **Delta**: PKPO is k-transform objective design; here objective is fixed and clustering substrate is ablated.
- **2602.06717 (F-GRPO)**: Difficulty-aware/focal-style reweighting protects rare-correct trajectories in RLVR. **Delta**: similar spirit on rarity weighting, but no explicit substrate-ablation story against LM-judge dependence.
- **2602.01062 (SetPO)**: Set-level diversity objective via embedding-kernel novelty and marginal set contributions. **Delta**: closest technical neighbor; this direction is explicitly minority-vote optimization plus compute-vs-performance frontier across substrates.
- **2503.14476 (DAPO)**: RLVR system-level stabilization and scaling tricks; strong practical baseline. **Delta**: infrastructure/training stability contribution, not minority-equivalence substrate question.
- **2503.06639 (GRPO dynamics)**: Formalizes GRPO effective loss/dynamics under verifiable rewards. **Delta**: theory for mean-baseline RLVR; does not tackle minority group definitions.
- **2504.16084 (TTRL)**: Test-time RL with unlabeled data using consensus-style rewards. **Delta**: adaptation at inference/test-time loop, not training-time minority-cluster substrate ablation.
- **2512.03847 (DVPO)**: Distributional value modeling and risk-aware policy optimization for LLM post-training. **Delta**: distributional/risk objective innovation; not an equivalence-structure substitution study.
- **2411.04109 (Self-Consistency Preference Optimization)**: Converts self-consistency signal into preference optimization training. **Delta**: consistency-as-supervision; does not isolate clustering substrate cost/performance in minority voting.

### 3. Novelty check
- **Specific claim**: For minority-voting RL on reasoning tasks, an LM judge is not load-bearing; cheap equivalence substrates recover most downstream gains at substantially lower training cost.
- **Closest existing work**: SetPO (embedding-based diversity credit) and Poly-EPO (set objective with LM-judge clustering). Difference: this project's primary contribution is a controlled substrate ablation under one minority objective, with explicit compute-normalized performance frontier.
- **Ifdita steelman objection (3 sentences)**: "This may just repackage existing diversity objectives under a different label. If cheap clustering works, reviewers may say the LM-judge baseline was over-engineered rather than scientifically wrong. If cheap clustering fails, you only learned that semantics-aware judging is necessary, which is useful but incremental unless your ablation is very clean."
- **Novelty rating**: **medium**.

### 4. Concrete training objective
Use inverse-frequency minority reweighting as the primary objective (cleanest for substrate ablation).

\[
\text{Given prompt }x,\; y_i \sim \pi_\theta(\cdot\mid x),\; i=1,\dots,N,\quad r_i \in \{0,1\}
\]
\[
c_i = S_\phi(x,y_i),\quad n_c=\sum_{j=1}^N \mathbf{1}[c_j=c],\quad
w_i=\frac{(n_{c_i}+\epsilon)^{-\beta}}{\frac{1}{N}\sum_{j=1}^N (n_{c_j}+\epsilon)^{-\beta}}
\]
\[
\bar r=\frac{1}{N}\sum_{j=1}^N r_j,\quad
A_i = w_i\,(r_i-\bar r)
\]
\[
\mathcal{L}_{\text{PG}}(\theta)= -\frac{1}{N}\sum_{i=1}^N
\min\!\Big(\rho_i A_i,\;\text{clip}(\rho_i,1-\eta,1+\eta)A_i\Big),
\quad
\rho_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}
\]
\[
\mathcal{L}(\theta)=\mathbb{E}_{x\sim\mathcal D}\!\left[\mathcal{L}_{\text{PG}}(\theta)\right]
+\lambda\,\mathbb{E}_{x}\!\left[\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\right]
\]

Per-batch aggregation is mean over prompts. Gradients flow only through \(\log \pi_\theta\) terms; \(r_i\), cluster assignments \(c_i\), and weights \(w_i\) are treated as stop-gradient statistics. The substrate ablation is exactly \(S_\phi\): LM judge vs answer-match vs embeddings vs n-gram fingerprint vs self-judge.

### 5. Experimental plan
- **Minimal headline experiment**: Train the same minority objective above under 5 substrates `{LM-judge, final-answer exact match, sentence-embedding clustering, n-gram fingerprint clustering, self-judge}` on Qwen-1.7B/DaPO-17k/400 steps; evaluate on AIME-25/26, Beyond-AIME, HMMT.
- **Baselines (required + extras)**: standard GRPO; majority-voting-trained model; minority objective with LM-judge (reference baseline); SetPO-style diversity baseline (if implementation effort stays low).
- **Ablations for defensibility**:
  1) substrate only (objective fixed),
  2) embedding model choice (all-MiniLM-L6-v2 vs jina-embedding variant),
  3) clustering threshold sensitivity,
  4) \(\beta\) (inverse-frequency strength),
  5) optional objective swap (inverse-frequency vs worst-subset) on top 2 substrates only.
- **Compute estimate (Modal)**: Modal pricing search gives A100-80GB at \(\$2.50\)/GPU-hour and H100 at \(\$3.95\)/GPU-hour (from Modal pricing page). Choose **A100-80GB** (1.7B model, lower cost, enough memory). Budget:
  - Main runs: 8 conditions (3 required baselines + 5 substrates) × 3 seeds × 14 GPU-h ≈ 336 GPU-h
  - Eval sweeps + long-sample Cover@\(\tau\): ≈ 60 GPU-h
  - Pilot tuning/failed runs buffer: ≈ 120 GPU-h
  - Total ≈ 516 GPU-h → **\$1290** on A100-80GB (fits \$1400), or \$2038 on H100 (does not fit).
- **Headline metric**: primary = Cover@\(\tau\) retention relative to LM-judge minority baseline at matched training steps; secondary = Pass@k and cost-per-1-point Cover@\(\tau\).

### 6. Failure modes and consolation result
- **Failure mode 1 (semantic collapse of cheap clusters)**: embedding/fingerprint substrates merge distinct valid solutions or split equivalent ones; observe large drop in Cover@\(\tau\) with small/no compute gain-adjusted benefit.
- **Failure mode 2 (reward hacking via self-judge substrate)**: policy exploits self-judge artifacts; observe train Pass@k rising while held-out verifier accuracy and hard-set Cover@\(\tau\) stagnate/fall.
- **Failure mode 3 (minority weighting too noisy)**: inverse-frequency amplifies spurious rare errors; observe higher variance across seeds and unstable KL/reward traces despite same compute.
- **Minimum publishable consolation result**: a compute-performance frontier figure showing exactly when LM-judge is load-bearing vs replaceable (Table: substrate × GPU-hours × Cover@\(\tau\) delta × Pass@k delta), even if no cheap substrate fully matches LM-judge.

### 7. Killer experiment
If one cheap substrate (embedding or exact-answer matching on verifiable tasks) matches at least ~90-95% of LM-judge Cover@\(\tau\) gain at <40% training-time compute, the direction is paper-worthy.
Expected figure: a Pareto plot (x=GPU-hours, y=Cover@\(\tau\) on hard OOD) where at least one cheap-substrate point dominates or nearly dominates the LM-judge point.

### 8. Overall verdict
- **Rating**: **promising**.
- **Worth running stage 3 (initial experimentation) on?** **yes, conditional on** enforcing strict objective-fixed substrate-only ablations first.
- This direction has real ceiling because it targets a load-bearing practical bottleneck (judge cost/dependency) while staying in the mentor's minority-voting framing. The risk is "incremental engineering ablation" if the claim is framed as Poly-EPO repair; avoid that by centering the minority-voting objective and presenting substrate as equivalence-structure science. If the ablation is clean, both positive and negative outcomes are defensible: either minority voting is robust to cheap clustering (strong practical thesis) or LM-judge semantics are necessary (clear boundary condition for the paradigm).
