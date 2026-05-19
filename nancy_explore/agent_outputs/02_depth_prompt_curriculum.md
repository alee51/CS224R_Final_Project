### 1. Direction restatement (<=150 words)
Prompt Minority Curriculum is a minority-voting instantiation where the contribution is not a new policy loss, but a new data-selection policy over prompts. For each prompt, maintain online answer-concentration statistics from current rollouts (e.g., answer entropy, top-1 vote margin, instability across steps). Sample prompts with probability increasing when answer mass is concentrated or minority-support is unstable/under-covered, and decreasing when one answer mode is already saturated. Keep the inner RL objective standard (GRPO-style policy gradient with verifiable reward), so the only changed lever is where training compute is spent. Relative to the mentor pitch, this still targets minority-voting optimization: it allocates optimization pressure to prompts where minority trajectories are currently weakest, then tests whether this improves harder-test-set generalization (AIME-25/26, Beyond-AIME, HMMT) versus majority-oriented training behavior.

### 2. Related work scan (6-12 papers, last ~18 months)
1. **arXiv:2604.17654 (Poly-EPO)** - Set-level objective combining reward and diversity improves pass@k coverage and exploratory reasoning.  
   **Delta:** Prompt curriculum does not change set objective/advantage; it changes prompt sampling probability under fixed RL loss.

2. **arXiv:2505.15201 (PKPO)** - Directly optimizes pass@k with low-variance estimators, showing harder-problem gains from set-utility optimization.  
   **Delta:** PKPO alters reward transform/loss; prompt curriculum keeps loss fixed and reallocates data budget to minority-weak prompts.

3. **arXiv:2602.06717 (F-GRPO)** - Difficulty-aware/focal reweighting reduces forgetting of rare-correct modes in group RLVR.  
   **Delta:** F-GRPO is gradient reweighting at update time; prompt curriculum is sampling reweighting before gradient computation.

4. **arXiv:2602.01062 (SetPO)** - Set-level diversity objective with marginal set contributions mitigates mode collapse.  
   **Delta:** SetPO injects diversity in the objective; prompt curriculum injects minority pressure via online prompt selection.

5. **arXiv:2503.14476 (DAPO)** - RL system-level improvements (including dynamic sampling) stabilize large-scale LLM RL and improve benchmark performance.  
   **Delta:** DAPO dynamic sampling is general stability/performance engineering; prompt curriculum explicitly conditions sampling on minority-answer concentration.

6. **arXiv:2504.16084 (TTRL)** - Uses test-time majority-voting-derived pseudo-rewards for RL on unlabeled data.  
   **Delta:** TTRL changes supervision source (pseudo-reward); prompt curriculum assumes standard labeled/verifiable reward and only changes prompt probabilities.

7. **arXiv:2505.24864 (ProRL)** - Prolonged RL training expands reasoning boundaries beyond base-model reachable solutions.  
   **Delta:** ProRL studies training duration/scale effects; prompt curriculum studies allocation of fixed training budget across prompts.

8. **arXiv:2409.10164 (Quantile Regression for Distributional Reward Models in RLHF)** - Models reward uncertainty/distribution to support risk-aware optimization.  
   **Delta:** Distributional reward modeling changes reward estimation itself; prompt curriculum leaves reward model unchanged and reweights which prompts are sampled.

9. **arXiv:2512.03847 (DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training)** - Distributional value modeling with risk-sensitive regularization improves robustness under noisy supervision.  
   **Delta:** DVPO changes critic/value modeling and regularization; prompt curriculum remains actor-only data-selection around standard RLVR.

### 3. Novelty check
- **Specific scientific claim:** Online prompt-level concentration-aware sampling improves worst-case/generalization performance (hard reasoning subsets, coverage metrics) versus uniform sampling, without modifying the RL loss.
- **Closest existing work:** F-GRPO and DAPO are closest (both adapt training signal/data by hardness/frequency). The differentiator is the unit of control: prompt-level concentration dynamics tied to minority answer support, rather than per-trajectory focal weighting or generic dynamic sampling.
- **Steelman Ifdita objection (3 sentences):** "This looks like a scheduler, not a minority-voting algorithmic contribution. F-GRPO/DAPO already tell a very similar story: spend gradient on hard or rare regions and avoid collapse. Unless you show a clear minority-coverage mechanism and a win over objective-level methods (Poly-EPO/SetPO/F-GRPO), this is incremental engineering."
- **Novelty rating:** **medium-low** (between medium and low; defensible only with strong mechanistic ablations).

### 4. Concrete training objective
Let prompt \(x\) have \(N\) sampled rollouts \(y_{1:N}\sim \pi_\theta(\cdot|x)\), rewards \(r_i\in\{0,1\}\), and GRPO-style advantage
\[
A_i(x)=r_i-\bar r(x),\qquad \bar r(x)=\frac{1}{N}\sum_{j=1}^N r_j.
\]
Define a prompt concentration score from answer histogram \(h_x\):
\[
c(x)=\lambda_1\big(1-\tfrac{H(h_x)}{\log M_x}\big)+\lambda_2\,\mathrm{margin}(h_x)+\lambda_3\,\mathrm{instab}(x),
\]
where \(H\) is entropy, \(M_x\) observed answer bins, margin \(=p_{(1)}-p_{(2)}\), and instab is EMA disagreement across recent visits.
Sampling distribution over prompts:
\[
q_t(x)=\frac{\exp(\beta_t c_t(x))}{\sum_{x'}\exp(\beta_t c_t(x'))}.
\]
Policy objective optimized under non-uniform prompt sampling:
\[
\mathcal{L}(\theta)= -\mathbb{E}_{x\sim q_t}\left[\frac{1}{N}\sum_{i=1}^N \rho_t(x)\,\mathrm{clip\_ppo}\!\left(\log\pi_\theta(y_i|x),A_i(x)\right)\right],
\]
\[
\rho_t(x)=\frac{p_{\text{target}}(x)}{q_t(x)} \;\;(\text{optional IS correction; }p_{\text{target}}\text{ uniform}).
\]
Gradients flow only through \(\pi_\theta\); \(q_t\) and \(c_t\) are stop-gradient online statistics updated after each batch (or every \(K\) steps).

This is precise enough; the underspecified part is not the loss but the choice of \(c(x)\) components/weights and whether IS correction is required for unbiasedness.

### 5. Experimental plan
- **Minimal headline experiment:** Train Qwen-1.7B on DaPO-17k for 400 steps with identical RL loss/hyperparams; compare uniform prompt sampling vs concentration-aware sampling. Report hard-set gains on AIME-25/26, Beyond-AIME, HMMT.
- **Baselines (required + useful):**  
  (i) Standard GRPO (uniform prompts),  
  (ii) Majority-voting-trained model (majority-winner supervision baseline),  
  (iii) F-GRPO,  
  (iv) SetPO-lite/Poly-EPO-lite objective baseline if implementable at same scale.
- **Defensibility ablations:**  
  (a) entropy-only vs margin-only vs instability-only curriculum signal;  
  (b) with vs without IS correction \(\rho_t\);  
  (c) online update frequency (every step vs every 20 steps);  
  (d) cap on max prompt probability (anti-overfitting);  
  (e) equalized-token-budget control.
- **Compute estimate (Modal):** Using current pricing references (H100 \(\$3.95\)/GPU-h, A100-80GB \(\$2.50\)/GPU-h; Modal pricing page), assume 1x H100 for stability and throughput.  
  Per 400-step run (train + periodic eval): ~14 GPU-h.  
  10 runs (2 core baselines + curriculum + key ablations, 2 seeds for core): ~140 GPU-h \(\approx \$553\).  
  Full expanded matrix 18 runs: ~252 GPU-h \(\approx \$995\).  
  Add contingency/re-runs 25%: \(\approx \$1244\).  
  **Fits in \$1400**, but only if ablation grid is pruned (do not run full Cartesian combinations).
- **Headline metric:** Primary = Cover@\(\tau\) on hard sets (minority-support claim). Secondary = Pass@k (k=8,16,32/64) and worst-case-subset accuracy stratified by high concentration prompts.

### 6. Failure modes and consolation result
- **Failure 1: Difficulty confound, not minority effect.**  
  Observation: curriculum mostly tracks low-reward prompts; gains disappear after matching for prompt difficulty bins.
- **Failure 2: Prompt overfitting / memorization.**  
  Observation: training concentration metrics improve, but OOD hard-set Cover@\(\tau\) is flat or worse; in-domain pass@k rises only on repeatedly oversampled prompts.
- **Failure 3: High-variance optimization from skewed sampling.**  
  Observation: unstable KL/reward curves, seed variance spikes, no statistically significant gain over uniform GRPO.
- **Minimum publishable result if null:**  
  A careful negative result: "Prompt-level concentration curricula do not improve OOD minority coverage after controlling for prompt difficulty and token budget." Publishable core artifact is a table crossing (sampling rule × confound controls × Cover@\(\tau\)/Pass@k), plus one trajectory figure of concentration metrics vs OOD performance.

### 7. Killer experiment
If positive, the killer result is: **with identical loss/compute/model, concentration-aware prompt sampling gives significant Cover@\(\tau\) and worst-case-subset gains on AIME-26/Beyond-AIME over both GRPO and majority-voting-trained baselines.**  
Expected figure: one Pareto plot (x: Pass@1 or compute-normalized cost, y: Cover@\(\tau\)) where prompt curriculum dominates baselines on hard-OOD points.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **conditional on X**.
- **X:** must pre-register confound controls (difficulty matching, token-budget matching, IS/no-IS ablation) and include at least one objective-level minority baseline (F-GRPO or SetPO-lite) to prove this is not just generic curriculum learning.
- **Honest summary:** This is in-direction and tractable, and it has a clean "data-selection layer" thesis distinct from loss engineering. But novelty is fragile because nearby work already adapts signal allocation for rare/hard cases; without crisp mechanism evidence, this will be read as scheduler tuning. If the team can show compute-matched, loss-invariant gains on hard-set minority coverage with robust controls, it becomes a legitimate contribution; if not, it likely downgrades to an ablation subsection.
