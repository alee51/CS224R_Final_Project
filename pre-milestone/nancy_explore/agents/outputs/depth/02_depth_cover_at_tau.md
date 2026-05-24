### 1. Direction restatement (<=150 words)
Train directly on `Cover@tau` instead of using it only at eval. For each prompt, sample a set of rollouts, cluster final correct answers into distinct modes, and reward the policy when multiple correct clusters each clear support threshold `tau` (rather than only rewarding majority success). This is a direct minority-voting instantiation: the objective pushes probability mass into underrepresented but correct answer clusters until they become reliably represented in the rollout set. Relative to the mentor pitch, this is exactly "set-based training objective for minority voting, then test OOD generalization on hard reasoning sets." The core framing is train-test alignment: if the paper claim is about broader correct coverage under test-time sampling, optimize that coverage metric during training instead of proxying with Pass@1 or plain GRPO rewards.

### 2. Related work scan (6-12 papers, last ~18 months)
1. **arXiv:2604.17654 (Poly-EPO)** - Set-RL objective multiplies reward and diversity, with subset-level credit assignment, improving pass@k coverage and diversity.  
   **Delta vs this direction:** Cover-at-tau uses an explicit coverage threshold on correct clusters; no reward-times-diversity product.

2. **arXiv:2505.15201 (PKPO)** - Optimizes Pass@k directly with unbiased reward transforms and k-annealing.  
   **Delta vs this direction:** PKPO is effectively a binarized "at least one success" tail metric; Cover-at-tau requires multiple distinct correct clusters to clear support `tau`.

3. **arXiv:2602.06717 (F-GRPO)** - Focal-style scaling protects rare-correct trajectories without increasing rollout count.  
   **Delta vs this direction:** F-GRPO reweights prompt difficulty; Cover-at-tau redefines the prompt reward itself around cluster-level coverage.

4. **arXiv:2602.01062 (SetPO)** - Adds set-level marginal diversity credit via kernelized similarity; preserves multi-modal outputs.  
   **Delta vs this direction:** SetPO optimizes semantic diversity generally; Cover-at-tau optimizes correctness-conditioned coverage beyond threshold.

5. **arXiv:2503.14476 (DAPO)** - Open large-scale RLVR system with stability tricks (decoupled clipping, dynamic sampling, token-level losses).  
   **Delta vs this direction:** DAPO is mostly systems/stability; Cover-at-tau is objective-level minority-voting signal that can be plugged into DAPO-style training.

6. **arXiv:2504.16084 (TTRL)** - Uses majority-vote pseudo-label rewards to do RL on unlabeled test-time data.  
   **Delta vs this direction:** TTRL uses self-generated labels; Cover-at-tau assumes verifiable correctness and changes reward aggregation over sampled sets.

7. **arXiv:2505.24864 (ProRL)** - Prolonged RL can expand reasoning boundaries under specific training controls.  
   **Delta vs this direction:** ProRL studies training horizon and capability expansion; Cover-at-tau is a specific minority-coverage objective.

8. **arXiv:2504.13837 (Does RL Really Incentivize Reasoning Capacity...)** - RLVR can improve pass@1 while concentrating modes already in base model.  
   **Delta vs this direction:** This paper is the adversarial baseline hypothesis Cover-at-tau must beat: prevent concentration by rewarding covered correct modes.

9. **arXiv:2505.22257 (Revisiting GRPO On/Off-Policy)** - Shows off-policy GRPO variants can match or beat on-policy in practice.  
   **Delta vs this direction:** Orthogonal training regime result; Cover-at-tau is about what scalar objective to optimize, not on/off-policy mechanics.

### 3. Novelty check
- **Specific scientific claim:** Replacing per-trajectory binary rewards with prompt-level `Cover@tau` reward improves hard-OOD minority-correct generalization (Cover@tau / worst-subset accuracy) at fixed rollout compute, relative to GRPO and majority-voting-trained baselines.
- **Closest existing work:** PKPO (arXiv:2505.15201), because both align training with set-level test-time metrics. Key difference is objective geometry: PKPO is "any success among k," while Cover-at-tau is "multiple correct clusters each above a support floor."
- **Ifdita-style 3-sentence objection (steelman):** "This is very close to PKPO plus a clustering heuristic, not a new principle. Thresholded coverage creates sparse and high-variance credit, and your denominator depends on noisy cluster extraction, so optimization may be brittle and hard to interpret. If gains disappear when compared to PKPO/F-GRPO under matched compute, the contribution collapses to metric engineering."
- **Novelty rating:** **medium**.

### 4. Concrete training objective
For prompt \(x\), sample \(N\) rollouts \(y_i \sim \pi_{\theta_{\text{old}}}(\cdot|x)\), with correctness \(r_i \in \{0,1\}\) and cluster ID \(c_i \in \{1,\dots,C_x\}\) from a fixed answer-clustering rule.

\[
n_{x,c}=\sum_{i=1}^{N}\mathbf{1}[c_i=c], \qquad
p_{x,c}=\frac{n_{x,c}}{N}, \qquad
q_{x,c}=\max_i \mathbf{1}[c_i=c]\cdot r_i
\]

\[
\mathcal{C}_x^{+}=\{c:\, q_{x,c}=1\}, \qquad
\text{Cover}_{\tau}(x)=
\begin{cases}
\frac{1}{|\mathcal{C}_x^{+}|}\sum_{c\in\mathcal{C}_x^{+}} \mathbf{1}[p_{x,c}\ge \tau], & |\mathcal{C}_x^{+}|>0\\[3pt]
0, & |\mathcal{C}_x^{+}|=0
\end{cases}
\]

Use prompt scalar \(R_x=\text{Cover}_{\tau}(x)\). Batch baseline and normalization:
\[
\mu_B=\frac{1}{B}\sum_{x\in B} R_x,\qquad
\sigma_B=\sqrt{\frac{1}{B}\sum_{x\in B}(R_x-\mu_B)^2+\epsilon},\qquad
A_x=\frac{R_x-\mu_B}{\sigma_B}
\]

Assign \(A_x\) to each rollout from prompt \(x\), and optimize GRPO/PPO clipped surrogate:
\[
\mathcal{L}_{\text{pg}}(\theta)=
\frac{1}{B}\sum_{x\in B}\frac{1}{N}\sum_{i=1}^{N}
\min\!\left(\rho_i(\theta)A_x,\ \text{clip}(\rho_i(\theta),1-\varepsilon,1+\varepsilon)A_x\right),
\quad
\rho_i(\theta)=\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}
\]

\[
\max_\theta\ \mathcal{J}(\theta)=\mathcal{L}_{\text{pg}}(\theta)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})
\]

Gradients flow only through \(\log\pi_\theta(y_i|x)\); clustering, thresholding, and \(R_x\) are stop-gradient statistics. If this objective is used, add LOO credit ablation \(A_{x,i}^{\text{LOO}}=\text{Cover}_\tau(x)-\text{Cover}_\tau(x\setminus y_i)-b\) because uniform within-prompt credit is a likely bottleneck.

### 5. Experimental plan
- **Minimal headline experiment:** Qwen-1.7B-Base, DaPO-17k, ~400 updates, same rollout budget and trainer across methods; train `cover_at_tau` vs GRPO vs majority-voting-trained baseline; evaluate on AIME-25, AIME-26, Beyond-AIME, HMMT.
- **Baselines (required + additional):** standard GRPO; majority-voting-trained model; PKPO (closest conceptual); F-GRPO (closest anti-collapse competitor); Poly-EPO (set-level diversity baseline, likely small-scale due complexity).
- **Ablations needed:** \(\tau \in \{0.05,0.1,0.2\}\); fixed vs scheduled \(\tau\); rollout count \(N\in\{8,16\}\); clustering substrate (exact final-answer match vs embedding clusters); prompt-uniform advantage vs LOO influence advantage; with/without KL tightening.
- **Compute estimate (Modal GPU-hours):** Modal pricing page currently lists A100-80GB at \$0.000694/s (\$2.50/hr) and H100 at \$0.001097/s (\$3.95/hr). For this scale, A100-80GB is sufficient for Qwen-1.7B RLVR. Budget rough-cut: 5 methods (GRPO, majority-train, PKPO, F-GRPO, cover-at-tau) x 3 seeds x 10 GPU-h/run \(\approx 150\) GPU-h; ablations + eval sweeps + reruns \(\approx 180\) GPU-h; total \(\approx 330\) GPU-h. Cost on A100 \(\approx 330 \times \$2.50 = \$825\); with 40% contingency \(\approx \$1155\), under \$1400.
- **Headline evaluation metric:** primary = Cover@tau on hard sets (Beyond-AIME/HMMT first), secondary = Pass@k curve and worst-quartile prompt accuracy. If primary gains come with severe Pass@1 collapse, claim fails.

### 6. Failure modes and consolation result
- **Failure mode 1: reward sparsity / flat gradients.**  
  **Observation:** slow or unstable learning curves; large seed variance; little movement vs GRPO despite similar KL and compute.
- **Failure mode 2: clustering noise dominates objective.**  
  **Observation:** Cover@tau (under one clustering rule) improves while exact-answer Pass@k and cross-clustering robustness do not; gains disappear under alternative clustering.
- **Failure mode 3: objective gaming via superficial mode splitting.**  
  **Observation:** model produces many superficially distinct but semantically redundant traces/answers that clear threshold without true hard-set accuracy gains.

- **Minimum publishable consolation result (if main hypothesis is null):**  
  A negative-result table showing that "train on eval metric" fails without robust credit assignment: compare GRPO, PKPO, cover-at-tau (uniform advantage), cover-at-tau (LOO advantage) on Cover@tau and Pass@k, plus a diagnostic figure of reward sparsity and gradient variance vs \(\tau\). That still yields a defensible claim about when metric-matching objectives break in RLVR.

### 7. Killer experiment
Under matched compute, `cover_at_tau` beats both PKPO and F-GRPO on Beyond-AIME and HMMT Cover@tau while maintaining near-par Pass@1.  
Expected figure: one Pareto plot (x-axis Pass@1, y-axis Cover@tau) with method points and confidence intervals, where cover-at-tau sits on the upper-right frontier.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **yes, conditional on** an early go/no-go against PKPO and F-GRPO in a 100-step pilot.
- This direction is on-pitch and high-ceiling because it directly operationalizes minority-voting coverage rather than proxying it. The main risk is not tractability but scientific compression: if it does not clearly beat PKPO/F-GRPO, it looks like thresholded reward shaping with noisy clustering. It is worth stage-3 only with strict falsification criteria (closest-baseline wins, not just GRPO wins) and early diagnostics on reward sparsity and clustering robustness.
# 1. Direction restatement (<=150 words)
Train directly on the metric you care about at test time: per-prompt Cover@τ over a rollout set. Instead of rewarding whether at least one sample is correct (Pass@k/PKPO) or multiplying reward by diversity (Poly-EPO), reward the policy when **multiple distinct correct answer clusters** each clear a support threshold τ. This is a concrete minority-voting instantiation because gradient mass goes to trajectories that help under-supported correct clusters cross the threshold, not just to the dominant correct mode. The claim is a smaller train-test mismatch for harder OOD reasoning sets where robust performance depends on covering several correct modes, not one lucky mode. Relative to the mentor pitch, this is still set-based minority optimization aimed at better worst-case generalization; it just chooses cluster-coverage-threshold as the minority target.

## 2. Related work scan (6-12 papers, last ~18 months)
- **arXiv:2604.17654 (Poly-EPO)** — set RL objective couples reward and diversity via product and set-level advantages. **Delta:** Cover@τ objective uses explicit cluster coverage threshold, not reward-diversity product/covariance.
- **arXiv:2505.15201 (PKPO)** — unbiased policy-gradient estimators for Pass@k; shows harder-task gains when optimizing pass@k directly. **Delta:** Cover@τ is multi-cluster coverage (how many modes clear τ), not single-event success probability.
- **arXiv:2602.06717 (F-GRPO)** — focal reweighting reduces sharpening of common solutions, recovering rare-correct modes at fixed group size. **Delta:** Cover@τ is a direct set-level target, not a prompt-difficulty weighting heuristic.
- **arXiv:2602.01062 (SetPO)** — set-level diversity advantage shaping from trajectory marginal contribution. **Delta:** SetPO optimizes semantic diversity globally; Cover@τ optimizes correctness-conditioned cluster support threshold.
- **arXiv:2503.14476 (DAPO)** — scalable GRPO system tricks (clip-higher, dynamic sampling, token-level PG) for stable RLVR. **Delta:** infrastructure baseline; no minority cluster-coverage objective.
- **arXiv:2504.16084 (TTRL)** — RL on unlabeled data using majority-vote pseudo-rewards. **Delta:** majority consensus reward; Cover@τ explicitly rewards minority-correct cluster support.
- **arXiv:2512.15146 (SCOPE)** — replaces plain majority pseudo-labeling with confidence-weighted subgroup consensus to reduce confirmation bias/sparsity. **Delta:** still pseudo-label aggregation; not direct training on correctness-cluster coverage threshold.
- **arXiv:2512.03847 (DVPO)** — distributional value modeling with risk-aware tail regularization for noisy supervision robustness. **Delta:** risk-shaped value estimation, not set-level cluster coverage as the optimization target.
- **arXiv:2502.06233 (CISC)** — weighted self-consistency reduces sample count via confidence-weighted voting. **Delta:** inference-time answer aggregation; Cover@τ is training-time RL objective over cluster support.

## 3. Novelty check
- **Specific claim:** Optimizing Cover@τ directly yields better OOD worst-case reasoning generalization than GRPO/majority-style objectives at matched compute, because it enforces support on multiple correct answer clusters during training.
- **Closest existing work:** PKPO (arXiv:2505.15201). PKPO trains on Pass@k, which is effectively "at least one success in k" and does not distinguish one dominant correct cluster from several sufficiently supported correct clusters.
- **Likely Ifdita objection (3 sentences):** "This looks like PKPO with a different reward transform plus brittle clustering. The hard part is not writing Cover@τ; it is making cluster assignment stable enough that the gradient is not pure noise. Unless you show gains over PKPO and Poly-EPO under equal rollout budget, this is metric overfitting, not a new algorithmic contribution."
- **Novelty rating:** **medium**.

## 4. Concrete training objective
For prompt \(x\), sample \(N\) rollouts \(y_{1:N}\sim \pi_\theta(\cdot|x)\).  
Let \(u_i=\text{cluster}(y_i)\), \(r_i\in\{0,1\}\) (verifiable correctness), and cluster support
\[
\hat p_c=\frac{1}{N}\sum_{i=1}^N \mathbf{1}[u_i=c].
\]
Define observed correct clusters
\[
\mathcal{C}^+(x)=\{c:\exists i,\ u_i=c,\ r_i=1\}.
\]
Per-prompt reward (training target):
\[
R_\tau(x,y_{1:N})=\frac{1}{\max(1,|\mathcal{C}^+(x)|)}\sum_{c\in\mathcal{C}^+(x)} \mathbf{1}\!\left[\hat p_c\ge \tau\right].
\]
Batch objective:
\[
J(\theta)=\mathbb{E}_{x}\ \mathbb{E}_{y_{1:N}\sim \pi_\theta}[R_\tau(x,y_{1:N})].
\]
REINFORCE estimator with prompt baseline \(b(x)\):
\[
\nabla_\theta J \approx \frac{1}{B}\sum_{b=1}^B\left(R_\tau^{(b)}-b(x_b)\right)\sum_{i=1}^{N}\nabla_\theta\log \pi_\theta(y_{b,i}|x_b).
\]
GRPO-style implementation (token-level clipped surrogate) uses
\[
A_{b,i}=\frac{R_\tau^{(b)}-\mu_B}{\sigma_B+\epsilon},\quad
\mathcal{L}_{\text{clip}}=-\frac{1}{BN}\sum_{b,i}\min\!\left(\rho_{b,i}A_{b,i},\operatorname{clip}(\rho_{b,i},1-\epsilon,1+\epsilon)A_{b,i}\right)+\beta\,\mathrm{KL}.
\]
Gradients flow only through \(\log\pi_\theta\); \(R_\tau\), clustering, and threshold indicators are treated as stop-gradient reward computations.

This is precise enough to implement; the true underspecification is not the loss formula but cluster definition quality (answer-only vs CoT-cluster).

## 5. Experimental plan
- **Minimal headline experiment:** Train Qwen-1.7B for 400 steps on DaPO-17k with identical trainer/hparams, swapping only objective: GRPO, majority-vote-trained baseline, PKPO, and Cover@τ. Report OOD Cover@τ on AIME-25/26, Beyond-AIME, HMMT.
- **Baselines (required + useful):** standard GRPO; majority-voting-trained objective; PKPO (closest); Poly-EPO-lite baseline (if available); F-GRPO (rare-mode rescue baseline).
- **Ablations needed:**
  - fixed \(\tau\in\{0.05,0.1,0.2\}\)
  - \(N\in\{8,16,32\}\) at fixed compute budget
  - clustering substrate (final-answer canonicalization vs embedding clustering)
  - reward normalization choice (denominator \(|\mathcal{C}^+|\) vs fixed target cluster count)
  - train on Cover@τ, eval on Pass@k and vice versa (explicit train-test gap check).
- **Compute estimate (Modal, checked):** A100-80GB is \$0.000694/s = **\$2.50/hour**.  
  Assume 10 training runs (main + baselines + key ablations), each ~40 GPU-hours (2xA100 for ~20h) \(\Rightarrow\) 400 GPU-hours = **\$1,000**.  
  Evaluation with multi-sample decoding + clustering: ~120 GPU-hours = **\$300**.  
  Buffer 10%: **\$130**.  
  **Total \(\approx \$1,430\)** (slightly over). Trim to 9 runs or lower eval sampling for **\(\approx \$1,280\)**, which fits \$1,400.
- **Headline metric:** primary **Cover@τ on OOD sets**; secondary Pass@k and worst-subset accuracy (hardest quartile prompts by base-model pass@64).

## 6. Failure modes and consolation result
- **Failure 1: threshold sparsity / high-variance reward.**  
  Observation: flat training curves, high variance across seeds, negligible improvement in either Cover@τ or Pass@k versus GRPO.
- **Failure 2: clusterer gaming.**  
  Observation: Cover@τ rises while human/answer-level distinctness does not; many "new clusters" are formatting artifacts or paraphrase-only variants.
- **Failure 3: diversity-forcing hurts primary correctness.**  
  Observation: Cover@τ up, but Pass@1 and majority-vote accuracy drop materially on easier sets; OOD gains vanish under strict answer canonicalization.

- **Minimum publishable result if null:** A controlled negative result figure showing that direct Cover@τ optimization is unstable/noisy relative to PKPO because thresholded cluster rewards induce poor credit assignment at small \(N\); include a table decomposing performance by \(N\), \(\tau\), and clustering method. This is still a defensible "when direct metric optimization fails" result.

## 7. Killer experiment
At matched rollout budget, Cover@τ-trained models beat PKPO and GRPO on OOD Cover@τ **without** significant Pass@1 regression.  
Expected figure: one Pareto plot (x=Pass@1, y=Cover@τ) where Cover@τ training shifts the frontier up-right on AIME-26/Beyond-AIME.

## 8. Overall verdict
- **Rating:** **promising**
- **Worth running stage 3 (initial experimentation) on?** **conditional on** a robust low-cost clustering substrate and a variance-reduced advantage estimator.

This direction is on-pitch and has a real upside: it tests the strongest possible "optimize what you evaluate" hypothesis for minority voting. The risk is substantial and specific: Cover@τ is thresholded, cluster-dependent, and likely noisier than PKPO under small rollout budgets. If stage-3 pilot shows stable gradients and at least one OOD Cover@τ gain at parity Pass@1, it deserves continuation; if not, it should be quickly downgraded to an ablation against PKPO rather than the project center.
