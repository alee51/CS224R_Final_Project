### 1. Direction restatement (<=150 words)
Inverse-frequency reweighting is a direct minority-voting variant: keep GRPO's base correctness reward, but multiply each rollout's advantage by an inverse frequency term computed from the prompt-local N-rollout answer distribution (or answer-cluster distribution). Rare correct trajectories then receive larger policy-gradient mass, while dominant correct trajectories are damped. This is distinct from QC-GRPO/PKPO-style tail objectives because "minority" is defined by low answer frequency, not reward quantiles. It remains a recognizable instantiation of the mentor pitch: both majority and minority behavior are induced from the same sampled set, then tested for harder-set generalization (AIME-25/26, Beyond-AIME, HMMT). Relative to Poly-EPO, this is intentionally lightweight: same clipping/KL machinery, no set-combinatorics objective, and no dependence on a heavy set-level judge loop.

### 2. Related work scan (6-12 papers, last ~18 months)
1. **arXiv:2604.17654 (Poly-EPO)** - Optimizes set-level reward x diversity with marginal set advantages; reports pass@k coverage and diversity gains.  
   **Delta vs this direction:** inverse-freq is per-trajectory GRPO reweighting with minority defined by local frequency, not set-level reward-diversity covariance.

2. **arXiv:2602.06717 (F-GRPO)** - Uses focal-style scaling to prevent policies from over-learning common/easy trajectories and forgetting rare-correct ones.  
   **Delta vs this direction:** closest neighbor; F-GRPO scales by prompt-level difficulty, while inverse-freq scales by within-prompt answer/cluster frequency.

3. **arXiv:2602.01062 (SetPO)** - Adds set-level diversity optimization via kernelized marginal contributions to preserve reasoning diversity.  
   **Delta vs this direction:** SetPO optimizes semantic set diversity directly; inverse-freq uses a cheaper frequency proxy and leaves objective form GRPO-like.

4. **arXiv:2505.15201 (PKPO)** - Provides unbiased estimators to optimize Pass@k directly and shows k-annealing improves harder reasoning.  
   **Delta vs this direction:** PKPO is tail-utility optimization; inverse-freq targets minority-by-rarity even when reward is binary and non-tail-differentiated.

5. **arXiv:2503.14476 (DAPO)** - Open RLVR system work (decoupled clipping, dynamic sampling, token-level updates) improves large-scale training stability/performance.  
   **Delta vs this direction:** DAPO is mostly training-system/optimization engineering; inverse-freq is an objective-level credit-allocation change.

6. **arXiv:2504.16084 (TTRL)** - Uses majority-vote pseudo-labels for test-time RL and demonstrates self-improvement on unlabeled data.  
   **Delta vs this direction:** TTRL changes supervision source (pseudo-labels); inverse-freq keeps verifiable rewards and changes gradient weighting only.

7. **arXiv:2512.15146 (Beyond Majority Voting...)** - Argues majority-vote pseudo-rewards are noisy and introduces finer-grained subgroup confidence signals for TTRL.  
   **Delta vs this direction:** both question majority-only signals, but this paper is test-time pseudo-reward refinement; inverse-freq is train-time minority weighting under RLVR.

8. **arXiv:2501.12948 (DeepSeek-R1)** - Shows GRPO-style RL can strongly improve reasoning but also motivates collapse/over-concentration concerns in RLVR practice.  
   **Delta vs this direction:** DeepSeek-R1 is the strong GRPO-family baseline paradigm; inverse-freq is a targeted anti-collapse minority credit mechanism on top.

### 3. Novelty check
- **Specific scientific claim:** Under binary RLVR, per-prompt inverse answer-frequency (or cluster-frequency) advantage weighting improves hard-set generalization (especially Cover@tau and worst-subset accuracy) at fixed compute versus standard GRPO and majority-voting-trained baselines.
- **Closest existing work and difference:** Closest is F-GRPO (arXiv:2602.06717). F-GRPO upweights hard prompts; this direction upweights rare answer clusters within each prompt's sampled rollouts, i.e., a finer-grained minority signal.
- **Ifdita 3-sentence objection (steelman):** "This is dangerously close to existing anti-collapse reweighting and may be novelty-thin unless the within-prompt frequency signal matters materially. Frequency over final answers is a brittle proxy for reasoning minority; formatting noise or weak clustering can produce fake rarity. If it does not clearly beat F-GRPO/SetPO-style baselines on hard OOD metrics, it is a variant, not a contribution."
- **Novelty rating:** **medium**.

### 4. Concrete training objective
For prompt \(x\), sample \(N\) rollouts \(y_i \sim \pi_{\theta_{\text{old}}}(\cdot|x)\), rewards \(r_i \in \{0,1\}\), and answer-cluster IDs \(c_i\).

\[
n_{x,c}=\sum_{j=1}^{N}\mathbf{1}[c_j=c],\qquad
\tilde w_i=(n_{x,c_i})^{-\gamma},\ \gamma\in[0,1].
\]

Per-prompt normalization (scale-preserving):
\[
w_i=\frac{N\tilde w_i}{\sum_{j=1}^{N}\tilde w_j}.
\]

Base GRPO advantage:
\[
b_x=\frac{1}{N}\sum_{j=1}^{N}r_j,\qquad A_i=r_i-b_x.
\]

Inverse-frequency weighted advantage:
\[
A_i^{\text{inv}}=w_iA_i.
\]

Clipped surrogate:
\[
\mathcal{L}_{\text{pg}}(\theta)=
\frac{1}{B}\sum_{x}\frac{1}{N}\sum_{i=1}^{N}
\min\!\left(
\rho_i(\theta)A_i^{\text{inv}},
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)A_i^{\text{inv}}
\right),
\]
\[
\rho_i(\theta)=\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}.
\]

Full objective:
\[
\max_\theta\ \mathcal{J}(\theta)=\mathcal{L}_{\text{pg}}(\theta)-\beta\,\mathrm{KL}\!\big(\pi_\theta\|\pi_{\text{ref}}\big).
\]

Gradient flows only through \(\log\pi_\theta(y_i|x)\). The statistics \(r_i,c_i,n_{x,c},w_i,b_x\) are stop-gradient within each update batch.

### 5. Experimental plan
- **Minimal headline experiment:** Qwen-1.7B-Base, DaPO-17k, ~400 steps, fixed rollout budget and KL target. Compare `inverse_freq` vs GRPO vs majority-voting-trained model on AIME-25/26, Beyond-AIME, HMMT; report Pass@k and Cover@tau.
- **Baselines (required + additional):** standard GRPO; majority-voting-trained model; F-GRPO (closest mechanism baseline); PKPO (tail baseline); SetPO (set-level diversity baseline, if implementation feasible).
- **Ablations for defensible claim:** answer-frequency vs cluster-frequency weights; \(\gamma\in\{0,0.5,1\}\); with/without per-prompt normalization; weight clipping \(w_i\le w_{\max}\); rollout count \(N\in\{8,16\}\); answer canonicalization quality (exact string vs normalized math form).
- **Compute estimate in Modal GPU-hours:** Modal pricing page reports A100-80GB at \$0.000694/s (\$2.4984/h). Use A100-80GB as default class for 1.7B RLVR. Assume one 400-step train run ~= 12 GPU-h. Main matrix: 5 methods x 3 seeds = 15 runs => 180 GPU-h. Ablations + eval sweeps + reruns: ~120 GPU-h. Total ~= 300 GPU-h => ~\$750; with 80% contingency: ~540 GPU-h => ~\$1349. Fits \$1400, but only if method count is kept tight.
- **Headline number:** primary = Cover@tau on Beyond-AIME/HMMT (minority-coverage claim); secondary = worst-quartile prompt accuracy and Pass@k frontier at matched Pass@1.

### 6. Failure modes and consolation result
- **Failure mode 1: proxy mismatch (fake rarity).**  
  **Observe:** higher rarity-weighted diversity metrics but flat/negative Cover@tau and hard-set Pass@k; qualitative audit shows many formatting variants of same reasoning.
- **Failure mode 2: weight-induced instability.**  
  **Observe:** elevated KL spikes, frequent clip saturation, high seed variance, and Pass@1 degradation without compensating hard-set gains.
- **Failure mode 3: nearest-baseline non-separation.**  
  **Observe:** GRPO < inverse-freq, but F-GRPO/SetPO match or beat inverse-freq, making contribution non-distinct.

- **Minimum publishable result if null:** a mechanistic negative result table + figure showing when within-prompt frequency weighting misallocates gradient mass (cluster-rank vs gradient-mass plots), plus benchmark outcomes where this misallocation predicts no hard-generalization gain.

### 7. Killer experiment
At matched compute and Pass@1, inverse-frequency reweighting produces statistically significant Cover@tau and worst-subset accuracy gains over both GRPO and F-GRPO on Beyond-AIME/HMMT (3 seeds).  
Expected figure: one panel with delta Cover@tau and CIs vs baselines, one panel linking gradient-mass shift toward rare-correct clusters to those gains.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **conditional on** early head-to-head win over F-GRPO on at least one hard benchmark.
- Direction fit is good: it is clearly minority-voting-centered and not a Poly-EPO scaling detour (consistent with `../archive/poly_epo/findings.md` and `../archive/poly_epo/why_stop.md`). The upside is fast implementation and a clean mechanistic story; the downside is novelty compression against nearby reweighting papers. This is worth a short, strict Stage-3 probe with explicit go/no-go criteria based on separation from F-GRPO, not just improvement over vanilla GRPO.
