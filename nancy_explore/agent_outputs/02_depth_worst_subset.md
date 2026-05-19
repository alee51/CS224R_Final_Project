### 1. Direction restatement (<=150 words)
Worst-Subset Set Objective is a direct minority-voting instantiation inside the same set-RL scaffold as Poly-EPO: sample \(N\) rollouts per prompt, form size-\(n\) subsets, score each subset by subset correctness, and optimize the policy against the worst subsets (minimum or lower-tail average) instead of a reward-diversity product. This operationalizes the mentor pitch as worst-case set performance optimization: not "is the majority correct?" but "how bad are the weakest voting coalitions?" If this works, the claim is stronger robustness/generalization on hard reasoning sets, since training pressure explicitly targets low-support failure modes rather than mean behavior. Relative to majority-voting optimization, this objective flips training pressure from central tendency to tail risk. Relative to Poly-EPO, it removes explicit diversity weighting and asks whether tail-subset pressure alone is sufficient.

### 2. Related work scan (6-12 papers, last ~18 months)
1. **arXiv:2604.17654 (Poly-EPO)**  
   Claim: set-RL objective \(\bar r \times\) strategy-diversity improves Pass@k/generalization and preserves diversity.  
   Delta vs this direction: same set-RL credit machinery, but aggregator is diversity-synergy; worst-subset replaces it with explicit lower-tail set risk.

2. **arXiv:2602.01062 (SetPO)**  
   Claim: leave-one-out set-level diversity contribution improves Pass@1/Pass@k while reducing mode collapse.  
   Delta vs this direction: SetPO is diversity-shaping over semantic similarity; worst-subset is correctness-tail shaping over subset outcomes.

3. **arXiv:2602.06717 (F-GRPO)**  
   Claim: focal-style difficulty weighting mitigates rare-correct miss at practical group sizes and boosts pass@256.  
   Delta vs this direction: F-GRPO is prompt-level reweighting in group-relative RL; worst-subset is prompt-internal subset-tail objective.

4. **arXiv:2505.15201 (PKPO)**  
   Claim: unbiased low-variance estimators directly optimize Pass@k and unblock harder-example learning.  
   Delta vs this direction: PKPO optimizes max-over-k sample success; worst-subset optimizes min/lower-tail subset success (risk-averse instead of optimistic).

5. **arXiv:2503.14476 (DAPO)**  
   Claim: scalable RLVR system (clip-higher, dynamic sampling, token-level PG, overlong shaping) improves large-scale reasoning RL reproducibility.  
   Delta vs this direction: DAPO is infrastructure/optimizer recipe; worst-subset is an objective-level scientific intervention.

6. **arXiv:2512.03847 (DVPO: Distributional Value Modeling-based PO)**  
   Claim: distributional value modeling with tail-aware regularization improves robustness under noisy/incomplete supervision.  
   Delta vs this direction: DVPO models return distributions via critic architecture; worst-subset keeps verifier rewards and injects risk sensitivity at set objective level.

7. **arXiv:2512.15146 (SCOPE, TTRL extension)**  
   Claim: confidence-weighted subgroup consensus improves pseudo-label quality beyond majority voting in test-time RL.  
   Delta vs this direction: SCOPE is test-time pseudo-labeling; worst-subset is train-time policy objective with verifiable rewards.

8. **arXiv:2508.11356 (ETTRL)**  
   Claim: entropy-aware rollout and advantage shaping improve exploration/exploitation in test-time RL with lower token budget.  
   Delta vs this direction: ETTRL modifies test-time adaptation; worst-subset modifies offline/online RLVR training objective directly.

### 3. Novelty check
- **Specific claim:** Optimizing the lower tail of subset-level correctness (via worst or CVaR-like subset aggregation) yields better worst-case reasoning generalization than majority-oriented training at equal model/data/compute.
- **Closest existing work:** Poly-EPO (arXiv:2604.17654). Difference: Poly-EPO's core signal is reward-diversity covariance; this direction's core signal is lower-tail subset correctness risk.
- **Ifdita pushback (steelman, 3 sentences):** "This is very close to existing set-RL machinery and may collapse to a harsher reweighting of binary rewards rather than a new learning principle. Without explicit diversity terms, the model can overfit to adversarially hard subsets and hurt both Pass@1 and practical Pass@k. You must show a regime where worst-subset pressure improves hard-set generalization beyond Poly-EPO/PKPO, not just different tradeoffs."
- **Novelty rating:** **medium**.

### 4. Concrete training objective
Let prompt \(x\) have \(N\) rollouts \(y_{1:N}\sim\pi_\theta(\cdot|x)\), verifier rewards \(r_i\in\{0,1\}\), and all (or sampled) size-\(n\) subsets \(\mathcal{G}_x=\{G_j\}_{j=1}^K\), \(K=\binom{N}{n}\).

\[
s(G_j)=\frac{1}{n}\sum_{i\in G_j} r_i
\]
(alternative: \(s(G_j)=\mathbf{1}[\text{minority-vote in }G_j\text{ is correct}]\)).

Define lower-tail selector \(w_q(G_j)\):
\[
w_q(G_j)=\frac{\mathbf{1}[s(G_j)\le Q_q(s)]}{\sum_{\ell=1}^K \mathbf{1}[s(G_\ell)\le Q_q(s)]}
\]
where \(Q_q\) is the \(q\)-quantile (\(q\!=\!1/K\) approximates hard minimum).

Tail set objective and baseline:
\[
J_{\text{tail}}(x)=\sum_{j=1}^K w_q(G_j)\,s(G_j), \qquad
b(x)=\frac{1}{K}\sum_{j=1}^K s(G_j)
\]

Set advantage and marginal rollout advantage:
\[
A^\#(G_j)=w_q(G_j)\big(s(G_j)-b(x)\big)
\]
\[
A_i(x)=\frac{1}{|\{j:i\in G_j\}|}\sum_{j:i\in G_j} A^\#(G_j)
\]

Batch policy loss (GRPO/PPO-style clip omitted for brevity; plug \(A_i\) into existing clipped surrogate):
\[
\mathcal{L}_{\text{WS}}(\theta)=-
\mathbb{E}_{x}\left[
\frac{1}{N}\sum_{i=1}^N
\operatorname{sg}(A_i(x))
\sum_{t}\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
\right]
+\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
\]
Gradient flows only through \(\log\pi_\theta\); \(A_i\) is treated as stop-grad, exactly like Poly-EPO's marginal set-advantage machinery.

### 5. Experimental plan
- **Minimal headline experiment:** Train Qwen-1.7B-Base on DaPO-17k (~400 steps) with identical trainer/hparams and three objectives: (i) standard GRPO, (ii) majority-voting-trained baseline, (iii) worst-subset objective (\(N=8,n=4\), \(q=0.1\)); evaluate on AIME-25/26, Beyond-AIME, HMMT.
- **Baselines (required + necessary):** standard GRPO; majority-voting-trained model; Poly-EPO (same \(N,n\), same rollout budget); PKPO (closest Pass@k-tail comparator).
- **Ablations to defend claim:** hard-min vs CVaR tail (\(q\in\{1/K,0.05,0.1,0.2\}\)); subset score definition (fraction-correct vs minority-vote-correct); sampled subsets vs full enumeration; \(n\in\{2,4,6\}\) at fixed \(N\); with/without KL tightening (tail objectives can destabilize).
- **Compute estimate (Modal):** Use A100-80GB at \$0.000694/s \(\approx\$2.50/h\) (fits 1.7B RL + rollouts with headroom, cheaper than H100). Assume ~18 GPU-h per 400-step run (conservative for rollout-heavy RLVR).  
  - Core comparison: 4 methods × 2 seeds × 18 h = 144 GPU-h \(\approx\$360\).  
  - Key ablations: 8 extra runs × 18 h = 144 GPU-h \(\approx\$360\).  
  - Eval/generation overhead (AIME/Beyond-AIME/HMMT at multi-sample): ~180 GPU-h on L40S (\$1.95/h) \(\approx\$351\).  
  - Buffer/failed runs contingency (35%): \(\approx\$375\).  
  - **Total \(\approx \$1,446\)** (slightly over); trim to 6 ablation runs or 1 seed on one baseline gives \(\approx\$1,300\), so **yes, fits \$1400 with a disciplined run matrix**.
- **Headline metric:** **worst-case-subset accuracy** on eval rollouts (same \(N,n\) as training) + Pass@k secondary. This direction is not convincing if it only moves mean Pass@1.

### 6. Failure modes and consolation result
- **Failure 1: Tail overfitting / mean collapse.**  
  Observable: worst-subset metric improves slightly, but Pass@1 and Pass@k on AIME/HMMT drop materially versus GRPO/majority.
- **Failure 2: Sparse/unstable gradients from hard minimum.**  
  Observable: high variance across seeds, oscillatory training reward, frequent divergence unless KL or \(q\) is softened.
- **Failure 3: Objective degeneracy under binary rewards.**  
  Observable: subset-tail score is nearly monotone in prompt difficulty only, giving little extra signal beyond prompt-level difficulty weighting (performance close to F-GRPO-style baselines).

- **Minimum publishable result if null:**  
  A clean negative-result table: "Tail set objectives (min/CVaR) vs Poly-EPO vs PKPO vs GRPO under fixed compute," showing where worst-subset helps/hurts, plus a diagnostic figure of tail-metric gain vs Pass@1 loss frontier. Even if no SOTA gain, this is a defensible contribution on objective geometry for minority-voting training.

### 7. Killer experiment
Under equal compute, worst-subset training beats majority-voting-trained and GRPO on Beyond-AIME/HMMT **worst-case-subset accuracy** while keeping Pass@1 within 1 point.  
Expected figure: one Pareto plot (x-axis Pass@1, y-axis worst-case-subset accuracy) where worst-subset is the only method on the upper-right frontier on hard OOD sets.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **conditional on X** (X = proving non-trivial gains on worst-case-subset metric without large Pass@1 collapse in a small pilot).
- This direction is on-pitch and scientifically sharper than another Poly-EPO knob sweep (already ruled out in `findings.md` / `why_stop_poly_epo.md`), because it tests a distinct risk-sensitive hypothesis at the set level. The risk is that it is "too close" to existing set-RL machinery and may collapse into a harsh reweighting trick under binary rewards. Still, as an objective-level comparator against Poly-EPO/PKPO with identical infrastructure, it has real ceiling: either a positive robustness result (best case) or a strong negative map of where lower-tail set training fails (still publishable at class-project scale). It is not a safe direction, but it is directionally high-value enough to merit a tightly scoped stage-3 pilot.
### 1. Direction restatement (<=150 words)
This direction is a direct minority-voting instantiation: instead of rewarding the policy for doing well on average over sampled rollouts, train it to improve the weakest subsets of rollouts for each prompt. Concretely, sample `N` rollouts, build size-`n` subsets, score each subset by subset correctness, and optimize the minimum or lower-tail subset score; then assign trajectory credit through Poly-EPO-style marginal set advantages. Relative to the mentor pitch, this is "minority voting at set level" rather than "majority voting at sample level": majority-like objectives tolerate many bad modes if enough rollouts are correct, while worst-subset objectives explicitly punish brittle pockets of failure. This aligns with the intended claim: better worst-case robustness and harder-test-set generalization (AIME-26, Beyond-AIME, HMMT), not just better Pass@1.

### 2. Related work scan (6-12 papers, last ~18 months)
1. **arXiv:2604.17654 (Poly-EPO)**  
Claim: Set-RL objective coupling reward and diversity improves pass@k coverage/generalization.  
Delta: Same set-RL machinery, but objective shifts from reward×diversity to worst-tail subset correctness.

2. **arXiv:2602.01062 (SetPO)**  
Claim: Set-level diversity marginal credit mitigates mode collapse and improves Pass@1/Pass@K.  
Delta: SetPO optimizes semantic diversity; worst-subset optimizes risk concentration in low-performing subsets.

3. **arXiv:2602.06717 (F-GRPO)**  
Claim: Difficulty-aware scaling prevents rare-correct forgetting at practical group sizes.  
Delta: F-GRPO is trajectory-level reweighting; worst-subset is explicit subset-level tail-risk optimization.

4. **arXiv:2505.15201 (PKPO)**  
Claim: Unbiased pass@k optimization unblocks learning on hard tasks by optimizing joint utility.  
Delta: PKPO targets best-of-k success; worst-subset targets low-tail subset performance (opposite tail).

5. **arXiv:2503.14476 (DAPO)**  
Claim: Scaled RLVR system improvements (sampling/clipping/token-level details) substantially boost reasoning.  
Delta: DAPO is systems/optimizer-centric; worst-subset is an objective-level scientific claim about minority robustness.

6. **arXiv:2504.16084 (TTRL)**  
Claim: Majority-vote pseudo-labeling enables unsupervised test-time RL gains.  
Delta: TTRL improves consensus training signal; worst-subset intentionally deviates from consensus toward adversarial minority sets.

7. **arXiv:2512.15146 (SCOPE / beyond majority voting for TTRL)**  
Claim: Confidence-weighted subgroup consensus improves over plain majority-vote pseudo-labeling.  
Delta: SCOPE refines pseudo-label quality; worst-subset changes the policy objective itself to optimize lower-tail subsets.

8. **arXiv:2512.03847 (DVPO)**  
Claim: Distributional value modeling with asymmetric tail regularization improves robustness/generalization under noisy supervision.  
Delta: DVPO is critic-side distributional risk shaping; worst-subset is actor-side set-risk shaping with no distributional critic.

9. **arXiv:2601.19280 (GDRO-driven RL for LLM reasoning)**  
Claim: Group DRO over prompt/rollout allocation improves pass@k by focusing on hard groups under fixed budget.  
Delta: GDRO reallocates sampling budget across groups; worst-subset changes within-prompt objective on fixed samples.

10. **arXiv:2502.06233 (CISC, confidence-informed self-consistency)**  
Claim: Confidence-weighted voting reduces self-consistency compute while improving accuracy.  
Delta: CISC is inference-time weighted voting; worst-subset is training-time RL objective over subset tails.

### 3. Novelty check
- **Specific scientific claim:** Training on lower-tail subset correctness (instead of mean/majority-like objectives) improves worst-case reasoning generalization on hard OOD math sets, at fixed model/data/compute.
- **Closest existing work:** Poly-EPO (set-level marginal credit assignment) + PKPO (set utility optimization).  
  Difference: Poly-EPO optimizes exploration-exploitation synergy via diversity; PKPO optimizes upper-tail success; this direction explicitly optimizes lower-tail subset performance.
- **Ifdita-style 3-sentence objection:** "This is mostly a different aggregator on top of the set-RL recipe we already wrote down. Unless you show a regime where worst-tail training beats both GRPO and majority-oriented set training on hard OOD while preserving useful pass@k behavior, this is just risk-sensitive reweighting. Also, hard-min objectives are noisy and may anti-train by overfitting to pathological subsets."
- **Novelty rating:** **medium**.

### 4. Concrete training objective
Let prompt \(x\) produce \(N\) rollouts \(y_1,\dots,y_N \sim \pi_\theta(\cdot|x)\), binary rewards \(r_i \in \{0,1\}\). Construct \(K\) size-\(n\) subsets \(G_1,\dots,G_K\) (all \(\binom{N}{n}\) or uniform sample).

\[
s_j \;=\; \frac{1}{n}\sum_{y_i\in G_j} r_i
\]
\[
m=\lceil \rho K\rceil,\quad s_{(1)}\le \cdots \le s_{(K)}
\]
\[
w_j \;=\; \frac{1}{m}\,\mathbf{1}[s_j \le s_{(m)}] \quad\text{(bottom-\(\rho\) tail weight)}
\]
\[
f_{\text{ws}}(x,G_j)=w_j s_j,\qquad
\hat f(x)=\frac{1}{K}\sum_{j=1}^K f_{\text{ws}}(x,G_j)
\]
\[
\hat A^\#_j = f_{\text{ws}}(x,G_j)-\hat f(x)
\]
\[
\hat A_i=\frac{1}{| \mathcal G(y_i)|}\sum_{j:\,y_i\in G_j}\hat A^\#_j
\]
\[
\mathcal L_{\text{policy}}(\theta)=
-\frac{1}{|B|}\sum_{x\in B}\frac{1}{N}\sum_{i=1}^N
\min\!\Big(r_i(\theta)\hat A_i,\;\text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)\hat A_i\Big)
\]
where \(r_i(\theta)=\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\).  
Per-sample advantage is \(\hat A_i\); baseline is \(\hat f(x)\); per-batch aggregation is mean over prompts and rollouts; gradients flow through \(\log\pi_\theta(y_i|x)\) terms only (treat \(w_j,s_j,\hat A_i\) as stop-grad REINFORCE weights).  
If \(\rho=1/K\), this becomes hard-min subset optimization.

### 5. Experimental plan
- **Minimal headline experiment:** Qwen-1.7B-Base, DaPO-17k, 400 steps, compare `GRPO` vs `majority-oriented set objective` vs `worst-subset objective` (same \(N,n\), same trainer), evaluate on AIME-25/26, Beyond-AIME, HMMT with Pass@k and worst-case-subset accuracy.
- **Baselines (required + additional):** standard GRPO; majority-voting-trained model (set objective targeting upper/mean utility); Poly-EPO; PKPO-style pass@k objective; optionally SetPO as diversity-focused control.
- **Ablations needed:** \(\rho\in\{1/K,0.1,0.2,0.3\}\); subset size \(n\in\{2,4,6\}\) at fixed \(N\); all-subsets vs sampled-subsets \(K\); binary hard-min vs lower-tail average; with/without entropy/KL stabilization.
- **Compute estimate (Modal):** use A100-80GB at current listed \( \$0.000694/s \approx \$2.50/h\) (Modal pricing page). Assume ~14 GPU-h per 400-step run at this setup.  
  Core matrix: 6 configs × 3 seeds = 18 runs \(\approx 252\) GPU-h (\(\$630\)).  
  Eval (64-sample decoding + metrics across 4 test sets/checkpoints): \(\approx 72\) GPU-h (\(\$180\)).  
  Debug/retries/extra ablations buffer: \(\approx 120\) GPU-h (\(\$300\)).  
  **Total \(\approx 444\) GPU-h, \(\$1110\)**. Fits under **\$1400** with ~\$290 margin.
- **Headline metric:** primary = **worst-case-subset accuracy** (bottom-\(\rho\) subset correctness on held-out prompts), secondary = Pass@1/Pass@k and Cover@\(\tau\) for comparability with mentor guidance.

### 6. Failure modes and consolation result
- **Failure 1: gradient brittleness from hard tail focus.**  
  Observation: unstable training curves, high variance across seeds, degraded Pass@1 without tail gain.
- **Failure 2: objective mismatch to eval reality.**  
  Observation: improves synthetic worst-subset metric but no gain (or regression) on AIME-26/Beyond-AIME/HMMT Pass@k or Cover@\(\tau\).
- **Failure 3: anti-learning via pathological subsets.**  
  Observation: model over-upweights rare wrong modes, answer entropy rises while correctness drops; tail metric noisy/improvable only in-train.

- **Minimum publishable result if null:** a clean negative result table showing that lower-tail subset objectives underperform majority-oriented objectives at equal compute, plus an ablation figure mapping when tail focus flips from robustness-improving to anti-learning as \((N,n,\rho)\) vary.  
  Concretely: one main figure "OOD worst-subset accuracy vs Pass@k tradeoff frontier" and one ablation table over \(\rho,n\).

### 7. Killer experiment
Show that, at equal training/eval compute, worst-subset training beats GRPO and majority-oriented set training on AIME-26 and Beyond-AIME in worst-case-subset accuracy **and** improves Cover@\(\tau\) without collapsing Pass@1.  
Expected figure: a 3-method Pareto plot (x: Pass@1 or Pass@k, y: worst-subset accuracy/Cover@\(\tau\)) where worst-subset dominates frontier points on hard OOD sets.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **yes, conditional on** using lower-tail average (\(\rho>1/K\)) instead of pure hard-min and pre-registering failure criteria.
- This is in-direction and scientifically sharper than "just tune Poly-EPO knobs," while still reusing established set-RL machinery (as already synthesized in `findings.md`/`simulation_results.md`). The upside is a clean, testable claim about minority robustness under fixed compute; the downside is novelty compression (it can look like "just an aggregator swap") and optimization fragility. It is worth a stage-3 shot because it has a falsifiable claim, a direct majority-vs-minority comparison story, and a credible negative-result fallback, but only if experiments are framed around hard OOD robustness rather than headline Pass@1 alone.
