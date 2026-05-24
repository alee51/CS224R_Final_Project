### 1. Direction restatement (<=150 words)
Per-token uncertainty minority is a valid minority-voting instantiation: instead of defining "minority" by low answer frequency in a sampled set, define it by low-probability token choices under the model's own policy. For each trajectory, compute a self-surprise score from per-token log-probs (high surprise = policy-tail trajectory). Then keep the mentor's majority/minority framing by upweighting **correct** trajectories that come from this low-confidence tail, i.e., trajectories that "beat the model's confident prior." This targets the same hypothesis as minority voting (improve hard/OOD robustness by preserving rare useful modes) but removes clustering, LM-judge dependencies, and answer-frequency estimation. The risk is conceptual drift: this is "minority under policy mass," not "minority under sampled vote counts." If it wins, the contribution is a cleaner, lower-variance minority signal with cheaper training infrastructure.

### 2. Related work scan (6-12 papers, last ~18 months)
- **arXiv:2604.17654** (Poly-EPO): set-level exploration-exploitation objective improves pass@k and diversity; **delta**: token-uncertainty minority is per-trajectory/token confidence shaping, no set clustering objective.
- **arXiv:2505.15201** (Pass@K Policy Optimization): directly optimizes pass@k via multi-sample reward transforms; **delta**: this direction does not estimate pass@k target statistics, it uses internal policy uncertainty as minority proxy.
- **arXiv:2602.06717** (F-GRPO): focal-style weighting preserves rare-correct modes under sampling bias; **delta**: rarity is estimated from prompt-level outcomes, not token-level surprisal.
- **arXiv:2602.01062** (SetPO): set-level kernel diversity objective prevents collapse; **delta**: requires set similarity machinery, while token-uncertainty needs only rollout log-probs.
- **arXiv:2503.14476** (DAPO): RLVR system innovations incl. token-level PG improvements and stability engineering; **delta**: systems/training stabilization focus, not minority definition via uncertainty.
- **arXiv:2504.16084** (TTRL): uses majority-vote pseudo rewards on unlabeled test data; **delta**: majority label source is external vote consensus, not model-internal uncertainty.
- **arXiv:2512.15146** (Beyond Majority Voting for TTRL): critiques coarse majority reward with finer pseudo-labeling; **delta**: still vote-aggregation centric, while this direction is policy-tail centric.
- **arXiv:2510.10649** (Uncertainty-aware Advantage Shaping / UCAS): leverages self-confidence for response/token credit assignment in RLVR; **delta**: closest prior, but this direction makes a sharper claim: uncertainty itself is the minority-voting surrogate and primary objective lens.

### 3. Novelty check
- **Specific claim:** In RLVR for reasoning, defining minority as high self-surprise (low policy-probability token trajectories) and upweighting high-surprise correct traces improves hard-set generalization (Cover@tau / high-k pass) at similar or better Pass@1 than frequency-based minority proxies.
- **Closest existing work:** UCAS (arXiv:2510.10649). Difference: UCAS is an uncertainty-shaped credit assignment method; this project's central scientific framing is a **minority-voting reinterpretation** where policy-tail membership is the minority variable replacing vote-frequency minority.
- **Ifdita steelman objection (3 sentences):** "This looks like uncertainty-weighted GRPO, not minority voting over sets. Minority voting in the pitch is about cross-sample answer structure; token surprisal can rise for stylistic noise and may decouple from reasoning diversity. Unless you show this proxy tracks low-frequency-correct answer modes better than explicit vote minority, this is rebranding, not a new project."
- **Novelty rating:** **medium**.

### 4. Concrete training objective
Let prompt \(x\) produce \(N\) rollouts \(y_i=(y_{i,1},...,y_{i,T_i})\) from \(\pi_{\theta_{\text{old}}}\), with binary reward \(r_i\in\{0,1\}\).

\[
s_i = -\frac{1}{T_i}\sum_{t=1}^{T_i}\log \pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})
\]

\[
\bar r = \frac{1}{N}\sum_{j=1}^N r_j,\quad
A_i^{\text{grpo}} = r_i-\bar r
\]

Prompt-local normalization:
\[
\mu_s=\frac{1}{N}\sum_j s_j,\quad \sigma_s^2=\frac{1}{N}\sum_j (s_j-\mu_s)^2,\quad
z_i=\frac{s_i-\mu_s}{\sigma_s+\epsilon}
\]

Minority weight (smooth):
\[
w_i = 1 + \lambda\cdot r_i\cdot \mathrm{softplus}(z_i-\tau)
\]

Shaped advantage:
\[
\tilde A_i = w_i\cdot A_i^{\text{grpo}}
\]

Token-level PPO/GRPO loss:
\[
\mathcal{L}(\theta)= -\frac{1}{|B|}\sum_{(x,i)\in B}\frac{1}{T_i}\sum_{t=1}^{T_i}
\min\!\left(\rho_{i,t}\tilde A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon_c,1+\epsilon_c)\tilde A_i\right)
+\beta\,\mathrm{KL}\!\left[\pi_\theta(\cdot|x,y_{i,<t})\|\pi_{\text{ref}}(\cdot|x,y_{i,<t})\right]
\]
with \(\rho_{i,t}=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\).

Gradient flow: through \(\rho_{i,t}\) and KL only; \(s_i,z_i,w_i,\tilde A_i\) are treated as detached statistics from rollout policy (\(\theta_{\text{old}}\)).

This is sufficiently specified for implementation.

### 5. Experimental plan
- **Minimal headline experiment:** Train Qwen-1.7B on DaPO-17k for 400 steps with (a) GRPO, (b) majority-voting-trained objective, (c) token-uncertainty minority objective above; evaluate on AIME-25/26, Beyond-AIME, HMMT with Pass@1/16/64 + Cover@tau.
- **Required baselines:** standard GRPO; majority-voting-trained model (set-level majority-consensus reward shaping). Add PKPO (arXiv:2505.15201) and F-GRPO (arXiv:2602.06717) if engineering budget allows.
- **Ablations:** (1) trajectory surprise metric (mean NLL vs fraction of tokens below per-position \(p\)-quantile), (2) only response-level weighting vs response+token-local weighting, (3) apply uncertainty bonus to all trajectories vs correct-only, (4) \(\lambda\in\{0,0.25,0.5,1.0\}\), \(\tau\in\{0,0.5,1.0\}\), (5) calibration check: correlation between \(s_i\) and empirical answer rarity.
- **Compute estimate (Modal):** From Modal pricing, A100-80GB is \( \$0.000694/s \approx \$2.50/h\). Qwen-1.7B RLVR should fit comfortably; A100 is cheaper than H100 and sufficient for this scale. Budget sketch: 12 full runs (4 methods/ablations x 3 seeds) x 30 GPU-h/run = 360 GPU-h (\$900), plus eval generation and metric sweeps ~120 GPU-h (\$300), plus 20% overhead (\$240) => **\$1,440** (slightly over). Trim to 10 full runs or 2 seeds for secondary ablations: 420 GPU-h total => **\$1,050**, within \$1400.
- **Headline metric:** **Cover@tau on hard splits (Beyond-AIME/HMMT)** at matched Pass@1 band; secondary: Pass@64.

### 6. Failure modes and consolation result
- **Failure 1: Uncertainty captures noise, not useful minority reasoning.** Observation: higher self-surprise raises output diversity but not correctness; Cover@tau flat/down while wrong-answer entropy increases.
- **Failure 2: Over-upweighting destabilizes PPO updates.** Observation: KL spikes, oscillatory reward curves, and degraded Pass@1 despite temporary Pass@k gains.
- **Failure 3: Proxy mismatch with minority-vote goal.** Observation: weak correlation between self-surprise and low-frequency-correct clusters; explicit frequency-based minority methods outperform on hardest subsets.
- **Minimum publishable result if null:** A clean negative table showing that policy-tail surprisal is a poor surrogate for minority-correctness compared with answer-frequency minority weighting, plus calibration plots (surprise vs correctness vs rarity). A defensible artifact is a "proxy validity" figure: x-axis surprisal decile, y-axis correctness and rarity rates across datasets.

### 7. Killer experiment
If token-uncertainty minority training improves Beyond-AIME/HMMT Cover@tau by a clear margin over both GRPO and majority-voting training at matched Pass@1, the project is paper-worthy.
Expected figure: one Pareto curve panel (Pass@1 vs Cover@tau) where the token-uncertainty frontier strictly dominates both baselines on hard OOD sets.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **conditional on X**.
- **X:** first-run proxy-validity check must show self-surprise is positively associated with low-frequency-correct trajectories; if not, stop quickly.
- The upside is real: this is the cleanest minority surrogate operationally (no clustering, no LM judge, no answer-frequency bookkeeping), and it stays aligned with the mentor's minority-voting spirit if framed as "minority under policy mass." The downside is novelty compression against uncertainty-aware RLVR work and a serious risk that token surprisal measures stylistic randomness rather than meaningful minority reasoning. So this is worth stage-3 only as a **high-speed falsifiable bet**: run proxy validation + one short training comparison first, then either scale or kill.
### 1. Direction restatement (<=150 words)
Per-Token Uncertainty Minority keeps the mentor’s majority/minority-voting spirit but changes the minority definition: instead of “minor answer class within sampled outputs,” minority means “trajectories drawn against the model’s own confident prior.” Concretely, each rollout gets a self-surprise score from token log-probs under the behavior policy at generation time; correct trajectories with unusually high surprise get extra weight in GRPO. The intended claim is: forcing learning signal toward rare-under-policy (but verifiably correct) trajectories should improve worst-case and hard-OOD reasoning coverage, analogous to minority voting improving tail behavior relative to majority-style optimization. This is still a recognizable instantiation of the pitch because training compares common vs rare solution modes within each sampled set, but rarity is now policy-density rarity rather than answer-frequency rarity.

### 2. Related work scan (6-12 papers, last ~18 months)
- **arXiv:2604.17654 (Poly-EPO)** — Set-level objective (reward × reasoning-strategy diversity) improves Pass@k coverage and diversity under RLVR. **Delta:** token-uncertainty uses no clustering or set objective; minority is per-token surprisal.
- **arXiv:2505.15201 (PKPO)** — Directly optimizes Pass@k with unbiased estimators and shows harder-task gains vs pass@1-style RL. **Delta:** PKPO is answer-set utility optimization; token-uncertainty is policy-density reweighting at trajectory level.
- **arXiv:2602.06717 (F-GRPO)** — Focal-style scaling downweights easy/high-success prompts to preserve rare-correct learning signal. **Delta:** F-GRPO’s rarity is prompt-difficulty/global success; token-uncertainty’s rarity is within-trajectory token probability.
- **arXiv:2602.01062 (SetPO)** — Set-level diversity functional with marginal contribution credit improves diversity-preserving reasoning RL. **Delta:** SetPO needs cross-sample similarity machinery; token-uncertainty needs only rollout logprobs.
- **arXiv:2503.19595 (Optimizing Inference-Time Objectives via RL)** — Trains for inference-time objectives (pass@k/majority variants) and demonstrates train–test objective coupling benefits. **Delta:** token-uncertainty does not optimize an explicit inference-time voting metric.
- **arXiv:2504.16084 (TTRL)** — Uses test-time consensus/majority signals as pseudo-rewards for unlabeled RL self-improvement. **Delta:** TTRL uses answer consensus labels; token-uncertainty uses no voting label at all.
- **arXiv:2509.06941 (Outcome-based Exploration for LLM Reasoning)** — Shows RL collapses diversity and adds outcome-level exploration bonuses to recover it. **Delta:** OBE defines novelty by answer/outcome counts; token-uncertainty defines novelty by policy surprisal.
- **arXiv:2503.14476 (DAPO)** — RL system-level improvements (dynamic sampling, clipping choices) materially improve math RL scaling. **Delta:** DAPO is systems/optimization stabilization; token-uncertainty is a new credit-shaping signal.
- **arXiv:2507.14843 (The Invisible Leash)** — RLVR often improves pass@1 while narrowing answer support; analyzes entropy/pass@k paradoxes. **Delta:** token-uncertainty explicitly targets this by rewarding correct low-density trajectories.
- **arXiv:2512.03847 (DVPO)** — Distributional value modeling with tail-aware regularization improves robustness under noisy supervision. **Delta:** DVPO is critic/value-distribution-centric; token-uncertainty is actor-side on-policy reweighting with no value model.

### 3. Novelty check
- **Specific claim:** In RLVR for reasoning, reweighting correct trajectories by within-trajectory self-surprise (under the behavior policy) improves hard-set tail generalization (Cover@tau / worst-subset accuracy) at similar pass@1 compared with standard GRPO and majority-voting-trained baselines.
- **Closest existing work:** F-GRPO (2602.06717) and OBE (2509.06941). Both protect rare signals, but neither defines minority as token-level low-probability mass under the generating policy itself.
- **Ifdita-style 3-sentence objection (steelman):** “This is not minority voting; it is confidence-penalized GRPO with a new weighting heuristic. High token surprise is entangled with length, formatting quirks, and harmless stylistic variation, so you may upweight noisy trajectories rather than meaningful minority reasoning modes. Unless you show gains over answer-frequency minority methods at equal compute, this reads as exploration regularization, not a new minority-voting algorithm.”
- **Novelty rating:** **medium**.

### 4. Concrete training objective
If this objective cannot be made this explicit, direction is underspecified; here it is explicit:

\[
\text{Sample } y_i \sim \pi_{\theta_{\text{old}}}(\cdot|x),\; i=1,\dots,N,\quad r_i\in\{0,1\}
\]
\[
s_i \;=\; -\frac{1}{T_i}\sum_{t=1}^{T_i}\log \pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t}) \quad\text{(self-surprise)}
\]
\[
\tilde s_i \;=\; \max\!\left(0,\frac{s_i-q_{1-\rho}(s_{1:N})}{\operatorname{MAD}(s_{1:N})+\epsilon}\right), \;\;\;
w_i \;=\; 1+\lambda\,\mathbf{1}[r_i=1]\cdot \mathrm{clip}(\tilde s_i,0,c)
\]
\[
A_i^{\text{GRPO}} = r_i-\bar r,\qquad \bar r=\frac{1}{N}\sum_{j=1}^N r_j
\]
\[
A_i^{\text{TU}} = w_i\cdot A_i^{\text{GRPO}}
\]
\[
\omega_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}
\]
\[
\mathcal L_{\text{clip}}(\theta)=
-\frac{1}{N}\sum_{i=1}^N\frac{1}{T_i}\sum_{t=1}^{T_i}
\min\!\Big(\omega_{i,t}A_i^{\text{TU}},\;\mathrm{clip}(\omega_{i,t},1-\epsilon,1+\epsilon)A_i^{\text{TU}}\Big)
\]
\[
\mathcal L(\theta)=\mathcal L_{\text{clip}}(\theta)+\beta\,\mathrm{KL}\!\left(\pi_\theta\|\pi_{\text{ref}}\right)
\]

Per-batch aggregation is the average above over prompts in the mini-batch. Gradients flow through \(\omega_{i,t}\) (policy) only; \(r_i,s_i,w_i\) are treated as stop-gradient statistics from \(\pi_{\theta_{\text{old}}}\), avoiding second-order terms.

### 5. Experimental plan
- **Minimal headline experiment:** Qwen-1.7B-Base, DaPO-17k, 400 RL steps, compare 3 objectives with matched rollout budget: GRPO, majority-voting-trained baseline, and token-uncertainty weighting.
- **Baselines (required + useful):** standard GRPO; majority-voting-trained model (PKPO-style high-k objective or sampled-majority-winner indicator objective); F-GRPO as strongest “rare-correct without token surprisal” control.
- **Ablations needed:** \(\lambda\) sweep; top-\(\rho\) threshold sweep; surprise metric (mean NLL vs fraction of tokens below per-position percentile); length-controlled variant (normalize by expected length bucket); “correct-only weighting” vs weighting all trajectories.
- **Compute estimate (Modal):** from current pricing snapshots, H100 \(\approx \$4.56/\text{GPU-hr}\), A100-80G \(\approx \$3.40/\text{GPU-hr}\).  
  Assume 1 run (400-step train + eval on AIME-25/26, Beyond-AIME, HMMT with Pass@k/Cover@tau) \(\approx 12\) H100-hr equivalent.  
  Core matrix: 4 methods × 3 seeds = 12 runs \(\approx 144\) H100-hr \(\approx \$657\).  
  Ablations: 6 extra runs \(\approx 72\) H100-hr \(\approx \$328\).  
  Total \(\approx 216\) H100-hr \(\approx \$985\) (comfortably < \$1400).
- **Headline metric:** **Cover@tau on hard sets** (AIME-26 + Beyond-AIME), with Pass@k as secondary; add worst-decile prompt accuracy to test the “minority/tail” claim directly.

### 6. Failure modes and consolation result
- **Failure 1: Surprise tracks verbosity, not minority reasoning.**  
  **Observation:** gains disappear after length-matching or brevity-controlled decoding; high-surprise trajectories are mostly longer/noisier CoTs.
- **Failure 2: Optimization instability from overweighted rare samples.**  
  **Observation:** rising KL spikes, reward variance blow-up, and degraded Pass@1 while Cover@tau does not improve.
- **Failure 3: No advantage over simpler rarity methods.**  
  **Observation:** F-GRPO / frequency-based reweighting matches or beats token-uncertainty on hard OOD, implying per-token uncertainty is unnecessary complexity.
- **Minimum publishable null result:** a table showing that token-level surprisal and answer-level rarity decouple (low correlation), plus an ablation figure where uncertainty weighting fails to improve hard-set Cover@tau once length and entropy confounds are controlled. That is still a defensible “what minority proxy is not” result for minority-voting optimization.

### 7. Killer experiment
Train-matched comparison on AIME-26 + Beyond-AIME showing token-uncertainty beats both GRPO and majority-voting training on Cover@tau and worst-decile accuracy without significant Pass@1 drop.  
Expected figure: one Pareto plot (Pass@1 vs Cover@tau) where token-uncertainty is the only frontier-improving point.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **conditional on X**.
- **One-paragraph honest summary of why:** This direction is one of the cleaner minority formulations operationally (no clustering, no judge, no answer-frequency estimator) and is scientifically nontrivial because it tests whether “minority” can be defined as low policy density rather than low answer count. It is not obviously already done, but novelty is only medium because it can collapse into generic exploration reweighting unless the paper proves a distinctly minority-voting benefit story. The must-have condition is a strong comparative result against majority-voting-trained and frequency-based minority baselines under strict confound controls (length/entropy). If that condition fails, this should not be the main project direction.
