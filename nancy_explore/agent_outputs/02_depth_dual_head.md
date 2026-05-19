### 1. Direction restatement (<=150 words)
This direction is a direct instantiation of Ifdita's pitch: train for both majority-voting and minority-voting objectives, then test harder-set generalization. Instead of forcing one scalar objective to interpolate, it uses a shared policy trunk with two heads. The majority head uses standard GRPO-style mean-baseline optimization; the minority head uses an explicit minority objective (e.g., inverse-frequency-weighted or worst-subset-weighted advantage). Training alternates modes so the trunk gets gradients from both objective families while each head specializes. At test time, you can deploy majority-only (safer), minority-only (exploratory), or interpolate between heads without retraining. Compared with prior single-objective methods, the scientific bet is that objective disentanglement in parameter space yields a better accuracy-diversity frontier on AIME-25/26, Beyond-AIME, and HMMT.

### 2. Related work scan (6-12 papers, last ~18 months)
1. **arXiv:2604.17654 (Poly-EPO)** — Set-level objective improves pass@k coverage by coupling reward and diversity.  
   **Delta vs dual-head:** Poly-EPO entangles goals in one objective; dual-head separates majority/minority objectives and only shares representation.

2. **arXiv:2505.15201 (PKPO)** — Directly optimizes pass@k via unbiased estimators and k-annealing.  
   **Delta vs dual-head:** PKPO still yields a single policy; dual-head keeps two deployable policies from one run.

3. **arXiv:2602.06717 (F-GRPO)** — Focal-style reweighting preserves rare-correct trajectories under finite group size.  
   **Delta vs dual-head:** F-GRPO reweights one stream; dual-head creates explicit specialization with controllable inference-time mixing.

4. **arXiv:2602.01062 (SetPO)** — Set-level diversity contributions mitigate mode collapse in RLVR.  
   **Delta vs dual-head:** SetPO modifies one advantage function; dual-head keeps diversity objective isolated from majority objective in separate heads.

5. **arXiv:2503.14476 (DAPO)** — RL system-level improvements (clipping/sampling/token-level choices) scale open RLVR training.  
   **Delta vs dual-head:** DAPO is mostly optimizer/system mechanics; dual-head is an objective/architecture claim.

6. **arXiv:2504.16084 (TTRL)** — Uses majority-vote pseudo-labeling for test-time RL self-improvement on unlabeled data.  
   **Delta vs dual-head:** TTRL improves via test-time adaptation; dual-head is train-time objective bifurcation for fixed-policy deployment.

7. **arXiv:2512.15146 (Beyond Majority Voting / SCOPE)** — Replaces plain majority pseudo-labels with finer confidence-weighted subgroup signals.  
   **Delta vs dual-head:** SCOPE refines pseudo-reward quality in TTRL; dual-head targets objective coexistence in offline/online RLVR training.

8. **arXiv:2505.23433 (Diversity-Aware Policy Optimization)** — Diversity-focused policy optimization improves reasoning robustness.  
   **Delta vs dual-head:** Diversity-aware PO is single-objective shaping; dual-head tests whether architectural disentanglement is superior.

9. **arXiv:2507.14843 (The Invisible Leash)** — RLVR often increases pass@1 while shrinking answer-level support/diversity.  
   **Delta vs dual-head:** Dual-head can be viewed as a direct intervention on this tradeoff: keep a high-pass@1 head and a support-preserving head jointly.

### 3. Novelty check
- **Specific claim:** A shared-trunk, dual-head policy trained under separate majority and minority objectives can dominate a single-head policy on the pass@1 vs worst-subset/coverage Pareto frontier at fixed compute.
- **Closest existing work:** PKPO (arXiv:2505.15201) is closest in spirit (majority-tail tension), but it is single-policy objective shaping; this direction's novelty is explicit two-policy specialization with shared representation and post-hoc interpolation.
- **Ifdita steelman objection (3 sentences):** "This might just be multitask training plus a tiny output fork, not a new minority-voting algorithm. If your gain comes from extra parameters or effectively more updates, the comparison is confounded and the claim collapses. Also, if head interpolation works only weakly, you have added complexity without proving better generalization than a strong single-head minority objective."
- **Novelty rating:** **medium**.

### 4. Concrete training objective
Let trunk params be \(\phi\), majority-head params \(\theta_M\), minority-head params \(\theta_m\), and routing variable \(z_t \in \{0,1\}\) for batch \(t\) (1 = majority mode).

\[
\pi_M(\cdot|x)=\pi_{\phi,\theta_M}(\cdot|x),\quad
\pi_m(\cdot|x)=\pi_{\phi,\theta_m}(\cdot|x)
\]

For a batch \(B_t=\{(x_j,\{y_{j,i}\}_{i=1}^N)\}\), rewards \(r_{j,i}\in\{0,1\}\), ratios \(\rho_{j,i}=\frac{\pi_h(y_{j,i}|x_j)}{\pi_{h,\text{old}}(y_{j,i}|x_j)}\), and head \(h\in\{M,m\}\):

\[
b^M_j=\frac{1}{N}\sum_{i=1}^N r_{j,i}
\]

Minority baseline (inverse-frequency example; \(a_{j,i}\)=final answer, \(c_j(a)\)=count in prompt-group):
\[
w_{j,i}=\frac{(c_j(a_{j,i})+\epsilon)^{-1}}{\sum_{u=1}^N(c_j(a_{j,u})+\epsilon)^{-1}},\quad
b^m_j=\sum_{i=1}^N w_{j,i}r_{j,i}
\]

\[
A^h_{j,i}=r_{j,i}-b^h_j
\]

\[
\mathcal{L}_h=
-\frac{1}{|B_t|N}\sum_{j,i}\min\!\Big(\rho_{j,i}A^h_{j,i},\,
\mathrm{clip}(\rho_{j,i},1-\eta,1+\eta)A^h_{j,i}\Big)
+\beta\,\mathrm{KL}\!\left(\pi_h(\cdot|x_j)\,\|\,\pi_{\text{ref}}(\cdot|x_j)\right)
\]

\[
\mathcal{L}_t=z_t\mathcal{L}_M+(1-z_t)\mathcal{L}_m,\quad z_t\sim\mathrm{Bernoulli}(p)
\]

Gradient flow:
\[
\nabla_\phi \mathbb{E}[\mathcal{L}_t]
=p\,\nabla_\phi\mathcal{L}_M+(1-p)\,\nabla_\phi\mathcal{L}_m,\;
\nabla_{\theta_M}\mathbb{E}[\mathcal{L}_t]=p\,\nabla_{\theta_M}\mathcal{L}_M,\;
\nabla_{\theta_m}\mathbb{E}[\mathcal{L}_t]=(1-p)\,\nabla_{\theta_m}\mathcal{L}_m
\]
with stop-gradient to the inactive head each step.

This is precise enough; direction is not underspecified.

### 5. Experimental plan
- **Minimal headline experiment:** Qwen-1.7B-Base, DaPO-17k, 400-step run, compare four policies under matched rollout budget: single-head GRPO, single-head majority-voting-trained model, single-head minority objective, and dual-head (evaluate majority head, minority head, and 3-point interpolation \(\lambda\in\{0.25,0.5,0.75\}\)).
- **Baselines (minimum + necessary):** standard GRPO; majority-voting-trained model; minority-only single-head counterpart with exactly same minority objective as dual-head minority branch; PKPO or F-GRPO as one strong external baseline.
- **Ablations for defensibility:** shared trunk vs fully separate trunks (parameter-matched); routing ratio \(p\) sweep; minority objective choice (inverse-frequency vs worst-subset); equalized token-budget control; interpolation at decode-time vs logit-level mixture.
- **Compute estimate (Modal):** use **A100-80GB** at about **$2.50/hr** (pricing page). Assume ~10 GPU-hours/run for 400-step Qwen-1.7B RLVR with \(N=8\), plus ~2 GPU-hours eval. Core grid: 5 methods x 3 seeds = 15 runs -> ~180 GPU-hours -> **$450**. Ablations + retries ~250 additional GPU-hours -> **$625**. Total ~430 GPU-hours -> **$1,075**, within $1,400. Equivalent on H100 (~$4.56/hr) would likely exceed budget for full ablations.
- **Headline metric:** primary = **worst-case-subset accuracy** (hard-strata slice on AIME-25/26 + Beyond-AIME + HMMT); secondary = Pass@1, Pass@k, Cover@tau. Headline figure should be a Pareto plot (Pass@1 vs worst-subset accuracy) including dual-head interpolation curve.

### 6. Failure modes and consolation result
- **Failure 1: Negative transfer in trunk.** Observation: both heads underperform their single-head controls, and representation similarity probes show conflicting gradients in late training.
- **Failure 2: Minority head learns "rare wrong" bias.** Observation: minority head increases answer entropy/diversity but drops both worst-subset accuracy and Cover@tau on hard sets.
- **Failure 3: Interpolation non-monotonic or degenerate.** Observation: mixed policies are dominated by one endpoint (no smooth frontier), implying the two-head story adds no practical control.

If main hypothesis is null, **minimum publishable result**: a controlled null showing that objective disentanglement does not beat single-head shaping at this scale, with a mechanistic analysis figure of trunk gradient cosine conflict over training plus endpoint/mixture frontier collapse. Concretely: one main Pareto figure + one diagnostics table (transfer/conflict metrics by method).

### 7. Killer experiment
Show that on Beyond-AIME + HMMT, the dual-head interpolation curve strictly dominates all single-head baselines in the Pass@1 vs worst-subset-accuracy plane at matched token budget; expected figure is a frontier plot where the dual-head curve forms the outer envelope and both endpoints are useful.

### 8. Overall verdict
- **Rating:** **promising**.
- **Worth running stage 3 (initial experimentation) on?** **conditional on X** (X = strict compute-matched controls and a pre-registered headline metric centered on worst-subset accuracy, not just pass@1).
- The upside is real: this is one of the few directions that literally instantiates "both methods" from the mentor's pitch in a single training system, and it creates an evaluation-time control knob without retraining. The downside is equally real: novelty can collapse into "just multitask with two heads" unless comparisons are extremely tight and the interpolation frontier is genuinely nontrivial. Given the project's current state and constraints in `findings.md` / `why_stop_poly_epo.md`, this is higher ceiling than safe single-objective tweaks, but only if the team treats confound control as the main experiment rather than an appendix.
