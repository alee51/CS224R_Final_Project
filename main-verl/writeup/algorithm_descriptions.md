# Algorithm descriptions (math-friendly, code-verified)

Each arm operates on a group of `N = 8` rollouts per prompt
(`main-verl/train/objective_minority.py:475`,
`N_ROLLOUTS = 8`). Set arms additionally enumerate the
`C(8, 4) = 70` size-4 subsets, with each rollout `i` appearing in
`C(7, 3) = 35` subsets (`main-verl/train/objective_minority.py:476-484`).
Let `r_i ∈ {0, 1}` denote the binary reward (Hendrycks `is_equiv` on the
last `\boxed{...}`), and `c_i ∈ ℤ` denote the judge-assigned cluster id
(`c_i = −1` for degenerate / non-math rollouts).

The advantage scalar `A_i` is broadcast to every response token of rollout
`i` (`_scatter_advantages_to_tokens`,
`main-verl/train/objective_minority.py:944`); the PPO clipped surrogate then
multiplies by `loss_agg_mode` (see `training.md` §3).

## GRPO

**Advantage.**

$$A_i^{\mathrm{GRPO}} \;=\; r_i \;-\; \frac{1}{N}\sum_{j=1}^{N} r_j.$$

Standard-deviation normalization is disabled
(`norm_adv_by_std_in_grpo: false`, `grpo_train_4b_1epoch.yaml:57`), matching
the Poly-EPO paper §A; the advantage is the raw mean-centered reward.

**Intuition.** Rollouts that beat the group's empirical mean reward get
positive advantage. When every rollout in a prompt receives the same reward
(all-correct or all-wrong), every `A_i = 0` — the prompt contributes no
gradient. This is the implicit zero-gradient filter, tracked at training
time as `train/fraction_filtered` (≈ 0.53 in production —
`main-verl/eval/PLAN.md:11`).

**Distinguishing knob.** `loss_agg_mode: seq-mean-token-mean`
(`grpo_train_4b_1epoch.yaml:31`), in contrast to the set arms'
`seq-mean-token-sum-norm`.

## Minority-CoT

**Subset score** (`main-verl/train/objective_minority.py:509-528`).
For a size-4 subset `G ⊂ {1, …, 8}` with rewards `r_G` and clusters `c_G`,
let `m(G)` be the rarest cluster id in `c_G` (random tiebreak among rarest
clusters, deterministic per `(problem_id, global_seed)`):

$$f_{\mathrm{min}}(G) \;=\; \mathrm{mean}\{\, r_i \;:\; i \in G,\; c_i = m(G)\,\}.$$

`m(G)` may equal `−1` (the degenerate id), in which case the rarest cluster
consists of degenerate rollouts — by Option A
(`main/docs/STANDARDS.md:99`), these are not filtered out.

**Marginal advantage** (`main-verl/train/objective_minority.py:502-506`,
`:531-630`). With $\mathcal{G}$ the 70 size-4 subsets and $\mathcal{S}_i$
the 35 subsets containing rollout `i`:

$$A_i^{\mathrm{min}} \;=\; \frac{1}{|\mathcal{S}_i|}\sum_{G \in \mathcal{S}_i} f_{\mathrm{min}}(G) \;-\; \frac{1}{|\mathcal{G}|}\sum_{G \in \mathcal{G}} f_{\mathrm{min}}(G).$$

Equivalently: average $f_{\mathrm{min}}$ over the 35 subsets containing `i`,
then subtract the average over all 70 subsets.

**Zero-gradient prompt filter** (`main-verl/train/objective_minority.py:580-586`).
If all rollouts in a prompt share one cluster id, every $f_{\mathrm{min}}(G)$
is the prompt's overall reward mean and the marginal advantage is identically
zero. The prompt's `keep_mask` is set to `False`; `train/fraction_filtered`
records the rate.

**Intuition.** A rollout earns positive advantage when it is *correct* and
sits in a *rare* cluster of its subsets: averaged over the 35 subsets it
belongs to, its rarest-cluster reward mean exceeds the baseline rarest-cluster
reward mean. The objective explicitly upweights the empirically-rare
reasoning strategies when those strategies are also correct.

**Distinguishing knob.** `cluster_source: judge` +
`judge_model: Qwen/Qwen3-4B-Instruct-2507`
(`minority_cot_train_4b_1epoch.yaml:60-62`); RNG tiebreak controlled by
`global_seed: 0` (`:67`).

## Poly-EPO-CoT

**Subset score** (`main-verl/train/objective_poly_epo.py:43-55`,
verbatim from Poly-EPO paper Appendix A.1, with the paper's `cluster 100`
mapped to our `−1`):

$$f_{\mathrm{poly}}(G) \;=\; \Bigl(\tfrac{1}{|G|}\textstyle\sum_{i \in G} r_i\Bigr) \cdot \frac{d(G)}{|G|}, \qquad d(G) \;=\; \bigl|\{\,c_i : i \in G,\; c_i \neq -1\,\}\bigr|.$$

That is: subset mean reward, scaled by the fraction of distinct
non-degenerate clusters present in the subset. Deterministic — no RNG
tiebreak.

**Marginal advantage.** Same kernel as Minority-CoT:

$$A_i^{\mathrm{poly}} \;=\; \frac{1}{35}\sum_{G \in \mathcal{S}_i} f_{\mathrm{poly}}(G) \;-\; \frac{1}{70}\sum_{G \in \mathcal{G}} f_{\mathrm{poly}}(G).$$

**Intuition.** A subset is "worth more" when (a) more of its rollouts are
correct and (b) those correct rollouts span more distinct reasoning clusters.
Every non-degenerate cluster contributes, including the most-common one —
the algorithm does not single out the rarest cluster.

**Distinguishing knob.** `cluster_source: judge` +
`adv_estimator: poly_epo_cot`
(`poly_epo_cot_train_4b_1epoch.yaml:55-66`). No `global_seed` field — the
kernel is deterministic given the cluster assignment.

## Verified properties (from code, not docs)

- All three arms share `set_based_marginal_advantages` for the set-arm
  marginal computation (`main-verl/train/objective_minority.py:531`); only
  the `subset_score_fn` differs.
- Both set arms use `loss_agg_mode: seq-mean-token-sum-norm`; GRPO uses
  `seq-mean-token-mean` (confirmed across the three YAMLs).
- The Minority-CoT advantage is **not** of the form `+1 / |rarest cluster| − mean`;
  it is the *marginal-over-subsets* form above, with the rarest cluster's
  mean reward as the subset score (`objective_minority.py:509-528`,
  `:502-506`).
- Both set-arm advantages are **not** z-score normalized within prompt; the
  Stage 8 v2 diagnostic memo flagged this as a concern, but the production
  config is locked at no-std-norm and matches the kernel as written.
  `norm_adv_by_std_in_grpo: false` controls only the GRPO path; the set-arm
  kernels do not read it
  (`main-verl/train/objective_minority.py:531`).
