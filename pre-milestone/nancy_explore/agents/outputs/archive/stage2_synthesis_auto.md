### Section 1: Comparison table

| Name | Slug | Novelty rating (section 3) | Verdict (section 8) | Worth stage 3? | Killer experiment (section 7, distilled) | Top failure mode (section 6, distilled) |
|---|---|---|---|---|---|---|
| Embedding-Based Minority Clustering | embedder_clustering | medium | promising | yes, conditional on strict objective-fixed substrate-only ablations first | Cheap substrate matches ~90-95% of LM-judge Cover@tau gain at <40% training compute on hard OOD | Cheap clustering collapses semantics (merges distinct solutions/splits equivalent ones), hurting Cover@tau |
| Per-Token Uncertainty Minority | token_uncertainty | medium | promising | conditional on proxy-validity check showing self-surprise tracks low-frequency-correct modes | Token-uncertainty frontier dominates GRPO and majority baselines on Pass@1 vs Cover@tau on hard OOD | Self-surprise tracks noise/verbosity rather than meaningful minority reasoning |
| Cover-at-Tau as Training Objective | cover_at_tau | medium | promising | conditional on robust low-cost clustering and variance-reduced estimator | Cover@tau-trained model beats PKPO/GRPO on OOD Cover@tau without significant Pass@1 regression | Thresholded cluster reward is sparse/high-variance, producing unstable or flat learning |
| Dual-Head Majority-Minority Policy | dual_head | medium | promising | conditional on strict compute-matched controls and pre-registered worst-subset metric | Dual-head interpolation curve dominates single-head baselines on Pass@1 vs worst-subset-accuracy | Negative transfer in shared trunk causes both heads to underperform |
| Prompt Minority Curriculum | prompt_curriculum | medium-low | promising | conditional on confound controls and at least one objective-level minority baseline | Concentration-aware prompt sampling improves Cover@tau/worst-subset under identical loss+compute | Difficulty confound: curriculum may just track hard prompts, not minority mechanism |
| Worst-Subset Set Objective | worst_subset | medium | promising | yes, conditional on lower-tail average and pre-registered failure criteria | Worst-subset objective improves worst-case-subset accuracy/Cover@tau on hard OOD without Pass@1 collapse | Tail-focused objective destabilizes training or causes mean-performance collapse |
| Inverse-Frequency Reweighting | inverse_freq | medium | promising | conditional on early head-to-head separation from F-GRPO | At matched compute/Pass@1, inverse-frequency beats GRPO and F-GRPO on Cover@tau + worst-subset | Frequency proxy mismatch (fake rarity from formatting/weak clustering) misallocates gradient mass |

### Section 2: Cross-cutting observations (1-2 paragraphs)

The seven depth evals cluster into two underlying mechanism families despite surface differences. One family changes **objective/credit geometry** (`inverse_freq`, `worst_subset`, `cover_at_tau`, plus the minority branch of `dual_head`); the other changes **where minority pressure enters the pipeline** (`embedder_clustering` via equivalence substrate, `prompt_curriculum` via data sampling, `token_uncertainty` via policy-density proxy). In practice, several directions will share components: answer canonicalization/clustering infrastructure is reused across `embedder_clustering`, `cover_at_tau`, and often `inverse_freq`; hard-tail evaluation (`worst-subset`, Cover@tau) is reused across most directions.

Budget-wise, most plans fit inside the $1400 envelope only with disciplined ablation pruning; multiple writeups explicitly show edge cases that drift near/over budget when grids expand. Citation overlap is heavy around `2604.17654` (Poly-EPO), `2505.15201` (PKPO), `2602.06717` (F-GRPO), `2602.01062` (SetPO), `2503.14476` (DAPO), and `2504.16084` (TTRL). Suspicious-looking IDs that appeared in only some evals (e.g., `2601.19280`, `2510.10649`, `2503.19595`) were checked and do resolve to real papers.

### Section 3: OFF-DIRECTION list

None of the seven completed depth evaluations is marked OFF-DIRECTION.

### Section 4: Notes for the human reviewer

- Several directions are "promising, conditional," but the conditions differ; compare each condition directly rather than the top-line verdict label.
- `dual_head` and `worst_subset` both target tail robustness but with different failure signatures (interference vs instability); keep these failure criteria distinct in stage-3 planning.
- `token_uncertainty` and `inverse_freq` both hinge on proxy validity; prioritize early calibration checks before expensive long runs.
- `embedder_clustering`, `cover_at_tau`, and `prompt_curriculum` all depend on cluster/answer statistics quality; one shared, audited implementation could de-risk multiple stage-3 experiments.
