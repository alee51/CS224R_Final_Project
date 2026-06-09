# Eval results — file index

For the at-a-glance overview, see [`../eval_summary.md`](../eval_summary.md).
For audit caveats + interpretive notes, see [`CAVEATS.md`](CAVEATS.md).

Each result file has a `## TL;DR` block at top and a `## How this was computed`
block at bottom; raw numeric tables in between.

## Held-out eval (4 arms × 6 datasets, n=64)

| File | What |
|---|---|
| [eval_complete.md](eval_complete.md) | **Single canonical completion record** — Phase 1/3 ledger + headline finding + 3-verification grader story + §8 spec compliance + bug ledger + what's NOT in the eval. Read first. |
| [comparison.md](comparison.md) | Cross-arm headline pass@k tables, 4 arms × 6 datasets. Polyepo×math500 still `_missing_` (re-run exists; see CAVEATS). |
| [auc_at_k.md](auc_at_k.md) | AUC@k + per-(arm,dataset) pass@k ladder, k∈{1..64}. |
| [coverage.md](coverage.md) | coverage / distinct_answers / entropy / majority @k. 5 datasets × 4 arms × k∈{1,2,4,8,16,32,64}. |
| [diff_at_k_split.md](diff_at_k_split.md) | distinct_answers@k partitioned by solved vs unsolved. 2 partitions × 5 datasets × 4 arms × k∈{1,4,8,16,32,64}. |
| [potential_at_k.md](potential_at_k.md) | Recoverable failure rate (budget-bound vs quality-bound). 5 datasets × 4 arms × k∈{1,4,8,16,32}. |
| [reflective_actions.md](reflective_actions.md) | Per-rollout count of 7 "reflective" lexical phrases. 5 datasets × 4 arms × 7 phrases + totals. |
| [self_bleu.md](self_bleu.md) | Self-BLEU + distinct-1/2/3-grams on rollout text. 5 datasets × 4 arms; subsampled. |
| [kl_summary.md](kl_summary.md) | Per-token KL(π_arm ‖ π_base) on rollouts. 3 arms × 6 datasets (polyepo×math500 skipped) × mean+median KL (bits). |
| [grader_sanity_all.md](grader_sanity_all.md) | 3-way grader verification (gt-in-preds, rescore, math_dapo tripwire). 4 arms × 5 datasets × 5 checks. |

## Training-time analysis (Phase 4)

| File | What |
|---|---|
| [training_dynamics.md](training_dynamics.md) | Loss / reward / pass@8 trajectories over 400 steps for all 3 arms. |
| [training_diff_at_k_split.md](training_diff_at_k_split.md) | distinct_answers@k on training rollouts, solved vs unsolved partition. The training-time companion to `diff_at_k_split.md`. |
| [cluster_correctness.md](cluster_correctness.md) | P(cluster = correct cluster) per rarity rank. The "rarity ≠ correctness" diagnostic for minority. |
| [u_correct_summary.md](u_correct_summary.md) | `|U_correct|` (number of distinct correct CoT clusters per prompt) trajectory over training. |

## Pipeline audit

| File | What |
|---|---|
| [eval_pipeline_bugs.md](eval_pipeline_bugs.md) | 5 eval-pipeline bugs hit + fixed this session. Root cause + lessons per bug. |
| [eval_pipeline_verification.md](eval_pipeline_verification.md) | Pipeline correctness audit (companion to bugs doc). |

## Paper-focused analysis (Anastasia, `main-verl/eval/results/`)

Separate analysis track on the **same locked abao GEN data**, focused on the
pass@k + CoT-diversity slice that backs `paper-overleaf/final_report.tex`.
**Pass@k numbers agree with this directory** (cross-checked). **CoT diversity
is NEW data not present here** — `eval.md §6.4` had marked judge clustering as
"SKIPPED for v1", but Anastasia ran it post-hoc against her own Qwen3-4B-Instruct
Modal endpoint.

| File | What |
|---|---|
| [`../../eval/results/diversity_analysis.md`](../../eval/results/diversity_analysis.md) | Paper-narrative writeup. Source for §Results/§Discussion in `final_report.tex`. |
| [`../../eval/results/passk_results.md`](../../eval/results/passk_results.md) | Pass@k tables AIME25/26/BeyondAIME. Matches `comparison.md`. |
| [`../../eval/results/cot_diversity_results.md`](../../eval/results/cot_diversity_results.md) | div@k for math500 + beyondaime, all 4 arms. Includes polyepo × math500 (CoT-only — see CAVEATS). |
| [`../../eval/results/coverage_results.json`](../../eval/results/coverage_results.json) | Answer-coverage@k JSON used by `plot_diversity.py`. |
| [`../../eval/results/cot_diversity_results.json`](../../eval/results/cot_diversity_results.json) | Raw judge-clustered output used by `plot_diversity.py`. |
| `../../eval/results/fig{1_passk, 2_cot_diversity, 3_summary_k16, 4_correctness_vs_diversity}.pdf` | The 4 paper figures (mirror PNGs also present). |
| [`../../eval/results/plot_diversity.py`](../../eval/results/plot_diversity.py) | Reproducible plotting script for all 4 figures. |
| [`../../eval/results/results_discussion.tex`](../../eval/results/results_discussion.tex) | LaTeX source of §Results + §Discussion in the paper. |

## Raw data + plots

- `GRPO_history.csv`, `Minority_CoT_history.csv`, `Poly_EPO_CoT_history.csv` — training history exports
- `metrics_summary.json`, `u_correct_trajectory.json` — raw metrics
- `wandb_plots/` — committed W&B PNG snapshots
