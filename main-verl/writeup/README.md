# `main-verl/writeup/` — poster + paper source notes

Tight, audit-clean notes for the CS224R minority-voting final project. Every
number, formula, and claim here is cross-checked against the actual code or
config; references are given as `<absolute path>:<line>`.

## Editorial principle

- **Positive statements only.** Describe what the system does.
- **Code is truth.** Where a prior doc disagrees with the YAML/Python, the YAML/Python wins.
- **No duplication.** Every fact lives in exactly one doc; others link.
- **No padding.** Each file is short enough to paste sections directly into the paper / poster.

## Files

| file | purpose |
|---|---|
| `eval.md` | **LOCKED eval spec.** 4 arms × 6 datasets × n=64 × logprobs=20. Scorer, sampling, metrics (Tier 1 / Tier 2 / training-time), reproducibility. |
| `eval_build.md` | **Eval run plan.** Cost ~$165, account assignments, 5 phases (pre-flight → GEN → analysis → KL → train-data → judge), and 20 implementation nuances. |
| `eval_harness.md` | How to launch the eval harness (env vars, output schema, analysis recipes). |
| `MODAL_STATUS.md` | **Operational source of truth** — Modal accounts, budgets, ckpt paths, per-rollout JSONL locations, eval-JSON inventory. |
| `training.md` | Auditable training spec: model, data, optimizer, PPO knobs, reward, judge, per-rollout JSONL schema. |
| `algorithm_descriptions.md` | Math-friendly description of GRPO, Minority-CoT, Poly-EPO-CoT; advantage formulas verified against `objective_*.py`. |
| `minority_diagnostic.md` | Why minority underperforms — cluster-correctness inversion, token-entropy gap. Source for diagnostic poster content. |
| `results/comparison.md` | Cross-arm pass@k table (auto-populated by `eval/analysis/compare.py` as runs land). |
| `results/u_correct_summary.md` | `|U_correct|` training-time trajectory for set arms (auto-populated by `eval/analysis/u_correct.py`). |

## Where every fact lives (canonical, NO duplication)

| fact | canonical location |
|---|---|
| Eval spec (datasets, metrics, scorer, sampling) | `eval.md` |
| Eval runs (accounts, sequencing, costs, nuances) | `eval_build.md` |
| Eval harness usage | `eval_harness.md` |
| Modal accounts, budgets, ckpt paths, eval-JSON inventory | `MODAL_STATUS.md` |
| Training spec | `training.md` |
| Algorithm math | `algorithm_descriptions.md` |
| Minority diagnostic findings | `minority_diagnostic.md` |
| Cross-arm result tables | `results/comparison.md` |

`main-verl/eval/` is now code-only — run_eval.py, launchers/, analysis/, and
JSON eval outputs under `eval/results/`. All human-curated docs and result
tables live here.

## Source-of-truth file pointers

- Training configs: `main-verl/configs/{grpo,minority_cot,poly_epo_cot}_train_4b_1epoch.yaml`
- Eval probe: `main-verl/eval/run_eval.py`
- Reward grader: `verl/utils/reward_score/math.compute_score` (Hendrycks `is_equiv`), pinned in fork `chicken602/maxrl@33873ec9`.
- Advantage kernels: `main-verl/train/objective_minority.py`, `main-verl/train/objective_poly_epo.py`.
- Judge prompt: `main-verl/judge/prompts/poly_epo_a1.md`.
