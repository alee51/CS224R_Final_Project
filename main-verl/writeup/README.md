# `main-verl/writeup/` — poster + paper source notes

Tight, audit-clean notes for the CS224R minority-voting final project. Every
number, formula, and claim here is cross-checked against the actual code or
config; references are given as `<absolute path>:<line>`.

## Start here

| If you want to... | Open |
|---|---|
| See what evals we ran at a glance | [`eval_summary.md`](eval_summary.md) |
| Write/edit the paper | [`paper.md`](paper.md) |
| Cite a specific result file | [`results/INDEX.md`](results/INDEX.md) |
| Understand audit caveats / interpretive notes | [`results/CAVEATS.md`](results/CAVEATS.md) |

## Editorial principle

- **Positive statements only.** Describe what the system does.
- **Code is truth.** Where a prior doc disagrees with the YAML/Python, the YAML/Python wins.
- **No duplication.** Every fact lives in exactly one doc; others link.
- **No padding.** Each file is short enough to paste sections directly into the paper / poster.

## Specs (locked, foundational)

| file | purpose |
|---|---|
| [`eval.md`](eval.md) | **LOCKED eval spec.** 4 arms × 6 datasets × n=64 × logprobs=20. Scorer, sampling, metrics, reproducibility. |
| [`training.md`](training.md) | **LOCKED training spec.** Model, data, optimizer, PPO knobs, reward, judge, per-rollout JSONL schema. |
| [`algorithm_descriptions.md`](algorithm_descriptions.md) | Math-friendly description of GRPO, Minority-CoT, Poly-EPO-CoT; advantage formulas verified against `train/objective_*.py`. |

## Active planning + state

| file | purpose |
|---|---|
| [`paper.md`](paper.md) | Final-report writing plan: section-by-section status, "can do now" vs "awaits eval", open decisions. |
| [`eval_summary.md`](eval_summary.md) | At-a-glance "what evals we have" — setup, metrics, coverage, headline pass@64. |
| [`MODAL_STATUS.md`](MODAL_STATUS.md) | Operational source of truth — Modal accounts, budgets, ckpt paths, per-rollout JSONL locations, eval-JSON inventory. |
| [`minority_diagnostic.md`](minority_diagnostic.md) | Why minority underperforms — cluster-correctness inversion, token-entropy gap. Source for diagnostic poster content. |

## Results

`results/` holds all eval result files (one fact per file). Start at
[`results/INDEX.md`](results/INDEX.md) for the file table, or
[`results/CAVEATS.md`](results/CAVEATS.md) for the 7 audit caveats + the
hmmt_nov25 crossover explanation + the polyepo×math500 reconciliation.

Anastasia's paper-focused analysis (figures, CoT diversity, LaTeX section
source) lives in [`../eval/results/`](../eval/results/) — also indexed from
`results/INDEX.md`.

## Archive

[`archive/`](archive/) holds completed-job docs (eval run plan, harness
how-to, session-state files, superseded result files) — kept for history,
not authoritative.

## Source-of-truth file pointers

- Training configs: `main-verl/configs/{grpo,minority_cot,poly_epo_cot}_train_4b_1epoch.yaml`
- Eval probe: `main-verl/eval/run_eval.py`
- Reward grader: `verl.utils.reward_score.math.compute_score` (Hendrycks `is_equiv`), pinned in fork `chicken602/maxrl@33873ec9`.
- Advantage kernels: `main-verl/train/objective_minority.py`, `main-verl/train/objective_poly_epo.py`.
- Judge prompt: `main-verl/judge/prompts/poly_epo_a1.md`.
