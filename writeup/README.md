# `writeup/` — poster + paper source notes

Tight, audit-clean notes for the CS224R minority-voting final project. Every
number, formula, and claim here is cross-checked against the actual code or
config; references are given as `<absolute path>:<line>`.

## Editorial principle

- **Positive statements only.** Describe what the system does. Negation lists
  ("we do NOT use X") belong in the engineering log, not the writeup.
- **Code is truth.** Where a prior doc (e.g. `STANDARDS.md`, `eval/README.md`,
  `PLAN.md`) disagrees with the actual YAML/Python, the YAML/Python wins, and
  the discrepancy is logged in `AUDIT_FINDINGS.md`.
- **No padding.** Each file is short enough to paste sections directly into the
  paper / poster.

## Files

| file | purpose |
|---|---|
| `training.md` | Auditable training spec: model, data, optimizer, PPO knobs, reward, judge, per-rollout JSONL schema. |
| `eval.md` | Auditable eval spec: dataset panel, prompt format, scorer, sampling, metrics (including new `|U_correct|@k`), procedure. |
| `algorithm_descriptions.md` | Math-friendly description of GRPO, Minority-CoT, Poly-EPO-CoT; advantage formulas verified against `objective_*.py`. |
| `results.md` | Headline pass@k table (5 datasets × 3 arms × k∈{1,4,8,16}) + cluster-correctness diagnostic. Cells marked `pending` where the eval has not landed. |
| `AUDIT_FINDINGS.md` | Discrepancies between docs and code; flips of "we do NOT X" → positive statements; numbers needing user confirmation; poster contradictions. |

## Source-of-truth file pointers

- Training configs:
  `main-verl/configs/{grpo,minority_cot,poly_epo_cot}_train_4b_1epoch.yaml`
- Eval probe: `main-verl/eval/run_eval.py`
- Reward grader: `verl/utils/reward_score/math.compute_score` (Hendrycks
  `is_equiv` on `last_boxed_only_string`), pinned in fork
  `chicken602/maxrl@33873ec9`.
- Advantage kernels: `main-verl/train/objective_minority.py`,
  `main-verl/train/objective_poly_epo.py`.
- Judge prompt: `main-verl/judge/prompts/poly_epo_a1.md`.
- Live results: `main-verl/eval/results/comparison.md`,
  `main-verl/eval/results/minority_diagnostic.md`.
