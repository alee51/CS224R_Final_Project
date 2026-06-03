# Cross-arm held-out eval — pending

No numbers here yet. Prior partial results were deleted because their grader /
sampling provenance was unclear. Re-run will populate this file via
`main-verl/eval/analysis/posthoc/auc_at_k.py` (pass@k table) plus
`main-verl/eval/analysis/posthoc/diff_at_k_split.py` (solved/unsolved partition).
The old `compare.py` is archived under `main-verl/eval/analysis/_legacy/`.

Authoritative spec: `main-verl/writeup/eval.md`.
Run plan: `main-verl/writeup/eval_build.md`.

## Schema (when populated)

For each dataset in the locked panel, one block:

```
## <dataset>
n=<n_prompts>

### pass@k
| arm      | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 |
|----------|--------|--------|--------|---------|---------|
| grpo     |        |        |        |         |         |
| polyepo  |        |        |        |         |         |
| minority |        |        |        |         |         |

### Diversity (eval-time)
| arm | k | coverage | distinct | entropy | majority@k |
|---|---|---|---|---|---|
```

Grader: `verl.utils.reward_score.math.compute_score` (Hendrycks `is_equiv`).
Sampling: `temp=1.0, top_p=1.0, max_tokens=4096`; `n` per dataset per the spec.
