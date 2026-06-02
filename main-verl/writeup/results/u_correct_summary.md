# |U_correct| training trajectory

Avg # distinct judge CoT-clusters among correct rollouts per prompt, averaged over prompts in the step. Mirrors Poly-EPO paper Fig 2 (left).

- Degenerate cluster (-1) excluded.
- **GRPO has no judge at training time → cluster_id=0 for everything → trivially 1.0**. Cross-arm vs GRPO requires a separate post-hoc judge pass on GRPO rollouts.
- **non_zero_rate**: fraction of prompts with >=1 correct rollout (Poly-EPO Fig 2 right).

Steps 0..400, sampled every 10. Source: per-rollout JSONLs under `main/data/probes/per_rollout_v2/`.

| arm | step bin | |U_correct| | non_zero_rate | n_prompts_with_correct |
|---|---|---|---|---|
| grpo | 0-99 | 1.000 | 0.414 | 477 |
| grpo | 100-199 | 1.000 | 0.455 | 582 |
| grpo | 200-299 | 1.000 | 0.470 | 602 |
| grpo | 300-399 | 1.000 | 0.493 | 631 |
| minority | 0-99 | 1.111 | 0.391 | 502 |
| minority | 100-199 | 1.168 | 0.432 | 553 |
| minority | 200-299 | 1.187 | 0.475 | 608 |
| minority | 300-399 | 1.178 | 0.458 | 586 |
| polyepo | 0-99 | 1.173 | 0.429 | 494 |
| polyepo | 100-199 | 1.236 | 0.469 | 600 |
| polyepo | 200-299 | 1.227 | 0.477 | 610 |
| polyepo | 300-399 | 1.186 | 0.480 | 614 |
