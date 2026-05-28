# BeyondAIME rollout qualitative analysis (20260527T223115Z)

- Rollouts source: `main/data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/rollouts_20260527T223115Z`
- Generated artifacts: `main/data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/qualitative_20260527T223115Z`
- Comparator: `base` vs each trained checkpoint label

## Core counts by arm

| Arm | base parse_ok | trained parse_ok | base_pass16_trained_fail | trained_only | base_any_correct_trained_none |
| --- | ---: | ---: | ---: | ---: | ---: |
| grpo_b200_s359 | 0.854 | 0.868 | 10 | 4 | 0 |
| minority_b200_s159 | 0.854 | 0.854 | 12 | 6 | 0 |
| poly_epo_b200_s133 | 0.854 | 0.871 | 11 | 6 | 0 |

## Directional conclusion checks

- `grpo_b200_s359`: parse_ok similar=True; base_pass16_trained_fail>trained_only=True; reasoning-error signal=True (147/160 fail rollouts had parse_ok+parsed_answer).
- `minority_b200_s159`: parse_ok similar=True; base_pass16_trained_fail>trained_only=True; reasoning-error signal=True (161/192 fail rollouts had parse_ok+parsed_answer).
- `poly_epo_b200_s133`: parse_ok similar=True; base_pass16_trained_fail>trained_only=True; reasoning-error signal=True (156/176 fail rollouts had parse_ok+parsed_answer).

## Findings

- Across all three arms, `parse_ok` remains close to base, so parsing collapse is not the main explanation for regressions.
- For each arm, `base_pass16_trained_fail` materially exceeds `trained_only`, matching prior asymmetry.
- Many trained failures in `base_pass16_trained_fail` still have parseable answers, supporting reasoning-error/mismatch rather than extraction-only failure.
