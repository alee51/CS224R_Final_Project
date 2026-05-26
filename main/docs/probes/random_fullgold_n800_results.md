# Random full-gold Polaris probe (n800)

Uniform random sample from Polaris-53K with **relaxed cleaning** (non-empty problem string + non-empty gold; **no** integer-gold filter).

- **Prompt arm:** `hybrid_answer_boxed`
- **Grading:** `extract_rank2` + `grade_parsed_answer` (mathd OR sympy)
- **Manifest:** `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-27/random_fullgold_n800/manifest.jsonl`
- **Rollouts:** `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-27/random_fullgold_n800/phase1_rollouts.jsonl`

## Run status

| Field | Value |
|-------|-------|
| Manifest prompts | 800 |
| Rollouts graded | 3840 |
| Expected rollouts | — |
| Complete | yes |
| Integer-gold prompts (manifest) | 452 |
| Non-integer-gold prompts (manifest) | 348 |

## Overall

| Metric | Value |
|--------|-------|
| Rollout pass@1 | 9.40% (3840 rollouts) |
| Prompt pass@8 (Chen mean) | 33.12% |
| Prompt pass@8 (any correct) | 33.12% |
| parse_ok_rank2 (rollout) | 86.25% |

## By difficulty band

| Band | Rollouts | pass@1 | pass@8 (mean) | pass@8 (any) | parse_ok |
|------|----------|--------|---------------|--------------|----------|
| 0/8 | 952 | 7.67% | 26.05% | 26.05% | 82.04% |
| 1/8 | 632 | 7.44% | 25.32% | 25.32% | 86.71% |
| 2/8 | 400 | 7.50% | 38.00% | 38.00% | 87.75% |
| 3/8 | 448 | 8.71% | 32.14% | 32.14% | 86.83% |
| 4/8 | 408 | 8.09% | 33.33% | 33.33% | 87.99% |
| 5/8 | 336 | 11.01% | 33.33% | 33.33% | 87.50% |
| 6/8 | 232 | 16.38% | 51.72% | 51.72% | 88.79% |
| 7/8 | 432 | 14.81% | 46.30% | 46.30% | 88.89% |

## By gold type (diagnostic)

| Gold type | Prompts | Rollouts | pass@1 | pass@8 (any) | parse_ok |
|-----------|---------|----------|--------|--------------|----------|
| Integer gold | 277 | 2216 | 9.30% | 35.38% | 86.10% |
| Non-integer gold | 203 | 1624 | 9.54% | 30.05% | 86.45% |
