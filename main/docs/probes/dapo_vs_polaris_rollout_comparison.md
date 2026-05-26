# DAPO pilot (Run 0) vs Polaris probe — unified rollout metrics

Same model (**Qwen3-1.7B-Base**), **8 rollouts/prompt**, **temp=1** (both runs).
Unified grader: `extract_rank2` + `dapo_answer_v1` + strict Minerva normalize string match.

## Metric definitions (comparable)

| Metric | Definition |
|--------|------------|
| **pass@1** | Rollout-level: (# correct rollouts) / (# rollouts) |
| **pass@8** | Prompt-level mean of Chen et al. unbiased Pass@k (k=8, n=8); equals **% prompts with ≥1 correct** |
| **mixed_reward** | % prompts with some but not all rollouts correct (0 < n_correct < 8) |
| **all_wrong** | % prompts with n_correct = 0 (zero GRPO gradient under standard filter) |
| **parse_ok_rank2** | % rollouts where Rank-2 extraction succeeded |

## Headline comparison (unified strict grader)

| Dataset | Prompts | Rollouts | pass@1 | pass@8 | mixed_reward | all_wrong | parse_ok |
|---------|--------:|---------:|--------|--------|--------------|----------|----------|
| DAPO pilot Run0 | 500 | 4000 | 8.05% (322/4000) | 32.60% (163/500) | 32.60% (163/500) | 67.40% (337/500) | 72.95% |
| Polaris n800 (arm A) | 800 | 6400 | 6.03% (386/6400) | 26.62% (213/800) | 26.50% (212/800) | 73.38% (587/800) | 84.81% |
| Polaris subsample n=500 | 500 | 4000 | 5.90% (236/4000) | 26.20% (131/500) | 26.20% (131/500) | 73.80% (369/500) | 84.67% |

### Ratio (Polaris 800 / DAPO pilot) — unified strict

- pass@1: 0.75×
- pass@8: 0.82×
- mixed_reward: 0.81×
- all_wrong: 1.09× (lower is better for GRPO signal)

## DAPO pilot — legacy labels (same 500×8 rollouts)

Pilot also recorded run-time and human-cleaned labels (different parsers).

| Label source | pass@1 (rollout) | pass@8 (prompt, any-correct) |
|--------------|------------------|------------------------------|
| Unified strict (rerank) | 8.05% | 32.60% |
| Stored at run time (`correct`) | 8.10% | 32.60% |
| Human-cleaned (`correct_clean`) | 8.25% | 33.00% |

Published pilot baseline ([`minority_metrics.md`](../../../pre-milestone/nancy_explore/run0_analysis/analysis_minority/minority_metrics.md)) uses **correct_clean**: pass@1 **9.03%**, pass@8 **34.40%**.

## Distribution: n_correct rollouts per prompt (unified strict)

### DAPO pilot (n=500)

| n_correct | # prompts | % |
|-----------|----------:|--:|
| 0/8 | 337 | 67.4% |
| 1/8 | 81 | 16.2% |
| 2/8 | 38 | 7.6% |
| 3/8 | 24 | 4.8% |
| 4/8 | 11 | 2.2% |
| 5/8 | 6 | 1.2% |
| 6/8 | 2 | 0.4% |
| 7/8 | 1 | 0.2% |

### Polaris n800

| n_correct | # prompts | % |
|-----------|----------:|--:|
| 0/8 | 587 | 73.4% |
| 1/8 | 128 | 16.0% |
| 2/8 | 39 | 4.9% |
| 3/8 | 24 | 3.0% |
| 4/8 | 12 | 1.5% |
| 5/8 | 4 | 0.5% |
| 6/8 | 3 | 0.4% |
| 7/8 | 2 | 0.2% |
| 8/8 | 1 | 0.1% |

## Takeaway

- **pass@8** on Polaris (26.62%) is materially lower than DAPO pilot unified (32.60%) and much lower than pilot human-cleaned pass@8 (34.4%).
- **mixed_reward** is lower on Polaris (26.50% vs 32.60% on DAPO): fewer prompts contribute GRPO signal.
- **all_wrong** is higher on Polaris (73.38% vs 67.40%): more wasted rollouts per step without dynamic sampling.
- Grader alignment matters for pilot: unified strict pass@1 (~9%) is close to stored run-time (8.1%); human-cleaned labels are more lenient on extraction.
