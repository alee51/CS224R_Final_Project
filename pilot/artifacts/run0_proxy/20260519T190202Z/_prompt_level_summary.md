# Run 0 prompt-level rollout summary

**Artifact:** `20260519T190202Z`  

**Prompts:** 500 × 8 rollouts = 4000 lines  

**Fields:** as-recorded `correct`, `cluster_id`, `parsed_answer` (no re-canonicalization).

**Dropped metrics:** `n_correct_clusters`, `correct_cluster_sizes`, `largest_correct_cluster_size`, `smallest_correct_cluster_size`, `minority_correct_cluster`, and `has_any_correct` — with this pilot's `is_correct()` / `cluster_id` rules, every correct rollout on a prompt shares one cluster, so those fields are derivable from `n_correct_rollouts`.

## Distribution: distinct clusters per prompt (0–8)


| n_distinct_clusters | count | %     |
| ------------------- | ----- | ----- |
| 0                   | 0     | 0.0%  |
| 1                   | 0     | 0.0%  |
| 2                   | 1     | 0.2%  |
| 3                   | 6     | 1.2%  |
| 4                   | 14    | 2.8%  |
| 5                   | 32    | 6.4%  |
| 6                   | 49    | 9.8%  |
| 7                   | 126   | 25.2% |
| 8                   | 272   | 54.4% |


## Distribution: correct rollouts per prompt (0–8)


| n_correct_rollouts | count | %     |
| ------------------ | ----- | ----- |
| 0                  | 337   | 67.4% |
| 1                  | 81    | 16.2% |
| 2                  | 35    | 7.0%  |
| 3                  | 28    | 5.6%  |
| 4                  | 10    | 2.0%  |
| 5                  | 6     | 1.2%  |
| 6                  | 2     | 0.4%  |
| 7                  | 1     | 0.2%  |
| 8                  | 0     | 0.0%  |


## Key aggregates


| Metric                       | Value             |
| ---------------------------- | ----------------- |
| Mean distinct clusters       | 7.18              |
| Median distinct clusters     | 8.0               |
| Mean distinct parsed answers | 7.19              |
| % prompts with ≥1 correct    | 32.6% (163/500)   |
| Total correct rollouts       | 324 / 4000 (8.1%) |


## Exemplar prompts

- **Max answer diversity (clusters/parsed):** `01677f18…` — clusters=8, parsed=8, correct=0/8, wrong_clusters=8
- **All 8 rollouts wrong, high wrong-cluster spread:** `01677f18…` — clusters=8, parsed=8, correct=0/8, wrong_clusters=8
- **Most correct rollouts in one prompt:** `ddd26788…` — clusters=2, parsed=2, correct=7/8, wrong_clusters=1
- **Single correct rollout among 8:** `07f5b71e…` — clusters=8, parsed=8, correct=1/8, wrong_clusters=7
- **Low diversity, many correct:** `ddd26788…` — clusters=2, parsed=2, correct=7/8, wrong_clusters=1

## Canonicalization / parse caveats (interpretation skew)

- Stored `cluster_id` is consistent with `hash(canonicalize(parsed))` in-process; **LaTeX/format splits** (e.g. `\(50\)` vs `50`) inflate `n_distinct_clusters` and can split semantically equivalent answers across clusters (`_audit_parse_cluster.md` §1–2).
- **Format false negatives** (~0.15% rollouts) depress `n_correct_rollouts` without changing cluster structure much.
- **Truncated `\boxed{...}`** parses create spurious clusters and wrong `correct` flags; true-equivalent parses in different clusters can make `n_wrong_clusters` look larger than true answer modes.

## Interpretation for next experiments

The rollout substrate shows **high within-prompt diversity**: median ~8 distinct clusters per 8 samples, with only **33%** of prompts seeing any correct answer at ~8% per-rollout accuracy. That pattern is viable for contrastive or cluster-based training signals—many wrong modes and occasional correct islands—but **semantic clustering bugs** mean cluster counts are a noisy superset of true answer modes. Before trusting cluster-level rewards, fix canonicalization (delimiters, `%`, brace handling) and boxed extraction; then re-run this table. Near-term experiments should treat **parsed string diversity** and **correct-hit rate** as primary viability metrics; use stored `cluster_id` only after canon fixes, or bucket by normalized parse in analysis.