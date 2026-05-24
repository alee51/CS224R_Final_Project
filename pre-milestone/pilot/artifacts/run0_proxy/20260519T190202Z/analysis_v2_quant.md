# Run 0 quantitative analysis v2 (completion-aware)

**Artifact:** `20260519T190202Z`  
**Inputs:** `raw_predictions.jsonl` (4000 rollouts), `prompt_inputs.jsonl` (500 prompts)  
**Script:** `analysis_v2_compute.py`  
**Per-prompt table:** `analysis_v2_prompt_stats.jsonl`

## Method

Each rollout's `**completion`** text was inspected. We computed **as-recorded** stats from stored `parsed_answer` / `correct` / `cluster_id`, and **completion-aware** stats by re-running `extract_answer`, `is_correct`, extract-path inference, boxed counts, and heuristic **semantic buckets** (`extract_answer priority + normalize_semantic: int n:, percent->n:, frac:a/b, else s:<lowercase<=120c>`). No LLM labeling.

**Cluster IDs:** cross-process `cluster_id()` hash mismatches are expected (Python hash salt). Within-artifact checks use **canonical string equality** `canonicalize_answer(parsed)` for grouping; `cluster_canon_split_in_prompt` flags when one canon maps to multiple stored `cluster_id`s.

## Global mismatch rates (4000 rollouts)


| Check                                                        | Mismatches | Rate    |
| ------------------------------------------------------------ | ---------- | ------- |
| `extract_answer(completion)` vs stored `parsed_answer`       | 126        | 3.15%   |
| `is_correct(completion, gold)` vs stored `correct`           | 58         | 1.45%   |
| `cluster_id(recomputed_parse)` vs stored `cluster_id` (hash) | 4000       | 100.00% |
| Loose: `canon(reparse)==canon(gold)` vs stored `correct`     | 3          | 0.07%   |


*Loose correct* uses canonical equality on the **re-extracted** parse, while production `is_correct` only accepts a **single shallow** `\boxed{...}` with int contents — so loose can exceed stored correct when boxed is missing but tail text matches gold.

**Storage consistency:** 92 prompts (18.4%) have ≥1 rollout where today's `extract_answer(completion)` ≠ stored `parsed_answer`; 47 prompts have ≥1 `correct` mismatch. An earlier audit note claiming 0% extract/correct mismatch appears stale; re-running `_audit_script.py` on this artifact reproduces the rates above (likely write-time vs current `answer_parse.py`, or artifact regeneration).

## Boxed & extract-path usage (completion text)


| Metric                                                 | Count | % rollouts |
| ------------------------------------------------------ | ----- | ---------- |
| Completions containing `\\boxed{`                      | 2023  | 50.6%      |
| Exactly one shallow-regex `\\boxed{...}`               | 1895  | 47.4%      |
| Multiple shallow-regex boxed                           | 34    | 0.8%       |
| Last boxed: brace-balanced inner ≠ shallow-regex inner | 88    | 2.2%       |


**Extract path** (which branch `extract_answer` uses):


| Path        | Count | %     |
| ----------- | ----- | ----- |
| boxed       | 1895  | 47.4% |
| last_line   | 1164  | 29.1% |
| answer_line | 941   | 23.5% |


**Answer-line + last-line fallback** (no single shallow boxed): 2105 (52.6%)

## Completion length

- Mean chars per completion (prompt-avg of rollout means): **1877**
- Mean approx tokens (chars/4): **469**

## Answer diversity per prompt (500 prompts)


| Metric                                    | Mean | Median |
| ----------------------------------------- | ---- | ------ |
| Distinct **semantic buckets** (heuristic) | 7.09 | 8.0    |
| Distinct stored `parsed_answer`           | 7.19 | 8.0    |
| Distinct stored `cluster_id`              | 7.18 | 8.0    |


- Prompts with **>1 semantic bucket** (8 rollouts): **500** (100.0%)
- Semantic buckets **>** distinct stored parsed: 0
- Semantic buckets **<** distinct stored parsed: 47
- Prompts where same `canon(parsed)` maps to **>1 stored cluster_id**: 3

### Distribution: distinct semantic buckets per prompt


| n_distinct_semantic_buckets | count | %     |
| --------------------------- | ----- | ----- |
| 0                           | 0     | 0.0%  |
| 1                           | 0     | 0.0%  |
| 2                           | 1     | 0.2%  |
| 3                           | 7     | 1.4%  |
| 4                           | 18    | 3.6%  |
| 5                           | 32    | 6.4%  |
| 6                           | 57    | 11.4% |
| 7                           | 131   | 26.2% |
| 8                           | 254   | 50.8% |


### Distribution: distinct stored parsed answers per prompt


| n_distinct_parsed | count | %     |
| ----------------- | ----- | ----- |
| 0                 | 0     | 0.0%  |
| 1                 | 0     | 0.0%  |
| 2                 | 1     | 0.2%  |
| 3                 | 5     | 1.0%  |
| 4                 | 14    | 2.8%  |
| 5                 | 33    | 6.6%  |
| 6                 | 45    | 9.0%  |
| 7                 | 128   | 25.6% |
| 8                 | 274   | 54.8% |


### Distribution: distinct stored clusters per prompt


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


### Distribution: correct rollouts per prompt (stored vs recomputed)

**Stored `correct`:**


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


**Recomputed `is_correct`:**


| n_correct_recomputed | count | %     |
| -------------------- | ----- | ----- |
| 0                    | 350   | 70.0% |
| 1                    | 87    | 17.4% |
| 2                    | 32    | 6.4%  |
| 3                    | 18    | 3.6%  |
| 4                    | 7     | 1.4%  |
| 5                    | 3     | 0.6%  |
| 6                    | 3     | 0.6%  |
| 7                    | 0     | 0.0%  |
| 8                    | 0     | 0.0%  |


## Correctness summary

- Rollouts with stored correct: **324/4000** (8.1%)
- Prompts with ≥1 correct rollout: **32.6%**

## Exemplar prompts

- **Max semantic diversity:** `01677f18-3dff-43b8-bf2d-6e4c9fa5cde6` — buckets=8 ['frac:1/2', 'frac:m/n', 'n:17', 'n:22', 's:area_mo', 's:m = sp.point(m[0].evalf() m[1].']…, stored parsed=8, correct=0/8
- **Most stored-correct rollouts:** `ddd26788-0e7c-4330-ae56-30b48f36c031` — 7/8

## Limitations (automated semantic bucketing)

- Buckets **collapse** format variants (e.g. `\(50\)` and `50` → `n:50`) that production clustering **splits** via `canonicalize_answer`.
- Buckets **do not** prove mathematical equivalence; different buckets can be wrong for the same reason, and one bucket can hide multiple reasoning errors.
- Nested `\boxed{...}` uses shallow regex in production; brace-balanced diagnostics show **2.7%** rollouts where regex inner ≠ balanced inner (88/4000).
- `is_correct` ignores non-boxed tails even when `Answer:` matches gold; loose canon match counts are **not** deployable accuracy.
- Prior audit (`_audit_parse_cluster.md`): stored fields are **internally consistent** with re-extraction; semantic defects are in canon/boxed rules.

## Interpretation for experiments

Completion reading confirms **high within-prompt diversity**: median 8 semantic buckets vs 8 stored parses. ~51% of completions mention `\boxed`, but only ~47% yield a single shallow boxed extract — the rest fall through to Answer:/last-line paths, which inflates parse diversity and depresses strict correct rate (8.1%). Treat **semantic bucket counts** as upper-bound answer-mode spread; fix boxed/canon before cluster-level RLVR rewards.