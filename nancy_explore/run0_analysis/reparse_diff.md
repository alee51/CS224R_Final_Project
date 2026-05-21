# Run 0 parser-fix re-score (0b)
**Date:** 2026-05-21  
**Source:** `data/raw_predictions.jsonl` + `data/prompt_inputs.jsonl`  
**Parser:** `pilot/train/answer_clean.py` — brace-balanced `\boxed` (C2), `normalize_answer_clean` (C3)  
**Output:** `data/predictions_reparsed.jsonl`

## Accuracy (rollout Pass@1)
| Version | Field | Rate | n correct / 4000 |
|---|---|---:|---:|
| v1 (stored) | `correct` | 8.10% | 324 |
| v2 (re-parsed) | `is_correct_v2` | 8.25% | 330 |
| Δ (v2 − v1) | | +0.15 pp | +6 / −0 flips |

## Cluster churn
| Metric | Count | Rate |
|---|---:|---:|
| `parsed_answer` → `parsed_answer_v2` changed | 514 | 12.85% |
| `cluster_id` → `cluster_id_v2` changed | 4000 | 100.00% |
| Prompts with different canon grouping (8 rollouts) | 311 | 62.2% |
| Mean distinct clusters / prompt (v1) | 7.18 |
| Mean distinct clusters / prompt (v2) | 6.86 |

## Unparseable (v2)
- Rollouts with empty `parsed_answer_v2`: **426** (10.65%)
- Breakdown by `extract_path_v2`:
| Path | Count | % |
|---|---:|---:|
| `boxed_balanced` | 2016 | 50.4% |
| `answer_line` | 853 | 21.3% |
| `last_line` | 705 | 17.6% |
| `runon_rejected` | 426 | 10.7% |

## Minority-correct prompt rate (bootstrap 95% CI, prompt-level)
Definition: among prompts with ≥1 correct rollout, fraction where correct rollouts span ≥2 clusters and at least one correct cluster is not the largest.
| Version | Rate | 95% CI |
|---|---:|---|
| v1 (`correct`, `cluster_id`) | 0.00% | [0.00%, 0.00%] |
| v2 (`is_correct_v2`, `cluster_id_v2`) | 0.00% | [0.00%, 0.00%] |

**Note:** Under exact-match clustering, correct rollouts that share the same canonical answer land in one cluster — minority-correct stays ~0% unless semantically equivalent answers split across clusters (parser) or substrate changes (Analysis A LLM clusters).

### Correct rollouts per prompt
**v1 (stored)**
| n_correct_v1 | count | % |
|---|---:|---:|
| 0 | 337 | 67.4% |
| 1 | 81 | 16.2% |
| 2 | 35 | 7.0% |
| 3 | 28 | 5.6% |
| 4 | 10 | 2.0% |
| 5 | 6 | 1.2% |
| 6 | 2 | 0.4% |
| 7 | 1 | 0.2% |
| 8 | 0 | 0.0% |

**v2 (re-parsed)**
| n_correct_v2 | count | % |
|---|---:|---:|
| 0 | 335 | 67.0% |
| 1 | 83 | 16.6% |
| 2 | 33 | 6.6% |
| 3 | 29 | 5.8% |
| 4 | 10 | 2.0% |
| 5 | 7 | 1.4% |
| 6 | 2 | 0.4% |
| 7 | 1 | 0.2% |
| 8 | 0 | 0.0% |
## Correct gained (v1 false → v2 true)
- `22063de2-a7a2-4214-895f-e015e0b78f87`
- `2e690d58-de84-4003-a33f-fbebdb71dae5`
- `65da7224-5f07-48e3-9b01-3c9ea1dfb036`
- `70aabfd8-5728-4d08-8363-94e175fc0632`
- `cfc7b48f-94bf-429f-b1c9-a7ac15e86b80`

## Implications for Analyses A–D
- **Analysis A/B/C/D** should use `data/predictions_reparsed.jsonl` (`is_correct_v2`, `cluster_id_v2`, `canonical_v2`).
- Parser fixes are **small** on accuracy (+6 rollouts) but **large** on parse/cluster hygiene (12.9% parsed changed; all 500 prompts re-grouped under deterministic SHA cluster ids).
- `minority_correct_prompt_rate_v2` remains 0% — proceed to **Analysis A** (LLM reasoning clusters) for the substrate-controlled gate metric.
