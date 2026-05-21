# Analysis B — Cheap-substrate comparison vs LLM reference

**Reference:** `llm_cluster_id` from Analysis A (treating `-1` as its own cluster, no rollouts dropped).

**LLM minority-correct prompt rate (headline):** 14.55% over 165 prompts with >=1 correct rollout.

## Aggregate substrate metrics

Columns: mean ARI [95% CI], mean V-measure [95% CI], mean |Δn_clusters|, minority-correct prompt rate, concordance accuracy (substrate yes/no vs LLM yes/no on the 500 prompts).

| Substrate | Mean ARI [95% CI] | Mean V-measure [95% CI] | Mean |Δn_clusters| | Minority-rate | Concordance acc |
|---|---|---|---|---|---|
| `answer_strict` | 0.188 [0.159, 0.218] | 0.814 [0.799, 0.828] | 2.144 | 1.82% | 0.954 |
| `answer_loose` | 0.173 [0.144, 0.206] | 0.798 [0.782, 0.813] | 1.950 | 0.00% | 0.952 |
| `completion_embedding@0.2` | 0.074 [0.056, 0.094] | 0.465 [0.440, 0.490] | 2.588 | 11.52% | 0.962 |
| `completion_embedding@0.3` | 0.036 [0.023, 0.052] | 0.201 [0.180, 0.225] | 3.604 | 3.03% | 0.954 |
| `completion_embedding@0.4` | 0.023 [0.012, 0.037] | 0.094 [0.078, 0.112] | 4.004 | 1.82% | 0.950 |
| `completion_embedding@0.5` | 0.018 [0.007, 0.031] | 0.059 [0.045, 0.074] | 4.120 | 1.21% | 0.952 |
| `completion_features` | 0.110 [0.090, 0.131] | 0.653 [0.638, 0.668] | 1.784 | 16.36% | 0.962 |

### Confusion matrices (substrate yes/no × LLM yes/no, 500 prompts)

| Substrate | TP (both yes) | FP (sub yes, LLM no) | FN (sub no, LLM yes) | TN (both no) |
|---|---|---|---|---|
| `answer_strict` | 2 | 1 | 22 | 475 |
| `answer_loose` | 0 | 0 | 24 | 476 |
| `completion_embedding@0.2` | 12 | 7 | 12 | 469 |
| `completion_embedding@0.3` | 3 | 2 | 21 | 474 |
| `completion_embedding@0.4` | 1 | 2 | 23 | 474 |
| `completion_embedding@0.5` | 1 | 1 | 23 | 475 |
| `completion_features` | 16 | 11 | 8 | 465 |

## Winner

Highest mean ARI: **`answer_strict`** (mean ARI = 0.188, 95% CI [0.159, 0.218]). All substrates score in absolute terms low — none exceeds 0.2 mean ARI vs the LLM reference.

Best embedding threshold (across `completion_embedding@*`): **0.2** (mean ARI = 0.074). Larger thresholds collapse all 8 rollouts into a single cluster, driving ARI to zero.

Note: `completion_features` produces a minority-correct prompt rate (16.36%) closest to the LLM headline (14.55%), but this is rate-level coincidence — its ARI is 0.110 so the *which prompts* labelled minority-correct only partially overlap (see TP/FN counts above).

## Can claim / cannot claim (per §B.6/§B.7)

- **Can claim:** No cheap substrate we tested matches the LLM reference well (best mean ARI = 0.188). The LM judge appears load-bearing on Run 0; Poly-EPO's original judge choice is empirically justified by this evidence.
- **Cannot claim:** That a high-ARI cheap substrate would *work as well as* an LM judge inside an RL training loop. ARI on base-model rollouts is correlational, not causal — substrate quality is necessary-not-sufficient for downstream policy gradient behavior.
- **Cannot claim:** That `completion_embedding` is "CoT clustering." It is text embedding clustering (MiniLM sentence embeddings + agglomerative on cosine distance), a different operation with different semantics from the macro/micro reasoning-strategy criterion the LLM judge applies.

## Best embedding threshold for Analysis D

Analysis D's Cover@τ should use `completion_embedding` at distance threshold **0.2** (highest mean ARI among the swept thresholds = 0.074).
