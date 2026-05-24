# Analysis D — Frozen-eval base-model baseline

> ⚠ **ARCHIVED (v2-era).** Cover@τ `n_eligible` fields in `baseline_metrics.json` use **165** prompts (`is_correct_v2`). Human labels → **172** eligible — recompute in Phase 5.

_500 prompts × 8 rollouts; bootstrap 95% CIs, prompt-level resampling, 1000 resamples, seed=0._

**Lock discrepancy note.** `pilot/preflight_lock.json` records `pass_at_k=16` (and `bootstrap_samples=2000`), but Run 0 produced only 8 rollouts per prompt. Per Analysis D design (§D.2) we report **Pass@8** and flag this discrepancy. Bootstrap is set to 1000 per the Analysis D spec.

## Headline table

| Metric | v1 (orig parser) | v2 (fixed parser) |
| --- | --- | --- |
| Pass@1 | 8.10% [6.85%, 9.35%] | 8.25% [6.97%, 9.53%] |
| Pass@8 | 32.60% [28.60%, 36.80%] | 33.00% [29.20%, 37.00%] |
| Cover@τ=0.15 (answer_loose) | n/a | 49.70% [41.72%, 57.50%] |
| Cover@τ=0.15 (completion_embedding) | n/a | 92.12% [87.95%, 96.05%] |
| Cover@τ=0.15 (llm_clusters) | n/a | 72.73% [66.07%, 79.65%] |
| worst_subset_accuracy | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |

## Notes on metric definitions

- **Pass@8** uses the Chen et al. (2021) unbiased estimator with n=8 sampled completions and k=8. With n=k=8 this collapses to `1 if any rollout correct else 0` per prompt; the prompt-mean is then `Pass@8`.
- **Cover@τ** is computed only over prompts with ≥1 correct rollout. Mass = `(count of rollouts in the largest cluster that contains any correct rollout) / 8`.
  - `answer_loose`: cluster substrate = `cluster_id_v2` (v2 canonical-answer hash).
  - `completion_embedding`: cluster substrate built from Analysis B's MiniLM embeddings at the best threshold per `substrate_comparison.md` (column = `completion_embedding@0.2`, source: `embed_clusters_at_best_threshold.parquet` produced by `build_embed_clusters.py`).
  - `llm_clusters`: cluster substrate = `llm_cluster_id` from Analysis A. Rollouts with `llm_cluster_id == -1` (degenerate) are treated as their own singleton clusters per rollout, so they cannot anchor the 'largest correct cluster'.
- **worst_subset_accuracy**: within each bootstrap resample, prompts are ranked by per-prompt Pass@1 and the worst 25% are selected; the reported value is the mean per-prompt Pass@1 over that worst quartile. Each bootstrap draw re-selects its own worst 25%.

## Can claim / cannot claim (per §D.4–D.5)

**Can claim:** the proxy base model (Qwen3-1.7B-Base) achieves the values in the table above on the Run 0 prompt set (rows 0–499 of `pilot/data/dapo_slice_3k.jsonl`) under the v2 parser; any training arm needs to clear these to be a real gain on this distribution.

**Cannot claim:** generalization to AIME-25 / HMMT / Minerva / MATH-500. Run 0 uses DaPO-3k proxy training prompts; this is a training-distribution baseline only. We also cannot claim anything about training-time behavior — these are base-model rollouts.

