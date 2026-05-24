# Run1b step-1 `mean_reward` anomaly (2026-05-19)

## Summary

**Hypothesis H1 (seed / prompt slice) is correct.** The gap is not a reward-pipeline bug. `run1b_grpo` uses `seed: 43` while `run2_inverse_freq` and `run3_f_grpo` use `seed: 42` (`pilot/configs/run1b_grpo.yaml:4`, `run2_inverse_freq.yaml:9`, `shared_train.yaml:16`). Each seed produces a different shuffle of `dapo_slice_3k.jsonl`, so step-1 sees **disjoint** 32-prompt batches.

## Evidence

### Config and sampling

- `RUNBOOK.md` documents seed 43 for run1b intentionally (diversity vs run1 at seed 42).
- `_load_train_prompts` shuffles the full slice with `random.Random(seed)` (`hf_grpo_train.py:69-76`).
- Step `k` takes a contiguous window: `start = (step * batch_prompts) % len(prompts)` (`hf_grpo_train.py:885-888`). Step 1 → rows `[0:32]` of the shuffled list.

Replaying shuffle: seed 42 first prompt `0b4478a7-…`; seed 43 first prompt `71fb6079-…` (matches first lines of each `raw_predictions.jsonl`). **0/32 prompt_id overlap** between run1b and run2/run3.

### Measured rewards (step-1 batch, 256 rollouts)


| Run               | `seed` | Step-1 prompts  | Mean correct (≈ `mean_reward`) |
| ----------------- | ------ | --------------- | ------------------------------ |
| run1b_grpo        | 43     | 32 unique       | **0.062**                      |
| run2_inverse_freq | 42     | 32 unique       | **0.172**                      |
| run3_f_grpo       | 42     | same 32 as run2 | **0.172**                      |


Run2/run3 agreement confirms identical prompt batch + identical base model at step 1; not objective-specific.

### H2 rejected — reward path is shared

All objectives build rewards in `_build_step_groups` before advantages:

```654:654:pilot/train/hf_grpo_train.py
            reward = 1.0 if is_correct(text, gold) else 0.0
```

Logged `mean_reward` is the unweighted rollout mean (`hf_grpo_train.py:941-943`). `weighted_advantages` in `objectives.py:57-69` only scales **advantages** (`inverse_freq`, `f_grpo`); it does not touch `group.rewards` or the log line.

### H3 rejected — canonicalization is objective-agnostic

`cluster_id(parsed)` runs for every run in the same block (`hf_grpo_train.py:655`). Clustering affects advantage weighting only, not the binary reward or `mean_reward` metric.

## Recommendation (diagnosis only)

- Treat step-1 `mean_reward` across run1b vs run2/3 as **not comparable** without matched seeds or eval on a fixed prompt set.
- For objective ablations, either align all matrix seeds to 42 or report per-seed / pooled tier-1 metrics—not single-step training reward.
- No code fix required for reward correctness; optional ops fix: document in launch checklist that run1b’s seed-43 batch is expected to differ from run2/3.

## Impact on pilot redesign

Does **not** invalidate the reward verifier, clustering hook, or objective implementations. It **does** invalidate any informal check that used “run1b step-1 reward ≈ run2/3” as proof of a shared reward signal—that equality would only hold with the same seed and step index. Decisions in `decision_memo.md` (tier-1 gate, Run0→Run1/1b/2/3 sequence) remain valid; do not use this anomaly to reject inverse_freq or F-GRPO on grounds of a broken baseline reward path.