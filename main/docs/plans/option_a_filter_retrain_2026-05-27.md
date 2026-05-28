# Option A — Filter-then-retrain plan (dry-run staging, do not execute)

> **Status: DRY-RUN ONLY.** This doc stages the artifacts and commands so Nancy can decide and execute in one shot when back. No money has been spent. The filter script (`main/scripts/filter_by_rollout_pass_rate.py`) has been smoke-tested on the existing n800 base rollouts and confirms ~33% mixed-reward yield (matches our 34% claim).

## Why

Synthesis entry in `timeline.md` and `fair_prompt_eval_summary_2026-05-27.md` establish:
1. Our 34% mixed-reward density matches the unfiltered GRPO baseline reported in [arxiv:2605.07689](https://arxiv.org/abs/2605.07689) (0.69 degeneracy at GS=4 on GSM8K).
2. Every successful published recipe at this regime applies rollout-pass-rate filtering (Polaris's 53K→30K refilter, DAPO dynamic sampling, [arxiv:2605.05112](https://arxiv.org/abs/2605.05112) Rollout Pass-Rate Control, [arxiv:2603.21177](https://arxiv.org/abs/2603.21177) Prompt Replay).
3. Fair-prompt eval shows `poly_epo_answer` already wins all three slices (+1.3/+0.8/+5.0 pp). Filtering the training data should give the hypothesis a clean chance to amplify, not just survive.

## Pipeline (3 phases)

### Phase 1 — Base-model rollout pass over 51K (one-shot, no policy update)

Generate N=8 base-model rollouts per prompt over the full `polaris_train.jsonl` (51,139 prompts), score with the production grader (mathd ∨ sympy), and dump per-prompt pass rates.

**Reuse the existing eval harness** (`main/probes/checkpoint_rollout_eval.py`) with `include_base: true` and no checkpoint variants — it already handles parallel B200 workers, prompt formatting, and reward scoring.

**Staged config to write** (not yet created): `main/configs/base_rollout_pass_polaris_51k_b200.yaml`. Template:

```yaml
# Single base-model rollout pass over the full prompt-filtered Polaris train set,
# N=8 per prompt. Used to compute per-prompt pass rates for the dynamic filter.
# Cost estimate: ~$200-300 + ~6-10 h wall on a single B200 (or shard across 4 GPUs).
operator: chicken602   # or anastasia — pick whichever has more credit
gpu_class: B200
global_seed: 42
modal_price_per_sec: 0.001736

eval:
  include_base: true
  report_by_band: true
  n_rollouts: 8
  rollout_chunk_prompts: 128   # tune for memory; 64 is the safe default
  output_dir: /vol/probes/base_rollout_pass_polaris_51k
  checkpoint_variants: []   # no trained checkpoints — base only
  datasets:
    polaris_train_full:
      kind: jsonl
      path: /root/main/data/polaris_train.jsonl   # the 51,139 prompt-filtered manifest
      use_all_rows: true
      n_prompts: 51139
      seed: 42
      prompt_variant: hybrid_answer_boxed
      persist_rollouts: true   # need per-rollout rewards for filtering

rollout:
  model: Qwen/Qwen3-1.7B-Base
  max_prompt_length: 1024
  max_response_length: 4096
  temperature: 1.0
  top_p: 1.0
  gpu_memory_utilization: 0.45
  max_model_len: 5120
  max_num_seqs: 128
  enable_prefix_caching: true
  logprobs: 1

artifacts:
  volume_name: main-artifacts
  volume_mount: /vol
```

**Caveat (confirmed 2026-05-27):** `checkpoint_rollout_eval.py` does **not** persist per-rollout records (it only writes summary JSONs to `partials/`). The proven path for "rollout over a manifest + dump per-rollout jsonl" is **`main/probes/group_a_rollout_judge.py`** — this is what wrote `main/data/probes/05-27/random_fullgold_n800/phase1_rollouts.jsonl` that the filter script consumes. Easiest Phase 1 = adapt `group_a_rollout_judge.py` to consume `polaris_train.jsonl` directly (it already reads a manifest at line 338 and writes per-rollout records at line 448). Skip the `persist_rollouts` flag on the eval harness.

**Phase 1a smoke (free, do this first):** rerun the existing `random_fullgold_n800` pass to completion (only 480/800 done per `grading_summary.json`). Same harness, same prompt — confirms the path works end-to-end and gives us another data point on filter yield. Config already exists: `main/configs/checkpoint_eval_random800_arms_b200.yaml`, just flip `include_base: true` and drop the trained variants.

### Phase 2 — Filter

```bash
main/.venv/bin/python3 main/scripts/filter_by_rollout_pass_rate.py \
  --manifest main/data/polaris_train.jsonl \
  --rollouts <pulled-from-Modal-volume>/base_rollout_pass_polaris_51k/<timestamp>/rollouts.jsonl \
  --out main/data/polaris_train_filtered_signal.jsonl \
  --meta main/data/polaris_train_filtered_signal.meta.json \
  --dropped-audit main/data/polaris_train_filtered_signal.dropped.jsonl \
  --min-pass 0.0 --max-pass 1.0 --require-min-rollouts 8
```

Expected output (extrapolating from n800 smoke): **~17K kept** (33% of 51K), with band distribution skewed away from extreme bands (0/8 and 7/8 dominate the dropped pile because they're closest to {0,1} pass-rate on 1.7B).

**Cutoff defensibility:** strict `pass_rate ∉ {0, 1}` (i.e., `min_pass=0.0`, `max_pass=1.0`) matches the gradient-starvation literature. Looser cutoffs like `min_pass=0.05, max_pass=0.95` (drop only "essentially impossible" and "essentially trivial") would keep more data — worth a 30-second dry-run to see how the yield curve looks before committing.

### Phase 3 — Retrain

Three options for what to retrain, in cost-ascending order:

**3a (recommended): retrain `poly_epo_answer` + GRPO only** on the filtered ~17K, 1 epoch, LR=1e-6 (or 2e-6 as a compromise — but stay conservative since the dataset is smaller and noisier per-batch). Skip minority — the across-slice fair-prompt evidence says it doesn't pay off, and a longer/higher-signal run isn't going to flip a structurally-flat curve. **Cost ~$450** (~1 epoch × 17K / 64 batch_size × 4 min/step for set-arms = 17.6 h wall × $0.46/step ≈ $200 per set-arm; GRPO is half that). Use **fresh checkpoint dirs** to keep this lineage isolated:

```yaml
# main/configs/train_real_b200_filtered_poly_epo.yaml
extends: train_real_b200.yaml
operator: chicken602   # pick based on remaining credit
train:
  dataset_jsonl: /root/main/data/polaris_train_filtered_signal.jsonl
  total_steps: 270   # ≈ 17000 / 64 ≈ 266 steps per epoch
arm_profiles:
  poly_epo_answer:
    train:
      checkpoint_dir: /vol/checkpoints/train_poly_epo_answer_b200_filtered/
```

(Plus parallel `train_real_b200_filtered_grpo.yaml` with the GRPO checkpoint dir.)

**3b: all three arms** — adds `train_real_b200_filtered_minority.yaml` for ~$200 more. Only justifiable if the LR=3e-6 probe shows minority moving up sharply in the next few hours; otherwise skip.

**3c: all three arms + 2 epochs.** Probably overkill given the timeline; revisit only if Phase 1+2 yield turns out way better than expected and we have budget.

## Launch commands (when ready — do not run now)

```bash
# Phase 1a (smoke, ~$15, 1 h):
# Make a 1-line edit to checkpoint_eval_random800_arms_b200.yaml (include_base: true,
# checkpoint_variants: []), then:
bash main/scripts/launch_checkpoint_eval.sh \
  --config main/configs/checkpoint_eval_random800_arms_b200.yaml --detach

# Phase 1 (51K base rollout pass, ~$200-300, ~8 h):
# After writing main/configs/base_rollout_pass_polaris_51k_b200.yaml:
bash main/scripts/launch_checkpoint_eval.sh \
  --config main/configs/base_rollout_pass_polaris_51k_b200.yaml --detach

# Phase 2 (filter, local, free):
MODAL_PROFILE=anastasia main/.venv/bin/modal volume get main-artifacts \
  probes/base_rollout_pass_polaris_51k/<TIMESTAMP>/rollouts.jsonl \
  /tmp/base_rollouts_51k.jsonl
main/.venv/bin/python3 main/scripts/filter_by_rollout_pass_rate.py \
  --manifest main/data/polaris_train.jsonl \
  --rollouts /tmp/base_rollouts_51k.jsonl \
  --out main/data/polaris_train_filtered_signal.jsonl \
  --meta main/data/polaris_train_filtered_signal.meta.json \
  --dropped-audit main/data/polaris_train_filtered_signal.dropped.jsonl

# Phase 3a (retrain poly_epo + GRPO on filtered, ~$450, ~18 h wall in parallel):
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm poly_epo_answer \
  --config main/configs/train_real_b200_filtered_poly_epo.yaml --no-resume --fresh-wandb
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm grpo \
  --config main/configs/train_real_b200_filtered_grpo.yaml --no-resume --fresh-wandb
```

## Risks / open questions

- **`checkpoint_rollout_eval.py` may not support persist_rollouts on a base-only config.** Confirm with the Phase 1a smoke before spending $200 on Phase 1. If it doesn't, ~30 min of plumbing.
- **The 51K dataset includes the n800 prompts** already rolled — could skip and save ~$3, but easier just to rerun for consistency.
- **Filter cutoff is empirical, not principled.** Polaris's own recipe may use a different threshold; not directly verified from the blog/repo (the citation-check subagent confirmed the recipe exists but did not pin the exact cutoff). Worth reading the [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/) before launching Phase 1 if cutoffs matter (which they will if the yield is materially different from ~33%).
- **Retraining poly_epo on filtered data may *not* widen its lead vs GRPO.** The set-clustering hypothesis requires multiple correct rollouts per prompt to cluster over; if the filter mostly admits prompts where pass_rate < 0.5, set-clustering may still have nothing to work with on the bulk of the data. Defensible either way for the writeup.
- **Time budget:** Phase 1 (8 h) + Phase 2 (instant) + Phase 3 (18 h in parallel) = ~26 h serial. Poster is 2026-06-03 (7 days out). Tight but feasible.

## Decision triggers (revisit when LR=3e-6 probe lands)

| LR=3e-6 outcome by step 200 | Recommended next move |
|---|---|
| Reward collapses or KL spikes >0.5 | Skip Option A; LR=1e-6 was already the right setting. Own-the-null with current data. |
| Flat curves, no minority/poly separation | Option A 3a (poly_epo+GRPO only); minority's structural failure is now established across two recipes. |
| poly_epo separates from GRPO by ≥1 pp | Option A 3a, possibly 3b — strong hypothesis-positive signal across two compute regimes is the headline. |
| minority separates from GRPO ≥1 pp | Revisit; this would be the biggest update. Probably Option A 3b (all three arms). |
