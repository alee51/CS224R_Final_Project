# `main-verl/eval/` — held-out evaluation

Probe + analysis scripts for the 3 Stage 8 step-400 checkpoints. The
authoritative spec for what we run and why is in `writeup/eval.md`. The
authoritative spec for the training setup the eval is comparing against is in
`main/docs/STANDARDS.md`. The active eval-decision doc (datasets + metric
candidates) is `writeup/eval_panel_candidates.md`.

## Layout

```
main-verl/eval/
├── README.md
├── run_eval.py            ← Modal probe (FSDP→HF merge, vLLM, score, write JSON)
├── launchers/{grpo,polyepo,minority}.sh
├── analysis/
│   ├── rescore.py             re-apply training grader to a saved eval JSON
│   ├── coverage.py            majority@k, distinct-answers@k, entropy@k
│   ├── compare.py             cross-arm pass@k markdown table
│   ├── per_rollout_diagnostic.py  training-time per-rollout JSONL analysis
│   ├── cluster_correctness.py     P(cluster correct | rank), training-time
│   └── u_correct.py           |U_correct| trajectory, training-time, set arms
└── results/
    ├── comparison.md          live cross-arm pass@k table
    └── minority_diagnostic.md why minority underperforms
```

## How to launch

```bash
bash main-verl/eval/launchers/grpo.sh
bash main-verl/eval/launchers/polyepo.sh
bash main-verl/eval/launchers/minority.sh
```

Each launcher pins the Modal account that owns its arm's ckpt
(anastasia / stonedpinecones / emma), so eval runs on the same volume as the
training run. Default `B200:1` (single GPU, TP=1). Override via env vars in
`launchers/*.sh`:

```bash
CS224R_EVAL_CKPT_PATH=/vol/checkpoints/main-verl/<run_name>/global_step_400/actor
CS224R_EVAL_LABEL=<arm>_step400
CS224R_EVAL_DATASETS=aime25,math500,hmmt_feb25,hmmt_nov25,beyondaime
CS224R_EVAL_N_ROLLOUTS=16            # 32+ for AIME-style; see writeup/eval.md
CS224R_EVAL_GPU_COUNT=1
CS224R_EVAL_OUTPUT_DIR=/vol/probes/eval_4b
```

## Output

Per-dataset JSON written incrementally to
`/vol/probes/eval_4b/<label>_<dataset>.json` (so a mid-run cancellation
doesn't lose completed datasets). Pull locally:

```bash
MODAL_PROFILE=<account> modal volume get --force main-artifacts \
  probes/eval_4b/<label>_<dataset>.json /tmp/<arm>_<dataset>.json
```

Each file has `{label, ckpt_path, n_rollouts, datasets: {<dataset>: {n_prompts,
pass_at_k, mean_reward_at_1, per_prompt: [{problem_id, ground_truth, n_correct,
rewards, preds, rollouts}]}}}`. `per_prompt.rollouts` holds the full text and
makes JSONs large (~100 MB per polaris_val run); slice it out before sharing.

## Analysis recipes

```bash
# Re-apply training grader to a saved eval JSON
python3 main-verl/eval/analysis/rescore.py /tmp/<arm>_<dataset>.json

# majority@k + distinct-answers@k + entropy@k
python3 main-verl/eval/analysis/coverage.py /tmp/<arm>_<dataset>.json

# Cross-arm pass@k markdown
python3 main-verl/eval/analysis/compare.py

# Training-time per-rollout diagnostic
python3 main-verl/eval/analysis/per_rollout_diagnostic.py --sample-every 5

# Per-rank cluster correctness (training, minority + polyepo)
python3 main-verl/eval/analysis/cluster_correctness.py --step-min 100 --step-max 400 --sample-every 10

# |U_correct| trajectory (training, minority + polyepo)
python3 main-verl/eval/analysis/u_correct.py --sample-every 10
```
