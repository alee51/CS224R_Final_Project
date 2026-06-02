# Eval spec (auditable)

Held-out evaluation pipeline for the three step-400 Stage-8 checkpoints.
Probe: `main-verl/eval/run_eval.py`.

## 1. Dataset panel

Five datasets, one row per problem, served as verl-format parquets from
`main-verl/data/` (registry at `main-verl/eval/run_eval.py:151-159`).
Polaris-val and DAPO-slice are in-distribution and redundant with training
validation; they are excluded from the headline panel.

| dataset | n | type | source | parquet |
|---|---|---|---|---|
| `aime25`     | 30  | hard OOD       | MathArena / AIME 2025                   | `main-verl/data/aime_val.parquet` |
| `math500`    | 500 | easy OOD       | MATH-500 (Hendrycks / lighteval)        | `main-verl/data/math500.parquet` |
| `hmmt_feb25` | 30  | hard OOD       | MathArena / HMMT Feb 2025               | `main-verl/data/hmmt_feb25.parquet` |
| `hmmt_nov25` | 30  | hard OOD       | MathArena / HMMT Nov 2025               | `main-verl/data/hmmt_nov25.parquet` |
| `beyondaime` | 100 | hard OOD       | ByteDance-Seed/BeyondAIME               | `main-verl/data/beyondaime.parquet` |

Each row carries the verl schema
`{prompt: [{role, content}], data_source, reward_model: {ground_truth, style},
extra_info: {problem_id}}`.

## 2. Prompt format

Identical to training. The eval probe (`main-verl/eval/run_eval.py:178-191`)
loads the parquet `prompt` field, renders it through the model's own
`AutoTokenizer.apply_chat_template(..., add_generation_prompt=True)`, and
passes the rendered string to vLLM. The user-turn content is

```
<problem text>
Please reason step by step, and put your final answer within \boxed{}.
```

The `\boxed{}` suffix is baked into the parquet at data-prep time, so the
eval probe does not add it a second time — the rendered string matches what
the policy saw during training.

## 3. Scorer

Same function as the training reward
(`main-verl/eval/run_eval.py:206-210`):

```python
from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed
```

`compute_score` runs `last_boxed_only_string` → `remove_boxed` → Hendrycks
`is_equiv` against the parquet `reward_model.ground_truth`. The probe also
saves the raw `remove_boxed(...)` output as `preds[i]` for offline
diversity analysis (`run_eval.py:220-222`). All `pass@k` headline numbers are
derived from this `1.0/0.0` grader; this matches the training grader exactly,
so eval and train pass@k are on the same scale.

## 4. Sampling

Held in `main-verl/eval/run_eval.py:143-148`:

| field | value | matches training? |
|---|---|---|
| `temperature` | `1.0` | yes (training: `:42`) |
| `top_p` | `1.0` | yes |
| `top_k` (vLLM default `-1`) | `-1` | yes |
| `max_tokens` | `4096` | yes (= `max_response_length`) |
| `n` (rollouts/prompt) | env `CS224R_EVAL_N_ROLLOUTS`, default `16` | rollouts/prompt at training time = 8; eval doubles it to halve the unbiased-pass@k variance |

**Recommended n_rollouts for the writeup panel.** Small datasets need more
statistical power per problem:

| dataset | recommended n | reason |
|---|---|---|
| `aime25`, `hmmt_feb25`, `hmmt_nov25` | **32** | 30 problems × 32 rollouts keeps pass@k variance manageable; report up to k=16 |
| `beyondaime` | 32 | 100 problems, hard OOD; same rationale |
| `math500` | 16 | 500 problems is enough breadth; k=16 fully reportable |

## 5. Metrics

Headline metric is **pass@k** with `k ∈ {1, 4, 8, 16}`. Diversity diagnostics
come from `main-verl/eval/analysis/coverage.py`.

| metric | definition | implementation |
|---|---|---|
| **pass@k** | `1 − C(n−c, k) / C(n, k)` per problem (`n` rollouts, `c` correct), averaged over problems. Unbiased estimator. | `run_eval.py:237-251` |
| **majority@k** | take the first `k` rollouts; majority vote on non-empty `parsed_answer`; correct if the modal answer equals the gold (under the same `is_equiv` grader). | offline via `eval/analysis/coverage.py` |
| **coverage@k** | per problem, number of distinct non-empty `parsed_answer` strings among rollouts with `reward = 1`, averaged over problems. | `eval/analysis/coverage.py:24-38` |
| **distinct_answers@k** | per problem, number of distinct non-empty `parsed_answer` strings in the first `k` rollouts (correct or not). | `eval/analysis/coverage.py:41-47` |
| **entropy@k** | Shannon entropy (bits) of the `parsed_answer` distribution over the first `k` rollouts. | `eval/analysis/coverage.py:50-61` |
| **`\|U_correct\|`@k** | per problem, the number of distinct judge CoT cluster IDs (degenerate excluded) among the first `k` rollouts with `reward = 1`, averaged over problems. Requires running the judge over the saved eval rollouts. The headline diversity metric per the Poly-EPO paper. | new — wire through the existing judge HTTP client in `main-verl/train/clusters_judge.py` against the saved eval JSON, all 3 arms |

## 6. Procedure

For each arm `<arm> ∈ {grpo, poly_epo, minority}`:

1. **Merge.** `python scripts/model_merger.py merge --backend fsdp
   --local_dir /vol/checkpoints/main-verl/<run>/global_step_400/actor
   --target_dir /tmp/merged_hf` (`run_eval.py:115-123`).
2. **Load vLLM.** `LLM(model=/tmp/merged_hf, tensor_parallel_size=1,
   gpu_memory_utilization=0.85, max_model_len=5120, enforce_eager=True,
   dtype=bfloat16)` on `B200:1` (`run_eval.py:133-141`, `:75`).
3. **Generate.** Per dataset: render prompts via the model's chat template,
   `llm.generate(rendered, sampling_params)` with the sampling spec in §4.
4. **Score.** Apply `math.compute_score` to each rollout; save
   `rewards`, `preds`, `rollouts` per prompt.
5. **Compute pass@k** (incremental dump per dataset to
   `/vol/probes/eval_4b/<label>_<dataset>.json`, `run_eval.py:265-275`).

Each arm's eval runs on the Modal account that owns its training ckpt
(GRPO → anastasia, Poly-EPO-CoT → stonedpinecones, Minority-CoT → emma) to
avoid cross-account ckpt copy. The `/tmp/merged_hf` directory is ephemeral.

## 7. Reproducibility

One bash launcher per arm at `main-verl/eval/launchers/{grpo,polyepo,minority}.sh`.
All knobs are env vars (`CS224R_EVAL_*`). The fork SHA
(`chicken602/maxrl@33873ec9`) and Modal image are pinned at
`main-verl/infra/modal_image.py`. A full panel costs ≈ 1 GPU-hour per arm on
B200:1.
