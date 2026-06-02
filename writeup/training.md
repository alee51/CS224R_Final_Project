# Training spec (auditable)

All numbers below are pulled directly from the production YAMLs or the verl
fork pinned at `chicken602/maxrl@33873ec9`. Where a number differs between
arms it is called out; otherwise the value is identical across arms.

## 1. Arms (one sentence + objective)

Three RL-from-verifiable-rewards arms, each fine-tuning the same Qwen3-4B-Base
checkpoint with the same prompt distribution, optimizer, batch, and reward
function. They differ only in how the per-rollout advantage is computed from
the group of `N=8` rollouts per prompt.

Let `r_i ∈ {0,1}` be the reward of rollout `i`, `c_i` be its judge-assigned
CoT cluster id, `G` denote a size-4 subset of the 8 rollouts (70 such subsets
per prompt), and `S_i` denote the 35 subsets containing rollout `i`.

- **GRPO** (`adv_estimator: grpo`, verl built-in):
  `A_i = r_i − (1/N) Σ_j r_j`. (Reference: paper-faithful GRPO with
  `norm_adv_by_std_in_grpo: false`; see
  `main-verl/configs/grpo_train_4b_1epoch.yaml:57`.)

- **Minority-CoT** (`adv_estimator: minority_cot`, kernel
  `main-verl/train/objective_minority.py:509`):
  `f_min(G) = mean(r_i : i ∈ rarest cluster of G)` (random tiebreak among
  rarest clusters); `A_i = mean_{G ∈ S_i} f_min(G) − mean_{G} f_min(G)`.

- **Poly-EPO-CoT** (`adv_estimator: poly_epo_cot`, kernel
  `main-verl/train/objective_poly_epo.py:43`):
  `f_poly(G) = (mean_{i∈G} r_i) · |distinct non-degenerate clusters in G| / |G|`;
  same marginal-over-subsets aggregation as Minority-CoT.

Both set-arm advantages route through the shared marginal kernel
`set_based_marginal_advantages` at
`main-verl/train/objective_minority.py:531`.

## 2. Model and data

| field | value | source |
|---|---|---|
| base model | `Qwen/Qwen3-4B-Base` | `grpo_train_4b_1epoch.yaml:24` |
| judge model (set arms) | `Qwen/Qwen3-4B-Instruct-2507` | `minority_cot_train_4b_1epoch.yaml:62` |
| train dataset | `polaris_train.parquet` (Polaris-51K filtered, 51,139 prompts) | `grpo_train_4b_1epoch.yaml:13`; corpus size cited in `main/docs/STANDARDS.md:54` |
| in-training validation | `polaris_val.parquet` (1024) + `aime_val.parquet` (30) | `grpo_train_4b_1epoch.yaml:15-16` |
| training length | 1 epoch ≈ 400 steps (forced) | `grpo_train_4b_1epoch.yaml:65` (`total_training_steps: 400`) |

## 3. Optimizer and PPO knobs (identical across arms)

Pulled from the three production YAMLs.

| knob | value | source line (all 3 YAMLs) |
|---|---|---|
| Optimizer | AdamW (verl default), no weight decay, no LR warmup | `actor.optim.lr` field |
| Learning rate | `3.0e-6` | `:38` (GRPO), `:39` (minority/poly_epo) |
| `train_batch_size` (prompts) | `128` | `:17` |
| `rollout.n` (rollouts/prompt) | `8` → 1024 rollouts/step | `:41` |
| `ppo_mini_batch_size` | `64` | `:28` (GRPO), `:29` (others) |
| `ppo_micro_batch_size_per_gpu` | `4` | `:36` (GRPO), `:37` (others) |
| `ppo_epochs` | `1` | `:29` (GRPO), `:30` (others) |
| `clip_ratio_low` / `clip_ratio_high` | `0.20` / `0.28` (DAPO asymmetric) | `:32-33` (GRPO), `:33-34` (others) |
| `use_kl_loss` | `false` | `:30` (GRPO), `:31` (others) |
| `use_kl_in_reward` | `false` | `:56` (GRPO), `:57` (others) |
| `entropy_coeff` | `0.0` | `:34` (GRPO), `:35` (others) |
| `norm_adv_by_std_in_grpo` | `false` (paper-faithful GRPO) | `:57` (GRPO), `:58` (others) |
| `max_prompt_length` | `1024` | `:18` (GRPO), `:19` (others) |
| `max_response_length` | `4096` | `:19` (GRPO), `:20` (others) |
| Rollout sampling | `temperature=1.0`, `top_p=1.0`, `top_k=-1` | `:42-44` (GRPO), `:43-45` (others) |
| Hardware | `B200:4` (NVLink, FSDP colocated rollout) | `:64,86` |

**Loss aggregation differs by arm** (the one knob that is intentionally
asymmetric):

| arm | `loss_agg_mode` | rationale | source |
|---|---|---|---|
| GRPO | `seq-mean-token-mean` | Paper-faithful GRPO uses `T_i=|y_i|` per-rollout token mean. | `grpo_train_4b_1epoch.yaml:31` |
| Minority-CoT | `seq-mean-token-sum-norm` | Closest verl knob to Dr.GRPO `T_max` normalization expected by the Poly-EPO paper. | `minority_cot_train_4b_1epoch.yaml:32` |
| Poly-EPO-CoT | `seq-mean-token-sum-norm` | Same. | `poly_epo_cot_train_4b_1epoch.yaml:32` |

## 4. Reward function

A **single grader** is applied at both training time and eval time:

> `verl.utils.reward_score.math.compute_score(rollout_text, ground_truth)` —
> extracts the last `\boxed{...}` substring (`last_boxed_only_string` →
> `remove_boxed`) and applies Hendrycks-style latex equivalence (`is_equiv`,
> which normalizes `\frac{1}{2}` ↔ `0.5`, strips `\text{...}` units, etc.).
> Returns `1.0` on equivalence, `0.0` otherwise.

This routing is the single math branch in
`verl/utils/reward_score/__init__.py:default_compute_score` (pinned fork). The
training-time grader and the eval-time grader (`main-verl/eval/run_eval.py:206`)
import the same function. The ground-truth string in
`reward_model.ground_truth` is passed through unmodified; all normalization is
done inside `is_equiv`.

## 5. Judge model and prompt (set arms only)

| field | value | source |
|---|---|---|
| model | `Qwen/Qwen3-4B-Instruct-2507` | `minority_cot_train_4b_1epoch.yaml:62` |
| serving | Modal-deployed vLLM, 2 containers, B200×2 each, `enforce_eager: false` | `main/docs/STANDARDS.md:95` |
| HTTP batch size | `judge_http_batch_size: 64` | `minority_cot_train_4b_1epoch.yaml:64` |
| concurrency | `judge_concurrency: 2` | `minority_cot_train_4b_1epoch.yaml:66` |
| prompt | Poly-EPO paper §A.1 instruction block verbatim + both paper few-shot examples | `main-verl/judge/prompts/poly_epo_a1.md` |
| cluster ID space | integers `≥ 0` for parsable judge output; `-1` (`DEGENERATE_CLUSTER_ID`, paper's `cluster 100`) for unparsable or degenerate (gibberish / code-only / no-answer) rollouts | `main-verl/judge/prompts/poly_epo_a1.md:34-37` and `main-verl/train/objective_minority.py:43` |

**Degenerate-rollout policy.** A `-1` cluster id is treated as an ordinary
cluster by both kernels: it can be selected as "rarest" by Minority-CoT
(`main-verl/train/objective_minority.py:521-528`). Poly-EPO removes `-1` from
the diversity numerator only, so a degenerate rollout still contributes to the
reward mean but not to `d(G)`
(`main-verl/train/objective_poly_epo.py:43-55`). About 25% of rollouts are
degenerate in production (W&B `train/degenerate_rollouts ≈ 250-280 / 1024`).

## 6. Per-rollout JSONL schema (training-time)

Every step writes one line per `(prompt × rollout)` to
`/vol/per_rollout/<wandb_run_id>/step_<n>.jsonl` (writer:
`main-verl/train/objective_minority.py:329-369`). Schema:

```
{ "global_step":     int,
  "prompt_id":       str,
  "rollout_idx":     int (0..7),
  "parsed_answer":   str (contents of last \boxed{}, empty if absent),
  "reward":          float (0.0 or 1.0; same grader as §4),
  "cluster_id":      int (-1 degenerate, ≥0 judge cluster; null on GRPO),
  "finish_reason":   "stop" | "eos" | "length",
  "response_length": int (unmasked response tokens) }
```

`parsed_answer` is symmetric across arms: the set arms get it from the judge
decode path; GRPO recovers it offline via a tokenizer stashed on the
DataProto (`main-verl/train/objective_minority.py:372-408`).

## 7. Quantities reconstructible offline from the JSONL

The per-rollout JSONL contains everything needed to compute, post-hoc, any
metric that depends only on per-rollout scalars. In particular:

- **`|U_correct|`** (writeup metric, see `eval.md`): per prompt, the number
  of distinct judge CoT cluster IDs (degenerate excluded) among rollouts
  with `reward = 1`, averaged over prompts. Free on set-arm JSONLs (judge
  was already running). For GRPO, requires a separate post-hoc judge pass
  on the saved rollouts.
- **Cluster-correctness alignment.** `P(rarest cluster = correct cluster | prompt)`
  is a join on `cluster_id` and `reward`. This is what
  `main-verl/eval/results/minority_diagnostic.md` uses.
- **Coverage and entropy trajectories.** All `coverage@k`, `entropy@k`, and
  `distinct_answers@k` quantities defined in `eval.md` have direct training-time
  analogs computable from the JSONL.

Run-id directories that landed under `unknown_run/` (Modal-retry containers
that lost `WANDB_RUN_ID`) should be merged with the proper `<wandb_run_id>/`
directory when stitching.
