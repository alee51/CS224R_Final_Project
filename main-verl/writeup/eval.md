# Eval spec (LOCKED — 2026-06-02)

Held-out evaluation pipeline for **4 arms** (3 trained step-400 ckpts + 1 base
model) on **6 datasets**.

- Probe: `main-verl/eval/run_eval.py`
- Implementation / run plan: `main-verl/writeup/eval_build.md`
- Operational state (Modal accounts, ckpt paths, budgets, eval-JSON inventory):
  `main-verl/writeup/MODAL_STATUS.md`

This doc defines **what we measure**. It does not restate run sequencing,
account assignments, or budget — those live in the build doc.

## 1. Arms (4, LOCKED)

| arm | ckpt | notes |
|---|---|---|
| **Base** | `Qwen/Qwen3-4B-Base` (HF) | added as a 4th arm to give a floor + free baseline pass@k + free difficulty-stratification bucketing |
| **GRPO** | step 400 | standard RL baseline; no judge during training |
| **Minority-CoT** | step 400 | reward = rarest cluster |
| **Poly-EPO-CoT** | step 400 | reward = mean × distinct-cluster-count over subsets |

## 2. Dataset panel (LOCKED — 6 datasets, 720 problems)

| dataset | n | type | parquet |
|---|---|---|---|
| `aime25` | 30 | hard OOD | `main-verl/data/aime_val.parquet` |
| `aime26` | 30 | hard OOD | `main-verl/data/aime26.parquet` |
| `hmmt_feb25` | 30 | hard OOD | `main-verl/data/hmmt_feb25.parquet` |
| `hmmt_nov25` | 30 | hard OOD | `main-verl/data/hmmt_nov25.parquet` |
| `beyondaime` | 100 | hard OOD | `main-verl/data/beyondaime.parquet` |
| `math500` | 500 | easy OOD | `main-verl/data/math500.parquet` |

Total: **720 problems**. Matches Poly-EPO Figure 1 panel **minus Minerva, plus MATH-500**.

**Dropped: Minerva.** 272-problem dataset audit (2026-06-02) found 42% decimal
answers + 25 grader-incompatible answers (2 arcsin variants, 23 scientific-
notation formats) on top of strict-decimal grading. Estimated 5–15% downward
bias under `is_equiv`. Re-add later only with a normalization wrapper for
decimals + trig variants + sci notation.

**Excluded as in-distribution:** Polaris-val, DAPO-slice. Polaris was the
training corpus; DAPO was used during early development. Neither belongs in
the OOD headline.

## 3. Prompt format (unchanged from training)

The eval probe loads the parquet `prompt` field, renders it through the model's
own `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True)`, and
passes the rendered string to vLLM. The user-turn content is:

```
<problem text>
Please reason step by step, and put your final answer within \boxed{}.
```

The `\boxed{}` suffix is baked into the parquet at data-prep time; the eval
probe does not add it a second time.

## 4. Scorer (LOCKED)

`verl.utils.reward_score.math.compute_score` — `last_boxed_only_string` →
`remove_boxed` → Hendrycks `is_equiv` against `reward_model.ground_truth`.
**Identical to the training reward.** NOT `math_dapo.compute_score(strict_box_verify=True)`.

The probe also saves the raw `remove_boxed(...)` output as `preds[i]` for
offline diversity analysis.

## 5. Sampling (LOCKED)

| field | value | rationale |
|---|---|---|
| `temperature` | `1.0` | matches training |
| `top_p` | `1.0` | matches training |
| `top_k` | `-1` (vLLM default) | matches training |
| `max_tokens` | `4096` | matches training (`max_response_length`) |
| `n` (rollouts/prompt) | **64** | training was n=8; eval extends n purely to study pass@k at k up to 64. **n=64 is NOT trying to match the training distribution.** |
| `logprobs` | **top-20 per token** | enables per-rollout token entropy (Tier 1) + KL from base (Tier 2). Top-5 would suffice for entropy alone; top-20 is the hedge for KL accuracy (minimizes tail bias). Adds ~2 GB total JSON across all 24 GEN runs. |

Pass@k ladder caps at **k=64**.

## 6. Metrics (LOCKED)

### 6.1 — Tier 1: free analysis on saved GEN rollouts

All computed offline from saved JSON `{rewards, parsed_answer, rollouts, n_correct, logprobs}`. No additional GPU or API spend.

| metric | definition | implementation |
|---|---|---|
| **pass@k** for k ∈ {1, 2, 4, 8, 16, 32, 64} | `1 − C(n−c, k) / C(n, k)`, unbiased, averaged over problems | `run_eval.py:237-251` |
| **majority@k** for same k ladder | take first k rollouts; modal `parsed_answer`; correct iff modal answer ≡ gold | `analysis/coverage.py` (extend) |
| **AUC@k** | `trapz(pass_at_k_vector, ks)` — single per-cell scalar | new, trivial |
| **coverage@k** | per-problem distinct correct `parsed_answer` strings, averaged | `analysis/coverage.py:24` |
| **distinct_answers@k** | per-problem distinct non-empty `parsed_answer` in first k | `analysis/coverage.py:41` |
| **entropy@k** | Shannon entropy (bits) over `parsed_answer` distribution in first k | `analysis/coverage.py:50` |
| **diff@k split by solved/unsolved** | distinct-answers metric, grouped by `n_correct > 0` vs `= 0` | new — load-bearing for minority story |
| **Potential@k** | fraction of failed problems solvable within k extra rollouts | new, trivial from `n_correct` |
| **Self-BLEU + distinct-n-gram** | rollout TEXT diversity (catches "different answers, identical reasoning") | new — sacrebleu, ~30 min |
| **reflective-action frequency** | counts of "wait", "however", "verify", "because" in rollout text | new — regex, trivial |
| **per-rollout token entropy split by correct/incorrect** | per-rollout mean token entropy from saved logprobs, grouped by reward | new — load-bearing for minority story |

### 6.2 — Tier 2: additional pass on saved rollouts

| metric | definition | cost |
|---|---|---|
| **KL(π_arm ‖ π_base) per token** | base-model teacher-force over saved policy rollouts; compute per-token KL using policy's saved top-20 logprobs + base's logprobs | 3 trained arms × 6 datasets = 18 forward passes, ~10 GPU-hr |
| **difficulty-stratified pass@k** | bucket problems by base-arm pass@k (since Base IS an arm, this is free) | analysis-only, no extra compute |

### 6.3 — Training-time metrics

| metric | source | status |
|---|---|---|
| `train/pass_at_8` trajectory | W&B | logged for all 3 trained arms |
| `train/fraction_filtered`, `train/prompts_unlocked` | W&B | logged |
| `actor/entropy`, `ppo_kl`, `critic/rewards/mean` | W&B (verl default) | logged |
| `train/distinct_clusters_mean`, `train/degenerate_rollouts` | W&B | set arms only |
| `train/judge_parse_ok_rate`, `train/judge_overflow_skipped` | W&B | set arms only |
| **`|U_correct|` trajectory** | `analysis/u_correct.py` on training-time per-rollout JSONLs | DONE for minority + poly_epo. **GRPO requires $15 judge pass over GRPO training rollouts** (Phase 5) to participate on the same axes — without it, GRPO is stuck at trivially `\|U_correct\|=1`. **LOCKED IN.** |
| Cluster-correctness by rank (rarest-correct rate by cluster size) | `analysis/cluster_correctness.py` | DONE for minority (35% rarest-correct, inverted from "rarest=correct" hypothesis). Refresh for poly_epo for parity. |
| Token-entropy gap (80–200× minority vs GRPO) | W&B `actor/entropy` | already documented |

### 6.4 — Judge-based eval-time

**SKIPPED for v1.** No eval-time `|U_correct|` parity exists for any arm; user
policy is all-or-nothing on paid metrics. Revisit only if Tier 1+2 leave the
eval-time diversity question unresolved.

## 7. Procedure (per arm × dataset)

1. **Merge** FSDP → HF (trained arms only): `python scripts/model_merger.py merge --backend fsdp --local_dir <ckpt> --target_dir /tmp/merged_hf`. Base arm skips this; loads HF directly.
2. **Load vLLM** with `logprobs=20`: `LLM(model=..., tensor_parallel_size=1, gpu_memory_utilization=0.85, max_model_len=5120, enforce_eager=True, dtype=bfloat16)` on `B200:1`.
3. **Generate** n=64 rollouts per prompt via chat template + `SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=4096, n=64, logprobs=20)`.
4. **Score** each rollout with `math.compute_score`; save `rewards`, `preds`, `rollouts`, **`logprobs`** per prompt.
5. **Compute pass@k** (incremental per-dataset dump to `/vol/probes/eval_4b/<label>_<dataset>.json`).

See `main-verl/writeup/eval_build.md` for which accounts run which arms.

## 8. Reproducibility & verification

- **Saved rollout JSON is the canonical artifact.** vLLM sampling is
  non-deterministic; do NOT expect bit-identical reruns. Rescore is
  deterministic from the saved JSONs.
- Fork SHA: `chicken602/maxrl@33873ec9`. Modal image pinned at `main-verl/infra/modal_image.py`.
- **Pre-headline grader sanity check is MANDATORY** before reporting any
  pass@k number (user has been burned by 2 silent-grader bugs):
  1. n_correct distribution histogram across prompts
  2. 2–3 sample `(problem, gt, parsed_pred, reward)` tuples
  3. Rescore on a held-out 20-problem subset to confirm grader stability
  4. **Belt-and-suspenders grader tripwire:** rescore the same 20 problems with `math_dapo.compute_score(strict_box_verify=True)` for comparison. The two graders agreeing on >90% of problems confirms our grader hasn't been silently swapped. Systematic disagreement = investigate before publishing.
- JSON size: per (arm × dataset) ≈ 150–200 MB with rollout text + top-20
  logprobs. Total ≈ 5–6 GB across 24 GEN runs. Strip rollout text only before
  any external upload, not at write time.
