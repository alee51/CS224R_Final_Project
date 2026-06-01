# Pre-relaunch change list

Final scope for the next Stage 8 launch. Reconciles the original draft against:
(a) the Ifdita Slack convo recorded in `main/docs/timeline.md` "Tweak list before re-launch" + "Ruled out",
(b) an independent audit of `main-verl/` against the Poly-EPO paper (arXiv:2604.17654),
(c) what already landed in the `chicken602/maxrl@cs224r-patches` fork.

## At a glance

**Landing before relaunch (10 items):**

1. **Force final-step checkpoint.** Small commit on the maxrl fork: in `_save_checkpoint`, if `global_steps == total_training_steps`, force a save regardless of `save_freq` and tag it permanent. Lets us use verl's 1-epoch flag (~397 steps) safely.
2. **Per-rollout detail logging.** For each rollout, persist: `prompt_id`, `parsed_answer`, `reward`, `cluster_id`, `finish_reason`. Then `|U_correct|`, distinct-real-cluster counts, length-clip breakdown, and anything else are reconstructable offline. Replaces the old `|U_correct|`-as-a-W&B-metric plan; W&B aggregate metrics are fine but the raw join is what we actually need.
3. **`finish_reason` → W&B.** Aggregate counts of `length` vs `eos` vs `stop` (Stage 7 punt). Separable from #2 because W&B aggregates are the cheap online signal; #2 is the offline backstop.
4. **Loss aggregation: `token-mean` → Dr.GRPO `T_max`.** Paper Appendix §A explicit: "we do not normalize by the individual generation lengths … we set `T_i = T_max`". Set `actor.loss_agg_mode: seq-mean-token-sum-norm` (verify exact verl knob name against `verl/trainer/ppo/core_algos.py:agg_loss` in the pinned fork).
5. **DAPO asymmetric clip: 0.20 / 0.28.** Paper Table 1: `ε_low = 0.20`, `ε_high = 0.28`. Set `actor.clip_ratio_low: 0.20`, `actor.clip_ratio_high: 0.28` in all three production yamls.
6. **`ppo_mini_batch_size: 32 → 64`.** Paper Table 1: minibatch 64. Verified safe — `ppo_micro_batch_size_per_gpu: 4` is the actual memory knob; verl just doubles grad-accumulation (8 → 16 per GPU). No VRAM change, total fwd/bwd work per train step unchanged.
7. **`norm_adv_by_std_in_grpo: true → false` on all three configs.** Paper Appendix §A: "we omit the standard deviation normalization term originally used in [GRPO]" — applies to **both** the GRPO baseline and Poly-EPO (Dr.GRPO-style throughout). Set-arm kernels already silently ignore the flag; GRPO config currently sets `true` and verl's GRPO path actually applies it, so the GRPO baseline today diverges from the paper. Flip all three to `false`.
8. **Host-clock verification post-launch.** abao 1.54× slowdown was host virtualization, not algorithm. Run the nvidia-smi probe after each container is up; kill + relaunch if any GPU shows `clocks.sm < 0.8 × max` with `throttle_reasons.active = 0x0`, or PCIe IDs span >1 domain.
9. **GRPO W&B parity.** Populate `train/pass_at_8`, `train/prompts_unlocked`, `train/fraction_filtered` on the GRPO advantage path so the three arms share W&B panels. Stage 7 forwarding patch is arm-agnostic — only the writer side is missing.
10. **Pin Table 1 defaults explicitly in yaml.** Don't rely on verl defaults for `actor.entropy_coeff: 0.0` and `actor_rollout_ref.rollout.temperature: 1.0` — verl's defaults have drifted across versions (`entropy_coeff` was 0.001 in older builds). Pin in all three production yamls. Other Table 1 values already present: `max_prompt_length: 1024`, `max_response_length: 4096`, `train_batch_size: 128`, `n_rollouts: 8`, `lr: 1e-6`.

**Already landed (do not re-do):**

- **Cluster-100 exclusion from poly-EPO diversity numerator.** `_poly_epo_subset_score` at `train/objective_poly_epo.py:54` filters `DEGENERATE_CLUSTER_ID`. Paper §A.1 compliant.
- **Permanent-ckpt logic.** Lives in `chicken602/maxrl@cs224r-patches` HEAD (`ce8f6740…`, pinned in `infra/modal_image.py:28`). Adds `permanent_ckpt_freq` knob and temp-ckpt pruning. The deleted `maxrl_permanent_ckpt.patch` file is obsolete.

**Explicitly NOT changing (decisions on file):**

| Item | Why not |
|---|---|
| Add per-prompt std-norm to set-arm kernel | Paper Appendix §A omits it; mentor confirms her setup also omits. Was the original draft's #1 "blocker" — overruled by both sources. |
| `math_verify` (SymPy) reward enrichment | Mentor-confirmed strict `math.py` (Hendrycks) is intentional. |
| LR change off 1e-6 | Mentor-confirmed; pre-milestone 3e-6 showed only a weak signal. |
| `use_dynamic_bsz` mode | Not needed; current static micro-batching is working. |
| Bump `ppo_micro_batch_size_per_gpu` above 4 | Real VRAM knob; don't touch without a profiling reason. |

---

## Details

### 1. Force final-step checkpoint

**Why.** Switching to verl's 1-epoch flag (`trainer.total_epochs: 1`) computes a true step count of ~397 from `train_dataloader_len`, but `save_freq: 10` then makes the last save land at step 390 — the actually-trained model is lost. The fork's current `_save_checkpoint` (see permanent_ckpt commit `ce8f6740…`) does not have a final-step branch.

**Fix.** Add a commit to `cs224r-patches`:

```python
# inside _save_checkpoint, before the save call
is_final_step = (self.global_steps == self.total_training_steps)
if is_final_step:
    is_permanent = True  # never prune the final ckpt
```

Then guard the temp-ckpt pruning at the bottom of the function so `is_final_step` always keeps the ckpt. Bump `MAXRL_BRANCH_COMMIT` in `infra/modal_image.py:28`.

**Config.** In each `*_train_4b_1epoch.yaml`:
- Remove `trainer.total_training_steps: 400`.
- Set `trainer.total_epochs: 1` (confirm canonical key against the pinned verl's `ppo_trainer.yaml`).

If for any reason we keep `total_training_steps`, set it to **397**, not 400, so the final step lands within the epoch.

### 2. Per-rollout detail logging

**Why.** Online aggregate metrics (`distinct_clusters_mean` etc.) under-specify everything that matters for the diversity story. The mentor explicitly asked for diversity logging, and the paper's `|U_correct|` curve is only one of several things we want offline — others include "do correct rollouts cluster differently from incorrect", length-vs-cluster joins, per-cluster reward distributions, etc. All of these are reconstructable from `{prompt_id, parsed_answer, reward, cluster_id, finish_reason}`.

**Schema (per rollout, one row per `prompt × rollout_idx × step`):**

| field | source |
|---|---|
| `global_step` | trainer |
| `prompt_id` | `problem_ids` already in `_build_step_metrics` |
| `rollout_idx` | 0..7 |
| `parsed_answer` | judge parse (`judge/parse.py`) — string after `\boxed{}` extraction |
| `reward` | `rewards_grouped[p, k]` |
| `cluster_id` | `cluster_ids[p, k]` (post-normalization, so `-1` = degenerate) |
| `finish_reason` | vLLM rollout output (see #3) |
| `response_length` | already available from generation output |

**Sink.** Persist to a Modal volume as JSONL or parquet, partitioned by step. Avoid W&B as the primary sink (130k rows/step × 397 steps would blow up artifact limits). W&B can carry a small artifact pointer per N steps if useful.

**Plumbing.**
1. `train/clusters_judge.py:295-306` returns aggregate `diagnostics`. Add `cluster_ids` (already a tensor) and `parsed_answers` (already in the judge response) to the returned dict.
2. In `_build_step_metrics` (`train/objective_minority.py:53-84`), join with `rewards_grouped`, `problem_ids`, and the rollout outputs from `batch.meta_info`. Write per-rollout rows to the Modal volume. Pop non-scalar fields before forwarding to W&B.
3. Same hook on the GRPO path (no cluster, but log `cluster_id = null` so the schema is uniform — makes cross-arm joins trivial).

**Resume note.** This sidesteps the original concern about state not surviving resume. We're training 1 epoch, so resume should not be needed; but if it happens, the JSONL on the Modal volume is append-only and survives.

### 3. `finish_reason` → W&B

Aggregate counters per step: `train/finish_length`, `train/finish_eos`, `train/finish_stop`. Cheap online signal that disambiguates `response_length/clip_ratio ≈ 5–6%` between "model rambled past 4096" vs "model emitted EOS at 4096".

`finish_reason` comes from vLLM rollout output. In the verl rollout adapter (`verl/workers/rollout/vllm_rollout/...` in the pinned fork), the reason is on each output sequence. Add counts to the `cs224r_metrics` dict in the same code path as `pass_at_8`.

### 4. Loss aggregation: `token-mean` → Dr.GRPO `T_max`

**Paper Appendix §A:** "for Poly-EPO, we do not normalize by the individual generation lengths. Instead, following the Dr.GRPO implementation in Verl, we set `T_i = T_max`, where T_max is the maximum response length."

**Current.** `actor.loss_agg_mode: token-mean` in all three `*_train_4b_1epoch.yaml`. Verl: `masked_mean(loss, mask)` over the batch's valid tokens — long sequences contribute proportionally more weight.

**Change.** Set `actor.loss_agg_mode: seq-mean-token-sum-norm` (verl's Dr.GRPO `T_max` mode). Per-sequence sum / `T_max`, then mean over sequences in the batch. **Verify the exact knob name** against `verl/trainer/ppo/core_algos.py:agg_loss` in the pinned fork before flipping — verl has at least three loss-agg modes and the names drift across versions.

**Applies to all three arms** (GRPO included — paper baseline uses Dr.GRPO-style `T_max` for the comparison to be apples-to-apples).

### 5. DAPO asymmetric clip 0.20 / 0.28

Paper Table 1: `ε_low = 0.20`, `ε_high = 0.28`. In all three production yamls:

```yaml
actor:
  clip_ratio_low: 0.20
  clip_ratio_high: 0.28
```

Currently no override → verl default `clip_ratio: 0.2` symmetric. `actor/pg_clipfrac ≈ 0.0006` on the failed runs means symmetric clip isn't engaging today, but post-#4 gradients change and asymmetric clip is a paper-canonical knob worth landing while we're touching the yamls.

### 6. `ppo_mini_batch_size: 32 → 64`

Paper Table 1: minibatch 64. Confirmed safe:
- `ppo_micro_batch_size_per_gpu: 4` is the memory-bounding knob; unchanged.
- Verl auto-computes `gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu` → 8 → 16 per GPU per optimizer update.
- `train_batch_size: 128` stays, so optimizer-updates-per-train-step halves (4 → 2). Total fwd/bwd work per train step unchanged.
- Verl config-load checks all pass: `train_batch_size (128) ≥ mini (64)`, normalized mini (`64 · n_rollouts / world_size = 128`) divisible by `micro_per_gpu (4)`.

Behavioral effect: fewer, larger Adam updates per train step → lower gradient noise, closer to the paper's optimization regime.

### 7. `norm_adv_by_std_in_grpo: true → false` on all three configs

Paper Appendix §A: *"Note that we omit the standard deviation normalization term originally used in [GRPO] … consistent with the Dr.GRPO implementation."* Applies to **both** the GRPO baseline and Poly-EPO. The "originally used in" is past-tense — describing stock GRPO, not the paper's GRPO run.

- **Set arms** (`minority_cot_*.yaml`, `poly_epo_cot_*.yaml`): flip to `false`. Custom kernels already silently ignore the flag; this is a yaml-honesty fix with no runtime change.
- **GRPO** (`grpo_*.yaml`): flip to `false`. Verl's GRPO path actually applies the flag, so today's GRPO baseline diverges from the paper. **This will change GRPO training behavior** — advantages will be larger (no per-prompt /std denominator), and `actor/grad_norm` should rise. Expect the GRPO curve to be different from the prior run; that's the point — it's the paper-faithful version.

### 8. Host-clock verification post-launch

The 1.54× slowdown on poly_epo_cot (abao) was entirely host:
- 3 of 4 GPUs pinned at exactly 1155 MHz (B200 base); GPU 0 at boost.
- `clocks_throttle_reasons.active = 0x0` (no recognized throttle).
- PCIe bus IDs spanning 2 domains (`00000002:`, `00000003:`) → IOMMU-passthrough virtualization, not bare-metal HGX.

**Probe (run after each container is up):**

```bash
modal container exec --no-pty <task-id> -- bash -c \
  "nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv && \
   nvidia-smi -q | grep '^GPU 0000' | head -8"
```

**Kill criteria:** any GPU with `clocks.sm < 0.8 × clocks.max.sm` and `throttle_reasons.active = 0x0`; **or** PCIe IDs spanning >1 domain. `modal app stop` + relaunch with same `JUDGE_BASE_URL` + `WANDB_TAGS` resumes from latest ckpt (cheap with `save_freq: 10`).

### 9. GRPO W&B parity

`_build_step_metrics` in `objective_minority.py` populates `batch.meta_info["cs224r_metrics"]` only on the set-arm path. The Stage 7 forwarding patch is arm-agnostic; the gap is the writer side.

In the GRPO advantage/outcome path (`compute_grpo_outcome_advantage`), populate:
- `train/pass_at_8` — compute the same way.
- `train/prompts_unlocked` — prompts with ≥1 correct rollout.
- `train/fraction_filtered` — zero-advantage / all-same-reward groups GRPO drops.

Leave `train/judge_*`, `train/distinct_clusters_*`, `train/degenerate_rollouts` unset on GRPO (no judge).

### 10. Pin Table 1 defaults explicitly

Verl defaults have drifted across versions — `actor.entropy_coeff` was `0.001` in older builds, `0.0` in newer; `rollout.temperature` defaults to `1.0` but isn't guaranteed. Pin both in all three production yamls to remove ambiguity:

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.0      # Table 1
  rollout:
    temperature: 1.0        # Table 1
    top_p: 1.0              # Table 1
    top_k: -1               # Table 1
```

Already-pinned Table 1 values: `data.max_prompt_length: 1024`, `data.max_response_length: 4096`, `data.train_batch_size: 128`, `actor_rollout_ref.rollout.n: 8`, `actor.optim.lr: 1e-6`. Verify each before launch.

---

## Audit findings deferred

These came out of the paper audit but aren't going into this relaunch:

- **Training steps 850 vs ~397.** Paper runs 2 epochs (Table 1); we're doing 1. Budget-constrained. Note for the writeup: our curves should be read at the 1-epoch mark, not directly compared to the paper's 850-step terminal numbers.
- **4× H200 vs 4× A100.** Hardware difference, not algorithmic. Affects wall-clock and possibly numerical precision but not training dynamics in any first-order way.

---

## Sources

- Paper: arXiv:2604.17654 (Poly-EPO), Table 1 (hyperparams), §A (Dr.GRPO `T_max`, std-norm omission), §A.1 (cluster 100 carve-out).
- Audit agent run, 2026-05-31 — verified subset score, marginal computation, cluster-100 filter, KL coef, reward; surfaced loss-agg + clip + mini-batch discrepancies.
- `main/docs/timeline.md` 2026-05-31: bring-up entry (permanent_ckpt status, W&B metric forwarding), Stage 8 diagnostic ("Tweak list before re-launch", "Ruled out" — incl. std-norm).
- Verl mechanics audit, 2026-05-31 — mini-batch / micro-batch / OOM check (`dp_actor.py:403,419,426`, `fsdp_workers.py:156-166`, `ray_trainer.py:550-554` in pinned fork).
- Fork HEAD `ce8f6740a2b81f4e7bf8685a5a326329198a1df6` — permanent_ckpt logic landed.
