# Main experiment plan

**Status:** working draft started 2026-05-23. Each section flagged with what's still TBD. Pilot postmortems are out of scope here — see `[LESSONS_FROM_PILOT.md](./LESSONS_FROM_PILOT.md)`.

---

## 1. Overview

**Goal:** Train Qwen3-1.7B with a set-based minority-voting objective on Polaris, and measure pass@k generalization to AIME-25/26, HMMT, and Beyond-AIME.

**Constraints:** $1,600 Modal credits confirmed (~640 hr on A100-80GB; GPU class flexible, see §7). Poster due 2026-06-03. Internal target: experiments done by 2026-05-31.

### Work split

Nancy code. Anastasia run monitoring, Emma evaluation maybe? 

### Doc layout

1. **Overview** — this section.
2. **Dataset** — Polaris sub-block selection, cleaning, eval splits, freeze policy.
3. **What we're testing** — training arms, objective math, hypothesis, success criteria.
4. **Evaluation** — pass@k harness, mid-training eval, figures we commit to.
5. **Codebase** — repo layout, vLLM/HF split, shared reward function, key design decisions.
6. **Operations** — Modal launch protocol, checkpointing, wandb, artifact handling, kill rules.
7. **Sizing & cost** — step time targets, batch/N/max_tokens, scoping probes, GPU-hr budget.

---

## 2. Dataset

**Source:** `[POLARIS-Project/Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)`. Only one version / one `train` split exists. Fields: `problem`, `answer`, `difficulty`. `difficulty` is 8 fractional bands (`1/8` easiest → `7/8` hardest), labeled by Deepseek-R1-distill-Qwen-7B pass rate.

**Train data (decided 2026-05-26):** **Polaris** (not DAPO-Math-17k) — mentor recommendation + `difficulty` bands for §2 sampling. Early DAPO-vs-Polaris comparison used **arm A** rollouts and overstated the gap; **arm C** hybrid on the same 800-problem manifest is ~1 pp below DAPO pilot pass@8 (33.1% vs 34.4%). See `[timeline.md](./timeline.md)` §2026-05-26 afternoon.

**Training sub-block size and difficulty band (decided 2026-05-26 late):** **16,000** rows, **stratified proportional** across difficulty bands `0/8`…`7/8` (mirrored-J shape at ~53k scale), seed **42**. Spec: [`polaris_preprocess_plan.md`](./polaris_preprocess_plan.md); script: `main/data/preprocess_polaris.py`.

**Cleaning (train freeze):** Polaris is already filtered from DeepScaleR + AReal-boba-Data. Apply only:

- drop rows where `problem` is empty or not a non-empty string
- drop rows where `answer` / gold is empty after strip

**Do not** filter to integer-only gold for the Polaris manifest — random full-gold n800 probe matched integer-stratified pass rates under arm C + `grade_parsed_answer` (see [`timeline.md`](./timeline.md) §2026-05-26 late night). Group A probes may still use integer gold for historical parity; that is not the train freeze policy.

**Still open (optional v2):** drop prompts where Qwen-1.7B gets 8/8 correct mid-training (Polaris dynamic recipe).

**Eval splits:** TBD — defer to §4 Evaluation. (Pilot frozen splits in `pre-milestone/pilot/data/` are a starting point but not load-bearing here.)

**Freeze policy:**

- **Train manifest (canonical):** `main/data/polaris_train.jsonl` + `polaris_train.meta.json` — 51,139 rows after prompt filter ([`data/README.md`](../data/README.md), [`decisions.md`](./decisions.md) §2026-05-27).
- **Full pool (source):** `main/data/source/polaris_train_full.jsonl` — from `preprocess_polaris.py`; input to `filter_polaris_train.py` only.
- Record provenance in each meta json: HF revision, seed, row counts, filters applied, materialization timestamp.
- Once frozen, **do not re-materialize** without writing a dated note in `main/docs/context.md` explaining why.
- Eval splits follow the same convention: one jsonl + meta.json per split.

## 3. What we're testing

**Hypothesis:** Training Qwen3-1.7B with a set-based minority-voting objective improves pass@k generalization on harder held-out reasoning evals compared to vanilla GRPO, by preventing collapse to a single output mode.

**Falsifies if:** minority-answer matches or underperforms GRPO on pass@k for k ∈ {1, 4, 16, 64} across the held-out evals in §4. (TBD: specific deltas / significance bar.)

### Arms


| Arm             | Status                       | Objective                                                  | Clustering substrate              | LLM judge required |
| --------------- | ---------------------------- | ---------------------------------------------------------- | --------------------------------- | ------------------ |
| GRPO            | must                         | per-trajectory advantage A_i = r_i − mean(r)               | none                              | no                 |
| Minority-answer | must                         | set-based minority voting, marginal per-rollout advantages | exact answer match                | no                 |
| Minority-CoT    | in scope; first cut if tight | same as minority-answer                                    | LLM-judged CoT clusters (in-loop) | yes (in-loop)      |
| Poly-EPO-answer | stretch                      | f_poly = mean(r) · diversity(G)                            | exact answer match                | no                 |


All arms share the same model (Qwen3-1.7B-Base), same training data, same N=8 rollouts/prompt, same reward, same eval suite. Only the advantage computation differs.

### Objective math (minority-answer / minority-CoT)

For each prompt, sample N=8 rollouts. For each of the C(8,4)=70 size-4 subsets G:

- `f(G) = r(minority(G))` where `minority(G)` is the cluster with lowest frequency in G (ties broken at random).
- Per-rollout marginal advantage A_i = (mean f(G) over the 35 sets containing rollout i) − (mean f(G) over all 70 sets).

Clustering substrate differs by arm (answer-hash vs. CoT). Poly-EPO-answer uses the same answer-hash substrate as minority-answer; only the subset score `f(G)` differs. Tiebreak settled in milestone: random pick (r=0.994 vs. averaging).

### Reward

**Locked (2026-05-26):** Rank-2 extraction (`extract_rank2`, arm C `hybrid_answer_boxed`) then **DeepScaleR / rLLM graders** — `grade_answer_mathd(parsed, gold) or grade_answer_sympy(parsed, gold)` (`grade_parsed_answer` in `main/train/reward.py`; vendored from [rLLM `math_utils/utils.py`](https://github.com/agentica-project/rllm/blob/main/rllm/rewards/math_utils/utils.py)). OOD eval remains Math-Verify (STANDARDS).  

### Success criteria

Primary outcome metric: **pass@k on held-out evals**, k ∈ {1, 4, 16, 64}. Comparison across arms on identical model, data, compute.

- **Pass criterion (strong):** minority-answer > GRPO on pass@4 and pass@16 on at least 2 of the held-out evals. TBD: how much margin counts as "real". 
- **Pass criterion (consolation):** minority-answer matches GRPO on pass@1 *and* improves cluster diversity on the same eval rollouts (less mode collapse, see "tokens to first branching" metrics used in Poly-EPO), even if pass@k is unchanged.
- **Failure mode:** minority-answer degrades pass@1 substantially → revisit hybrid objective (milestone §3 next-steps item).

### Open

- Significance bar for "real" improvement (bootstrap CI, paired t-test, raw delta).
- Minority-CoT judge: locally hosted vs. API (cost / latency / throughput tradeoff).
- Poly-EPO-answer: confirm `diversity(G)` matches Run 0 (`distinct answer-hash clusters in G / 4`) vs. paper's LLM-judged CoT clusters (we use answer-hash only for cost parity).
- Hybrid-objective fallback definition if pass@1 collapses.

## 4. Evaluation

Kept deliberately open — flesh out once arms are training and we know the noise floor.

**OOD evals we can use** (final mix TBD): AIME-25, AIME-26, HMMT (Nov / Feb 2025), Beyond-AIME, Polaris hard split. Optional: MATH-500, Minerva.

**Metrics to consider:**

- **pass@k** for k ∈ {1, 4, 16, 64} — primary.
- **Cluster diversity at eval** — distinct answer hashes per prompt, distinct CoT clusters per prompt. Already reported in milestone baseline table.
- **Cover@τ** — Anastasia's earlier exploration; revisit as a generalization metric if it tells a different story than pass@k.
- **Poly-EPO-style diversity diagnostics** — e.g. tokens-to-first-branching, output entropy. Pull from their paper.

### Open

- Answer-format cleanliness across the OOD evals — uncertain. Non-integer answers (LaTeX, symbolic, tuples) likely need a math-equivalence grader (Ifdita mentioned MathReward package); integer-only grading will silently under-report. Audit before locking the eval mix.
- Which metrics actually go in the poster / paper vs. supplementary. Borrow framing from Poly-EPO and adjacent set-RL papers.
- Bootstrap CI methodology for pass@k (how many resamples, paired or unpaired across arms).
- Eval at final checkpoint only vs. multiple checkpoints (best-on-eval vs. last vs. trajectory plot).

## 5. Codebase

Lightweight VeRL-flavored trainer; not the full VeRL stack. Nancy owns all `.py`. Target: small enough that one person holds the whole thing in their head. All code follows `[STANDARDS.md](./STANDARDS.md)` (reproducibility, wandb, Modal, checkpointing) — this section is just the trainer-specific architecture on top.

The binding constraint is **wall-clock, not $$.** Internal target is ~7 days from skeleton-complete to results across all arms. $1,600 on A100-80GB ≈ 640 GPU-hr ≈ 26 days sequential / ~6 days at 4-way parallel — and that's before judge cost for Minority-CoT / Poly-EPO-CoT and before any re-runs. **GPU class resolved 2026-05-26: H200** (Group B readout — H100 OOMs at `batch_size: 64` because `_completion_logprobs_hf` is one-shot; H200 fits with 25% VRAM headroom and is 25% cheaper / 34% faster per prompt). Trainer remains GPU-class-agnostic in code (A100 / H100 / H200 / B200, single-GPU or 2-GPU) — only Modal `gpu=` strings and `modal_price_per_sec` change between SKUs.

### Proposed

- **Rollout engine:** vLLM, in-process. Returns per-token logprobs so they can be reused as `old_logprobs` (no second HF forward over rollouts).
- **Training model:** HF transformers, separate from the vLLM engine. Per-step weight sync from HF → vLLM via vLLM's `update_weights` API (no engine reload).
- **No reference model.** KL coef = 0, following Poly-EPO. Drops a 1.7B frozen copy off the GPU and removes the ref-logprob forward pass. Revisit only if we observe collapse / reward hacking.
- **Repo layout:**

```
main/
  train/
    rollout.py     # vLLM engine + HF→vLLM weight sync + logprob capture
    reward.py
    objective.py   # grpo / minority-answer / minority-cot / poly-epo-answer advantages
    loss.py        # clipped surrogate, microbatched backward
    trainer.py     # main loop
  eval/
    passk.py
  judge/           # only loaded by minority-cot / poly-epo-cot arms
    client.py      # local-vLLM judge or API client (decided by §7 probe)
  infra/
    modal_app.py
  configs/
  data/            # frozen jsonl + meta.json per §2
  docs/
```

### Optimization knobs

Two layers: **architecture** decisions (load-bearing, commit at codebase-build time, affect what the code looks like) and **size/throughput** decisions (mostly config, dialed by §7 probes).

Aggressive optimization here is the only place we can buy back wall-clock before probes happen. Goal: get per-arm cost low enough that all 4 arms train *in parallel within the 7-day wall-clock window*, leaving slack for re-runs and the judge subsystem.

**Calibration: Poly-EPO mathematical-reasoning config** (their Table 1, at Qwen-3-4B on 4× H200) — what a real run of this objective looks like. Not values we copy directly; useful as the sanity check our scaled-down config benchmarks against.

| Param | Poly-EPO | Note for us (1.7B) |
| --- | --- | --- |
| N rollouts / set size n / K | 8 / 4 / 70 | Same — already locked |
| Max prompt / response | 1024 / 4096 | **4096 locked** (Group A; ~1.25% hit cap) |
| Prompts / batch / microbatch | 128 / 64 | Locked **64** on single H200† |
| LR / KL / clip low/high | 1e-6 / 0.0 / 0.20 / 0.28 (DAPO-asym) | Adopt KL=0 and asym clip; LR sweep if needed |
| Entropy / rollout temp | 0.0 / 1.0 | Adopt |
| Training steps | 799 | One epoch on filtered Polaris (51,139 rows / 64-prompt batch ≈ 799) |
| Codebase | VeRL via Tajwar et al MLRL fork | Read-and-lift, not import |

† Poly-EPO **128 prompts / batch 64** is on **4× H200** (4B, VeRL) — not our single-GPU collocated 1.7B stack. We lock **`train.batch_size: 64`** on one H200; bs=128 OOMs in `logprob_fwd` after rollout ([`decisions.md`](./decisions.md) §2026-05-26).

**Architecture (commit now):**

- **No reference model / KL=0** (also in Proposed above). ~25–35% VRAM freed, one fewer forward per step. `loss.py` has no KL term; `rollout.py` has no ref-logprob path.
- **Reuse vLLM rollout logprobs as `old_logprobs`.** Skip the separate HF forward over rollouts. ~25–30% step time saved. `rollout.py` requests and returns logprobs; `loss.py` consumes them directly.
- **Inner PPO epochs = 1.** REINFORCE-with-clip semantics. ~3× cheaper on the update phase vs textbook 4 inner epochs. Matches DAPO / Dr.GRPO / Poly-EPO's stack. Affects `trainer.py` loop shape.
- **Filter zero-advantage prompts before backward.** When all N rollouts get the same reward (GRPO) or all 70 sets get the same f (set-RL arms), the advantage is zero and the prompt contributes nothing to the gradient. Skip it. Particularly relevant for minority-answer: collapsed prompts (all 8 rollouts agree on one answer) produce no signal by construction. `objective.py` returns a mask; `loss.py` honors it.
- **Judge as a sidecar vLLM engine** (Minority-CoT, Poly-EPO CoT variant). Run Qwen-3-4B-Instruct via vLLM locally — same machine if VRAM allows, second GPU otherwise. `judge/client.py` is a thin wrapper that's swappable to API if probes show local hosting is infeasible. **Group A n800:** judge ≈ rollout wall-clock and ~$0.0014/call on H100 — CoT arms are ~2× inference GPU vs rollout-only arms, not blocked on $/latency; collocated train+policy+judge still unmeasured.
- **vLLM prefix caching ON.** Shared system-prompt prefix is reused across all N rollouts per problem. Free.
- **wandb logging** from `trainer.py`; project `cs224r-minority-voting`. Log step time, mean reward, loss, and advantage stats each training step — plus **training-dynamics panel** (§5 below) required for Poly-EPO Fig. 2–style curves.
- **Async rollout / train overlap** — stretch. vLLM rolls out batch t+1 while HF does backward on batch t. Up to ~30% wall-clock saved. Defer unless §7 step-probe shows it's the binding constraint; the code complexity isn't worth it otherwise.

### Training-time reporting (Poly-EPO Fig. 2 parity)

Poly-EPO reports **in-training** curves on the **training set** (Fig. 2), separate from held-out **pass@k** (Fig. 1). We need the same split: wandb during train; `eval/passk.py` only post-train (or rare offline eval jobs).

**Source:** [`pre-milestone/pilot/docs/analysis/0519_poly_epo_methodology.md`](../../pre-milestone/pilot/docs/analysis/0519_poly_epo_methodology.md) §6 — Ifdita Hasan Orney et al., Poly-EPO (May 2026).

| Poly-EPO Fig. 2 panel | Definition | Required for v1? | How we implement |
| --- | --- | --- | --- |
| **Right — training coverage** | Fraction of **training prompts** in the step with **≥1 correct** rollout (of N=8), using the **train reward** (mathd∨sympy on Rank-2 parse) | **Yes — all arms** | `trainer.py` wandb: `train/prompt_coverage` (= mean over batch of `max(reward_row) > 0`). **Not** the same as `train/mean_reward` (mean over all rollouts). |
| **Right — related** | Fraction with **mixed** correct/incorrect rollouts (minority / GRPO signal density) | **Yes — all arms** | `train/mixed_reward_rate` (= fraction of prompts with `0 < sum(rewards) < N`). Maps to probe-plan **C1**. |
| **Left — strategy diversity** | Mean **unique LM-judge reasoning clusters** among **correct** rollouts only | **Poly-EPO / Minority-CoT only** | In-loop judge (`judge/` + `poly_epo_a1.md`) each step: cluster correct rollouts, log `train/mean_unique_strategy_clusters_correct`. **GRPO baseline does not log this** — no judge in GRPO loop. |
| **Left — answer diversity (our analogue)** | Unique **answer-hash** clusters among correct rollouts | **Minority-answer / Poly-EPO-answer** when implemented | Cheap hash clusters on parsed answers; log `train/mean_unique_answer_clusters_correct`. Not identical to paper's CoT clusters — cite separately in writeup. |
| **Diagnostics (PLAN §5)** | `extract_path` distribution, parse failures | **Yes — all arms** | Per-step counts: `train/parse_ok_rate`, `train/extract_path_{hybrid,boxed,answer_line,none}` fractions. |
| **Length / collapse (C2)** | Mean completion tokens per step | **Yes — all arms** | `train/mean_completion_tokens`, optional p95. |
| **Set-arm advantages (C3)** | Distribution of marginal subset advantages | **Minority-* / Poly-EPO arms** | Log at step 100 (and every 100): histogram or percentiles of per-rollout marginal advantages — not meaningful for pure GRPO. |

**Retention policy:** These metrics are **wandb scalars only** unless we add an explicit decision to flush rollout jsonl to the volume (expensive). **Checkpoints do not store rollouts** — you cannot reconstruct Fig. 2 from `step_*.pt` alone. If we need paper-faithful strategy clusters on a GRPO run post hoc, run an **offline judge pass** on a saved checkpoint's eval rollouts, not from training checkpoints.

**Implementation status (2026-05-26):** **All-arm rows implemented** in `trainer.py` via `aggregate_train_step_wandb_metrics` (C1, C1b, C2). Still missing: C3 (set-arm advantages), C4/C4b (cluster diversity — judge / answer-hash). Core loop also logs `loss`, `mean_reward`, `fraction_filtered`, `n_kept`.

**What we explicitly do not emulate in-loop:** Held-out AIME/HMMT **pass@k** during training (paper Fig. 1 is end-of-run). Optional later: sparse offline eval on checkpoint every K steps — separate Modal job, not in `train()` loop.

**Size / throughput (set by §7 probes):**

- **bf16 everywhere.** No fp16, no fp32 fallback.
- **FlashAttention-2** (FA-3 if we land on H100/H200).
- **Gradient checkpointing on** — ~50% activation memory for ~30% recompute. Standard at this scale.
- **8-bit AdamW** (bitsandbytes) if VRAM-tight. Drops optimizer state from ~4× model size to ~1× (~10 GB freed for Qwen3-1.7B). Cheap to A/B.
- **Fused AdamW** for speed once memory is sorted.
- **`max_response_length`** — **4096 locked** (Group A). vLLM decode compute scales with tokens actually generated per sequence (PagedAttention, incremental KV blocks), not full `max_tokens` matmul on every rollout. **`max_model_len`** (1024+4096=5120) sets the KV memory pool and concurrency budget — separate from per-token decode cost.
- **vLLM `gpu_memory_utilization`** — tuned to leave VRAM for HF model + grads + optimizer + activations if collocated; maximized if vLLM has its own GPU.
- **HF → vLLM weight sync cadence.** Every step by default; batch updates over N steps if `update_weights` is slow.
- **Microbatch shapes** (rollout, logprob forward, backward) + gradient accumulation — sized to fit VRAM with everything above. Defer to probes.

### Mentor-suggested VeRL references

- Data preprocessing example: `[examples/data_preprocess/gsm8k.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py)`
- Core RL algos (GRPO / PPO): `[verl/trainer/ppo/core_algos.py](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)`

We're not importing VeRL — these are read-and-lift references.

### Prompt + parser (locked 2026-05-25)

Resolved by Group A offline analysis, prompt A/B/C probe, and Rank-2 implementation — see [`probes/group_a_results.md`](./probes/group_a_results.md) addendum and [`timeline.md`](./timeline.md).

| Knob | Lock | Fallback |
| --- | --- | --- |
| **Train prompt** | **`hybrid_answer_boxed`** (arm C) — `Answer: \boxed{N}` hybrid; no validated upstream recipe | `dapo_answer_v1`, `verl_math_boxed` in `main/train/prompts.py` |
| **Parser** | **Rank-2** — hybrid regex (arm C) → last `\boxed{}` → Minerva `Answer:` line; `extract_path` logged | Revert variant in yaml if train diverges |
| **`max_response_length`** | **4096** (Group A: 1.25% cap hits at 4096) | — |

**Monitor in training:** per §5 **Training-time reporting** — `extract_path` fractions + `train/prompt_coverage`; if `extract_path_none` climbs above ~20% mid-run, revisit prompt. OOD eval stays Math-Verify (format-agnostic).

### Open

- Exact vLLM version + `update_weights` API signature — **pinned 0.8.5** in image; spike in `test_weight_sync.py`.
- Judge hosting: same GPU vs second GPU vs API — Group A answered $/latency; **collocated** three-way VRAM still open.
- Config schema — production yaml: [`configs/train_real.yaml`](../configs/train_real.yaml); launch via `launch_train.sh --mode smoke|full`.
- Whether async rollout/train overlap is worth the code complexity — **Group B answered:** rollout is 73% of step on H200 @ bs=64, so overlap could save ~25–30% wall-clock. Defer to **after** first real training run lands; implementation complexity not worth bundling into the first launch.
- Whether to re-introduce `kl_coef` if we see mode collapse / reward hacking under KL=0.

## 6. Operations

Anastasia-owned; protocol to be finalized when training is ready to launch. Engineering rules (wandb conventions, Modal image, checkpointing cadence, artifact routing) live in `[STANDARDS.md](./STANDARDS.md)` — this section is operational protocol on top of those rules.

### Decided (from pilot)

- **No shared Modal workspace.** Each person launches detached jobs on their personal Modal profile. Artifacts shared via HuggingFace Hub / git, not cross-workspace volumes. (See `pre-milestone/nancy_explore/narrative/decisions.md` 2026-05-19.)
- **wandb project:** `cs224r-minority-voting`. Run names must include operator; should include other modal-relevant IDs. Must allow for mid-training monitoring. wandb team is: "[https://wandb.ai/224r-project](https://wandb.ai/224r-project)"

### Needs a protocol

- How does a member pull the repo and set up? 
- **Modal launch:** standard command (likely `modal run --detach …`), how to verify profile, smoke before matrix launch.
- **Checkpointing:** cadence (wall-clock or step-based), where saved (Modal volume? HF Hub push on completion?), how a resume actually works.
- **Artifact handling:** what gets pulled locally, what stays on the volume, how the team accesses each other's runs.
- **Kill rules:** budget cap per run, what triggers an auto-kill, when we should consider killing manually (flat eval signal, OOM loop, runaway cost).
- **Run log:** spreadsheet or markdown index of every launched run (operator, config, status, cost, link to wandb + artifacts). Raw data lives on wandb and modal; we can autorun a script to automatically update our doc.

### Open

- Final form of all of the above — defer until §7 sizing is settled and trainer is real.

## 7. Sizing & cost

Where every knob that affects cost or step time gets enumerated. **Not comprehensive** — add to it as more knobs surface during implementation.

**How we decide these:** build the codebase skeleton first → apply known cheap perf wins → run the scoping probes below to measure step time, throughput, parse rate, and reward density → only then lock the training-matrix config. No values are committed here yet; this is the menu of things that need values.

### Knobs to set

**Per-step shape:**

- prompts per training step (batch size)
- N rollouts per prompt (milestone used 8)
- max_new_tokens (response length cap — single number for train and eval)
- max prompt length
- microbatch sizes (rollout, logprob forward, backward)
- gradient accumulation steps

**Training horizon:**

- total training steps
- epochs over the training data
- learning rate + schedule + warmup
- optimizer (AdamW vs fused AdamW vs 8-bit)
- weight decay
- gradient clipping
- seed

**RL-specific:**

- kl_coef / whether to keep a ref model
- PPO clip ratio (symmetric vs asymmetric à la DAPO)
- inner PPO epochs per step
- advantage normalization / standardization
- reuse rollout logprobs as `old_logprobs` (vs separate forward)

**Sampling at rollout:**

- temperature
- top_p / top_k
- repetition penalty
- stop sequences

**Hardware / system:**

- GPU class (A100-80GB baseline; H100 / multi-GPU if math indicates it's better)
- precision (bf16 baseline; fp16, fp32, quantization as alternatives)
- FlashAttention on/off
- gradient checkpointing on/off
- vLLM `gpu_memory_utilization`
- HF → vLLM weight sync cadence (every step vs every N steps)

**Eval-time:**

- pass@k sample count (≥64 to compute pass@64)
- eval temperature (often different from train)
- eval max_new_tokens
- eval batch size
- mid-training **held-out** eval cadence + which slice (optional; **not** required for Poly-EPO Fig. 2 parity — see §5 Training-time reporting)

### Scoping probes (run before locking the matrix)

- **vLLM throughput probe** — tokens/sec for `n_prompts × N × max_new_tokens` at the candidate batch shape.
- **Weight sync probe** — wall-clock to push HF state dict → vLLM.
- **End-to-end step probe** — one full step (rollout → reward → advantage → backward → optimizer → sync). Gates whether the matrix is affordable.
- **Answer extractability check** — % of baseline rollouts that parse to a final answer.
- **Reward density check** — fraction of prompts with ≥1 correct rollout (GRPO signal) and with mixed correct/incorrect (minority-answer signal).
- **Minority-signal sanity** — distribution of per-rollout marginal advantages under the real subset-of-4 averaging.

### Outputs to compute

- step time (s)
- tokens/sec at rollout
- $/step
- $/arm (full training)
- $/full eval matrix
- total $/matrix incl. buffer

### Probe status (2026-05-26)

| Probe | Status |
| --- | --- |
| Group A (rollout + judge) | **Done** — H100; ~4.5k tok/s policy; judge ≈ rollout time; see `group_a_results.md` |
| Prompt A/B/C | **Done** — arm **C** (`hybrid_answer_boxed`) locked for train |
| Group B (collocated GRPO step) | **Done** — H200 chosen; bs=64 fits with 75% VRAM peak; rollout=73% / backward=25% of step (~0.41 s/kept seq); $0.0023/prompt; see `timeline.md` 2026-05-26 |
| GPU SKU (H100 vs H200) | **Done** — H100 ruled out (OOMs at bs=64); H200 locked. B200 not run; optional ~1 hr smoke per `probes/B200_migration_analysis_2026-05-26T034425Z_b01999f.md` |

### Locked from Group B (2026-05-26)

- **GPU**: H200 (Modal `gpu="H200"`, `modal_price_per_sec: 0.001261`).
- **`batch_size`**: **Locked 64** ([`decisions.md`](./decisions.md) §2026-05-26). Fits with ~75% VRAM peak on H200; theoretical ceiling ≈ bs=80–96 before OOM at this stack shape — **do not use 128** on single-GPU collocated train (bs=128 probes OOM in `logprob_fwd` after rollout).
- **`gpu_memory_utilization`**: 0.45 on H200 (gives vLLM ~63 GB / trainer ~77 GB). **Do not raise blindly** — raising starves the trainer and re-creates the bs=64 OOM.
- **Microbatch**: capped by `n_kept_sequences` per step (~72 in the probe). All sequences fit in one forward+backward at this shape; no gradient accumulation needed unless we scale batch_size past the VRAM ceiling.
- **Per-step economics**: ~$0.150/step at this shape (~$0.0023/prompt). $/arm depends on §7 horizon (TBD).
- **Backward (Phase 1, canonical `66g5uyt6`)**: 29.5 s @ `n_kept=72` → **~0.41 s / kept sequence**. Phase 1b backward is not comparable (max microbatch + fresh rollouts).
- **`rollout.gpu_memory_utilization: 0.45` on H200**: vLLM ~63 GB after rollout; trainer peak ~105 GB on 140 GB → ~35 GB headroom. Raise util only after re-probing trainer peak.
- **Open knob to revisit later (not for first launch)**: async rollout/train overlap — rollout is 73% of step, so overlap pays. Implementation complexity ≠ trivial; defer to post-first-run.

### Reward density / batch utilization (flag)

Group A n800 (arm A): **73.4%** of prompts all-wrong (0/8 rollouts correct) → under zero-advantage filtering only **~27%** of prompts contribute gradient per step. Hybrid arm C: **~65.9%** all-wrong (~30% more contributing prompts). **Consider for v1 train:** DAPO-style dynamic sampling (oversample prompts with mixed rewards) or curriculum — not implemented yet; log `fraction_filtered` / `n_kept` each step (Group B + Group C instrumentation).

### Open

- ~~Whether **H100** stays default after Group B readout~~ — **Resolved 2026-05-26: H200 locked** (H100 OOMs at bs=64).
- Whether to chase the stretch +$400 credits proactively or only if probes say we need them.
- Whether some arms (e.g. Poly-EPO-answer stretch) get a smaller training horizon / fewer prompts than the headline arms for cost parity.
- §2 Polaris freeze (size, bands, drop-easy) — still blocks final train matrix; now the **binding blocker** for first training launch.

