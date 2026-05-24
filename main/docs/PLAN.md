# Main experiment plan

**Status:** working draft started 2026-05-23. Each section flagged with what's still TBD. Pilot postmortems are out of scope here — see `[LESSONS_FROM_PILOT.md](./LESSONS_FROM_PILOT.md)`.

---

## 1. Overview

**Goal:** Train Qwen3-1.7B with a set-based minority-voting objective on Polaris, and measure pass@k generalization to AIME-25/26, HMMT, and Beyond-AIME.

**Constraints:** $1,600 Modal credits confirmed (~640 hr on A100-80GB; GPU class flexible, see §7). Poster due 2026-06-03. Internal target: experiments done by 2026-05-31.

### Timeline

TBD.

### Work split

TBD.

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

**Training sub-block size and difficulty band:** TBD. Target ≈ 16k problems (mirrors DAPO-17k baseline); final size depends on §7 cost. Three coupled decisions, all open:

- Whether to drop problems Qwen-1.7B gets 8/8 correct (Polaris recipe).
- Probe baseline pass rate across difficulty bands to see where reward signal actually exists.
- Final distribution: low-difficulty bias (milestone framing) vs. mirrored-J per Polaris vs. something in between.

**Cleaning:** Polaris is already filtered from DeepScaleR + AReal-boba-Data and should be clean. Apply only:

- drop rows where `answer` doesn't parse as an integer
- drop rows where `problem` is empty or malformed

**Eval splits:** TBD — defer to §4 Evaluation. (Pilot frozen splits in `pre-milestone/pilot/data/` are a starting point but not load-bearing here.)

**Freeze policy:**

- Once difficulty band + size are chosen, materialize a single jsonl file at `main/data/polaris_train.jsonl` with a deterministic seed.
- Record provenance in `main/data/polaris_train.meta.json`: HF dataset ID, dataset revision SHA, difficulty bands kept, sampling seed, row count, cleaning filters applied, materialization timestamp.
- Once frozen, **do not re-materialize** without writing a dated note in `main/docs/context.md` explaining why.
- Eval splits follow the same convention: one jsonl + meta.json per split.

## 3. What we're testing

**Hypothesis:** Training Qwen3-1.7B with a set-based minority-voting objective improves pass@k generalization on harder held-out reasoning evals compared to vanilla GRPO, by preventing collapse to a single output mode.

**Falsifies if:** minority-answer matches or underperforms GRPO on pass@k for k ∈ {1, 4, 16, 64} across the held-out evals in §4. (TBD: specific deltas / significance bar.)

### Arms


| Arm             | Status                       | Objective                                                  | Clustering substrate                                   | LLM judge required |
| --------------- | ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ | ------------------ |
| GRPO            | must                         | per-trajectory advantage A_i = r_i − mean(r)               | none                                                   | no                 |
| Minority-answer | must                         | set-based minority voting, marginal per-rollout advantages | exact answer match                                     | no                 |
| Minority-CoT    | in scope; first cut if tight | same as minority-answer                                    | LLM-judged CoT clusters (in-loop)                      | yes (in-loop)      |
| Poly-EPO        | stretch                      | f_poly = mean(r) · diversity(G)                            | LLM-judged CoT clusters (or answer-hash variant — TBD) | yes if CoT variant |


All arms share the same model (Qwen3-1.7B-Base), same training data, same N=8 rollouts/prompt, same reward, same eval suite. Only the advantage computation differs.

### Objective math (minority-answer / minority-CoT)

For each prompt, sample N=8 rollouts. For each of the C(8,4)=70 size-4 subsets G:

- `f(G) = r(minority(G))` where `minority(G)` is the cluster with lowest frequency in G (ties broken at random).
- Per-rollout marginal advantage A_i = (mean f(G) over the 35 sets containing rollout i) − (mean f(G) over all 70 sets).

Clustering substrate differs by arm (answer-hash vs. CoT). Tiebreak settled in milestone: random pick (r=0.994 vs. averaging).

### Reward

0/1 integer match between the model's `\boxed{...}` answer and Polaris gold. Answer parsing and grading depends on the answer-extraction method (consider VeRL-inspired, or MathReward package).  

### Success criteria

Primary outcome metric: **pass@k on held-out evals**, k ∈ {1, 4, 16, 64}. Comparison across arms on identical model, data, compute.

- **Pass criterion (strong):** minority-answer > GRPO on pass@4 and pass@16 on at least 2 of the held-out evals. TBD: how much margin counts as "real". 
- **Pass criterion (consolation):** minority-answer matches GRPO on pass@1 *and* improves cluster diversity on the same eval rollouts (less mode collapse, see "tokens to first branching" metrics used in Poly-EPO), even if pass@k is unchanged.
- **Failure mode:** minority-answer degrades pass@1 substantially → revisit hybrid objective (milestone §3 next-steps item).

### Open

- Significance bar for "real" improvement (bootstrap CI, paired t-test, raw delta).
- Minority-CoT judge: locally hosted vs. API (cost / latency / throughput tradeoff).
- Poly-EPO arm: their exact formula or an answer-hash variant for cost parity?
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

Lightweight VeRL-flavored trainer; not the full VeRL stack. Nancy owns all `.py`. Target: small enough that one person holds the whole thing in their head.

### Proposed

- **Rollout engine:** vLLM, in-process.
- **Training model:** HF transformers, separate from the vLLM engine. Per-step weight sync from HF → vLLM.
- **Repo layout:**

```
main/
  train/
    rollout.py     # vLLM engine + HF→vLLM weight sync
    reward.py
    objective.py   # grpo / minority-answer / minority-cot / poly-epo advantages
    loss.py        # clipped surrogate, microbatched backward
    trainer.py     # main loop
  eval/
    passk.py
  infra/
    modal_app.py
  configs/
  data/            # frozen jsonl + meta.json per §2
  docs/
```

### Mentor-suggested VeRL references

- Data preprocessing example: `[examples/data_preprocess/gsm8k.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py)`
- Core RL algos (GRPO / PPO): `[verl/trainer/ppo/core_algos.py](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)`

We're not importing VeRL — these are read-and-lift references.

### Open

- HF → vLLM weight sync mechanism (exact API path, sync every step vs. every N steps).
- Answer-extraction prompt template + parser (VeRL trick Ifdita mentioned — borrow from their data preprocess example).
- kl_coef / ref model — keep textbook or simplify.
- Quantization — default bf16, revisit if VRAM-tight.
- Microbatching, gradient checkpointing, flash-attn, optimizer choice — defer to §7 once probes give numbers.
- Config schema (flat yaml fields).

## 6. Operations

Anastasia-owned; protocol to be finalized when training is ready to launch.

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
- mid-training eval cadence + which slice

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

### Open

- Whether single A100-80GB is the right baseline GPU class or whether H100 / smaller-quant pays off.
- Whether to chase the stretch +$400 credits proactively or only if probes say we need them.
- Whether some arms (e.g. Poly-EPO stretch) get a smaller training horizon / fewer prompts than the headline arms for cost parity.

