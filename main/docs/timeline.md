# Timeline

Chronological narrative of major work, decisions, and pivots on the main experiment. Pilot timeline lives in `pre-milestone/nancy_explore/narrative/timeline.md`.

This doc records the **journey** — what we tried, what we learned, what we decided. For the static rules of the project, see `[STANDARDS.md](./STANDARDS.md)`; for the strategic plan, `[PLAN.md](./PLAN.md)`.

**Checkpoint eval (canonical):** [B200 three-arm — Polaris 2k / DAPO 2k / BeyondAIME](#2026-05-27-wednesday--b200-three-arm-checkpoint-eval-canonical) (May 27, 2026). Local JSON under `main/data/probes/checkpoint_eval_*_arms_latest/`.

---

## 2026-05-24 (Sunday) — repo bootstrap

- Drafted `[PLAN.md](./PLAN.md)`, `[STANDARDS.md](./STANDARDS.md)`.
- Designed Group A / B / C probe plan: `[probes/05-24_probe_plan.md](./probes/05-24_probe_plan.md)`.
- Settled on DAPO `Answer:` + `math_dapo` Minerva parser as the Rank-1 train-time reward stack (`[probes/prompt_extraction_research.md](./probes/prompt_extraction_research.md)`).
- Implemented Group A probe (Phase 1 rollout + Phase 2 judge); locked Modal app-name / volumes / secrets conventions.

---

## 2026-05-25 (Monday)

### Morning — Group A 200-prompt run

- Ran Group A on 200 Polaris problems (25/band × 8 bands × 8 rollouts = 1600 completions) on H100.
- Phase 2 judge ran on all 200 problems; roughly same cost and time as rollouts — Minority-CoT arm not blocked on judge cost.
- Wrote up readout in `[probes/group_a_results.md](./probes/group_a_results.md)`.

### Afternoon — parser concern → diagnosis → A/B/C ablation

**Problem.** Group A's `parse_ok` came in at **55.9%** — far below the ~90% soft target and into the "<70% → format SFT cold start" tier in the pre-decided escalation rule (`[probes/prompt_extraction_research.md](./probes/prompt_extraction_research.md)` §11b).

**Diagnosis (not escalation).** Before committing to SFT, pulled `phase1_rollouts.jsonl` from the volume and did an offline analysis (`/tmp/group_a_analysis/`):

- `parse_ok` (55.9%) ≈ `has_answer_line` (56.4%) → **parser is fine; format compliance is the problem**. Escalating the parser without fixing format wouldn't help.
- `has_boxed` (34.6%) was largely *disjoint* from `has_answer_line` (only 7% overlap) → **a multi-path Rank-2 parser (Minerva → last-`\boxed{}` fallback) could lift `parse_ok` from 56% → 84%** "for free" on the same data.
- When both formats are present and disagree on value (65/113 cases), `\boxed{}` matched gold 6 times, `Answer:` matched 0 — so **Rank-2 order is boxed-first**, not the obvious Minerva-first.
- Residual 16% is dominated by semantic failures (wrong answers, model rambling, code-block dropouts) — not parser failures. Prompt changes alone won't recover them.

**Implemented Rank-2 in `main/train/reward.py`** with `extract_rank2()` and `extract_path` diagnostic field. `compute_reward()` kept stable for callers.

**Pivot to prompt ablation.** Original recommendation was "lock Rank 2, move on." User pushed back: are we sure DAPO `Answer:` is the right prompt, or is the model "fighting" it because pretraining bias prefers `\boxed{}`? Mentor (Ifdita) had separately flagged `\boxed{}` worth considering.

**Designed A/B/C prompt probe** (`[probes/prompt_probe.md](./probes/prompt_probe.md)`):


| Arm | Prompt                      | Source                                                     |
| --- | --------------------------- | ---------------------------------------------------------- |
| A   | DAPO `Answer:` (control)    | `BytedTsinghua-SIA/DAPO-Math-17k` parquet — verbatim       |
| B   | VeRL MATH `\boxed{}` suffix | `verl/examples/data_preprocess/math_dataset.py` — verbatim |
| C   | Hybrid `Answer: \boxed{N}`  | constructed for this probe, no validated lineage           |


Locked: paired comparison on same 800-problem manifest, H100 only, Phase-1-only, offline Rank-2 rescore on saved `completion` text.

### Late afternoon / evening — runs and verdict

**Ran 800-prompt Arm A rerun first** (n=6400) to get tighter band-level statistics than the original 200-run.


| Metric                  | n=200 | n=800     | Δ                                   |
| ----------------------- | ----- | --------- | ----------------------------------- |
| `parse_ok_rank2`        | 83.8% | 84.8%     | +1.0pp (stable)                     |
| `parse_ok_minerva`      | 55.9% | 60.3%     | +4.4pp (n=200 was slightly unlucky) |
| `mixed_reward` fraction | 15.0% | **26.5%** | **+11.5pp**                         |


Surfaced one PLAN-level finding: **73.4% of prompts are all-wrong** (0/8 correct). Under GRPO zero-advantage filtering, ~27% of each training batch contributes gradient. May want DAPO dynamic sampling or curriculum — flag for PLAN §7 once B/C land.

**Ran Arms B and C in parallel on the same 800-problem manifest** (Modal Phase 1 rollouts → volume `probes/05-25/prompt_b/`, `prompt_c/`). Built `[main/scripts/compare_prompt_arms.py](../scripts/compare_prompt_arms.py)` (paired prompt-level comparison + prompt_probe.md §5 decision verdict) while waiting.

**Final A/B/C numbers (n=6400 each):**


| Metric                          | A (DAPO) | B (VeRL MATH) | C (Hybrid) |
| ------------------------------- | -------- | ------------- | ---------- |
| has_answer_line                 | 60.7%    | 2.2%          | 42.9%      |
| has_boxed                       | 33.7%    | 89.3%         | **90.2%**  |
| **parse_ok_rank2**              | 84.8%    | 79.0%         | **87.6%**  |
| pass_rate                       | 6.0%     | 8.6%          | 8.3%       |
| **mixed_reward**                | 26.5%    | 30.9%         | **33.9%**  |
| residual (`extract_path: none`) | 15.2%    | 21.0%         | **12.4%**  |


Extract paths for C: 33% hit the strict `Answer: \boxed{N}` hybrid pattern, 53% fall back to boxed-only, 1.6% answer-line only, 12.4% nothing.

**Key reads:**

- **B confirmed the pretraining-bias hypothesis** (boxed compliance jumped 34% → 89%), but the model abandoned `Answer:` entirely (2%) so it has no fallback when boxed parsing fails. Net: B's Rank-2 actually *underperforms* A.
- **C wins on every metric that matters for RL**: highest parse_rank2, highest mixed_reward density (+28% relative vs A), lowest residual, paired wins 1.8× more often than A.
- **The strict prompt_probe.md §5 decision rule said "A wins by default"** (no arm hit both the +5pp parse_rank2 AND +2pp mixed_reward thresholds — C's parse lift was 2.8pp, below 5pp). The rule was designed conservatively to prevent adopting a *worse* parser; in fact C's parser is the best, just not by the threshold margin.

### Decision — adopt C (hybrid)

**Rationale:**

1. Highest parse_rank2 (87.6%), lowest residual (12.4%).
2. Highest mixed_reward density (33.9% vs A's 26.5%) — directly improves GRPO signal-per-step efficiency by ~28%.
3. All-wrong fraction drops 73.4% → 65.9% → effectively 30% more batch contributes gradient under zero-advantage filter.
4. Honors mentor's `\boxed{}` signal (90% boxed compliance) while keeping `Answer:` structure for parser robustness.
5. Hybrid format is *conservative* — gives the model permission to do what it already wants (boxed) inside a parseable wrapper, rather than fighting pretraining bias.

**Risks accepted (documented for fallback):**

1. **Validation lineage.** C is our invention; A (DAPO) and B (VeRL MATH) are battle-tested at scale. If training shows convergence issues, swap config to `dapo_answer_v1` or `verl_math_boxed` (both kept in `main/train/prompts.py`).
2. **Format inconsistency during training.** Three accepted formats → no gradient pressure to converge on one. Likely benign (all parse), but monitor `extract_path` distribution per step in wandb; if "none" rate climbs above ~20% mid-training, revisit.
3. **Not teaching new looseness.** The base model already had weak instruction-following; C accepts that rather than punishing it. OOD eval uses Math-Verify (format-agnostic per STANDARDS), so looseness doesn't propagate downstream.
4. **Non-risk:** minority-voting clustering — three formats all normalize to the same answer string via our parser, so set-based arms cluster correctly.

### Late evening — trainer skeleton + Group B probe

- **Trainer skeleton landed** — `main/train/{rollout,objective,loss,trainer,weight_sync}.py`, `main/data/dataset.py`, `configs/train_grpo_05-25.yaml`, `launch_train.sh`; `run_one_grpo_step(..., instrument=True)` with per-phase timers + VRAM peaks; CPU tests green (~20 pass).
- **Group B probe implemented** — `main/probes/group_b_step_probe.py` + `configs/probe_step_b_05-25{,_smoke}.yaml` + `launch_probe_step_b.sh`; three Modal phases (timed @ mb=1 → microbatch OOM sweep on cache → Phase 1b full timed step @ max mb with fresh rollouts).
- **Training path uses arm C** — `prompt_variant: hybrid_answer_boxed` in Group B config (not DAPO A).
- **Group B smoke** — passed (4×2, all phases); launch via `main/.venv` (script needs venv for Modal import when loading config).
- **Group B full H100 run** — detached launch; Modal app `ap-CDWaOaDdYVLSNyRsqxjxtd` (wandb group `probe-B-05-25`; fill run URL after Phase 1 inits).

### Artifacts shipped today

- `[main/docs/probes/prompt_probe.md](./probes/prompt_probe.md)` — probe design + decision rule
- `[main/train/reward.py](../train/reward.py)` — Rank-2 parser with `extract_rank2()` and hybrid path
- `[main/scripts/rescore_rollouts_rank2.py](../scripts/rescore_rollouts_rank2.py)` — offline rescore of saved jsonl
- `[main/scripts/compare_prompt_arms.py](../scripts/compare_prompt_arms.py)` — A/B/C paired comparison driver
- `[main/docs/trainer_skeleton.md](./trainer_skeleton.md)` — GRPO trainer build doc
- `[main/docs/probes/group_b_impl.md](./probes/group_b_impl.md)` — Group B step-probe spec (aligned with shipped code)
- Trainer + Group B code (see late evening above)
- Volume artifacts: `probes/05-25/group_a_n800/`, `probes/05-25/prompt_b/`, `probes/05-25/prompt_c/`; `probes/05-25/group_b/` pending full run completion

### Open (next-up)

- **Group B readout** — when full run finishes: pull `probes/05-25/group_b/`, write results, update PLAN §7 (microbatch, collocated util, step time, $/step, async go/no-go).
- **§2 Polaris freeze** — materialize `polaris_train.jsonl` + meta (size, bands, drop-easy).
- **First GRPO train run** — set `prompt_variant: hybrid_answer_boxed` in `train_grpo_05-25.yaml` (skeleton default; yaml may still say `dapo_answer_v1` until flipped at launch).
- **Eval harness** — pass@k (§4 still TBD).
- **Optional (not blocking training):** H100 vs H200 $/throughput — thin re-run of Group B toy slice; see `[probes/group_b_impl.md](./probes/group_b_impl.md)` §1.
- Potentially look into **B200** -- h100, h200, b200 are all around the same cost per step (from initial calculations); the decrease in wall clock time in b200 could be meaningful, if we don't take too long to get the architecture set up.

**Doc hygiene done (2026-05-25):** `group_a_results.md` addendum, `trainer_skeleton.md` §2 resolved, `PLAN.md` §5 prompt/parser + §7 density flag, `STANDARDS.md` reward section.

---

## 2026-05-26 (Tuesday)

### Early morning — Group B rerun, H100 OOM, switched to H200

- **Group B rerun OOM'd at batch_size=64 on H100** in `_completion_logprobs_hf` (vLLM holds ~~35 GB KV after rollout; one-shot logprob forward at n_kept~~110 won't fit in remaining 45 GB). Switched probe GPU to H200 across `group_b_step_probe.py:350`, `trainer.py:691`, and `configs/probe_step_b_05-25.yaml`. Modal app `ap-Uo8iajUVI3CHaxlycxFwNv` (H100 failure); rerun `ap-L7YjvrKS6ICh3rOKz9OAE9` (H200 success, wandb `66g5uyt6`).
- **Probe-side code fixes landed before the rerun** (agent-led): (a) microbatch sweep clamp visibility — break out when requested mb ≥ `n_kept`, persist `sweep_limited_by` + `sweep_n_kept` to `phase1_done.json`; (b) Phase 1b warmup — one untimed full `run_one_grpo_step` before the timed step so kernels are warm; (c) probe `batch_size: 32 → 64` and `toy_batch.problem_ids` extended to 0–63. Smoke config untouched.
- **H200 readout at batch_size=64 (Phase 1, warm step at mb=1):**

  | Metric                  | H100 bs=32 (prior)  | H200 bs=64 (new)                     | Read                                                                           |
  | ----------------------- | ------------------- | ------------------------------------ | ------------------------------------------------------------------------------ |
  | Step time               | 90s                 | 118.7s                               | +32% raw — but 2× the prompts                                                  |
  | $/step                  | $0.099              | $0.150                               | +52% raw                                                                       |
  | **$/prompt**            | **$0.0031**         | **$0.0023**                          | **−25% per useful unit**                                                       |
  | **wall/prompt**         | **2.81s**           | **1.85s**                            | **−34% per useful unit**                                                       |
  | VRAM peak               | 70 / 80 GB (88%)    | 105 / 140 GB (75%)                   | Real headroom on H200 (room for bs ~80–96)                                     |
  | Rollout share           | 60%                 | 73%                                  | Rollout dominates again at bs=64                                               |
  | Backward time           | 29.4s @ n_kept=56   | 29.5s @ n_kept=72 (~0.41 s/kept seq) | Use **s / n_kept** for planning — not Phase 1b (different mb + fresh rollouts) |
  | n_kept (group-survival) | 56 (22% group-kept) | 72 (14% group-kept)                  | Lower survival on problem_ids 32–63 — batch variance, not a code issue         |

- **Decision: H200 locked as the training-arm baseline.** Per useful unit of work, H200 beats H100 on both $ and wall-clock. H100 ruled out — it physically cannot run bs=64 with this code path because `_completion_logprobs_hf` doesn't microbatch. B200 remains a *potential* further upgrade (~1 hr stack debug per `B200_migration_analysis_*.md`; ~+20–30% on top of H200 if it works), but H200 alone is good enough to ship.
- **Knob caution noted:** `gpu_memory_utilization: 0.45` was sized for H100 (gave vLLM ~36 GB, left ~44 GB for trainer). Unchanged on H200, it gave vLLM ~63 GB and left ~77 GB for trainer — accidentally correct because both partitions scaled proportionally. **For any future SKU change, re-size the partition explicitly** rather than carrying the knob over blindly. Bumping to 0.55 on H200 would cannibalize trainer-side headroom and re-OOM at bs=64.
- **Bandwidth speedup smaller than projected.** H200 rollout output throughput peaked at ~4.6k tok/s vs H100's ~4.0k tok/s — a +15% bump, not the +30% the 4.8/3.35 TB/s ratio would suggest. Likely because vLLM isn't saturating BW at `gpu_memory_utilization=0.45`. Raising util would help rollout but break the trainer-side budget at bs=64. Tolerable; the per-prompt win comes mostly from being able to run the bigger batch at all.
- **Probe regression flagged (non-blocking):** microbatch sweep on H200 logged only `mb=1` in `microbatch_sweep.jsonl` — should have logged the full ladder (mb=2, 4, 8, …) up to `n_kept=72`. Either the sweep schedule jumped from 1 straight to a value ≥ `n_kept` and broke on the clamp-visibility fix, or that fix has a loop-exit bug. Headline `max_microbatch_ok=72 / sweep_limited_by=n_kept` is still correct; only the intermediate ladder data is lost. Tracked in `[issues.md](./issues.md)` #4.

### Next up (as of 2026-05-26 early morning)

Ordered by what unblocks training launch first:

1. **§2 Polaris freeze** — **done** (full pool + `polaris_train.jsonl` train manifest). Upload train jsonl to Modal before first real run.
2. **First GRPO train run on H200** at `prompt_variant: hybrid_answer_boxed`, `batch_size: 64`, `gpu_memory_utilization: 0.45`. Sanity-check the bs=64 economics at training horizon. Per-prompt cost ≈ $0.0023 → estimate $/arm once total step count is set.
3. **Eval harness** — pass@k scaffold; still TBD per PLAN §4.
4. **DAPO-style dynamic sampling** to lift the 14–22% group-survival rate (PLAN §7 density flag). Cheapest remaining throughput lever — bigger payoff than any further GPU upgrade.
5. **Async rollout/train overlap** — Group B confirms rollout is 73% of step time on H200 at bs=64; overlap could save another ~25–30% wall-clock. Worth doing **after** the first real run lands and we have a stable baseline to measure against. Implementation complexity is non-trivial; don't bundle into the first launch.
6. **Fix microbatch sweep schedule regression** in `group_b_step_probe.py` — small, do alongside any future probe rerun.
7. **Optional: B200 smoke** (~1 hr budget). Only chase if calendar pressure spikes after first real arm runs. Bring-up checklist already drafted in `[probes/B200_migration_analysis_2026-05-26T034425Z_b01999f.md](./probes/B200_migration_analysis_2026-05-26T034425Z_b01999f.md)`.

### Afternoon — DAPO vs Polaris train-data decision (arm C numbers)

**The scare.** While sizing the first training run, we compared DAPO pilot Run 0 against Polaris Group A n800 and got a **~8 pp gap on pass@8** (pilot human labels **34.4%** vs Polaris **26.6%** under a unified strict re-grade). That looked like Polaris was too hard / too sparse for 1.7B GRPO — and mentor's scaled recipe for this model size had been **DAPO-Math-17k**, not full Polaris-53k. Briefly considered switching train data to DAPO.

**What was wrong.** The Polaris side of that comparison used **prompt arm A** (`dapo_answer_v1` rollouts + Rank-2 with the DAPO extraction order), not the **locked train arm C** (`hybrid_answer_boxed`). Arm A also matched the scary **73.4% all-wrong** prompt fraction from the 05-25 Group A readout. We had already decided to train with hybrid C on 05-25; the dataset debate accidentally re-used pre-C numbers.

**Recomputed on the same 800-problem manifest** (`probes/05-25/prompt_c/phase1_rollouts.jsonl`, offline `extract_rank2(..., hybrid_answer_boxed)` + strict normalize):


| Metric       | DAPO pilot (human `cleaned_answers.parquet`, 500×8) | Polaris n800 **arm C** |
| ------------ | --------------------------------------------------- | ---------------------- |
| pass@1       | 9.03%                                               | **8.45%**              |
| pass@8       | 34.40%                                              | **33.12%**             |
| mixed_reward | ~34%                                                | **33.0%**              |
| all_wrong    | ~65.6%                                              | **66.9%**              |


Polaris with arm C is **~1 pp below** the DAPO pilot on both pass@1 and pass@8 — not the ~8 pp gap from arm A. Canonical pilot analysis remains `pre-milestone/nancy_explore/run0_analysis/` (dashboard + `minority_metrics.md`); see also `[probes/dapo_vs_polaris_rollout_comparison.md](./probes/dapo_vs_polaris_rollout_comparison.md)` (note: that doc's Polaris row is arm A only — do not use it for train-data decisions).

**Decision — train on Polaris.** Mentor recommendation + `**difficulty` bands** (8-way mirror-J labels) are worth the small baseline gap vs DAPO. PLAN §2 freeze (`polaris_train.jsonl`) is still the binding blocker; target ~16k stratified rows, **full gold types** (not integer-only), arm C prompt + Rank-2 + mathd∨sympy reward at train time.

**Arm C implementation check (same day):**


| Layer                                 | Status                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Prompt template `hybrid_answer_boxed` | **Done** — `main/train/prompts.py`; Group B + probe C yaml                                                         |
| Rank-2 parser hybrid path             | **Done** — `extract_rank2()` in `main/train/reward.py`                                                             |
| Trainer rollouts                      | **Done** — `format_problem(..., variant=cfg.prompt_variant)` in `trainer.py`                                       |
| Train yaml default                    | **Fixed** — `configs/train_grpo_05-25.yaml` → `hybrid_answer_boxed` (was still `dapo_answer_v1`)                   |
| Train-time reward                     | **Fixed** — `compute_reward()` → Rank-2 + `prompt_variant`; grading **mathd OR sympy** via `grade_parsed_answer()` |


Fallback if convergence issues: swap yaml to `dapo_answer_v1` or `verl_math_boxed` (both kept in `prompts.py`).

### Evening — train grader: mathd OR sympy (DeepScaleR / rLLM)

**Decision:** Train reward = Rank-2 extract + `**grade_answer_mathd ∨ grade_answer_sympy`** on `parsed_answer` (same rule as rLLM `grade_answer_verl`; vendored in `math_grade_deepscaler.py`). See `[decisions.md](./decisions.md)` §2026-05-26.

**Why:** Matches DeepScaleR/Polaris upstream; SymPy rescues strict/format false negatives on probes (`01`/`1`, commas) → better GRPO signal. On n800 parsed rollouts mathd added 0 passes beyond SymPy, but OR is cheap and covers rare Hendrycks-only extractions.

**Shipped:** `grade_parsed_answer()` in `reward.py`; wired through `extract_rank2` / `compute_reward` / trainer / Group A judge.

### Late evening — `batch_size: 128` probes (OOM); lock **bs=64**

**Motivation.** Poly-EPO Table 1 uses **128 prompts / batch 64** on **4× H200** (4B, VeRL). We asked whether single-GPU collocated train+vLLM could match **128 prompts/step** on H200 to improve utilization and `n_kept` (more surviving GRPO groups per step). No dynamic sampling planned — larger batch was the lever under consideration.

**Canonical bs=64 on H200 (volume `probes/05-25/group_b/phase1_done.json`, wandb `g0hrklub`).** Same Group B pipeline as early-morning readout; stochastic variance vs other wandb ids on the same config. Measured: **n_kept=96** (12/64 prompts × 8 rollouts), **VRAM peak ~115 GB / 140**, rollout **~67%** of step, `max_microbatch_ok=96` (limited by `n_kept`).

**bs=128 @ `gpu_memory_utilization: 0.45`** — `configs/probe_step_b_05-26_bs128.yaml`, artifacts `probes/05-26/group_b_bs128/` (no `phase1_done` — failed). Modal `ap-SwkHA9fDlPgLCYugbI9YZL`, wandb `1burtfuq`.


| Stage                         | Result                                                                                        |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| Rollout (128×8 = 1024)        | ✅ ~3 min                                                                                      |
| Phase 1 train (`logprob_fwd`) | ❌ OOM in warmup — **139.7 / 139.8 GB** used, needed **+176 MiB** in `_completion_logprobs_hf` |


**bs=128 util sweep (same stack; lowering vLLM cap did not free post-rollout memory).**


| `rollout.gpu_memory_utilization` | Modal                       | Wandb      | OOM after rollout                                               |
| -------------------------------- | --------------------------- | ---------- | --------------------------------------------------------------- |
| **0.38**                         | `ap-zS1o9oJTat5ZWMmDee8wdV` | `6zhrsrc3` | ✅ 1024/1024 rollouts → ❌ **139.4 GB** used, needed **+394 MiB** |
| **0.40**                         | `ap-xOSMb7WRVxnVMVK5pphPKa` | `4wkoecge` | Same                                                            |


**Readout.** Collocated single H200: doubling prompts fills GPU during/after rollout (KV + dual model copies); HF `logprob_fwd` has no headroom. `**gpu_memory_utilization` is a vLLM pool ceiling**, not a train/rollout split — lowering 0.45→0.38/0.40 did not materially change ~139 GB footprint after rollout. Fitting 128 would need **structural** changes (vLLM sleep/KV release before train, microbatched `logprob_fwd`, or 2-GPU), not yaml util tweaks alone.

**Infra fix (same session):** `pylatexenc` added to `main/infra/modal_image.py` — fresh Modal image builds failed reward import until fixed.

**Decision:** Lock `**train.batch_size: 64`**, `**gpu_memory_utilization: 0.45`**, H200 — see `[decisions.md](./decisions.md)` §2026-05-26 batch size.

### Evening — GRPO train smoke + pylatexenc log noise

**Run.** `launch_train.sh --mode smoke` → Modal `ap-SPj5QSem9RFgU9602NthEF`, wandb `yfmhev1g`. Step 0 completed (W&B logged); step 1 OOM in `_completion_logprobs_hf` (~139.7 GB) — same collocated VRAM story as Group B bs=128.

**Weird log.** Bursts of `macro '\frac' failed its substitution` (~3% of grades per rollout batch) right after each 512-completion vLLM batch — pylatexenc during sympy grading on **malformed frac LaTeX in policy extractions**, not gold or infra failure.

**Decision.** Silence pylatexenc warnings; policy garbage is expected early — see `[decisions.md](./decisions.md)` §2026-05-26 pylatexenc.

### Late night — random full-gold n800 vs integer stratified (train-data sanity check)

**Question.** For PLAN §2 `polaris_train.jsonl`, should we drop non-integer gold (like Group A probes) or keep all Polaris answers? Would a random 800 with full gold look harder/easier than the stratified integer n800?

**Experiment.** Built `scripts/build_polaris_random_manifest.py` (relaxed clean: non-empty problem + gold only; seed 42). Phase-1 rollouts on Modal: `configs/probe_random_fullgold_n800.yaml`, arm C, 800×8 → `probes/05-27/random_fullgold_n800/`. Analysis: `scripts/analyze_random_fullgold_rollouts.py`.

**Unified grader (both runs).** Offline re-score saved completions with `**extract_rank2(..., hybrid_answer_boxed)` + `grade_parsed_answer`** (mathd OR sympy) — not the jsonl `reward` column on the May 25 integer run.


| Run                             | Sample                         | pass@1                           | pass@8 (any) | parse_ok_rank2 |
| ------------------------------- | ------------------------------ | -------------------------------- | ------------ | -------------- |
| Integer stratified n800 (arm C) | 100/band, integer gold         | **8.50%**                        | **33.25%**   | 88.0%          |
| Random full-gold n800           | uniform random, all gold types | ~9.4% (partial) / matches at 60% | ~33.1%       | ~86%           |


**Readout.** Headline difficulty is **the same** whether we integer-filter at sample time or include LaTeX/fraction/string gold — sampling pool choice is not moving baseline pass rates much for 1.7B + arm C. Safe to drop integer-only filter for the 16k freeze unless we want probe parity with Group A manifests.

**Gotcha logged.** `05-25/prompt_c/phase1_rollouts.jsonl` stored `reward` at **2.77%** pass@1 (old probe grading); offline unified regrade is **8.50%**. Always re-score completions for train-aligned metrics. Write-up: `[probes/integer_vs_random_fullgold_unified_grade.md](./probes/integer_vs_random_fullgold_unified_grade.md)`.

**Frozen** full clean Polaris-53K (53,291 rows) → `[source/polaris_train_full.jsonl](../data/source/polaris_train_full.jsonl)` + meta via `preprocess_polaris.py --n 53291` (seed 42, full gold). **Canonical train manifest:** `[polaris_train.jsonl](../data/polaris_train.jsonl)` (51,139 rows after prompt filter) — see `[data/README.md](../data/README.md)`.

---

## 2026-05-27 (Wednesday) — Polaris prompt filter (proof / gold-leak)

**Motivation.** The full pool (`source/polaris_train_full.jsonl`) includes proof-style prompts (“Prove that …”) and cases where the HF gold string appears verbatim in the problem. Train stack is arm C (`hybrid_answer_boxed`) + Rank-2 + **mathd∨sympy** on a **parsed final answer** — not proof grading. Bad rows add noise (model writes proofs, boxed extract fails) or fake reward (model copies gold from the stem).

### Heuristic labeling (full 53,291 rows)

Shipped `main/data/prompt_heuristics.py` + `main/scripts/label_polaris_prompts.py` → `main/data/polaris_train_labeled.jsonl`, `polaris_train_heuristic_summary.json`.


| Flag                  | Count  | %     | Notes                                                            |
| --------------------- | ------ | ----- | ---------------------------------------------------------------- |
| `last_starts_prove`   | 1,507  | 2.8%  | Last sentence matches `^prove\b` after split on `.!?` / newlines |
| `last_contains_prove` | 1,854  | 3.5%  | `prove` in last sentence (includes starts)                       |
| `contains_show_that`  | 720    | 1.4%  | `\bshow\s+that\b` anywhere                                       |
| `gold_in_prompt`      | 10,826 | 20.3% | Stripped gold substring of problem (case-insensitive)            |
| Any of four flags     | 12,466 | 23.4% | —                                                                |


**Manual spot check (n=80):** `[probes/prove_prompt_spotcheck_80.md](./probes/prove_prompt_spotcheck_80.md)`. Pools: 40 with `prove` anywhere, 40 with last sentence starting `Prove`. ~~70% of “contains prove” sample are genuine proof tasks; ~88% of “last starts Prove” are proof / show-equality. “Contains prove” is **broader** than “ends with Prove” (~~35% of A-only sample are find-all + prove or split-sentence artifacts). Many `Prove`-ending rows have gold **not** in the prompt (938 / 1,507) — still proof-style, not leakage.

### Predicate definitions (frozen spec)

Locked 2026-05-27. `main/data/prompt_heuristics.py` is the current source-of-truth code, but **these definitions take precedence** if the module is refactored — re-deriving the 2,152-drop / 51,139-keep counts requires this exact semantics.

**Field provenance.** Both inputs come from frozen `main/data/source/polaris_train_full.jsonl` (post-`clean_rows`, pre-template-wrap):

- `problem` — raw HF `problem` string.
- `gold` — `normalize_train_gold(answer)` = `str(answer).strip()`. **Whitespace-only normalization** — no `\boxed{}` strip, no LaTeX canonicalization, no comma removal.

`**last_sentence(problem) -> str`**

- `problem.strip()` first.
- Split by regex `(?<=[.!?])\s+|\n+` — whitespace following `.!?`, OR one-or-more newlines.
- Each chunk stripped; empty chunks dropped.
- Return last chunk, or `""` if none.
- Does **not** strip leading `$`, parentheses, list markers (`(a)`), or other punctuation from the returned chunk. (Deferred relaxation noted in `decisions.md` §2026-05-27.)

`**last_starts_prove(problem) -> bool`** — outer-arm trigger

- `re.match(r"^prove\b", last_sentence(problem), re.IGNORECASE)`.
- Anchored at start of last sentence; word-boundary after; case-insensitive.
- Will NOT fire on `"$\\,$ Prove …"`, `"(b) Prove …"`, or any last sentence that doesn't *begin* with the literal token `prove`.

`**contains_show_that(problem) -> bool`** — inner-arm keyword

- `re.search(r"\bshow\s+that\b", problem, re.IGNORECASE)`.
- Anywhere in the full problem; word-boundaries on both sides; `\s+` between tokens (matches `show  that`, `show\nthat`); case-insensitive.

`**gold_in_prompt(problem, gold) -> bool`** — inner-arm gate

- Let `g = str(gold).strip()`. If `g == ""`, return `False`.
- Otherwise return `g.lower() in problem.lower()`.
- Case-insensitive substring. **No length floor. No word boundary. No `\boxed{}` strip.** `gold='1'` substring-matches any `"1"` in the problem.

`**"prove" in problem.lower()`** — inner-arm keyword (inline; **not** a labeled function in `prompt_heuristics.py`)

- Plain case-insensitive substring on the full problem.
- **NOT word-bounded** — also fires on `improve`, `approve`, `proven`, `disprove`, `disproven`. Asymmetric with `contains_show_that` (which IS `\b`-bounded) **by design**: catches morphological variants `proves` / `proved` / `disprove` inside the gated branch, where the gate (`gold_in_prompt`) already filters out incidental matches.
- The filter applier must evaluate this inline; do not substitute `last_contains_prove` (different semantics — last-sentence-only).

### Composition (the adopted rule)

```python
drop = last_starts_prove(problem) or (
    gold_in_prompt(problem, gold) and (
        "prove" in problem.lower() or contains_show_that(problem)
    )
)
```

Accepted false-positive rate: ~12% on the outer arm (n=80 spot check — ~88% of `last_starts_prove` hits are genuine proof/show tasks). Inner arm not separately spot-checked; gated by gold-leak so false-positive cost is bounded.

### Filters we considered


| Policy                                             | Drop   | Keep   | Verdict                                                                                                                                   |
| -------------------------------------------------- | ------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `gold_in_prompt` alone                             | 10,826 | 42,465 | **Reject** — removes ~9.9k MCQs / logic puzzles with answer text in clues                                                                 |
| Any of four flags                                  | 12,466 | 40,825 | **Reject** — same MCQ problem                                                                                                             |
| `prove` anywhere **OR** `(gold ∧ show_that)`       | 2,988  | 50,303 | **Reject** on outside arm — +836 mostly find-all / multi-part with numeric gold                                                           |
| `last_contains_prove` **OR** `(gold ∧ prove/show)` | 2,388  | 50,303 | **Reject** vs `last_starts` outside — +236 rows; mostly “Given …, prove”, `(b) Prove`, find-all completeness; still valid math, not leaks |
| `last_starts` **OR** `(gold ∧ prove anywhere)`     | 2,152  | 51,139 | **Adopted**                                                                                                                               |
| `last_starts` only                                 | 1,507  | 51,784 | Too narrow for mid-body “Prove X” with gold X in stem (+292 leaky rows beyond starts-only)                                                |


**Gold-leak branch nuance.** `gold_in_prompt ∧ prove anywhere` catches mid-body “Prove that a^2+b^2…” with gold `a^{2}+b^{2}` (+292 vs `gold ∧ prove/show in last sentence` only). We kept **prove anywhere** in the **inner** branch (not only last sentence) so leaks in part (a) still drop when gold appears in the stem. We did **not** require gold leak for the outer `last_starts` arm — most proof endings (62%) have no substring leak but are still poor boxed-answer targets.

`**last_starts` vs `last_contains` on the outside.** `last_contains` adds 236 rows where the last sentence has `prove` mid-sentence (`Given …, prove`, `$$  Prove`, “Provide all answers and prove no others”). Spot check + samples showed these are often still real proof tasks or formatting variants — not worth the extra cut vs tightening `^prove` to strip `$`/whitespace (deferred).

`**show that` without `prove`.** ~453 rows kept (no gold leak). ~232 with gold leak dropped via inner branch.

### Decision — locked filter

See `[decisions.md](./decisions.md)` §2026-05-27. Predicate:

```text
DROP  last_starts_prove
   OR (gold_in_prompt AND ("prove" in problem OR contains_show_that))
```

**Result:** **2,152 dropped (4.0%)**, **51,139 kept (96.0%)** on the frozen 53,291 pool.

**Materialized (frozen):** `[polaris_train.jsonl](../data/polaris_train.jsonl)` (51,139 rows) + `[polaris_train.meta.json](../data/polaris_train.meta.json)` via `filter_polaris_train.py` from full pool; dropped audit `[polaris_train_dropped.jsonl](../data/polaris_train_dropped.jsonl)` (2,152 rows). `train_real.yaml` → `/vol/data/polaris_train.jsonl`. Upload train jsonl to Modal `main-artifacts` before full train.

---

## 2026-05-28 (Thursday) — GRPO full run launch, mid-epoch handoff

### Trainer fix + token-budget packing (commit `039ad38`)

Pre-launch, fixed a structural OOM in `_train_step_microbatched`: previously the per-sequence forwards' autograd graphs stayed alive simultaneously across all `n_kept` until a single backward at the end, so the `microbatch` knob did not actually bound VRAM. Refactored to interleave forward+backward per chunk (graphs freed after each chunk's `.backward()`). Added **token-budget packing** (`train.token_budget=90000` in yaml) — greedy first-fit-decreasing by completion length so each chunk holds ≤ budget total tokens, decoupling peak VRAM from per-step completion-length variance. Variable-chunk loss weighting (chunk_size / n_kept) verified equivalent to full-batch mean (within 1e-5).

Also added VRAM telemetry to the train loop: `vram_peak_gb_step`, `vram_headroom_gb_step`, per-phase peaks, `t_train_fwd_bwd_s`, `num_chunks`, `max_chunk_size`, `effective_microbatch`.

### Full run launched — wandb `8qesa78k`, app `ap-yzf7ep0GD5mA2m5boc88JR`

Config: `train_real.yaml`, H200, bs=64, n_rollouts=8, microbatch=64, `token_budget=90000`, lr=1e-6, warmup_steps=20, total_steps=850, checkpoint_every_steps=10, `polaris_train.jsonl` (51,139 rows).

### Observations through step 141 (Modal 8h timeout cut us off mid-step 142)


| Metric            | Value                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| Step time         | median 197s, mean 201s, max 287s                                                                |
| VRAM peak         | 109–126 GB (~14–31 GB headroom on 140 GB device) — token-budget validated                       |
| Chunks per step   | 2 typical; 3 when `n_kept` × completion length spilled the budget; 1 when both were small       |
| n_kept range      | 96–216 (mean ~160)                                                                              |
| t_train_fwd_bwd   | dominated by backward (gradient checkpointing recompute) — ~7s forward vs ~65s backward typical |
| Checkpoints saved | step 9, 19, 29, …, 139 (14 ckpts) on `/vol/checkpoints/train_real/`                             |


**Reward trajectory (20-step windows):**


| Window              | n   | mean reward | sd           |
| ------------------- | --- | ----------- | ------------ |
| s 0–19 (pre-warmup) | 20  | 0.0864      | 0.027        |
| s 20–39             | 20  | 0.0817      | 0.020        |
| s 40–59             | 20  | 0.0800      | 0.016        |
| s 60–79             | 20  | 0.0951      | 0.023        |
| s 80–99             | 20  | 0.0971      | 0.018 ← peak |
| s 100–119           | 20  | 0.0888      | 0.024        |
| s 120–139           | 19  | 0.0855      | 0.021        |


U-shape early dip + recovery, then regression to baseline by step 140. No collapse, no instability. n_kept stable; `mean_completion_tokens` flat (831 → 850); no length blowup. No OOM, no errors — exit was Modal's 8-hour function-call timeout, not a code failure.

### Handoff

Stopping here at the natural Modal-timeout break to transfer to **Emma or Anastasia** to continue the run on their Modal credits. Checkpoint at step 139 (~13.7 GB) + filtered Polaris jsonl + handoff doc shipped. They resume via `resume: auto` which picks up step 140 from `step_000139.pt`. Wandb runs will be discontinuous (two run IDs) but step numbers stay continuous; stitch in post via `wandb.Api`.

### Pre-handoff monitoring patch (commit `9b9e104`)

Added 6 health metrics before handoff so the next operator has better signal during their leg:

- **Importance ratio stats** (`ratio_mean/max/p95`, `clipped_low_frac`, `clipped_high_frac`) — DAPO clip stress diagnostic
- `**grad_norm_preclip`** — instability / clipping-attenuation watch
- `**mean_neg_logprob`** — cheap entropy proxy for mode-collapse early warning
- `**finish_reason` distribution** (`frac_finish_stop/length/other`) — catches length blowup
- **Reward by extract path** (`mean_reward_extract_{hybrid,boxed,answer_line,none}`) — separates "learning math" from "learning format compliance"
- **Sample completions every 50 steps** — visual sanity check for reward hacking / garbage tokens

All 53 CPU tests still pass; smoke-tested via direct `grpo_loss(..., return_stats=True)` and `aggregate_train_step_wandb_metrics` invocations.

### Open follow-ups for next operator

- Bump `timeout=60*60*8` → `24` in `trainer.py:974` to halve relaunch count per epoch (currently ~5 legs needed for 41h epoch).
- Consider raising `token_budget` to 105–110k once a few steps of the new run confirm peak VRAM stays comfortable; reduces multi-chunk steps (each chunk has ~30s recompute overhead).
- Consider deferred experiment: `gradient_checkpointing=False` with smaller `token_budget` to halve backward time — biggest possible single win (~25% step time), but needs careful VRAM calibration.

---

## 2026-05-26 (Tuesday) — efficiency restart: FA2 + token_budget 105k + self-spawn

### Motivation

Step time on the live run (`8qesa78k`, app `ap-ZPXuQuUDfEHb7p5aQhj15N`) was 3–4 min/step at bs=64; at 2 epochs (~1600 steps) and Modal's 24h function cap, that's ~4 days wall-clock and ~7 manual relaunches per branch. With up to 4 arms planned (GRPO, Minority-answer, Minority-CoT, Poly-EPO-answer), per-step savings compound; restart cost is one-time and only 12h was sunk. Three open follow-ups from the 28th entry were landed in one batch. Full reasoning + branch-savings math in `[efficiency_wins_2026-05-26.md](./efficiency_wins_2026-05-26.md)`.

### Changes landed


| Change                                          | File                                          | Expected win                                                                                                                                     |
| ----------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `attn_implementation="flash_attention_2"`       | `train/trainer.py` (`build_hf`)               | ~~20–30% on HF fwd+bwd (~~27% of step) → ~5–8% step time                                                                                         |
| `flash-attn==2.7.4.post1` prebuilt wheel        | `infra/modal_image.py`                        | (enabler; source build failed — `debian_slim` has no nvcc, switched to GitHub release wheel matched to torch 2.6 + cu12 + cp311 + cxx11abiFALSE) |
| `token_budget 90000 → 105000`                   | `configs/train_real.yaml`                     | ~25% step time when chunks drop 2 → 1 (often, given FA2's activation savings)                                                                    |
| Self-spawn auto-relaunch                        | `train/trainer.py`, `scripts/launch_train.sh` | eliminates manual relaunches; `train_remote` chains itself via `.spawn(...)` at `CS224R_LEG_HOURS` (default 23h)                                 |
| `--fresh-wandb` flag + `CS224R_FRESH_WANDB` env | `train/trainer.py`, `scripts/launch_train.sh` | escape hatch when resume ckpt < live wandb step (wandb rewind requires private-preview access we don't have)                                     |
| Wandb rewind helper (unused but kept)           | `scripts/wandb_rewind.py`                     | one-shot rewind via `resume_from`; returns 400 on current plan, kept for future tier                                                             |


### Restart procedure executed

1. `modal app stop --yes ap-ZPXuQuUDfEHb7p5aQhj15N` — old run had reached wandb step 155, latest ckpt was `step_000149.pt`.
2. Attempted `wandb_rewind.py --run-id 8qesa78k --step 149` → `wandb: Rewind is in private preview -- contact support@wandb.com to enable it.` Backed out.
3. Added `CS224R_FRESH_WANDB` env override + `--fresh-wandb` flag (~5 lines total).
4. Image rebuild failed first time on flash-attn source build (no nvcc). Switched to prebuilt wheel URL.
5. `bash main/scripts/launch_train.sh --mode full --fresh-wandb` → app `ap-ojqOqa0PgKoHk6O5QWmVw1`. Container started post-image-build and **died ~90s later, before any wandb logs were emitted**. Modal's 100-line log buffer was saturated with image-build output by the time we checked, so no stack trace was recovered.

### Root-cause narrowing (incomplete; FA2 deferred)

The four changes ride together. We can isolate by **when each one can crash**:


| Change                                    | When it executes                                                                        | Could have killed a 90s init?                   |
| ----------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Self-spawn (`leg_budget_s` check)         | Bottom of step loop; default 23h                                                        | **No** — never reached in 90s                   |
| `token_budget=105000`                     | First call to `_train_step_microbatched`, i.e. after first rollout completes            | **No** — first rollout alone is ~2 min on bs=64 |
| `--fresh-wandb`                           | `setup_wandb` (skips ckpt's wandb_run_id)                                               | **No** — just None assignment, no I/O risk      |
| `attn_implementation="flash_attention_2"` | `build_hf` → `AutoModelForCausalLM.from_pretrained(...)` → triggers `import flash_attn` | **Yes** — exactly in the dead window            |


High-confidence inference: **FA2 is the culprit.** Likely a runtime `import flash_attn` failure (CUDA lib resolution, ABI mismatch, or transformers↔flash_attn version incompatibility for Qwen3) despite the wheel installing cleanly at image-build time. The wheel was matched to `torch 2.6 + cu12 + cp311 + cxx11abiFALSE`; the failure mode is consistent with cuda driver / loader mismatch on the H200 worker, not the wheel itself.

### Decision (2026-05-26): drop FA2, keep the rest

Reverted `build_hf` to default attn (no `attn_implementation` arg). Kept:

### FA2 re-enabled (2026-05-27)

`main/probes/smoke_flash_attn.py` passed all stages on H200 (import, HF FA2 load/forward, collocated vLLM+HF). FA2 is not broken on the current image; ~90s death on `ap-ojqOqa0PgKoHk6O5QWmVw1` was likely vLLM init timing or unrelated. Re-enabled `attn_implementation="flash_attention_2"` in `build_hf`. Smoke: `bash main/scripts/launch_smoke_flash_attn.sh`.

Previously kept after 2026-05-26 revert:

- `token_budget=105000` (still ~25% win on multi-chunk steps; doesn't depend on FA2)
- Self-spawn (~5 manual relaunches saved per branch)
- `--fresh-wandb` flag (escape hatch utility)
- `flash-attn==2.7.4.post1` wheel still in `infra/modal_image.py` (no removal — image already cached; re-enabling FA2 later is a one-line trainer change, not an image rebuild)

Net expected step-time savings vs pre-restart: **~25% from token_budget** (lost the ~5–8% from FA2 until we debug it). Still well worth the restart cost.

### Open: FA2 debug (deferred)

To re-enable FA2 cleanly:

1. Smoke-test in isolation: launch a Modal function that does only `from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()` and reports success or stack trace. If this fails, we get the actual error.
2. Common fixes if `import flash_attn` fails: add `nvidia-cuda-cccl-cu12` to image; try a different wheel ABI (`cxx11abiTRUE`); pin transformers to a tested version.
3. Once smoke passes, re-enable `attn_implementation="flash_attention_2"` in `build_hf` and relaunch.

The flash-attn wheel install in `modal_image.py` is deliberately left in place so re-enabling is a one-line change and doesn't trigger image rebuild.

### Verification plan (during live run)

Watch these for the first ~10 post-resume steps:

- **VRAM** — `train/vram_peak_gb_step` must stay < 140 GB at `token_budget=105000`. If it touches 140, dial back to 95–100k.
- **Step time** — target 150–180s (vs. 197s median pre-change). If unchanged, FA2 may not be active — check `train/t_logprob_fwd_s` and `train/t_backward_s` for ~20–25% drop.
- **Numerical stability** — `train/ratio_max` should be similar or *lower* than pre-change (FA2 is closer to vLLM's attn than SDPA was). Spike to >10 = numerical mismatch in the importance ratio; rollback to SDPA if so.
- **Reward continuity** — `train/mean_reward` should resume near 0.085–0.10. Sharp drop to 0 = state-dict mismatch from FA2 weight-shape expectations (unlikely for Qwen3 but possible).
- **Self-spawn** — first leg should spawn leg 2 around 23h elapsed. Visible in modal app list as a new app with `leg_number=2` tag in wandb.

### Branch-cost arithmetic


| Branches              | Hours saved by FA2 + 105k token_budget (~12%) | Modal $ saved (~$0.001261/s × 64-prompt step) |
| --------------------- | --------------------------------------------- | --------------------------------------------- |
| 1 (current GRPO)      | ~13h                                          | ~$60                                          |
| 2 (+ Minority-answer) | ~26h                                          | ~$120                                         |
| 4 (all arms)          | ~52h                                          | ~$240                                         |


### Notes for next operator

- Old wandb run `8qesa78k` has the pre-change baseline through step 155; the post-revert run `pcas3emd` covers steps 150–159 with 105k token_budget (FA2 reverted). Use these two runs as the A/B baseline if/when FA2 lands cleanly.
- The 2026-05-28 entry's "Open follow-ups" are mostly resolved here (Modal 24h timeout already set in `trainer.py`; token_budget bumped; FA2 not in that list but landed). Remaining deferred: `gradient_checkpointing=False` experiment (next-biggest win, requires careful VRAM calibration; do as an A/B on a fresh branch, not mid-run).
- `--fresh-wandb` is a one-shot. Don't bake it into normal launches — auto-resume after a crash should preserve history.

### Handoff to Anastasia (post-restart, step 159)

After the post-revert relaunch (app `ap-TPL9A7X2WjbAHFzcibHfq5`, wandb run `pcas3emd`) ran cleanly through step 159, Nancy stopped the run to hand off. Live-run snapshot at stop time:


| Metric                 | Value                                                                     |
| ---------------------- | ------------------------------------------------------------------------- |
| Steps trained this leg | 150–159 (10 steps post-resume)                                            |
| `vram_peak_gb_step`    | ~125 GB (well under 140 GB threshold)                                     |
| `t_rollout_s`          | ~101s                                                                     |
| `t_train_fwd_bwd_s`    | ~102s                                                                     |
| `num_chunks`           | 2 (token_budget=105k didn't drop to 1 at `n_kept=192`; would need ~170k+) |
| `mean_reward`          | 0.064–0.078 (within historical noise band)                                |
| Step time              | ~210s — **similar to pre-restart 197s**                                   |


**Honest revision of the token_budget projection:** at the n_kept range we're seeing (mean ~160, occasionally 192+), total per-step tokens often exceed 105k, so chunks stay at 2. Realistic gain from the 90k→105k bump is **5–10% on average**, not the 25% I projected upfront. The bigger lever in retrospect is still self-spawn (eliminates ~5 manual relaunches per branch) and FA2 once we get it working.

**Handoff packet:**

- Latest checkpoint: `step_000159.pt` on volume
- Code on `origin/main` at `97236a8` (efficiency knobs + self-spawn, FA2 deferred)
- Anastasia launches with `bash main/scripts/launch_train.sh --mode full --fresh-wandb` on her Modal account; resumes at step 160; self-spawn handles legs 2+ across the remaining ~~640 steps (~~37h wall-clock at current pace ≈ 2 chained 23h legs).

---

## 2026-05-26 (Tuesday) — set-RL infra: arm 2 done, arm 4 config ready

While Anastasia's GRPO continuation ran, built the set-based RL infrastructure. **Arm 2 (`minority_answer`) code is complete** (kernel + clustering + trainer + config + CPU tests). Arm 4 (`poly_epo_answer`) shares the same kernel — YAML exists; dispatch is a thin delta. Arm 3 (`minority_cot`) infra deferred — needs a separate Modal GPU plan for the judge.

Spec: `[docs/build_spec/remaining_arms.md](./build_spec/remaining_arms.md)`. Clustering substrate details and deferred refinement: `[docs/build_spec/answer_clustering.md](./build_spec/answer_clustering.md)`.

### Built

**Set-RL kernel (`main/train/objective.py`).** One shared core: `set_based_marginal_advantages(rewards, clusters, subset_score_fn, ...)`. Enumerates C(8,4)=70 size-4 subsets per prompt at module load, computes `f(G)` per subset, baselines against per-prompt mean, returns each rollout's mean over the 35 subsets containing it. `_minority_subset_score` and `_poly_epo_subset_score` plug into the same kernel. `compute_advantages` dispatches by arm. `keep_mask` filters single-cluster prompts (collapsed-mode, zero marginal by construction). Math ported verbatim from `pre-milestone/nancy_explore/run0_analysis/analysis_c/set_score_simulation.py`.

**Clustering substrate (`main/train/clustering.py`) — v1.** Two-pass:

1. `canonicalize_answer` — hardened textual normalization (strip `\(...\)`, `$...$`, trailing `.`/`}`/`\]`, unwrap `\boxed{}`, int-valued-float coercion via `math.isfinite` guard, lowercase).
2. `sympy_equiv` union-find over canonicalize buckets, using `grade_answer_mathd_or_sympy(a,b) or grade_answer_mathd_or_sympy(b,a)` (symmetric OR). Asymmetric branches in the grader (unreduced-fraction rejection, int-strictness) are **preserved**, not bypassed — prompts ask for simplified answers, so `1/2 ≢ 2/4` is the right behavior for clustering. Blocklist regex skips sympy on set-operator / `\text{}` content to avoid the `\inA`/`\notinA` false-positive class.

**Trainer wiring (`main/train/trainer.py`).**

- `TrainCfg.from_dict` enforces `loss.length_norm: batch_max` for set arms (Dr.GRPO / Poly-EPO style) with a warning if YAML disagrees. Locks the GRPO YAML mistake-mode.
- `run_one_grpo_step` builds `clusters_grid` from `reward_meta` for set arms, threads `global_seed + problem_ids` into `compute_advantages` for reproducible minority tiebreak rng.
- `aggregate_train_step_wandb_metrics` extended with C4b (`train/mean_unique_answer_clusters_correct`); set-arm C3 percentiles (`adv_marginal_p05/p50/p95`) surfaced from `adv_out.diagnostics`.
- `minority_cot` arm currently raises `NotImplementedError` — judge client is the remaining work.

**Configs.** `configs/train_real_minority_answer.yaml` and `configs/train_real_poly_epo_answer.yaml`, each a thin fork of `train_real.yaml` (arm + `checkpoint_dir` + `wandb.group` + explicit `length_norm: batch_max`). `scripts/launch_train.sh` now accepts `--config <path>`.

**Tests.** `main/tests/test_objective_minority.py` (8 cases: collapsed cluster filtered; 7+1 minority sign; advantages match independent numpy reference; poly-EPO scoring; reproducible tiebreak; missing-clusters guards; GRPO unchanged) and `main/tests/test_clustering.py` (16 cases covering canonicalize hardening, sympy equivalences, blocklist, union-find determinism, unreduced-fraction rejection). 62/62 unit tests pass.

### Investigated and deferred: sympy clustering refinement

Started with the spec's recommendation (canonicalize-only string hashing). User flagged this misses sympy-equivalent answers in different forms (`8/5` vs `\frac{8}{5}` etc.). Investigated.

**On local probe + Run 0 rollouts (2,580 prompts × 8):**


| metric                                                                      | value                                                            |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| prompts where sympy-union-find disagrees with old string-canonicalize       | 5.81%                                                            |
| prompts where minority-cluster identity flips (real training-signal change) | 5.78%                                                            |
| distinct sympy-merged pairs observed                                        | 142                                                              |
| fraction caught by hardened canonicalize alone (no sympy needed)            | ~80%                                                             |
| fraction requiring sympy (real LaTeX equivalence)                           | ~20%                                                             |
| sympy false-positives observed                                              | 1 class: `\inA` vs `\notinA` (LaTeX parser strips set operators) |
| per-prompt sympy overhead                                                   | ~6 ms — negligible vs ~180 s/step                                |


**On the live GRPO run** (12 sample completions pulled from wandb `8qesa78k` + `pcas3emd`): the trained model produces visibly cleaner answer strings than base-model probes — bare integers, well-formed LaTeX, no `\(...\)` or trailing-period boilerplate. So the 5.78% impact on probe data is likely an upper bound on the live distribution.

**Transitivity check.** Initially worried sympy was non-transitive. Re-tested with grader's asymmetric branches bypassed (pure `simplify(a-b)==0`): zero non-transitive triples across 6 families. Earlier "non-transitivity" was actually the grader's intentional unreduced-fraction rejection. Sympy.simplify is weak (misses `sin(π/6) ≡ 1/2`, `\arctan(1) ≡ π/4`, etc.) but **transitive within its competence** — safe for clustering.

**Decisions logged:**

- Keep grader's asymmetric branches (don't merge `1/2 ≡ 2/4`).
- Ship v1 with blocklist for now; switch to allowlist (built from sympy's LaTeX parser supported-commands set) before first long minority-answer run. Failure mode shift: blocklist → silent wrong merge (unsafe); allowlist → missed merge (safe).
- Clustering algorithm choice deferred until 10-step smoke rollouts are analyzed offline (no separate checkpoint probe).

### What's left

1. **Smoke arm 2** — **done** (10 steps); see **2026-05-27** for step-time, clustering ablation, rollouts on volume.
2. **Pre-long-run review flags** (no change without sign-off): cluster ids use built-in `hash()` (STANDARDS prefers stable digest); sympy clustering on by default; `train/mean_unique_clusters_kept` not wired yet (`[build_spec/train_wandb_metrics_verdict.md](./build_spec/train_wandb_metrics_verdict.md)`).
3. **Allowlist vs blocklist default** — smoke offline compare done (`compare_clustering_methods.py`); pick default before long `minority_answer` run (allowlist safer on malformed LaTeX; blocklist catches more equivalences).
4. **Arm 3 (`minority_cot`):** `main/judge/client.py`, Modal GPU plan, `train_real_minority_cot.yaml`, C4 wandb. See `[docs/build_spec/remaining_arms.md](./build_spec/remaining_arms.md)` §4.
5. **Arm-2 / arm-4 full runs** once GRPO completes — see **2026-05-27** entry below for revised $/step (~**$0.48/step @ ~380s**, not GRPO's ~$0.25 @ ~200s).

---

## 2026-05-27 (Wednesday) — minority smoke: ~2× step time (budget impact)

**Context.** Compared wandb step-time panels for the 10-step `minority_answer` smoke vs ongoing GRPO full trains. Step time looked ~2× higher on the smoke run — this **drastically changes** arm-2 / arm-4 wall-clock and Modal $ estimates if it persists on long runs.

**Runs compared**


| Run                 | wandb                                                                                                                                 | Group                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| Minority smoke      | `[q6m0tmiu](https://wandb.ai/224r-project/cs224r-minority-voting/runs/q6m0tmiu)` (`train-minority_answer_nancy`, `launch_mode=smoke`) | `train-minority-answer` |
| GRPO full (ongoing) | e.g. `train-grpo_anastasia` / `8qesa78k`                                                                                              | `train-real`            |


**Measured (steps 0–5 smoke vs typical GRPO steps)**


| Metric                        | Minority smoke               | GRPO full                    |
| ----------------------------- | ---------------------------- | ---------------------------- |
| Step time (sum `train/t_*_s`) | **~310–430s** (median ~380s) | **~140–215s** (median ~190s) |
| `train/t_rollout_s`           | ~82–100s                     | ~90–112s                     |
| `train/t_train_fwd_bwd_s`     | **~226–330s**                | **~46–114s**                 |
| `train/n_kept_sequences`      | **504–512** (≈ 64×8)         | **112–208**                  |
| `train/fraction_filtered`     | **0–1.6%**                   | **59–78%**                   |
| `train/num_chunks`            | **4–5**                      | **2**                        |


Rollout time is essentially unchanged (both arms always generate 512 completions). The ~2× wall-clock is almost entirely **HF train** (forward + backward + grad checkpointing recompute).

### Root cause — not sympy clustering CPU

Earlier profiling (§2026-05-26 set-RL infra) already had sympy clustering at **~6 ms/prompt** — negligible vs ~180–200s/step. **Do not attribute the 2× to clustering compute.**

The driver is **different `keep_mask` semantics** after answer clustering:


| Arm                                   | Prompt dropped when…                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| **GRPO**                              | All 8 rollouts share the same **binary reward** (includes all-wrong → all 0). |
| **minority_answer / poly_epo_answer** | All 8 rollouts share one **answer cluster id** (collapsed mode only).         |


On Polaris early in training, wrong answers are often **diverse strings** (`36` vs `36.0` vs `\frac{36}{1}`, etc.) → **different clusters** → prompt **kept** with full 8-rollout gradient. Smoke run: `fraction_filtered ≈ 0`, `n_kept ≈ 512`. GRPO on the same data filters ~65% all-wrong prompts (Group A) → `n_kept` ~100–200.

This is **by design** (`objective.py` + `[build_spec/train_wandb_metrics_verdict.md](./build_spec/train_wandb_metrics_verdict.md)` §4.1): set arms can train with `mixed_reward_rate == 0` as long as answer diversity exists. It is not a performance bug.

Train cost then compounds: 512 kept sequences × ~850 mean completion tokens exceeds `token_budget=105000` → **4–5 chunks/step** vs GRPO's **2** at ~160 kept. Backward scales with `**n_kept × num_chunks`**.

### Revised budget math (if `n_kept` stays near 512)

Assume H200 `modal_price_per_sec=0.001261` from `train_real.yaml`:


| Arm                     | s/step | $/step | 799 steps (1 epoch in yaml) | vs GRPO   |
| ----------------------- | ------ | ------ | --------------------------- | --------- |
| GRPO (observed)         | ~200   | ~$0.25 | ~$200                       | 1×        |
| minority_answer (smoke) | ~380   | ~$0.48 | ~$380                       | **~1.9×** |


**Compounding across planned branches** (`[efficiency_wins_2026-05-26.md](./efficiency_wins_2026-05-26.md)` style):


| Branches                                  | Extra wall-clock vs budgeting GRPO rates for set arms | Extra Modal $ (rough) |
| ----------------------------------------- | ----------------------------------------------------- | --------------------- |
| GRPO only                                 | —                                                     | —                     |
| + minority_answer @ 2× step               | +~27h per 799-step epoch                              | +~$180/epoch          |
| + poly_epo_answer (same kernel/filtering) | another +~27h/epoch                                   | another +~$180/epoch  |


Prior plan assumed set arms cost ~~the same as GRPO per step (~~210s, ~$0.50/step in §2026-05-26 "What's left"). **That assumption is wrong** at low `fraction_filtered`.

### What might bring step time back down

- `**fraction_filtered` rising** as the policy collapses or repeats one wrong answer → `n_kept` falls → train time approaches GRPO. Watch `train/n_kept_sequences` and `train/fraction_filtered` on any long minority run; do not assume smoke rates persist.
- **Not fixed by:** DAPO dynamic sampling (GRPO's lever for all-wrong prompts), FA2, token_budget bump alone (smoke already at 105k with 4–5 chunks), or allowlist vs blocklist clustering (CPU noise only).

### Open decisions (not implemented)

- Accept ~2× $/step as the cost of training on diverse wrong answers.
- Or spec a cap / subsample on `n_kept` per step (changes training dynamics — needs sign-off before full arm-2 launch).

### Cross-refs

- Implementation: `main/train/objective.py` (`keep_mask`), `main/train/trainer.py` (`clusters_grid` → `compute_advantages`).
- Metrics interpretation: `[build_spec/train_wandb_metrics_verdict.md](./build_spec/train_wandb_metrics_verdict.md)` §4.1–4.2.
- Dashboard: `[monitoring/wandb_dashboard_full.md](./monitoring/wandb_dashboard_full.md)` §4.2–4.3 (`n_kept`, `num_chunks`).

### Arm 2 smoke complete + offline clustering ablation

**Run.** Modal `ap-rTxKthqOS0zytVb3iiQiSa`, wandb `[q6m0tmiu](https://wandb.ai/224r-project/cs224r-minority-voting/runs/q6m0tmiu)` — **10 steps**, exit 0 (~68 min). Rollouts: `/vol/probes/05-26/minority_answer_smoke/train_rollouts.jsonl` (**5,120 rows** = 10 steps × **512 rollouts/step**).

**Scale (don't confuse rollouts with prompts).** `batch_size=64` → **64 problems/step**, **8 rollouts each** → **512 completion rows/step**. Clustering runs per **(step, problem_id)** on those 8 rollouts → **640 prompt-groups** total (64 × 10 steps), not 64 or 512 prompts. All 5,120 rollouts are in the comparison.

**Tool.** `main/scripts/compare_clustering_methods.py` → `main/data/probes/05-26/minority_answer_smoke/clustering_compare_{detail,summary}.json(l)`. Four methods: `old_canon`, `hardened_canon`, `hardened_sympy_blocklist` (smoke train default), `hardened_sympy_allowlist`. See `[build_spec/answer_clustering.md](./build_spec/answer_clustering.md)`.

**Partition agreement** (640 prompt-groups; “differ” = different 8-way cluster groupings):


| Pair                              | Same partition | Differ |
| --------------------------------- | -------------- | ------ |
| hardened_canon vs sympy blocklist | 95.9%          | 26     |
| hardened_canon vs sympy allowlist | 97.8%          | 14     |
| blocklist vs allowlist            | 98.1%          | 12     |
| old_canon vs hardened_canon       | 98.9%          | 7      |


**Readout.** Hardened canonicalize alone almost matches pilot `old_canon` on partitions (7 prompts). Sympy blocklist moves **~4%** of prompts vs canon-only — same order of magnitude as probe-era ~5.8% minority-identity flip on base rollouts (not re-measured here). Allowlist is **less aggressive** than blocklist (12 vs 26 prompts differ from canon-only); e.g. step 0 `problem_id` 600: blocklist sympy-merges malformed `(1+sqrt{(}3))*a` with `(1+\sqrt{3})a`; allowlist keeps them separate. Mean distinct clusters/prompt ~5.8 → ~5.76 (blocklist); clustering CPU remains negligible.

**Deferred before long minority run:** ~~minority-**cluster identity** flip rate (subset `f(G)`), not just partition change~~ (deprioritized — we care about getting the upstream **correct truth**, not relative minority-identity stability); `hash()` → stable digest; ~~allowlist vs blocklist default for production~~ (decided below).

### Production decision — hardened canon + expanded allowlist as default

**Decision (2026-05-26):** ship **hardened canonicalize + sympy with the expanded allowlist** as the production clustering for `minority_answer` + `poly_epo_answer`. Wired in `[main/configs/train_real.yaml](../configs/train_real.yaml)` (`clustering.sympy_mode: allowlist`) and `[main/train/trainer.py](../train/trainer.py)` (dispatch on `sympy_mode ∈ {allowlist, blocklist, off}`). Legacy `clustering.use_sympy` key is no longer read.

**Why allowlist over blocklist.** The failure-mode asymmetry matters more than the partition-agreement delta:

- **Blocklist** = sympy by default, skip ~25 known-unsafe commands. Expanding the list is *reactive* — every unknown LaTeX command sympy supports is enabled and can silently merge distinct answers (the `\inA` / `\notinA` failure class). Corrupts the reward signal at the substrate.
- **Allowlist** = no sympy unless the string is built only from vetted commands + safe chars. Expanding is *proactive*. Failures are silent **false splits** (same answer in two forms not merged) — under-counts agreement but never poisons the signal. Self-heals as the policy converges to a consistent format.

**Expansion of the allowlist** (`main/train/clustering.py::_ALLOWED_LATEX_CMD` + `_ALLOWED_SAFE_CHARS`):


| added                                                                                                                                                                                                                                                                        | category                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `\infty`, `\tfrac`                                                                                                                                                                                                                                                           | constants / structural                 |
| `\sin \cos \tan \sec \csc \cot \arcsin \arccos \arctan \sinh \cosh \tanh`                                                                                                                                                                                                    | trig + hyp + inverse                   |
| `\log \ln \exp`                                                                                                                                                                                                                                                              | logs / exp                             |
| `\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \vartheta \iota \kappa \lambda \mu \nu \xi \rho \varrho \sigma \varsigma \tau \upsilon \phi \varphi \chi \psi \omega` + uppercase `\Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega` | Greek letters                          |
| safe chars: `^`, `_`, `[`, `]`                                                                                                                                                                                                                                               | powers / subscripts / closed intervals |


Each candidate was checked against representative positive (must merge) and negative (must NOT merge) sympy pairs before adding — see `main/tests/test_clustering.py::test_allowlist_`*. Word-boundary regex changed from `\b` to `(?![a-zA-Z])` so `\log_2` and similar still match while `\logarithm` is protected from being chopped. Empty-residual case (`\pi` alone) fixed via `*` quantifier on safe-char regex.

**Uplift on smoke rollouts.** Re-running the allowlist gate over the 4,479 parseable rollouts in `data/probes/05-26/minority_answer_smoke/`:


| variant                       | rollouts admitted | unique strings |
| ----------------------------- | ----------------- | -------------- |
| allowlist v1 (pre-expansion)  | 3,407 (76.1%)     | 1,202          |
| **allowlist v1.5 (shipped)**  | **3,461 (77.3%)** | **1,249**      |
| blocklist (rejected for prod) | 4,397 (98.2%)     | 1,907          |


Small but real (+47 unique strings unlocked: `\infty` intervals, bracket intervals like `[-1,1]`, numeric powers `2^{2006}`, numeric trig `\sin(\pi/2)`, pure-Greek expressions). The 21% gap to blocklist is almost entirely bare-letter expressions (`(0,0,d)`, `2x+3`, etc.) — intentionally not unlocked, see limitations.

### Known limitations of the shipped clustering (will not block runs)

This substrate is imperfect by design — we accept the floor, log the limitations, and don't iterate further before launching.

1. **Bare-letter expressions don't sympy-merge.** The allowlist does not admit `[a-zA-Z]` in residual chars. Vetting found two false-positive classes:
  - `xy ≡ x*y` — sympy applies juxtaposition-as-multiplication, so `xy` and `x*y` merge.
  - `ABC ≡ BCA` — same plus commutativity, so multi-letter labels (triangle vertex names, sequence labels) merge across orderings.
   Polaris contains geometry problems where vertex labels matter, so this would be a real signal-corruption. The cost we pay: variable-bearing expressions like `2x+3`, `(a+b)^2`, `x^2`, `\sin(\theta)` cluster purely by hardened canonicalize — equivalent forms (`2x+3` vs `3+2x`) land in different clusters. Falls under "false split" — under-counts agreement, doesn't poison.
2. `**\boxed{...}` malformed survivors.** Hardened canon unwraps one level of `\boxed{}`. Truncated/nested boxed strings (`\boxed{1`, `\boxed{...`) are not normalized — visible in the smoke detail as 13 unique strings. Not worth fixing pre-launch; these are rare and split safely.
3. **Cluster IDs are `hash(str)` and not stable across Python invocations.** `PYTHONHASHSEED` randomization would break cross-run comparison of saved cluster IDs. Within a single run, partitions are deterministic. Stable digest is still on the backlog.
4. **Grader's int-strictness branch is preserved on purpose.** `2^3` and `8` do NOT merge (the asymmetric `_str_is_int(a) != _str_is_int(b)` branch fires). This is documented behavior — we want different *forms* of the answer in distinct clusters since prompts ask for simplified output. See `[build_spec/answer_clustering.md](./build_spec/answer_clustering.md)` §"Why we kept the grader's asymmetric branches".
5. **Trig/log don't simplify to numerics through the grader.** `\sin(0) == 0`, `\log_2(8) == 3` return False. Same root cause as (4) — grader's conservative branches. Pure-symbolic equivalences (`\sin(x)+\cos(x)` reordered) still work.
6. **One-shot vetting, not continuous.** When the policy starts producing new answer formats during training, we won't notice silent under-merging unless we re-survey. Acceptable for the upcoming runs; revisit if the trained model output distribution shifts substantially.

### Cross-refs

- Implementation: `main/train/clustering.py` (`_ALLOWED_LATEX_CMD`, `_ALLOWED_SAFE_CHARS`, `sympy_equiv_allowlist`).
- Trainer dispatch: `main/train/trainer.py` (`sympy_mode` config switch).
- Tests: `main/tests/test_clustering.py::test_allowlist_`* (10 new tests covering admission + sympy merge/non-merge).
- Smoke data behind the decision: `main/data/probes/05-26/minority_answer_smoke/clustering_compare_{detail,summary}.{jsonl,json}`.

---

## 2026-05-27 (Wednesday) — GRPO checkpoint slice eval (H200 only; precursor)

**Superseded for three-arm numbers** by **[B200 three-arm checkpoint eval (canonical)](#2026-05-27-wednesday--b200-three-arm-checkpoint-eval-canonical)** below. Keep this entry for the **H200 GRPO-only** learning-curve question (“flat wandb ≠ flat learning?”).

**Context.** ~275 H200 GRPO steps (`pcas3emd`); live W&B looked flat → fixed-slice Polaris 2k, **base vs multiple GRPO ckpts only** (not minority/poly).

**Method.** `[checkpoint_rollout_eval.py](../probes/checkpoint_rollout_eval.py)`; config `checkpoint_eval_2k_polaris_aime_b200.yaml`; run `20260527T060234Z` (Modal `ap-pF7iDkRVy6L8QBqtBW8QOe`). Polaris 2000 prompts, seed 42, `hybrid_answer_boxed`, 8 rollouts/prompt, train grader.

| Checkpoint (H200 GRPO) | pass@8 | frac 0/8 |
| ---------------------- | ------ | -------- |
| base | 0.306 | 0.694 |
| step 49 | 0.315 | 0.686 |
| **step 99** | **0.324** | **0.676** |
| step 149 | 0.323 | 0.677 |
| step 239 | 0.314 | 0.686 |
| step 339 | 0.320 | 0.681 |

**H200 GRPO verdict:** Small on-distribution lift (best **+1.8 pp** at step 99), then plateau ~0.31–0.32 — not a wiring failure (`ratio_max` healthy). Artifacts: [polaris_summary_20260527T060234Z.json](../data/probes/checkpoint_eval_2k_polaris_aime/polaris_summary_20260527T060234Z.json), [polaris_summary_20260527T074137Z.json](../data/probes/checkpoint_eval_2k_polaris_later/polaris_summary_20260527T074137Z.json).

**H200 GRPO DAPO 2k only** (`dapo_n2000_seed43`, `20260527T090530Z`): base **0.274**, step 339 **0.259** (−1.6 pp OOD regression). Partials under [checkpoint_eval_ood_aime_dapo_99_339/](../data/probes/checkpoint_eval_ood_aime_dapo_99_339/20260527T090530Z/partials/dapo/).

---

## 2026-05-27 (Wednesday) — B200 sleep + `gc_off`: stop here (for now)

**Context.** On B200, `minority_answer` step time is dominated by HF backward (~200s). The tempting bundle was `vllm_sleep=1` (evict vLLM KV during HF train) + `gradient_checkpointing=false` (store activations, reduce recompute) + tuned `token_budget`.

**What we observed.**

- Sleep works mechanically: vLLM logs show it frees **~82–85 GiB** during the train window.
- However, `sleep + gc_off` smokes repeatedly hit **true CUDA OOMs at the device cap (~178.35 GiB)**, followed by allocator-abort fallout.
- Modal GPU memory plots can still spike near the cap during **awake** phases because vLLM re-reserves its full KV pool on wake; wandb “VRAM” is PyTorch-only and does not include vLLM’s non-torch reservations.

**Decision.** **Stop iterating** on sleep+gc_off unless a single follow-up smoke with **reduced vLLM KV reservation** (`rollout.gpu_memory_utilization <= 0.35`) goes green. Proceed with B200 runs using `vllm_sleep=0`, `gradient_checkpointing=true`, and optionally `token_budget=130k` for a modest win.

Write-up: `docs/efficiency/B200_sleep_gc_off_give_up_2026-05-27.md`.

## 2026-05-27 (Wednesday) — GRPO smoke: H200 vs B200 step-time A/B

**Context.** Needed a clean GPU comparison before committing to B200 for production arms. Earlier wandb spot-checks against long H200 full runs (`8qesa78k`, `pcas3emd`) were misleading — those are not paired with the B200 smokes.

**Runs (paired GRPO smoke only).** Same config (`batch_size=64`, `n_rollouts=8`, `token_budget=105k`, 10 steps); only `gpu_class` differs:


| GPU  | wandb                                                                            | run name           |
| ---- | -------------------------------------------------------------------------------- | ------------------ |
| B200 | `[1hg8fs5u](https://wandb.ai/224r-project/cs224r-minority-voting/runs/1hg8fs5u)` | `train-grpo_nancy` |
| H200 | `[5sekbfnq](https://wandb.ai/224r-project/cs224r-minority-voting/runs/5sekbfnq)` | `train-grpo_nancy` |


**Metric.** `train/t_rollout_s` + `train/t_train_fwd_bwd_s` = instrumented wall-clock **per GRPO step** (not cumulative). Score/advantage/optimizer/sync add <1s combined.

**Raw data.** All 10 steps × 2 runs exported to `[efficiency/grpo_smoke_h200_vs_b200_times.csv](./efficiency/grpo_smoke_h200_vs_b200_times.csv)` (pulled via `main/.venv/bin/python` + wandb API).

**Per-step rollout + train (s):**


| step | B200 | H200 |
| ---- | ---- | ---- |
| 0    | 137  | 190  |
| 1    | 113  | 147  |
| 2    | 134  | 202  |
| 3    | 121  | 183  |
| 4    | 86   | 163  |
| 5    | 127  | 211  |
| 6    | 132  | 201  |
| 7    | 91   | 159  |
| 8    | 100  | 205  |
| 9    | 93   | 212  |


**Medians (rollout + train):**


| window                  | B200 | H200 | ratio |
| ----------------------- | ---- | ---- | ----- |
| steps 0–9               | 117s | 196s | 1.7×  |
| steps 1–9 (drop cold 0) | 113s | 201s | 1.8×  |
| steps 4–9 (steady)      | 97s  | 203s | 2.1×  |


**Decomposition (median, steps 0–9):** rollout B200 **58s** vs H200 **89s**; train B200 **60s** vs H200 **105s**. `n_kept` similar (~136–208); H200 slowness is not explained by keeping more sequences alone.

**Verdict.**

- On this 10-step GRPO smoke, **B200 is ~2× faster per step** than H200 in steady state (~97s vs ~203s median, steps 4–9).
- Train phase drives most of the gap; rollout is also faster on B200.
- **Do not** use long H200 production step times (~180–200s) as the H200 baseline for GPU A/B — use this paired smoke pair or re-run a fresh B200/H200 smoke after stack changes.
- **$/step at list rates** still slightly favors H200 on paper, but paired smokes show B200 often wins **both** wall-clock and $ when step time is ~~2× shorter — aligns with time-first preference (~~25% higher $/s acceptable).

---

## 2026-05-27 (Wednesday) — minority_answer B200 smoke: interim step-time (steps 0–2)

**Context.** While the B200 **10-step checkpoint smoke** was still running, pulled early step-time from the first three logged steps to sanity-check the ~2× GRPO budget story (**minority smoke: ~2× step time** and **GRPO smoke: H200 vs B200** above). The B200 run below is **not** a production full train — it is the same smoke that later finished as **B200 training infra validated** (10 steps, `launch_mode=smoke`).

**Runs (partial pull).**


| GPU  | wandb                                                                            | notes                                                                    |
| ---- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| B200 | `[wdl3fczm](https://wandb.ai/224r-project/cs224r-minority-voting/runs/wdl3fczm)` | **10-step smoke** (ckpt-resume fresh phase); steps 0–2 only at pull time |
| H200 | `[w9z6boek](https://wandb.ai/224r-project/cs224r-minority-voting/runs/w9z6boek)` | parallel H200 smoke (in flight at pull time)                             |


Same train stack as GRPO smoke (`batch_size=64`, `n_rollouts=8`, `token_budget=105k`, allowlist clustering) but arm `minority_answer` → set-RL `keep_mask` keeps prompts with ≥2 answer clusters, not GRPO mixed-reward filter.

**Steps 0–2 — rollout + train (s/step):**


| step | B200 roll | B200 train | **B200 total** | H200 roll | H200 train | **H200 total** |
| ---- | --------- | ---------- | -------------- | --------- | ---------- | -------------- |
| 0    | 61        | 223        | **284**        | 86        | 306        | **392**        |
| 1    | 55        | 207        | **261**        | 86        | 304        | **391**        |
| 2    | 54        | 191        | **245**        | 92        | 303        | **395**        |


**Shared shape (all three steps):** `n_kept_sequences = 512` (64×8, no prompts collapsed), `fraction_filtered = 0%`, `num_chunks = 5`. Train cost is dominated by training **every** rollout — not the ~~160 kept on GRPO (~~70% filtered).

**Vs GRPO smoke (step 2, same day):**


|                 | B200      | H200      |
| --------------- | --------- | --------- |
| minority_answer | 245s      | 395s      |
| GRPO smoke      | 134s      | 202s      |
| ratio           | **~1.8×** | **~2.0×** |


Matches the earlier minority-smoke vs GRPO-full observation: rollout time is similar; the extra wall-clock is almost all **HF train** on 512 sequences.

**Vs prior minority smoke on H200** (`[q6m0tmiu](https://wandb.ai/224r-project/cs224r-minority-voting/runs/q6m0tmiu)`, steps 0–2): 430 / 318 / 384s — current H200 leg (~392 / 391 / 395s) is the same ballpark, slightly more stable across steps 0–2.

**GPU A/B on minority (steps 0–2):** B200 **~1.5× faster** than H200 (~~260–285s vs ~390–395s), same pattern as GRPO smoke — mostly **train** (B200 191–223s vs H200 303–306s), rollout also cheaper on B200 (~~54–61s vs 86–92s). B200 train time **trended down** 223 → 207 → 191s (three points only; may be warmup).

**Rough $/step (step 2, Modal list rates):** B200 245s × $0.001736 ≈ **$0.43**; H200 395s × $0.001261 ≈ **$0.50** — at these early steps B200 wins both wall-clock and $ despite higher $/s.

**Verdict / watch.**

- Early vibe confirms budget: plan for **~4–6 min/step on H200**, **~4 min on B200** while `n_kept ≈ 512` and `fraction_filtered ≈ 0`.
- Revisit $/epoch once `fraction_filtered` rises (policy collapses to one wrong answer per prompt → `n_kept` drops → train approaches GRPO timing).
- Too few points to call steady state at pull time; **full 10-step fresh + resume** recorded in **B200 training infra validated** below.

---

## 2026-05-27 (Wednesday) — B200 training infra validated (smoke ladder green)

**Context.** Optional B200 bring-up from 05-26 (`B200_migration_plan.md`, ~1 hr budget) became a full **opt-in GPU flag** path (`--gpu-class h200|b200`) so H200 stays the default fallback while B200 is tested. Goal: same collocated train+vLLM stack (FA2, `token_budget=105k`, set arms, checkpoint/resume, self-spawn) on Modal **B200 (~178 GB usable VRAM)** without changing training semantics.

**Verdict (end of day).** Full B200 bring-up ladder is **green**: probe smokes (vLLM, FA2, weight sync) plus **GRPO** and **minority_answer** each passed **10-step fresh → checkpoint at step 9 → resume to step 10+** on `chicken602`. Safe to launch production training with `--gpu-class b200` when wall-clock matters.

**Infra shipped (same day).**

- `train_remote_h200` / `train_remote_b200` Modal entrypoints; `launch_train.sh --gpu-class`.
- `train_real_b200.yaml` overlay (`extends: train_real.yaml`, `modal_price_per_sec: 0.001736`).
- `load_cfg()` recursive `extends:` merge (fixed path: `extends: train_real.yaml`, not `configs/...`).
- Probe launchers + smokes: vLLM generate, FlashAttention, HF→vLLM weight sync — each with `_h200` / `_b200` entrypoints.
- `launch_smoke_ckpt_resume.sh` — 10-step fresh (ckpt at step 9) + resume to ≥step 10 for GRPO and `minority_answer`.
- Image pin: `transformers<4.54.0` (unblocks `vllm==0.9.0` / `aimv2` AutoConfig clash).

**B200 smoke gates (all passed).**


| Gate                                     | Modal / wandb                                                                                                                 | Result                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| vLLM generate                            | B200 probe smoke                                                                                                              | ✅                        |
| FlashAttention + collocated HF           | B200 probe smoke                                                                                                              | ✅                        |
| HF → vLLM weight sync                    | B200 probe smoke                                                                                                              | ✅                        |
| GRPO 10-step fresh                       | `ap-FeIG4QuMsmkYjIXD3093op`, `[1hg8fs5u](https://wandb.ai/224r-project/cs224r-minority-voting/runs/1hg8fs5u)`                 | ✅ exit 0                 |
| GRPO resume from `step_000009.pt`        | `ap-v3tF5WQ3iz1HUOr2UaqHdF`, `[jg92ywy3](https://wandb.ai/224r-project/cs224r-minority-voting/runs/jg92ywy3)`                 | ✅ exit 0 (~3 min)        |
| GRPO 10-step fresh + resume (H200)       | `ap-x8GRQv1x` / `ap-ZM4xcvn22gvj3XyV8bqmRj`, `[5sekbfnq](https://wandb.ai/224r-project/cs224r-minority-voting/runs/5sekbfnq)` | ✅ (paired baseline)      |
| **minority_answer 10-step fresh (B200)** | `ap-VlVMq3eC1g4TsmhDRRaLYU`, `[wdl3fczm](https://wandb.ai/224r-project/cs224r-minority-voting/runs/wdl3fczm)`                 | ✅ **finished**, 10 steps |
| **minority_answer resume (B200)**         | `ap-pN5RJ8dlBOkL46brMo8OjR` (chicken602), W&B [`w9z6boek`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/w9z6boek) resumed | ✅ exit 0 (~8.8 min); `_step=10` |


**Resume gotcha (minority).** First resume attempt used Modal workspace **`alee72`** (profile `anastasia`) while `step_000009.pt` lives on **`chicken602`** — volumes are not shared, so that run could not load the fresh ckpt. Relaunch after `modal profile activate chicken602` succeeded.

**minority_answer B200 — fresh (`wdl3fczm`).**

- W&B: `state=finished`, tags `gpu_class=B200`, `launch_mode=smoke`, `total_steps=10`.
- Shape: `n_kept_sequences=512`, `fraction_filtered=0`, `num_chunks=5` (expected for set-arm early training).
- Timing (last step): `t_rollout_s≈58`, `t_train_fwd_bwd_s≈208` → **~266 s/step** (~4.4 min); see interim entry above for H200 A/B at steps 0–2.
- VRAM peak ~148 GB / 178 GB device.
- Wrote `/vol/checkpoints/train_minority_answer/step_000009.pt` on **chicken602** volume.

**minority_answer B200 — resume (`ap-pN5RJ8dlBOkL46brMo8OjR`).**

- Loaded `step_000009.pt`, ran one post-ckpt step, exited cleanly.
- W&B resumed existing run `w9z6boek` (wandb id stored in checkpoint) through `_step=10`.

**Decision — B200 is production-ready for training launches.**

- Use `**bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm <grpo|minority_answer|...>`** (or explicit `train_real_b200.yaml`) when calendar time matters; keep `**--gpu-class h200**` as the safe default until an operator opts in.
- Paired GRPO smokes: **~2× faster per step** on B200 vs H200 in steady state; minority smokes show the same GPU A/B pattern at ~1.5–1.8× with full `n_kept`.
- **Not a wiring experiment** — same trainer, arms, clustering allowlist, and checkpoint format as H200; only SKU + yaml pricing overlay differ.

**Still open (non-blocking).**

- H200 minority 10-step fresh/resume smokes (if not already finished).
- B200 **efficiency** matrix (`vllm_sleep`, `gc_off`, higher `token_budget`) — see [`reference/efficiency/B200_efficiency_smoke_plan.md`](./reference/efficiency/B200_efficiency_smoke_plan.md); separate from bring-up gates.

**Operator note.** Resume smokes must use the **same Modal workspace** as the fresh leg that wrote the checkpoint (`modal profile current` before `launch_smoke_ckpt_resume.sh --phase resume`).

**Cross-refs.** Runbook: `[efficiency/B200_build_notes.md](./efficiency/B200_build_notes.md)`; audit: `[efficiency/B200_readiness_audit.md](./efficiency/B200_readiness_audit.md)`; GRPO A/B CSV: `[efficiency/grpo_smoke_h200_vs_b200_times.csv](./efficiency/grpo_smoke_h200_vs_b200_times.csv)`.

---

## 2026-05-27 (Wednesday) — pre–full-run audit triage (session)

**Context.** Both arms (`grpo`, `minority_answer`) code-ready; B200 smokes green (~2× GRPO step time vs H200). Before launching full runs on B200, reconciled scattered audits ([Critical pass P0/P1](a427c82c-8155-4a9f-8514-e179befcb183) + [Pre-launch readiness](71068bb3-fac3-4c13-87e9-f502f284e1c2); no single checklist doc — also `issues.md`, `B200_readiness_audit.md`, `train_wandb_metrics_verdict.md`).

**Launch gate (operator).** Per arm: `bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm <grpo|minority_answer>`; default config `train_real_b200.yaml`. Sanity on W&B: finite loss, sensible `fraction_filtered`, checkpoints written.

**Decisions this session.**

| Topic | Decision |
| --- | --- |
| **`n_kept` cap / subsample** | **No** — keep full set-arm `n_kept` (~512); accept ~2× step cost vs GRPO; do not change objective for schedule. |
| **Checkpoint `weights_only=False` (critical-pass P0 #1)** | **Defer** — pickle/RCE risk only if ckpt file is untrusted; not a training-correctness issue; fix later with base64 RNG + `weights_only=True`. |
| **Critical-pass P0 #2–3, P1 #5–6c** | Already landed (OOM log, Phase-2 rollout assert, `extends` cycles, `_extract_old_logprobs` tests). |
| **Solo fixes queued (not started here)** | `hash()` → `hashlib` in clustering; `mean_unique_clusters_kept`; set-arm error copy; `n_rollouts==8` assert; `--gpu-class` vs yaml preflight. |
| **Config validator (P1 #4)** | **Small preflight landed** — `preflight_train_launch.py` + [`launch_training.md`](./launch_training.md) for agent commands. |
| **Weight-sync automated test (P1 #6a)** | **Modal smoke is the gate** (`launch_smoke_weight_sync.sh` passed on B200 bring-up); CPU unit test stays skip until someone runs spike. |
| **Zero-advantage trainer (P1 #6b)** | **Done** — all-filtered batch **skips** step (`skipped_no_kept` on W&B); `test_trainer_zero_kept.py`. |

**Closeout.** Verified in-loop weight sync on B200 via 5-step GRPO smoke W&B run [`fh63ww4z`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/fh63ww4z): `train/weight_sync_s` present on steps 0–4 (nonzero ms-scale), no sync errors.

**Still open (non-blocking).** Batched logprobs / B200 efficiency ablations (science-neutral speed, not correctness). B200 `minority_answer` resume smoke **done** — see **B200 training infra validated** entry above.

---

## 2026-05-27 (Wednesday) — `poly_epo_answer` full B200 launch (chicken602)

**Context.** Stretch arm 4 shares the set-RL kernel with `minority_answer` (answer-hash clusters; `f(G) = mean(r)·distinct_clusters/4`). No new trainer code beyond existing `poly_epo_answer` dispatch — only config + launch.

**Credit / workspace.** First submit briefly targeted **alee72** (`anastasia` profile); stopped **`ap-Iug2ChLptNCAYybkbrlLqQ`** (0 tasks, never trained). Relaunched on **chicken602** (Nancy credits).

| Item | Value |
|------|--------|
| Arm | `poly_epo_answer` |
| Modal workspace | **chicken602** |
| Modal app | [`ap-rzTnv1IwgUhcbqeNas4lRY`](https://modal.com/apps/chicken602/main/ap-rzTnv1IwgUhcbqeNas4lRY) |
| Config | `main/configs/train_real_b200_fresh_poly_epo.yaml` (`extends: train_real_b200.yaml`) |
| Checkpoint dir | `/vol/checkpoints/train_poly_epo_answer_b200/` |
| W&B group | `train-poly-epo-answer` (fresh run via `--fresh-wandb`) |

**Launch (replay, chicken602):**

```bash
modal profile activate chicken602
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm poly_epo_answer \
  --config main/configs/train_real_b200_fresh_poly_epo.yaml --no-resume --fresh-wandb
```

**Parallel production on alee72 (unchanged).** B200 GRPO `ap-VBmgTVFefkECyZa0r52RMb`, minority `ap-3Acz8FrtQY4D4ubqkzJ4jB` — see [`handoff/b200_production_launch_2026-05-27.md`](./handoff/b200_production_launch_2026-05-27.md).

**Monitor:** `main/.venv/bin/modal app logs ap-rzTnv1IwgUhcbqeNas4lRY -f` (profile `chicken602`).

---

## 2026-05-27 (Wednesday) — Modal credit budget snapshot (B200 prod)

W&B `_runtime × modal_price_per_sec` on active prod runs; **799 steps = 1 epoch**. Modal volumes are per-workspace (`chicken602` ≠ `alee72`).

### `chicken602` (Nancy) — **$381 credits now**

| Run | Step | Calibrated $/step | min/step |
|-----|------|-------------------|----------|
| B200 `poly_epo_answer` [`fdx95beu`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/fdx95beu) | ~146 / 799 | **~$0.46** | **~4.2** |

**Spent (May 25+ on this volume, W&B):** ~$201 (incl. H200 GRPO $40, ablations/smokes ~$55, poly ~$66). **H200 GRPO ckpts:** `checkpoints/train_real/` through `step_000239.pt`.

**To finish:** ~$303 more → **1 epoch** poly · ~$674 more → **2 epochs**.

**Surplus / shortfall (vs $381 now):** **~+$78** after 1 epoch poly · **~−$293 short** for 2 epochs.

### `alee72` (Anastasia) — **~$650 starting credits**

| Run | Step | Calibrated $/step | min/step |
|-----|------|-------------------|----------|
| B200 GRPO [`t11jct0t`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/t11jct0t) | ~326 / 799 | **~$0.22** | **~2.1** |
| B200 `minority_answer` [`o5ypkzja`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/o5ypkzja) | ~148 / 799 | **~$0.43** | **~4.1** |
| H200 GRPO [`pcas3emd`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/pcas3emd) (crashed) | 537 | ~$0.25 | ~3.1–3.3 |

**Spent (W&B on this volume):** ~$231 (H200 GRPO ~$93 + B200 GRPO/minority ~$69 each) + in-flight 4×B200 checkpoint eval (~$40–80, not in W&B). **Implied balance:** ~$340–420 left from $650.

**To finish (GRPO + minority):** ~$383 more → **1 epoch** both · ~$900 more → **2 epochs** both.

**Surplus / shortfall (implied ~$380 mid):** **~break-even to ~−$45 short** for 1 epoch · **~−$520 short** for 2 epochs.

### Combined (both workspaces)

**1 epoch all three B200 arms:** ~$686 more needed vs ~$761 credits (~$380 + $381) → **~+$75** team slack (tight; eval overhead can erase). **2 epochs all three:** ~$1,574 needed → **~−$813 short**.

**Ref:** H200 GRPO was ~3.1–3.3 min/step; B200 GRPO ~2.1 min/step at ~$0.22/step. Set-arms ~4.1 min (~$0.43–0.46/step) dominate spend. Latest H200 ckpt on alee72: `step_000529.pt`.

---

## 2026-05-27 (Wednesday) — B200 three-arm checkpoint eval (canonical)

**Canonical entry** for **base vs trained** comparisons across **GRPO**, **minority_answer**, and **poly_epo_answer** on fixed eval slices (May 27, 2026). Use this section for writeups / agent context — not the scattered run logs elsewhere in this file.

**Context.** Mid-training on B200 (~**step 299** GRPO, ~**step 133** minority/poly at Polaris/DAPO launch; BeyondAIME used **later** ckpts — see table). Live W&B looked flat; needed decision-grade **pass@k** on training distribution + OOD. Model: **Qwen3-1.7B-Base**; harness: [`checkpoint_rollout_eval.py`](../probes/checkpoint_rollout_eval.py) (HF load ckpt → vLLM weight sync → rollouts → train grader).

**Training checkpoints evaluated**

| Arm | Label | Checkpoint path (alee72 unless noted) | ~Training progress |
| --- | --- | --- | --- |
| — | `base` | `Qwen/Qwen3-1.7B-Base` (no ckpt) | — |
| GRPO | `grpo_b200_s299` | `/vol/checkpoints/train_real_b200/step_000299.pt` | ~299 / 799 |
| minority_answer | `minority_b200_s133` | `/vol/checkpoints/train_minority_answer_b200/step_000133.pt` | ~133 / 799 |
| poly_epo_answer | `poly_epo_b200_s133` | `/vol/checkpoints/train_poly_epo_answer_b200/step_000133.pt` | ~133 / 799 (synced from chicken602) |
| GRPO (BeyondAIME only) | `grpo_b200_s359` | `.../train_real_b200/step_000359.pt` | newer than Polaris/DAPO row |
| minority (BeyondAIME only) | `minority_b200_s159` | `.../train_minority_answer_b200/step_000159.pt` | newer than Polaris/DAPO row |

**Workspace.** GRPO + minority train on **alee72**; poly trains on **chicken602** (poly ckpt copied to alee72 for cross-arm eval).

### Polaris 2k — training distribution

- **Run:** `20260527T213611Z` · Modal `ap-E5xvFQaZCRV7vMn4b750DS` (~49 min)
- **Config:** `checkpoint_eval_2k_polaris_arms_latest_b200.yaml`
- **Slice:** 2000 prompts from `polaris_train.jsonl`, seed **42**
- **Prompt:** `hybrid_answer_boxed` (arm C, matches train)
- **Rollouts:** 8 / prompt · train grader (mathd ∨ sympy, Rank-2)

| Variant | pass@8 | Δ vs base | frac 0/8 correct |
| ------- | ------ | --------- | ---------------- |
| **base** | **0.306** | — | 0.694 |
| grpo_b200_s299 | 0.317 | +1.1 pp | 0.683 |
| minority_b200_s133 | 0.307 | +0.1 pp | 0.693 |
| poly_epo_b200_s133 | 0.319 | +1.3 pp | 0.682 |

**Readout:** No collapse on train distribution. **Minority flat** (~noise). GRPO/poly **+1–1.3 pp** — same ballpark as H200 GRPO-only best (+1.8 pp at step 99 on same slice; see [precursor entry](#2026-05-27-wednesday--grpo-checkpoint-slice-eval-h200-only-precursor)).

### DAPO 2k — easier OOD

- **Run:** `20260527T203133Z` · Modal `ap-VsJNlyGdseXSWmByy8OiO1`
- **Config:** `checkpoint_eval_ood_aime_dapo_arms_latest_b200.yaml`
- **Slice:** 2000 prompts, seed **43**
- **Prompt:** `dapo_answer_v1` (≠ train prompt)
- **Rollouts:** 8 / prompt

| Variant | pass@8 | Δ vs base | mean_reward |
| ------- | ------ | --------- | ----------- |
| **base** | **0.248** | — | 0.051 |
| grpo_b200_s299 | **0.272** | **+2.4 pp** | 0.056 |
| minority_b200_s133 | 0.252 | +0.5 pp | 0.052 |
| poly_epo_b200_s133 | 0.263 | +1.5 pp | 0.055 |

**Readout:** **GRPO** only arm with a clear (still small) OOD gain. Minority **flat**.

> **Prompt confound (flag for redo):** Eval used `dapo_answer_v1`, but trained arms use `hybrid_answer_boxed` (arm C). Rank-2 parser still catches `\boxed{}` as a fallback so grading is fair, but rollout *generation behavior* under a non-train prompt is unknown — model has internalized the train template and may produce different reasoning/stop behavior here. Direction of bias is unclear: GRPO's +2.4 pp could be larger under fair prompt, or could shrink. Queued for rerun with `hybrid_answer_boxed`.

### BeyondAIME — hard OOD

- **Run:** `20260527T221956Z` · Modal `ap-Vl16FgmiDkRIgUxtv909Ce` (~6 min, 4× B200)
- **Config:** `checkpoint_eval_beyondaime_pass16_arms_latest_b200.yaml`
- **Slice:** 100 problems, seed **42**
- **Prompt:** `dapo_answer_v1` (≠ train prompt)
- **Rollouts:** 16 / prompt · pass@16 primary metric

| Variant | pass@1 | pass@4 | pass@8 | pass@16 | Δ pass@16 |
| ------- | ------ | ------ | ------ | ------- | --------- |
| **base** | 0.009 | 0.035 | **0.068** | **0.130** | — |
| grpo_b200_s359 | 0.005 | 0.020 | 0.038 | 0.070 | **−6.0 pp** |
| minority_b200_s159 | 0.005 | 0.020 | 0.038 | 0.070 | **−6.0 pp** |
| poly_epo_b200_s133 | 0.005 | 0.020 | 0.040 | 0.080 | **−5.0 pp** |

(pass@4 / pass@8 recomputed from histograms in saved partials; harness now logs `pass_at_{1,4,8,16}_mean` on future runs.)

**Readout:** **Base beats every trained arm at every k** — clearest negative result. All arms regress similarly (not minority-specific). Likely mix of **real hard-OOD gap** and **eval prompt ≠ train prompt**; rollout qualitative check queued.

> **Prompt confound (flag for redo):** Same issue as DAPO 2k above — eval prompt `dapo_answer_v1` ≠ train prompt `hybrid_answer_boxed`. With base pass@16 only ~13%, even a small format-induced perturbation could explain the sign flip. **This is the most important eval to rerun with the matching prompt before believing the regression.** If trained arms still regress under fair prompt → real hard-OOD capability loss (paper-worthy). If they recover → it was the template.

### AIME-25 — exploratory only

- **Run:** `20260527T211739Z` · `checkpoint_eval_ood_aime_only_arms_latest_b200.yaml`
- **Slice:** 30 problems · `dapo_answer_v1` · 8 rollouts

| Variant | pass@8 | Δ vs base |
| ------- | ------ | --------- |
| base | 0.033 | — |
| grpo_b200_s299 | 0.033 | 0.0 |
| minority_b200_s133 | 0.067 | +3.3 pp (~1 problem) |
| poly_epo_b200_s133 | 0.000 | −3.3 pp |

**Not decision-grade** (n=30).

### Cross-slice summary

| Eval slice | minority Δ | GRPO Δ | poly Δ | Use in writeup |
| ---------- | ---------- | ------ | ------ | -------------- |
| **Polaris 2k** pass@8 | +0.1 pp | +1.1 pp | +1.3 pp | Primary (on-distribution) |
| **DAPO 2k** pass@8 | +0.5 pp | **+2.4 pp** | +1.5 pp | Primary (easier OOD) |
| **BeyondAIME** pass@16 | −6.0 pp | −6.0 pp | −5.0 pp | Primary (hard OOD; note prompt) |
| AIME-25 pass@8 | +3.3 pp | 0 | −3.3 pp | Exploratory only |

### Verdicts (project-level)

1. **Training is not obviously broken** — Polaris pass@8 does not regress; optimization was stable in W&B (`ratio_max` &lt; 3, low clip, sane grad norms).
2. **minority_answer hypothesis not supported** at these ckpts — flat on Polaris and DAPO; no win vs GRPO decision-grade.
3. **GRPO** — small Polaris lift; best OOD signal on DAPO; **regresses vs base on BeyondAIME** (with newer ckpts).
4. **poly_epo_answer** — similar to GRPO on Polaris/DAPO; slightly better on BeyondAIME pass@16 but still **far below base**.
5. **Do not chase Poly-EPO Fig. 2 curves** on 1.7B / this stack — gains are ~1–2 pp, within eval noise; see PLAN success criteria / consolation diversity path.

### Follow-ups

| Item | Status |
| ---- | ------ |
| **BeyondAIME with `hybrid_answer_boxed` (fair OOD vs train prompt)** | **Done** — `ap-yHQQhuvFOUmQ7nrJC0eGNY`, run `20260528T023940Z`; local [beyondaime_hybrid_summary_20260528T023940Z.json](../data/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/beyondaime_hybrid_summary_20260528T023940Z.json). |
| **DAPO 2k with `hybrid_answer_boxed` (fair OOD)** | **Done** — `ap-5JsNBqCp1k6hqMS4RAenQy`, run `20260528T024146Z`; local [dapo_hybrid_summary_20260528T024146Z.json](../data/probes/checkpoint_eval_2k_dapo_hybrid_arms_latest/dapo_hybrid_summary_20260528T024146Z.json). |
| BeyondAIME rollouts + `analyze_beyondaime_rollouts.py` (`parse_ok`, `extract_path`) | **Done (recomputed + persisted)** — rollout dump `rollouts_20260527T223115Z/`; durable artifacts in `main/data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/qualitative_20260527T223115Z/` (`qualitative_summary.md`, `qualitative_metrics.json`, per-arm logs). Recompute keeps the same conclusion: `parse_ok` stays close to base across arms, `base_pass16_trained_fail > trained_only`, and many fails are parsed-but-wrong (reasoning-quality errors). |
| More training epochs | Deprioritized (budget; flat marginal returns) |

### Artifacts (local JSON = source of truth)

| Slice | Volume dir | Local summary |
| ----- | ---------- | ------------- |
| Polaris 2k | `/vol/probes/checkpoint_eval_2k_polaris_arms_latest/20260527T213611Z/` | [polaris_summary_20260527T213611Z.json](../data/probes/checkpoint_eval_2k_polaris_arms_latest/polaris_summary_20260527T213611Z.json) |
| DAPO + AIME | `/vol/probes/checkpoint_eval_ood_aime_dapo_arms_latest/` (`20260527T203133Z`, `20260527T211739Z`) | partials under `partials/dapo/`, `results.json` |
| BeyondAIME | `/vol/probes/checkpoint_eval_beyondaime_pass16_arms_latest/20260527T221956Z/` | [beyondaime_summary_20260527T221956Z.json](../data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/beyondaime_summary_20260527T221956Z.json) |

**Launch replay:**

```bash
bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_2k_polaris_arms_latest_b200.yaml --detach
bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_ood_aime_dapo_arms_latest_b200.yaml --detach
bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_beyondaime_pass16_arms_latest_b200.yaml --detach
```

---

## 2026-05-27 (Wednesday) — BeyondAIME rollout qualitative recompute (persisted)

Re-ran `main/scripts/analyze_beyondaime_rollouts.py` on `main/data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/rollouts_20260527T223115Z/` for:

- `grpo_b200_s359`
- `minority_b200_s159`
- `poly_epo_b200_s133`

Saved durable outputs under `main/data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/qualitative_20260527T223115Z/`:

- `qualitative_summary.md` (human-readable counts + checks)
- `qualitative_metrics.json` (machine-readable metrics, overlap counts, PID sets)
- `*_analysis.log` (raw analyzer stdout per arm)

Result is directionally unchanged from prior notes: parsing quality is similar between base/trained, `base_pass16_trained_fail` exceeds `trained_only` in every arm, and failure tails remain dominated by parsed-but-wrong reasoning rather than parser collapse.

## 2026-05-27 (Wednesday) — eval prompt-mismatch confound flagged; rerun spec

**Issue.** The canonical three-arm eval above used **`dapo_answer_v1`** for both **BeyondAIME** and **DAPO 2k**, but all three arms were trained with **`hybrid_answer_boxed`** (arm C — `Answer: \boxed{N}`, 90% boxed compliance after training). Trained checkpoints have been pushed toward boxed-first output; grading them under a prompt that elicits `Answer:`-line format under-counts correct rollouts whenever the boxed fallback fails. This is the most likely contributor to the **−5–6 pp pass@16 sign-flip vs base on BeyondAIME** and partially confounds the **GRPO +2.4 pp** on DAPO 2k. **Polaris 2k is unaffected** — it already used `hybrid_answer_boxed` (matches train). Do not cite BeyondAIME / DAPO 2k regressions in the writeup until reruns land.

**Rerun spec (priority order; ~$80–120 total).**

1. **BeyondAIME pass@16 rollout dump — diagnostic first** (`checkpoint_eval_beyondaime_pass16_rollouts_b200.yaml`, was queued). **Completed; superseded by persisted recompute artifacts** in `main/data/probes/checkpoint_eval_beyondaime_pass16_arms_latest/qualitative_20260527T223115Z/`. Directional readout: (b) `parse_ok=true ∧ reward=0` dominates over parser-collapse-only failures.
2. **BeyondAIME pass@16 with `hybrid_answer_boxed`** — new config `checkpoint_eval_beyondaime_pass16_hybrid_arms_latest_b200.yaml` (copy from `..._pass16_arms_latest_b200.yaml`, flip `prompt_variant`). Same 100-problem slice, seed 42, 16 rollouts/prompt, train grader. Same three checkpoints as canonical entry (`grpo_b200_s359`, `minority_b200_s159`, `poly_epo_b200_s133`). **Decision-grade for writeup**; ~$40–60 (4× B200, ~6 min).
3. **DAPO 2k pass@8 with `hybrid_answer_boxed`** — new config `checkpoint_eval_2k_dapo_hybrid_arms_latest_b200.yaml`. Same 2000-problem slice, seed 43, 8 rollouts/prompt. **Same three checkpoints** as canonical (`grpo_b200_s299`, `minority_b200_s133`, `poly_epo_b200_s133`). ~$40–60.
4. **Diversity panel (offline, ~$0)** — compute `unique_answer_clusters_correct` per prompt on the Polaris 2k saved rollouts (`/vol/probes/checkpoint_eval_2k_polaris_arms_latest/20260527T213611Z/`). Supports the PLAN consolation criterion (minority matches GRPO on pass@1 *and* improves diversity) without spending compute. Add to canonical eval entry as a new sub-section once computed.

**Explicitly not doing (and why).** No 4th arm (`minority_CoT` / `poly_epo_CoT`) — the headline minority hypothesis is flat on the fair-prompt eval (Polaris 2k, +0.1 pp); the cluster substrate (answer-hash vs LLM-judge CoT) is not what's failing. The failure mode is **low signal density at 1.7B** (66% all-wrong prompts, few correct rollouts to cluster over). CoT arms require the unimplemented judge sidecar (~2× inference GPU) and would hit the same wall. No 2nd epoch on existing arms — slopes are shallow on flat curves; budget better spent on (1)–(3) above + diversity. No train-data switch to DAPO — mentor-blessed, sunk-cost loss, and arm C closed the gap to ~1 pp vs DAPO pilot on n800.

**Status updates required after reruns.**

- Update **canonical entry** above (§ `2026-05-27 (Wednesday) — B200 three-arm checkpoint eval (canonical)`) BeyondAIME and DAPO 2k tables in-place; add a `prompt: hybrid_answer_boxed` row alongside the existing `dapo_answer_v1` row (do not delete the old numbers — they're the apples-to-apples reference for the prompt-mismatch effect size).
- Update **`ta_discussion.md` §1** "Where we are" table with fair-prompt BeyondAIME Δ.
- If BeyondAIME regression survives the fair-prompt rerun: that's the headline negative result, frame in writeup as "base &gt; trained on hard OOD at 1.7B even under matched prompt" (see Q2(b) in `ta_discussion.md`).

**Rerun status (landed 2026-05-28 ~03:30 UTC / 2026-05-27 ~20:30 PDT):**

- BeyondAIME fair-prompt (`ap-yHQQhuvFOUmQ7nrJC0eGNY`, run 19:38–19:46 PDT) **finished**. `results.json` at `/vol/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/20260528T023940Z/results.json` (alee72 / `main-artifacts` volume); mirror at `/tmp/beyondaime_hybrid_results.json`.
- DAPO 2k fair-prompt (`ap-5JsNBqCp1k6hqMS4RAenQy`, run 19:40–20:29 PDT) **finished**. `results.json` at `/vol/probes/checkpoint_eval_2k_dapo_hybrid_arms_latest/20260528T024146Z/results.json`; mirror at `/tmp/dapo_hybrid_results.json`.

**Fair-prompt deltas vs base (`hybrid_answer_boxed`, n_rollouts=16/8, train-grader):**

| Slice | base | GRPO Δ | minority Δ | **poly_epo Δ** |
|-------|------|--------|------------|-----------------|
| BeyondAIME pass@16 (n=100, hard OOD) | 0.070 | +1.0 pp | +1.0 pp | **+5.0 pp** |
| DAPO 2k pass@8 (n=2000, easier OOD) | 0.313 | −0.5 pp | −0.65 pp | **+0.8 pp** |
| Polaris 2k pass@8 (n=2000, train-dist) ‡ | 0.306 | +1.1 pp | +0.1 pp | **+1.3 pp** |

‡ Polaris 2k was already `hybrid_answer_boxed`; included for cross-slice synopsis. Full delta-summary: [`fair_prompt_eval_summary_2026-05-27.md`](../data/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/fair_prompt_eval_summary_2026-05-27.md).

**Headline.** The BeyondAIME −5 to −6 pp regression was **entirely a prompt artifact**. Under matched prompts, the trained arms are at-or-above base on hard OOD, and **`poly_epo_answer` is the best arm on all three slices**. The prompt-mismatch effect size was +7–10 pp on BeyondAIME (inflating base, deflating trained) and −1–3 pp on DAPO 2k (the opposite direction).

**Significance.** No individual Δ is significant at >1σ given slice sizes (BeyondAIME n=100 gives SE≈0.04 on pass@16; DAPO 2k n=2000 gives SE≈0.015). The load-bearing claim is **across-slice consistency**: poly_epo best 3/3, minority worst 2/3. A diversity panel on saved Polaris 2k rollouts would add an orthogonal axis at $0.


**Supersedes** the May 27 scattered eval narrative in this file (merged here). **Related:** [H200 GRPO-only Polaris learning curve](#2026-05-27-wednesday--grpo-checkpoint-slice-eval-h200-only-precursor) (multi-step GRPO, not three-arm).

## 2026-05-27 (Wednesday late) — structural diagnosis: model/data mismatch, signal-density benchmark, LR first principles

> **Why this entry exists.** All three arms underperformed on a fair-prompt evaluation (Polaris 2k: GRPO +1.1 pp, minority +0.1 pp, poly +1.3 pp at ~1/6–1/3 epoch; BeyondAIME pass@16 −5 to −6 pp across arms before the prompt-mismatch confound was found). The prompt-mismatch entry above explains BeyondAIME; it does **not** explain why minority/poly hypotheses are flat on the fair-prompt slice (Polaris 2k). This entry is the deeper diagnosis: an independent audit of the training setup, a benchmark of our signal density against published baselines, a first-principles ranking of remediation options, and an honest analysis of the LR=3e-6 hedge that is currently running on `chicken602`. Length is deliberate — the next 24 h decides whether we retrain or own the negative result for the poster.

### 1. Independent audit (model / data / hyperparameter mismatch)

Spawned a separate audit agent without sharing our hypothesis-favoring framing. Key findings (sources and verification flags inline):

- **Polaris-Dataset-53K was calibrated by Deepseek-R1-distill-Qwen-7B**, not by a 1.7B model. The HKU NLP team's published recipe also **refilters 53K → ~30K specifically for their 4B model** by running rollout-pass-rate filtering and dropping always-solved + never-solved prompts before RL ([Polaris blog, HKU NLP](https://hkunlp.github.io/blog/2025/Polaris/); [ChenxinAn-fdu/POLARIS](https://github.com/ChenxinAn-fdu/POLARIS) — **verified by citation-check subagent**). We are running the unfiltered 51K on a model **2.3× smaller than the official refiltered recipe's target**, which is the canonical setup for the signal-starvation failure mode the Polaris team's own filtering step was designed to prevent. **Confidence: high** for the existence of the recipe; **confidence: medium** that the 4B → 1.7B gap is the single dominant cause (model-size scaling for refilter thresholds is not separately ablated in the public material).

- **~~BeyondAIME regression is mostly real, not purely a prompt artifact.~~ RETRACTED 2026-05-27 ~21:30 PDT.** The fair-prompt rerun (above) shows the regression is **entirely a prompt artifact**: under matched `hybrid_answer_boxed`, BeyondAIME pass@16 goes from {GRPO −6, minority −6, poly_epo −5} to {+1, +1, **+5**} pp vs base. The qualitative-recompute "parse_ok roughly comparable" read was correct on its face but I over-weighted it: parse_ok-equal does not imply downstream-correct-equal when the prompt format influences which reasoning paths the model commits to. **Lesson:** trust matched-prompt eval before trusting parser-quality proxies for end-to-end correctness.

- **Hyperparameters don't look broken per se.** N=8 rollouts/prompt, KL=0, LR=1e-6, batch_size=64 are all in the published-recipe range for GRPO on 1–4B models. The interaction that hurts us is **the combination of (unfiltered, mid-difficulty-for-7B) data × (small model) × (low N, fixed)**: the same recipe that worked at 4B produces a near-zero advantage signal at 1.7B because most groups land in {0/8, 8/8}.

### 2. Signal-density benchmark — we are in-range with unfiltered baselines, but every successful published recipe filters

Our `random_fullgold_n800` probe at base showed **34% mixed-reward density** (i.e., 34% of prompts have at least one correct and at least one incorrect rollout under N=8; 66% are degenerate). I went looking for a published target.

- **arxiv:2605.07689** ("Gradient Starvation in Binary-Reward GRPO," Nie et al. — **verified to exist**) reports a degeneracy rate of **0.69 at group size 4** on GSM8K with vanilla GRPO. That is **~31% productive groups**, which is essentially our 34% number. So our setup is **not pathologically broken** — it matches the published unfiltered baseline almost exactly.
- The same paper's headline result is that fixing the gradient-starvation degeneracy (their proposal: replace group-mean-centered advantage with `A = 2r − 1`) lifts GSM8K accuracy from **28.4% → 73.8%** at group size 4. This is consistent with our null result being a **signal-density problem, not an arm-hypothesis problem** — minority and poly_epo are set-based reweightings of an advantage that is mostly zero, so they have nothing to amplify or differentiate from GRPO.
- The standard fixes in the literature are (a) **dynamic rollout filtering** (Polaris, DAPO dynamic sampling, [arxiv:2605.05112 "Rollout Pass-Rate Control"](https://arxiv.org/abs/2605.05112) — **verified**), (b) **prompt replay** of medium-pass-rate prompts ([arxiv:2603.21177 "Prompt Replay"](https://arxiv.org/abs/2603.21177) — **verified**), and (c) **fixed-reference advantage** (Nie et al.). We are using (none of the above).

**Implication for the writeup.** The honest framing is **not** "our setup is uniquely broken." It is **"we matched the published unfiltered baseline at the model-size-extrapolated signal density, then skipped the filtering step that every successful published recipe at this regime applies."** That story is defensible; it also explains why the set-based-clustering arms can't beat GRPO at this scale.

### 3. First-principles fix ordering (why filtering > more epochs > … > LR bump)

Working from "what does the gradient actually see" rather than what's easiest to launch:

1. **Filter the dataset to medium-pass-rate prompts (best).** Run one rollout pass over 51K with N=8 on the base model, drop 0/8 and 8/8 prompts, retrain on the surviving ~17K (back-of-envelope: 34% × 51K ≈ 17.4K). This is the **only** option that increases the fraction of training steps with non-zero advantage — it directly attacks the root cause. Cost: one rollout pass over 51K ≈ same as ~2 training steps of compute (rollout-dominated), plus a fresh ~17K × 1 epoch run. ETA ~$200–300 + ~12–18 h wall on B200. **Confidence this works at our scale: medium-high** (matches Polaris's own recipe; matches Nie et al.'s "fix the degeneracy" finding empirically).
2. **More epochs on the same data (second-best, but limited by signal density).** Two epochs ≈ doubles the chance a flat-curve arm separates. But if 66% of gradient steps are zero, doubling steps still leaves you with most steps doing nothing. Helps the trained-model curve climb slightly but doesn't fix the structural mismatch. **Confidence: low-medium** that it moves minority/poly from "flat" to "above GRPO." Cost: ~$813 over current 1-epoch budget for all 3 arms.
3. **Curriculum (sort by base pass-rate, train easy→hard).** Same total data, but the early signal density is higher. Standard pre-DAPO trick. **Confidence: medium**, but doesn't reach the upper ceiling of (1) because hard-prompt steps still have zero gradient.
4. **Increase N (rollouts per prompt).** N=8 → N=16 halves the probability that a borderline-difficulty prompt lands at 0/N or N/N. But Bernoulli variance argument: doubling N only square-roots the chance of escaping degenerate groups for prompts where the model's true accuracy is far from 0.5. Costs 2× rollout time per step; on a fixed budget this means halving the number of update steps. **Confidence: low** that it dominates filtering.
5. **SFT cold start on Polaris solutions before RL.** Lifts base accuracy distribution upward, shifts more prompts into the mixed-reward band naturally. This is what most strong published recipes do (DeepSeek-R1, Polaris). Out of scope for our timeline (no SFT pipeline; ≥2 days of bring-up).
6. **Switch base model to a distilled / instruct variant** (e.g., Qwen3-1.7B-Instruct or a distillation of R1-distill-Qwen-7B). Same effect as SFT cold start. Also out of scope; would invalidate every prior probe.
7. **Higher LR.** See §4 below — this is what's actually running right now on `chicken602`. It is a **hack, not a structural fix**.

The Pareto-frontier choice in our time budget is **(1)** if we have the wall-clock and a willingness to spend ~$300, **(7)** as a cheap parallel hedge to see if there's any signal at all in the existing arms before committing to (1), and **own-the-null with the mismatch story** if both come back flat.

### 4. Honest LR analysis (what raising 1e-6 → 3e-6 can and cannot do)

The user pushed back, correctly, on the earlier audit's framing of LR=3e-6 as a "(B)-tier hedge fix." Reframing from first principles:

- **What higher LR does:** for the same advantage signal, takes a larger step in policy-parameter space per update. Amplifies whatever gradient is present.
- **What higher LR does NOT do:** create gradient where there is none. If 66% of prompts in a batch produce zero centered advantage, those steps contribute nothing regardless of LR. The 34% of productive prompts get larger updates, which **can** translate to (a) faster separation of arms if the arms differ structurally, or (b) instability / mode collapse if the arms over-commit to the noisy minority signal.
- **Why it is still worth running:** the cost is small (~$60–80 for 200 steps × 3 arms on B200), the upside is "we see daylight between minority/poly and GRPO in the first 100 steps and gain a concrete training-curve figure for the poster," and the downside is bounded (we know the curve goes flat or diverges by step 200 — we don't waste 800 steps to find out).
- **Why it is not the structural answer:** if it works, it works because the existing dataset has enough signal that we just weren't pushing hard enough on it. That's defensible but weak as a method contribution — the paper story degrades to "we tuned LR upward and the arms separated marginally," which doesn't add to the set-clustering hypothesis. If it doesn't work, we've ruled out "we just needed more aggression" and the only remaining honest path is filter-then-retrain or own-the-null.

**Confidence that LR=3e-6 moves the needle on minority/poly vs GRPO separation: ~25%.** The mechanism by which set-based clustering should help is "when there are multiple correct answers, reward the rarer cluster more." If the rare-correct-cluster events are themselves rare (which is what signal starvation implies), no amount of LR amplification creates more of them.

### 5. Live status — LR=3e-6 short runs on `chicken602` (launched 2026-05-27 20:53 PDT)

User stopped the prior LR=1e-6 fresh runs at ~20:05–20:15 PDT (no measurable progress) and relaunched 3 arms at LR=3e-6 / total_steps=200 on `chicken602`. Apps and W&B runs:

| Arm | App | W&B | Config |
|-----|-----|-----|--------|
| GRPO | `ap-7BigFBD8Qu5aZnjRCG8giF` | [ik4imyoq](https://wandb.ai/224r-project/cs224r-minority-voting/runs/ik4imyoq) | `train_real_b200_lr3e6_s200_grpo.yaml` |
| minority_answer | `ap-y2xM0dwBR3Aj5QylTEnqEV` | [ib9n7akg](https://wandb.ai/224r-project/cs224r-minority-voting/runs/ib9n7akg) | `train_real_b200_lr3e6_s200_minority.yaml` |
| poly_epo_answer | `ap-b6PRqstNcizgMUo0Bhc6xe` | [kau6lbl2](https://wandb.ai/224r-project/cs224r-minority-voting/runs/kau6lbl2) | `train_real_b200_lr3e6_s200_poly_epo.yaml` |

All three are mid-rollout on first/second training step as of 21:07 PDT (~14 min in); no crashes, no NaN/inf in tailed logs. Checkpoint dirs are isolated from the prior B200 runs (`_lr3e6_s200/` suffix). **Stop condition to watch for:** reward going to zero, KL spike to >0.5, or any NaN — none observed yet. **Decision point:** after step ~50 (≈1 h wall) check whether minority/poly are tracking above GRPO; if not, by step 200 the result is a clean negative.

### 6. Recommended next moves (for Nancy to pick when back)

Two coherent paths, plus a hybrid:

**Option A — Filter-then-retrain (recommended if you have ~$700 + ~3 days budget).**
1. Run a 1-pass rollout over Polaris 51K with base model, N=8, no policy update, dump per-prompt pass rates. Script does not exist yet; **dry-run artifact to be staged in this branch** (see task #5).
2. Drop prompts with pass_rate ∈ {0, 1}; keep the ~17K remainder. Save as `polaris_filtered_n17k.parquet`.
3. Retrain all three arms 1 epoch on the filtered set at LR=1e-6 (or LR=2e-6 as a compromise — keep this conservative on smaller data).
4. Re-evaluate at the same three checkpoints, same eval slices. **Defensible publication either way:** if arms separate, story is "the published recipe works once you apply the published filtering"; if flat, story is "even with proper signal density, set-based clustering doesn't beat GRPO at 1.7B."

**Option B — Own the null result, no further retraining.**
1. Wait for current LR=3e-6 probe to land (free information, cheap).
2. Wait for fair-prompt BeyondAIME + DAPO 2k reruns to land.
3. Spend remaining time on diversity panels (PLAN consolation criterion), an ablation figure showing flat minority/poly curves, and writing the mismatch story carefully (Polaris-7B calibration, our 1.7B, 34% signal density, why this kills set-based clustering specifically). **Risk:** TA/grader expects to see at least one positive lever pulled; pure null with no remediation attempt reads as "did not engage with the problem."

**Option C (what's de facto happening) — LR=3e-6 probe in parallel with the prompt-fair reruns.**
- By tomorrow morning (~2026-05-28 ~07:00 PDT), all three of {LR probe, BeyondAIME fair-prompt, DAPO 2k fair-prompt} should be in hand or near-complete. Decide then between A and B with full information.
- Estimated cost of waiting until then: ~$60–80 (LR probe is the dominant spend; the two eval reruns are $40–60 each per the prior entry).

**My recommendation, Nancy-back-from-meeting version:** Let C complete overnight. If LR probe shows zero arm-separation by step 200 *and* fair-prompt BeyondAIME still shows >3 pp regression for trained arms, commit to **B** (clean null, mismatch framing, diversity panel for set arms) — A's $700 / 3 days will not save the hypothesis at this scale and the poster timeline is the real binding constraint. If LR probe shows >1 pp minority-over-GRPO separation by step 200, **A** becomes a high-EV bet because the hypothesis has a pulse and the filtered-data run is the cleanest way to give it room to breathe.

**UPDATE 2026-05-27 ~21:30 PDT — fair-prompt eval landed.** Both BeyondAIME and DAPO 2k fair-prompt reruns finished and are summarized in the rerun-status block of the prior entry. Across-slice picture: **`poly_epo_answer` wins all three eval slices** (+1.3 / +0.8 / +5.0 pp vs base on Polaris 2k / DAPO 2k / BeyondAIME), `minority_answer` is flat-to-negative (+0.1 / −0.65 / +1.0), GRPO is intermediate (+1.1 / −0.5 / +1.0). No individual Δ is >1σ but across-slice consistency for poly_epo is the headline.

**This shifts the recommendation.** The hypothesis-positive signal exists for at least one set arm, just not the headline minority arm. Updated path:

- **Best path now: hybrid of A and C.** Let the LR=3e-6 probe finish (free information by ~07:00 PDT 2026-05-28). If poly_epo separates further from GRPO at step 200, commit to Option A *focused on poly_epo only* — retrain just poly_epo + GRPO on filtered ~17K, skip minority unless time permits. That's ~$450 not $700, and keeps the falsifiable comparison (`poly_epo > GRPO under fair signal-density`) intact. Skip retraining minority: the across-slice evidence already says it doesn't pay off, and a longer run isn't going to flip a structurally-flat curve.
- **Compute diversity panel tonight (offline, $0).** PLAN consolation criterion is `matches GRPO on pass@1 AND improves cluster diversity`. With minority near pass@1-parity on Polaris 2k, the diversity number determines whether minority is "consolation-passing" or "not supported." This is the cheapest decision-grade artifact remaining and should happen regardless.
- **Writeup framing if Option A runs:** "We observed that set-based RL underperforms unfiltered baselines as expected per gradient-starvation literature, then re-ran on rollout-pass-rate-filtered data matching Polaris's own recipe; poly_epo_answer separates by X pp at matched compute." This is a *much* stronger story than the null framing.
- **Writeup framing if Option B (no further training):** lead with the cross-slice fair-prompt table (poly_epo 3/3) + signal-density mismatch as the why-not-more story. Still defensible; less surprising.

### 7. Verification flags (where I'm not confident)

- **Polaris team's exact refilter cutoffs** are not directly read from the blog/repo by me — citation-check subagent confirmed existence of the recipe; I have not independently confirmed the 53K → ~30K number or that the cutoffs are pass-rate ∈ {0,1}-style rather than something subtler. **Action if it matters:** read the [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/) directly before drafting the writeup paragraph.
- **The 4B → 1.7B scaling argument** is intuitive but I have no published ablation showing model-size sensitivity of the optimal refilter cutoff. Treating it as "directionally true" not "quantitatively pinned."
- **Whether retraining on filtered data lifts arm separation specifically** (vs. just lifting all arms together) is the empirical question and the paper-worthy one. Nie et al.'s 28.4% → 73.8% lift is for the GSM8K + fixed-reference-advantage fix, not for the filter-then-vanilla-GRPO recipe; my "confidence: medium-high" is interpolated, not directly evidenced.
- **The qualitative `parse_ok ≈ similar across base/trained` claim** is from the persisted `qualitative_summary.md` per the audit agent's read; I haven't re-derived the underlying numbers. Worth a 2-minute sanity check on the .md before citing in writeup.
- **LR=3e-6 outcome at 200 steps** is the active experiment — if reward collapses or KL spikes early, the option ordering above changes (mode collapse pushes us back to LR=1e-6 + filter, ruling out the "just push harder" interpretation).
- **Cost estimates for Option A** assume rollout-only first pass is ~2 training-step-equivalents; actual rollout-only pass without weight-sync overhead may be cheaper (closer to $100), but I'm rounding up to be safe.

**Related:** [`probes/random_fullgold_n800_results.md`](./probes/random_fullgold_n800_results.md) (34% mixed-reward density measurement), [`ta_discussion.md` §Q1–Q4](./ta_discussion.md), [`handoff/b200_production_launch_2026-05-27.md`](./handoff/b200_production_launch_2026-05-27.md) (LR=1e-6 lineage that these LR=3e-6 runs are isolated from), [`plans/option_a_filter_retrain_2026-05-27.md`](./plans/option_a_filter_retrain_2026-05-27.md) (Option A pipeline + commands, dry-run staged), [`../data/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/fair_prompt_eval_summary_2026-05-27.md`](../data/probes/checkpoint_eval_beyondaime_pass16_hybrid_arms_latest/fair_prompt_eval_summary_2026-05-27.md) (fair-prompt cross-slice summary).

---

## 2026-05-28 (Thursday) — LR=3e-6 checkpoint eval (DAPO + Polaris stratified 2k) + AIME deep dive

**Context.** Overnight follow-up to the May 27 LR=3e-6 probe: re-eval both checkpoint families on higher-signal slices (`n_rollouts=16`, `hybrid_answer_boxed`), replacing BeyondAIME with **Polaris stratified 2k** (250/band), **DAPO 2k**, **AIME-25**, and **MATH-500**. Writeup: [`probes/checkpoint_eval_morning_2026-05-28.md`](./probes/checkpoint_eval_morning_2026-05-28.md).

### Runs landed

| Family | Profile | App | Bundle stamp |
|--------|---------|-----|--------------|
| LR=3e-6 redo (base + 3 arms) | `chicken602` | `ap-kINjUu8IcD3ckvokrvSVQC` | `20260528T083158Z` |
| LR=1e-6 resolved (3 arms, no base) | `anastasia` | `ap-IwCEmOJ2WhYI5RojYlItfV` | `20260528T083202Z` |
| AIME + MATH-500 calibration | `chicken602` / `anastasia` | (earlier apps) | `20260528T082033Z` / `20260528T082039Z` |

DAPO then Polaris ran **sequentially per GPU** (4 GPUs on `chicken602`, 3 on `anastasia`). Volume paths: `main-artifacts/probes/checkpoint_eval_lr3e6_latest_dapo2k_polaris2k_b200/` and `.../checkpoint_eval_lateckpt_resolved_nobase_dapo2k_polaris2k_b200/`.

### Headline (decision-grade slices)

**GRPO wins on every slice that matters at n=2000+.** LR=3e-6 `grpo_lr3e6_s59` beats base on DAPO 2k (+2.0 pp pass@8), Polaris stratified 2k (+2.1 pp), and MATH-500 (+4.4 pp pass@16). Minority/poly are close on DAPO but trail GRPO on Polaris. On 1e-6 resolved ckpts, GRPO still leads everywhere. 3e-6 GRPO vs 1e-6 GRPO: flat on DAPO (+0.2 pp), modest lift on Polaris (+1.1 pp) and MATH-500 (+1.6 pp).

**Recommendation:** continue LR=3e-6 GRPO; do not pivot to minority/poly from these evals; future eval menu = Polaris stratified 2k + DAPO 2k + MATH-500 @ 16 rollouts.

### AIME-25 — base vs GRPO/minority (qualitative deep dive)

Overnight AIME summary showed base pass@16 **13.3%** vs GRPO **10.0%** vs minority **6.7%** (n=30, high variance). Initial read looked like a trained-arm regression on hard OOD.

**Follow-up:** original run did not save per-rollout completions. Re-ran AIME-only with `save_rollouts: true` (`checkpoint_eval_aime25_rollouts_diagnostic_b200.yaml`, app `ap-n0MKKITYLsM3w3XqGrKJkB`, bundle `20260528T184145Z`). Analyzed via [`scripts/analyze_aime_rollouts.py`](../scripts/analyze_aime_rollouts.py); rollouts at [`data/probes/checkpoint_eval_aime25_rollouts_diagnostic_b200/20260528T184145Z/rollouts/`](../data/probes/checkpoint_eval_aime25_rollouts_diagnostic_b200/20260528T184145Z/rollouts/).

**Conclusion: wrong math, not parsing.**

| Arm | parse_ok (rollout-level) | reward rate |
|-----|---------------------------|-------------|
| base | 88.3% | 1.0% |
| grpo_lr3e6_s59 | 87.3% | 0.6% |
| minority_lr3e6_s54 | 88.5% | 0.6% |
| poly_epo_lr3e6_s39 | 84.0% | 0.8% |

All arms parse ~84–89% of completions; only ~1% are correct. Base "winning" AIME is **stochastic luck on 1–2 hard problems** (4/30 with pass@16>0 for base vs 2–3/30 for trained), not a grader/parser artifact.

**Regressions where base pass@16=1.0 and trained=0.0** — breakdown on trained rollouts is overwhelmingly **parsed_but_wrong**, not parse_fail:

| problem_id | gold | base | trained failure mode |
|------------|------|------|----------------------|
| 20 | 81 | `\boxed{81}` | GRPO `\boxed{24}`, minority `\boxed{729}`, poly `\boxed{0}` — wrong counting / template bleed |
| 26 | 60 | `\boxed{60}` | GRPO `\boxed{15361}` (bogus m+n+p); minority often gibberish tail; poly wrong radical form |
| 23 | 49 | `\boxed{49}` | poly only: `\boxed{6}` — flawed divisibility case analysis |

Shared easy win: **problem 5** (gold 70) — all arms pass@16=1.0. GRPO hits `length` stop more often (16/480 vs 3/480 for base) — secondary, not the main story.

**Implication for writeup.** AIME-25 is **sanity-only** (±3 pp per problem at n=30). Do not use it to rank arms. The BeyondAIME-style "base beats trained on hard OOD" narrative does **not** reproduce under matched prompt when you inspect rollouts: trained models emit **confident wrong boxed answers** (often DAPO-style prime-factor / m+n+p templates), consistent with training on easier Polaris/DAPO where GRPO actually wins.

**Related:** [`probes/checkpoint_eval_morning_2026-05-28.md`](./probes/checkpoint_eval_morning_2026-05-28.md), [`configs/checkpoint_eval_aime25_rollouts_diagnostic_b200.yaml`](../configs/checkpoint_eval_aime25_rollouts_diagnostic_b200.yaml).

## 2026-05-31 (Saturday → Sunday) — Stage 6/7 main-verl bring-up: judge batching, W&B metrics, broken permanent_ckpt patch

### Judge throughput: 95s → 45s per step (2.1×)

Stage 6 was deliberately bypassed (4B already fits at bs=128); focus shifted to judge call latency.

- **Single-container batched judge** (ladder1d, 16 prompts/POST × 8 concurrent): 95s wall_s on 128 prompts/step. Step 1 = 291s (with trace overhead).
- **2-container batched judge** (ladder1e, `max_containers=2` on `judge/server.py`, same client knobs): 69s wall_s. Step 1 = 246s.
- **2-container + bigger batches** (`minority_cot_train_4b_1epoch.yaml`, `judge_http_batch_size=64, judge_concurrency=2, JUDGE_MAX_BATCH_SIZE=128`): **45s wall_s**. Step 1 = 208s, `timing_s/adv=51s` (down from 88s).
- Key insight: when 1 container saw 8 POSTs of 16 they were serialized through FastAPI; with `requests[]` batching the in-flight batch is what vLLM continuously batches. Increasing per-POST size unlocks denser KV-cache packing per container.

**Production sizing (`*_train_4b_1epoch.yaml`):** Polaris-51K filtered, 400 steps/epoch/arm. Estimated wall: GRPO ~18hr, CoT arms ~21hr each. Parallel across 2 accounts (chicken602 = GRPO + shared judge 6 GPU; second account = both CoT arms 8 GPU) ~$1,760 at $6.25/B200-hr. Judge collisions are ~8% of calls (estimated ~3% net step slowdown when CoT arms share).

### W&B metric forwarding (Stage 7) — patch added, image baked

`maxrl_cs224r_metrics_ray_trainer.patch` forwards `batch.meta_info["cs224r_metrics"]` into the metrics dict. Minority/poly_epo hooks populate it via `_build_step_metrics` in `train/objective_minority.py`:
- `train/pass_at_8`, `train/prompts_unlocked`, `train/fraction_filtered` (all arms)
- `train/distinct_clusters_mean`, `train/degenerate_rollouts`, `train/judge_parse_ok_rate`, `train/judge_overflow_skipped` (judge-only)

Also fixed `clusters_judge.py` to `_strip_left_pad(ids, pad_id)` before decode, so the judge sees actual rollout text not pad tokens.

Initial patch had wrong line numbers (computed against original maxrl, not post-prior-patches state) and wrong hunk counts — fixed in commits `4dae097`, `c30045b`. Image rebuilds now apply it cleanly.

### W&B tag propagation bug

`+trainer.wandb_kwargs.tags` in yaml never reached W&B — every prior run had `tags=[]`. Workaround: pass `WANDB_TAGS=tag1,tag2,...` via Modal Secret (wandb reads it at `wandb.init()`). Probe (`probes/minority_cot_judge_smoke_4b.py`) now wires this through.

### Probe bug: module-level env reads vs Modal Secret injection order

`CONFIG_NAME = os.environ.get("CS224R_SMOKE_CONFIG", DEFAULT)` was at module load. On the Modal container, module load happens BEFORE Modal Secrets are injected, so the env var (passed through `modal run` shell env) was empty and the probe always loaded the DEFAULT yaml — silently shipping the wrong config. Same bug would have affected `CS224R_SMOKE_STEPS`. Fix: move the reads inside the function body, AND pass both keys through the Modal Secret dict so they're guaranteed present at function-call time. WANDB_TAGS worked accidentally because wandb itself reads it at `init()` time (post-injection).

### BROKEN: `maxrl_permanent_ckpt.patch` (other agent's patch)

Stage 8 patch adding `permanent_ckpt_freq` support to `ray_trainer._save_checkpoint`. Multiple bugs:

1. **Blank context lines missing leading space** → unified-diff parser miscounts hunks (`patch: malformed patch at line 9`). Fixed by writing single-space prefix.
2. **Hunk counts** off in H1 (claimed 6/7, actual 4/5) and H3 (claimed 29/48, actual 31/50).
3. **Line numbers** computed against original maxrl, not post-prior-patches state — needs +4 shift before line 1424 (empirical from H2's `offset 4`).
4. **Context lines lack trailing whitespace** that exists in actual maxrl source (`None ` vs `None`) — added `-l` flag to the patch step in `modal_image.py`.

**Even with all four fixes applied, H3 still fails at line 1430.** The `-l` flag should have handled the trailing-whitespace issue but didn't, suggesting a deeper content mismatch (possibly the patch was generated against a different commit / pre-applied state than what we have).

**Recommended fix:** regenerate the patch from scratch by applying the desired `_save_checkpoint` rewrite against a freshly-patched maxrl tree (i.e., maxrl@pinned + all prior 6 patches applied), then `diff` to produce a clean patch. Alternatively, rewrite the `_save_checkpoint` change as inline `sed`/Python in `modal_image.py` and drop the patch entirely.

**Workaround for now:** comment out the `permanent_ckpt` patch step in `modal_image.py` to unblock the verification run. Lose ckpt-keeping logic (every ckpt persists, no temp-pruning) but get back to validating Stage 7 W&B metrics. Not yet executed.

### Production-launch readiness

**Ready (committed `4665573`, `c30045b`, `4dae097`):**
- 3 production yamls (`{grpo,minority_cot,poly_epo_cot}_train_4b_1epoch.yaml`) at gpu_mem 0.85, save_freq 15, permanent_ckpt_freq 60 (yamls expect the broken patch to be live)
- Judge config (batch 64, conc 2, 2 containers, MAX_BATCH_SIZE 128)
- W&B metrics patch + decode fix
- Probe fix for env var propagation

**Not ready:**
- `permanent_ckpt` patch (broken — see above)
- Multi-account dispatch script
- Stage 7 `finish_reason="length"` wiring (optional pre-launch per STATUS.md, but flagged for eval story)

## 2026-05-31 (Sunday) — Stage 8 launched; GPU clock lottery on abao

All three production runs launched at 04:09 PDT — `chicken602` (GRPO + judge), `emma` (`minority_cot` + judge replica), `abao` (`poly_epo_cot`). All three healthy on W&B; tags propagating via `WANDB_TAGS` env var. Created the "Stage 8 production (verl)" saved view (`main/scripts/setup_wandb_production_view.py`) filtering on tags `verl + production`.

### `poly_epo` step wall ~1.8× the other two — diagnosed to host-side GPU clock cap, not code

W&B `perf/time_per_step` showed `poly_epo_cot ≈ 360 s` vs `minority_cot ≈ 200 s` vs `grpo ≈ 155 s`. All three runs use identical model / batch / FSDP / rollout config and were launched in the same minute.

**What it wasn't:**
- Not a crash: `modal app logs` clean on all three; no tracebacks.
- Not the judge: `clusters_judge` wall ≈ 43 s on poly_epo vs ≈ 40 s on minority_cot. Parse-OK 1.000 on both.
- Not the algorithm: subagent diff'd `objective_minority.py` vs `objective_poly_epo.py` + both adv-est patches + both probes + both yamls. Implementation parity confirmed; if anything `poly_epo_cot`'s adv kernel is slightly cheaper (no `Counter` / no `rng.choice`).
- Not NVLink: `nvidia-smi nvlink -s` inside both containers showed all 18 links at 53.125 GB/s = full 956 GB/s aggregate B200 spec. Identical on abao and emma.
- Not a different GPU SKU: both probes request `gpu="B200:4"`; B200 ships SXM-only so 4× B200 is always on an HGX baseboard with NVSwitch.

**Smoking gun (single nvidia-smi query inside each running container):**

```bash
modal container exec --no-pty <task-id> -- bash -c \
  "nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv"
```

| GPU | poly_epo (abao) `clocks.sm` | minority_cot (emma) `clocks.sm` |
|-----|-----------------------------|----------------------------------|
| 0   | 1965 MHz (max)              | 1927 MHz                          |
| 1   | **1155 MHz**                | 1965 MHz                          |
| 2   | **1155 MHz**                | 1882 MHz                          |
| 3   | **1155 MHz**                | 1965 MHz                          |

Three of `poly_epo`'s four B200s pinned at exactly **1155 MHz** (≈ the B200 base clock) vs `minority_cot`'s near-boost ≈ 1900–1965 MHz. `1155 / 1965 = 0.59` — matches the observed wall-time ratio. All GPUs at 99–100% util, so they're *doing* work, just at base clock.

**Critical:** `clocks_throttle_reasons.active = 0x0` on every poly_epo GPU. No thermal flag (temp 38–44 °C), no power-brake flag (440–660 W of 1000 W limit), no SW slowdown. The clock is held down with **no documented NVIDIA-side throttle reason**, and three GPUs at the *exact same* 1155 MHz is not stochastic boost behavior — it's the fingerprint of an admin `nvidia-smi --lock-gpu-clocks` set on the underlying host (often done when rack PSU can't sustain 4× 1000 W and the operator clamps clocks instead of risking power events). Modal's NCCL healthcheck (per their gpu-health blog) catches link failures but does not catch admin clock locks. No first-party Modal report of this symptom found in docs / HN / GitHub issues — undocumented but real.

### Re-diagnosis (later that day): not a `-lgc` lock — host-virtualization layer

The "admin clock lock" framing above was a guess; deeper probing rules out every standard NVIDIA clock-cap mechanism and points instead to a **host/hypervisor-level cap invisible to the guest**.

**Evidence that rules OUT all guest-visible clock caps:**

```bash
modal container exec --no-pty <task-id> -- bash -c \
  "nvidia-smi -q -d TEMPERATURE,POWER,CLOCK -i 0,1,2,3"
```

On abao's slow GPUs (1, 2, 3 — pinned at 1155 MHz):

| Field | Value | What a `-lgc`/`-ac`/`-pl` lock would show |
|---|---|---|
| `Applications Clocks: Graphics` | 1965 MHz | The locked value (would be 1155) |
| `Default Applications Clocks: Graphics` | 1965 MHz | The locked default |
| `Max Clocks: SM` | 1965 MHz | The `-lgc` ceiling (would be 1155) |
| `Max Customer Boost Clocks: Graphics` | 1965 MHz | Lower if customer boost capped |
| `Current Power Limit` | 1000 W | Lower if `-pl` set |
| `Performance State` | P0 | P2/P3 if PerfState locked |
| `Persistence Mode` | Enabled | — |
| `GSP Firmware Version` | 580.95.05 | Same as healthy hosts |
| `Clocks Event Reasons: HW Slowdown / SW Thermal / Sync Boost` | Not Active | Any would set throttle bits |
| `GPU Current Temp` | 31–33 °C | Cool |
| `Memory Current Temp` | 33–36 °C | Cool |

(The `Shutdown T.Limit Temp: -5 C`, `Slowdown T.Limit Temp: -3 C`, `Max Operating T.Limit Temp: 0 C` fields look alarming but are **driver display placeholders on Blackwell** — emma's fully-healthy GPUs report the identical `-5/-3/0` values. Not real thermal violations.)

So: no `-lgc` lock, no `-ac` lock, no `-pl` cap, no thermal violation, no recognized throttle, persistence on, P0, firmware matches. The cap is **not enforced anywhere the guest can observe.**

**Evidence that the algorithm is NOT the cause** (rules out the earlier "rank-0 imbalance / DVFS-from-low-util" hypothesis):

| Run | Setup | actor `update_policy` s/round (verl progress bar, n≈30 samples) |
|---|---|---|
| anastasia GRPO | no judge | **1.17** (range 1.14–1.37) |
| emma minority_cot | judge + CoT + set-based marginal adv kernel + rank-0 judge HTTP block ~43s/step | **1.17** (range 1.11–1.20) |
| abao poly_epo_cot | **same** judge + CoT + set-based marginal adv kernel + rank-0 judge HTTP block ~43s/step | **1.80** (range 1.71–1.91) |

emma's `minority_cot` runs the *same critical-path code* as abao's `poly_epo_cot` (same set-based marginal advantage kernel `set_based_marginal_advantages`, same `assign_judge_clusters` HTTP path, same FSDP config, same vLLM rollout config, same yamls modulo `adv_estimator` name + presence of `global_seed`). It runs at GRPO speed. The slowdown is therefore **entirely the host**, not the arm. Update_policy is pure FSDP forward/backward/optimizer with no judge or rank-0 host work, and the 1.80/1.17 = 1.54× ratio is consistent with 3-of-4 GPUs clamped at 1155 MHz while the slowest GPU sets the step time.

**The actual fingerprint — PCIe topology:**

```bash
modal container exec --no-pty <task-id> -- bash -c "nvidia-smi -q | grep '^GPU 0000'"
```

| Container | PCIe Bus IDs | Layout |
|---|---|---|
| anastasia (healthy) | `00000000:03:00.0`, `:73:00.0`, `:93:00.0`, `:E3:00.0` | Single domain `0000:`, widely-spaced bus numbers, function `00.0` — **bare-metal HGX SXM baseboard**, one PCIe root complex per GPU |
| emma (healthy) | `00000000:03:00.0`, `:63:00.0`, `:73:00.0`, `:83:00.0` | Same — bare-metal HGX |
| **abao (slow)** | `00000002:00:01.0`, `00000002:00:02.0`, `00000002:00:04.0`, `00000003:00:03.0` | **Two PCIe domains** (`0002:` + `0003:`), all on bus `00`, sequential device numbers `:01/:02/:03/:04` — signature of **GPUs presented through IOMMU passthrough / virtualization layer**, not bare-metal HGX |

Modal's B200 fleet is heterogeneous: most nodes are bare-metal HGX, some fraction are virtualized passthrough. The latter enforces a clock cap on 3 of 4 GPUs at the host nvidia driver / GSP firmware / hypervisor layer (any of these would be invisible to the guest in the same way). GPU 0 stays at boost in all three theories because it's the bootstrap/primary device at the lowest PCIe address — host policies typically leave the primary uncapped to keep latency-sensitive driver init paths fast.

We can't fix this from inside the container. Only mitigation is host re-lottery (kill + relaunch).

### How to diagnose this if it recurs

**Minimal probe (one CSV line tells you everything):**

```bash
modal container exec --no-pty <task-id> -- bash -c \
  "nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv"
```

Compare across the two containers' identical workloads. Red flag = current SM clock < ~80% of max with `throttle_reasons.active = 0x0`. If a throttle bit *is* set (thermal `0x4/0x8`, power `0x40`, SW `0x100`), it's a recognized condition — usually self-resolving or fixable. If `0x0` and clock is low, it's a host-virtualization cap; only fix is kill + relaunch onto a different host.

Full diagnostic chain that ruled out everything else:
1. `modal app logs <app-id>` → confirm no traceback / crash.
2. Compare `clusters_judge wall_s` between arms → rule out judge.
3. Diff per-arm `objective_*.py` / patches / yamls / probes → rule out impl.
4. Cross-arm `update_policy s/round` from the verl progress bar → rule out algorithm/sequence-length effects. If two arms with identical code paths (e.g. minority_cot + poly_epo_cot, both judge+CoT) diverge in update_policy time, the slowdown is host, not arm.
5. `nvidia-smi nvlink -s` → rule out degraded NVSwitch.
6. `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv` → low clock + `0x0` throttle is the first signal.
7. `nvidia-smi -q -d TEMPERATURE,POWER,CLOCK` → confirms no guest-visible cap (Max Clocks SM, Applications Clocks, Default Applications Clocks all at boost default; Current Power Limit at default; no thermal violation).
8. `nvidia-smi -q | grep '^GPU 0000'` → PCIe bus IDs. Bare-metal HGX = single domain `0000:` with widely-spaced bus numbers. Virtualized passthrough = multiple domains (`0002:`, `0003:`) with sequential `:01/:02/:03/:04` device numbers on bus `00`. This is the *actual* fingerprint of the bad host class.

### Mitigation: kill after next checkpoint + relaunch (host re-lottery)

`save_freq: 10`, verl `resume_mode: auto` (default in `ppo_trainer.yaml`, not overridden) → just re-running the same probe re-reads `latest_checkpointed_iteration.txt` and continues. `modal app stop` does NOT trigger `modal.Retries`. Sequence: wait for `global_step_10/` on the volume → confirm `latest_checkpointed_iteration.txt` contents → `modal app stop <app-id>` → relaunch with same `JUDGE_BASE_URL` + `WANDB_TAGS`. Cost: ~5–10 steps of lost progress + ~10 min restart vs ~2× wall going forward if the new host has uncapped clocks. Verify the new container's `clocks.sm` *before* trusting the relaunch.

---

## 2026-05-31 (Sunday, late) — Stage 8 diagnostic: poly_epo is slow, minority is regressing, cluster-100 bug fixed

### TL;DR (final picture after iterating with Ifdita on Slack)

- **Poly-EPO is learning, just ~3× slower than GRPO at the same nominal hyperparams.** Reward slope +0.015%/step vs GRPO's +0.047%/step; entropy 0.85 → 0.55 over 145 steps; pass@8 0.36 → 0.41; degenerate-rollouts count dropping over time (535 → 473) — all signs of real, slow learning, not a stall. Ifdita confirms her POLARIS Poly-EPO is also slower than her GRPO at this step count and that high early degenerate-rate is expected.
- **Minority-CoT is actively regressing.** Reward slope **−0.016%/step**, entropy not dropping (0.89 → 0.81), `bin[0.0,0.0]` rising — the policy is getting *worse* at math. Distinct mechanism from poly_epo's slowness; see "minority regression" below.
- **Std-norm theory retracted.** Ifdita confirmed she runs `norm_adv_by_std_in_grpo: false` for both arms, so adding std-norm would *diverge* from her setup, not replicate it. Our set arms already match her on this axis (custom estimators ignore the flag, net behavior = false). Big gradients are *not* the goal.
- **Cluster-100 inclusion was a real bug, now fixed.** Paper App. A.1: "we remove any cluster assignment to cluster 100 from being in the set in the numerator." Our `_poly_epo_subset_score` and `_minority_subset_score` were counting `-1` (our internal mapping of paper's `100`) as a real distinct cluster. Fixed in `objective_minority.py` and `objective_poly_epo.py`; unit tests added.

### W&B trajectory comparison (smoothed, 15-step buckets)

Pulled via `wandb.Api()` against runs `edna0184` (poly_epo, 145 steps), `4n1z6bdl` (minority, crashed at step 46), `bf9j82gh` (GRPO, crashed at step 63).

| metric | **GRPO** | **poly_epo** | **minority** |
|---|---|---|---|
| reward linear slope | **+0.000469/step** (+0.047%) | **+0.000149/step** (+0.015%) | **−0.000161/step** (−0.016%, regressing) |
| reward range (start → late) | 0.109 → 0.143 | 0.115 → 0.132 (peak 0.135 @ step 105) | 0.105 → 0.100 |
| entropy (early → late) | 0.84 → 0.60 | 0.85 → 0.55 | 0.89 → 0.81 (stuck) |
| pass@8 trajectory | n/a (rises via rewards) | 0.36 → 0.41 | 0.38 → 0.36 |
| `bin[0.0,0.0]` (all-wrong prompts) | 0.62 → 0.56 | 0.64 → 0.59 | 0.62 → 0.64 (*rising*) |
| `bin(0.5,1.0]` (mostly-right) | 0.04 → 0.08 (doubling) | 0.04 → 0.06 | 0.04 → 0.03 (falling) |
| `degenerate_rollouts/1024` | n/a | 535 → 473 (~−12%) | 539 → 519 (~−4%) |
| `actor/grad_norm` (std-normed adv vs raw) | 0.20 | 0.0068 | 0.0181 |
| `critic/advantages/max` | 2.47 (std-normed) | 0.12 (r̄·k/4 ∈ [0, 0.25]) | 0.41 (rarest-cluster mean ∈ [0, 1]) |
| `actor/ppo_kl` median | 2e-5 | 2e-5 | 5e-5 |

Note: `actor/ppo_kl` is the same (~2e-5) for *all three arms including learning-GRPO*, so it's a misleading "is it learning" indicator (it's per-mini-batch approx-KL inside one update, not policy drift). Use `critic/rewards/mean` slope + `actor/entropy` trend.

### Why minority is regressing (separable from poly_epo's slowness)

`_minority_subset_score` (`main-verl/train/objective_minority.py`) picks the rarest cluster in a 4-rollout subset and returns *that cluster's mean reward*. Pre-fix, if `-1` (degenerate, the paper's `cluster_id: 100` for code/gibberish/non-math) happened to be the rarest in a subset — and Qwen-3-4B-Base writes Python *often* early, so degenerate rollouts are ~50% of rollouts on day zero — and a degenerate rollout happened to box the correct numeric answer (rare but not impossible if the Python output is right), the subset score becomes 1.0 and the marginal for that degenerate rollout is positive. **The algorithm rewards code-writing.** That positive feedback on degeneracy accumulates and the model regresses on math.

Poly-EPO's diversity term doesn't have this pathology — `len(set(clusters)) * r̄ / 4` only ever rewards *more diversity*, not specifically the degenerate cluster. Hence poly_epo learns (slowly) while minority regresses.

### Cluster-100 fix (landed)

Per paper Appendix A.1: "when computing the diversity of any set using Eq. (11), we remove any cluster assignment to cluster 100 from being in the set in the numerator."

- `main-verl/train/objective_poly_epo.py:_poly_epo_subset_score` — now excludes `DEGENERATE_CLUSTER_ID` (= -1) from the unique-cluster count: `r̄ · |{c in subset : c != -1}| / SUBSET_SIZE`.
- `main-verl/train/objective_minority.py:_minority_subset_score` — now excludes `-1` from the rarest-cluster selection entirely. If every rollout in the subset is degenerate, returns 0.
- `main-verl/judge/types.py` already defines `DEGENERATE_CLUSTER_ID = -1` and maps raw `100` → `-1` in `parse.py`; both objective files now import this constant.
- 2 new unit tests in `tests/test_objective_poly_epo.py` and `tests/test_objective_minority.py` cover the degenerate-cluster cases (including the minority "reward-the-code" failure mode). All 15 tests pass.

### Open policy question — minority-CoT degenerate handling (decide before re-launching minority)

**The question**: should the minority-CoT subset score ignore degenerate rollouts (cluster_id=100, internally `-1`) when picking the rarest cluster AND when computing reward, or only one of those, or neither?

**Why this matters**: `_minority_subset_score` picks the rarest cluster in a 4-rollout subset and returns *that cluster's mean reward*. If `-1` is rare and a degenerate rollout happens to box the right answer (e.g., model wrote Python that printed the right boxed value), pre-fix code returns 1.0 → marginal advantage rewards the code-writer. This is unique to minority's "amplify the rare cluster" mechanism; poly-EPO's diversity term doesn't have this pathology.

**The paper doesn't address this** — minority-CoT is our own algorithm, not from the Poly-EPO paper. Paper's cluster-100 prescription is only about poly-EPO's diversity numerator. So minority is a judgment call.

**Options (with example: clusters=[-1, 0, 0, 0], rewards=[1, 0, 0, 0])**:

| option | "can `-1` be the rarest?" | "do `-1` rollouts contribute to reward?" | score |
|---|---|---|---|
| **A. Pre-fix (status quo)** | yes | yes (if `-1` is selected) | 1.0 — reinforces code-writer |
| **B. Exclude from both** (what I tentatively implemented, then reverted) | no | no | 0.0 — cluster 0 selected, all zeros |
| **C. Exclude from selection, keep in reward** | no | yes — if a degenerate rollout happens to share the rarest real cluster | n/a here (no overlap); generally near-B |
| **D. Force-zero reward for `-1` regardless** | yes (any) | force 0 | 0.0 |

**Resolution status (2026-05-31, pre-relaunch review)**: STILL OPEN. Current code at `main-verl/train/objective_minority.py:509-528` is **Option A** — `-1` is treated as a regular cluster ID, can be picked as rarest, degenerate rollouts contribute reward. The function's docstring explicitly flags this as an open policy decision.

**Recommended (not yet landed): Option B.** Reasoning:

1. *Symmetric with poly-EPO.* That arm excludes `-1` from the diversity numerator per paper §A.1. Minority excluding `-1` from "rarest valid reasoning cluster" is the philosophical mirror — same `DEGENERATE_CLUSTER_ID` semantics, same exclusion policy.
2. *Removes the only exploit vector.* Today a model that emits Python that happens to print a correct boxed answer gets *amplified* by the minority kernel (rare-degenerate → rarest selected → reward propagates). B kills this.
3. *`-1` is definitionally "couldn't be clustered as a reasoning trace"* (Ifdita: code/gibberish/non-math). Selecting it as "rarest valid reasoning" is a category error.
4. *D is strictly worse than B* when `-1` ties with a real rarest cluster — D randomly destroys real signal half the time; B always uses it.

**Implementation sketch (if B is chosen):**
- `_minority_subset_score`: filter `-1` before `Counter`; return `0.0` if no real clusters remain.
- `set_based_marginal_advantages.keep_mask`: extend to also drop prompts where, after excluding `-1`, all rollouts share one real cluster.
- Test cases: `[-1,-1,-1,-1]` → 0.0; `[-1,0,0,0]` → mean of cluster 0; `[-1,-1,0,1]` → rarest among real {0,1} tiebreak.

**Why not landing for this relaunch:** flipping minority A → B is a real training-behavior change (unlike the rest of the relaunch list, which is observability + paper-faithful knob alignment). Existing minority data is under Option A; switching now mixes regimes. Decision still deferred to Nancy — pending Ifdita's view if she replies, otherwise a judgment call.

### Outstanding question for Ifdita (Slack-ready)

> At step ~50–100 of your Poly-EPO POLARIS run, do you remember:
> 1. `actor/grad_norm` — ballpark 0.01, 0.1, or 1?
> 2. `critic/advantages/max` — order 0.1 or order 1?
> 3. `critic/rewards/mean` slope — from step 0 to step 100, did it rise by ~5%, ~25%, or more?
>
> Ours: grad_norm ≈ 0.007, adv/max ≈ 0.12, rewards/mean +12% over 140 steps. GRPO at same settings: grad_norm 0.20, adv/max 2.5, rewards +28% over 60 steps. Want to confirm the set arms are just naturally slower at LR=1e-6.

Three worlds depending on her answer:
1. **Her set arms also have tiny grad_norm and slow reward slope** → we're on the same trajectory; just need more steps. Total_training_steps may need to extend beyond 400.
2. **Her set arms have tiny grad_norm but faster reward slope** → cluster-100 fix should help close the gap; re-launch and check.
3. **Her set arms have noticeably larger grad_norm** → something in advantage normalization or loss aggregation differs; dig further.

### Tweak list before re-launch

1. **Cluster-100 exclusion (DONE, this commit).** Expected effect: faster slope on poly_epo (less noise in diversity term); stops minority from regressing (no more "reward the code-writer" pathology). Magnitude unknown — re-launch to measure.
2. **Re-launch minority first.** It's the only arm where current trajectory is going the wrong way; fix is most likely to bite there. Watch for `bin[0.0, 0.0]` to start *falling* (vs current rising) within 30 steps.
3. **Keep `norm_adv_by_std_in_grpo: true` on GRPO** (matches verl default; GRPO is working). For the set arms it's a no-op anyway.
4. **Defer DAPO clip-higher** (ε_high=0.28) — `pg_clipfrac=0.0006`, irrelevant until/unless gradients get a lot bigger.
5. **Don't add std-normalization to set arms** — Ifdita's setup runs without it.
6. **Don't change reward** — mentor confirmed `math.py` (strict Hendrycks) is the intended scorer.

### Ruled out

| candidate | verdict | evidence |
|---|---|---|
| Reward strict-string vs sympy | not the issue | Mentor confirmed `math.py` (strict Hendrycks) is intentional. GRPO uses same reward and learns; per-arm gap can't be reward. |
| LR=1e-6 too low | not the issue | Mentor used 1e-6 without problem. Pre-milestone 3e-6 signal weak. |
| Std-norm missing on set arms | not the issue | Ifdita's setup also runs without std-norm. |
| Base model | ruled out | Paper Table 1 uses Qwen-3-4B-Base, same as us. |
| vLLM sampling | ruled out | T=1.0, top_p=1, top_k=-1 matches paper. |
| `max_response_length` | ruled out | 4096 matches; mean ~1100, clip 5%. |
| Judge degenerate-rate | ruled out by mentor | Ifdita: "in the beginning degenerate rollouts will be high due to code/non-english generations, but it will decrease over time." Confirmed in our data: 535 → 473 over 145 steps. |
| Judge parse rate | ruled out | `judge_parse_ok_rate = 0.999`. |
| `loss_agg_mode: token-mean` | **OPEN — flagged 2026-05-31** | Verl configs use `token-mean` (masked_mean over all batch tokens); pre-milestone `main/train/loss.py` used `length_norm: batch_max` (Dr.GRPO `T_max`). Different per-sequence weighting. The "matches paper" framing was based on the pre-milestone port, which is just citing Nancy's prior interpretation — the paper's `T_i` definition has not been independently re-verified. Don't change without reading the Poly-EPO paper directly. See [`main-verl/docs/build/relaunch_changes.md` §10](../../main-verl/docs/build/relaunch_changes.md). |
| Custom Poly-EPO subset math | ruled out (modulo the cluster-100 fix now landed) | `r̄ · k/4` and 35-subset marginal match pre-milestone `main/train/objective.py`. |

---

## 2026-05-31 (Sunday, later) — Step-0 gap vs paper; reopen reward strictness

### Picture after pulling the paper

Overlaid `bin[0.0, 0.0]` (= 1 − non-zero pass rate) on Poly-EPO paper Fig. 2 right at our actual step counts:

| step | paper Poly-EPO non-zero | ours Poly-EPO non-zero |
|---|---|---|
| 0 | ~43% | ~36% |
| ~100 | ~52% | ~40% |
| 145 | ~55% | ~41% |
| 200 (paper peak region) | ~57% | — |
| 800 (paper endpoint) | ~55% | — |

**Roughly half the gap is already present at step 0** — same Qwen3-4B-Base, ~7pp behind before a gradient step has been applied. The other half opens over 145 steps at ~1.5× slower slope. Algorithm changes (the relaunch list in `main-verl/docs/build/relaunch_changes.md`) only address the slope half. The step-0 half lives in rollout + grading, not in the trainer.

### Grad-norm sanity (the "is it even learning" question)

`grad_norm ≈ 0.007` on Poly-EPO looked alarming. Worked through it:
- Set-RL marginal advantage is bounded above by `r̄ × k/n ≈ 0.13 × 0.5 ≈ 0.07` early in training. Our `critic/advantages/max = 0.12` is at the algorithm's theoretical ceiling. Not a bug.
- Adam normalizes the update by `m̂ / √v̂` ≈ sign(grad), so per-parameter step ≈ LR regardless of grad magnitude. The model is moving by ~1e-6 / parameter / step either way; over 397 steps that's ~4e-4 cumulative weight drift — enough to shift policy by a few pp, which is exactly what we observe.
- The real risk with tiny grads is **SNR**, not magnitude: when grads are tiny *and* noisy, `√v̂` (which sees noise²) doesn't shrink while `m̂` does, so Adam's effective step collapses below LR. We're not in that regime yet (loss is monotonically improving), but it's the failure mode to watch.

Net: tiny gradients are **intrinsic to poly-EPO at this reward level** and not, by themselves, evidence of a misconfiguration. Ifdita's runs almost certainly have grad_norm in the same 0.005–0.02 ballpark.

### Reopening reward strictness (was "ruled out" — now open)

The earlier "reward isn't the gap" verdict rested on "Ifdita said use `math.py` strict." But:
- Our own reward-decision smoke shows `\boxed{\frac{1}{2}}` vs gold `0.5` → 0.0 under `math.py` strict, 1.0 under `math_verify`. Plausibly a ~5–10% slice of correct-but-unrewarded rollouts on Polaris.
- Most published math-RL setups in 2025–26 (DeepScaleR, rLLM, DAPO, Open-Reasoner-Zero) use `mathd ∨ sympy` or `math_verify`. Strict Hendrycks `==` for training reward is unusually conservative.
- Pre-milestone analysis already showed `mathd ∨ sympy` lifts pass@1 ~4–6pp over strict on the same n800 rollouts (see 2026-05-26 evening entry — "Train grader: mathd OR sympy").
- Mentor's "use strict" advice may have been a post-hoc recipe, not what her paper runs actually used. **Follow-up Slack to her: did the paper training reward include the SymPy fallback?**

### Proposed probe: 4-grader offline rescore (no new Modal cost)

Existing artifact: `probes/05-25/prompt_c/phase1_rollouts.jsonl` (800 prompts × 8 rollouts, raw-ish Qwen, arm C prompt). Already on Modal volume.

Rescore the saved completions with four graders side-by-side:
1. `math.py` strict (current locked baseline)
2. `mathd ∨ sympy` (DeepScaleR / rLLM — already in `main/train/math_grade_deepscaler.py`)
3. `math_verify` (HuggingFace SymPy — already in `.venv`)
4. Rank-2 hybrid extract + grader-of-choice (already in `main/train/reward.py`)

Report per grader: pass@1, pass@8, non-zero pass rate, lift over strict. Hand-check 50 disagreements (grader-X correct, strict wrong) — is grader-X actually right or sneaking false positives?

**Caveat:** arm-C rollouts use the hybrid `Answer: \boxed{}` prompt, not the verl-locked plain `\boxed{}` prompt. So this answers "does lenient grading help on arm-C completions" — directionally informative but not the canonical step-0 number for the verl stack. If lift ≥3pp here, spend ~$15 on fresh raw-Qwen rollouts under the verl-locked prompt for an apples-to-apples re-run.

**Decision tree:**
- **Lift ≥5pp** → reward is the gap; switch verl reward to `mathd ∨ sympy` (or `math_verify`) before relaunching; ping Ifdita for confirmation.
- **Lift 2–5pp** → low-cost correctness fix worth landing, but won't close the 14pp endpoint gap; relaunch anyway.
- **Lift <2pp** → reward isn't it; the 7pp step-0 deficit lives in prompt format or generation, separate probe needed.

### Status

- Probe not yet implemented; this entry documents the plan.
- Existing scripts to extend: `main/scripts/rescore_rollouts_rank2.py`, `main/scripts/rescore_mathd_sympy.py`.
- Moves `Reward strict-string vs sympy` from "ruled out" to **open** in the diagnostic table.

### Result — reward strictness is NOT the gap (re-closed)

Implemented as `main/scripts/rescore_four_graders.py`; ran on
`probes/05-25/prompt_c/phase1_rollouts.jsonl` (800 prompts × 8 rollouts,
arm-C `hybrid_answer_boxed`). Manifest gold lookups via
`probes/05-25/group_a_n800/manifest.jsonl` (same 800-problem manifest).

| grader | pass@1 | pass@8 (non-zero rate) | lift vs strict |
|---|---|---|---|
| G1 legacy strict (Rank-2 + normalize ==) | 8.45% | **33.12%** | +0.00 pp |
| G2 mathd OR sympy (Rank-2 extract) | 8.50% | 33.25% | +0.13 pp |
| G3 math_verify (Rank-2 extract) | 8.89% | 34.38% | +1.25 pp |
| G4 math_verify on raw `\boxed{}` (no fallback) | 9.52% | 35.38% | +2.25 pp |

Legacy strict pass@8 = 33.12% reproduces the May 25 prompt-C number
exactly — harness is sound.

**Hand-check of G3 disagreements (15 of 28 samples reviewed):**
- ~6/15 are **legit rescues** — degree symbols (`90°` vs `90`), text
  suffixes (`1011 people at positions...` vs `1011`), zero-padded gold
  (`01` vs `1`), `= 4` inside an equation.
- ~9/15 are **false positives** — `2,-2` vs gold `2`; `4√3` vs gold `4`;
  `24π` vs gold `24`; `2n` (with variable) vs gold `2`; `4s−4` vs gold `4`;
  `1 + 1/n` vs gold `1`. math_verify approximates or extracts substrings
  too loosely on multi-answer / parametric responses.

Net real lift after false-positive correction ≈ **+0.5 pp**, well below
the 2pp threshold. mathd∨sympy is even smaller (+0.13 pp nominal,
3 total rescues across 6400 rollouts). G4's extra +1 pp over G3 is almost
entirely more false positives — skipping Rank-2 discipline grabs anything
in the last `\boxed{}` and grades it leniently.

**Decision gate result: lift <2pp → reward strictness is NOT the gap.**
Ifdita's "use strict math.py" was correct. Do not change verl reward
before relaunching. The 7pp step-0 deficit vs paper Fig. 2 lives in
prompt format, generation, or dataset sampling — not grading.

**Remaining suspects for the step-0 gap:**
1. Prompt template — verl uses plain `\boxed{}`-only; paper template
   may have differed (no published prompt for paper's POLARIS runs).
2. Tokenization / chat template — Qwen3 base vs chat template subtleties
   at the boundary between system prompt and user turn.
3. Dataset sampling — our filter drops prove/gold-leak (~23.4% of
   Polaris-53k); if paper kept those rows, "non-zero rate" includes
   easier gold-in-prompt cases.

Caveat already noted in the plan: this rescore is on arm-C prompt
completions, not the verl-locked plain `\boxed{}` prompt. A fresh
raw-Qwen rollout pass under the verl prompt would be the apples-to-apples
step-0 number — but with grader-strictness ruled out as the lever,
that probe is no longer worth ~$15. Skipped.

Diagnostic-table update: moves `Reward strict-string vs sympy` back to
**ruled out** with stronger evidence (offline 4-grader rescore + hand-check,
not just verbal mentor recall).

### Reversal — both decisions wrong; we ran the canonical probe anyway

Spent ~$5 on chicken602 B200 to generate raw Qwen3-4B-Base rollouts under
the actual verl-locked prompt (`\nPlease reason step by step, and put your
final answer within \boxed{}.` — verbatim from
`main-verl/data/preprocess_polaris_verl.py:INSTRUCTION_SUFFIX`). Wired
this in as new prompt variant `verl_polaris_maxrl` in
`main/train/prompts.py`. Probe config: `configs/probe_verl_prompt_4b_n800.yaml`.
Reused the random_fullgold n=800 manifest from 05-27.

Modal app `ap-2GuVS6L5ORetb9oYZMIwYs`, wandb `w57tqx8a`. ~25 min wall.

**Result — both prior conclusions overturned:**

| grader | pass@1 | pass@8 (non-zero) | lift vs strict |
|---|---|---|---|
| G1 legacy strict | **15.66%** | **44.25%** | +0.00 pp |
| G2 mathd ∨ sympy | 16.56% | **47.62%** | **+3.38 pp** |
| G3 math_verify (Rank-2 extract) | 14.03% | 38.62% | **−5.63 pp** |
| G4 math_verify on raw `\boxed{}` | 15.72% | 41.00% | −3.25 pp |

### Finding 1: the 7pp step-0 gap was the prompt template, not the grader

Strict pass@8 under verl prompt = **44.25%**, essentially identical to
paper Fig. 2 right step-0 ≈ 43%. The 33% number we'd been comparing was
from arm-C (`hybrid_answer_boxed`) rollouts — a different (and worse)
prompt than what verl training actually uses. **Under the canonical
training prompt, our step-0 reproduces paper.** The earlier "step-0
deficit" framing was an artifact of using arm-C as the baseline.

Mechanical interpretation: the verl maxrl suffix
(`\nPlease reason step by step, and put your final answer within \boxed{}`)
elicits more disciplined boxing and longer step-by-step reasoning from
Qwen-3-4B-Base than arm-C's prefix-style template, doubling pass@1
(8.45% → 15.66%) and lifting pass@8 by 11 pp.

### Finding 2: mathd∨sympy is a real ≈+3 pp lever (vs the 0.13 pp I claimed)

Earlier rescore on arm-C said G2 lift was negligible (+0.13 pp, only 3
rescues total). That was because arm-C completions concentrate on short
integer boxes where Hendrycks strict and mathd∨sympy agree.

Under the verl prompt, Qwen-3-4B-Base produces a much wider answer
distribution — multiple choice (`\textbf{(D)}`), word answers (`Water`,
`Ranch`, `positive`), latex equations (`S_1 \geq \frac{2}{3} S_2`),
fractions written as `\dfrac` vs `\frac` — exactly the territory where
strict-equality misses correct answers and SymPy/mathd normalization
catches them.

**Confusion matrix (n=6400 rollouts):**

| (G1, G2, G3) | count | meaning |
|---|---|---|
| (1, 1, 1) | 742 | all three pass |
| (1, 1, 0) | 202 | strict+G2, math_verify chokes |
| (0, 1, 1) | 50 | G2 and math_verify catch what strict misses |
| (0, 0, 1) | 84 | math_verify only (often FPs) |
| (0, 1, 0) | 66 | mathd∨sympy only |
| (1, 0, 0) | 36 | strict only — G2 regression |
| (1, 0, 1) | 22 | strict + math_verify, G2 regression |
| (0, 0, 0) | 5198 | all fail |

G2 vs G1: 116 rescues, 58 regressions, **net +58 rollouts (+0.91 pp pass@1)**;
at pass@8 (per-prompt max), net +27 prompts unlocked (+3.38 pp).

**Hand-check of 25 G2 rescues (verl prompt): 24/25 legit.**
- Case sensitivity: `Water` ↔ `water`, `Ranch` ↔ `ranch`, `Positive` ↔ `positive`
- Multiple choice: `\boxed{D}` ↔ gold `\textbf{(D)}`, `\boxed{E}` ↔ `(E)\9`
- LaTeX equivalences: `\dfrac{100}{3}` ↔ `\frac{100}{3}`
- Symbolic identity: `S_1 \geq \frac{2}{3} S_2` ↔ same
- Whitespace: `17` ↔ `17\,`
- Single FP: pid=33 (function-form answer, structurally different)

**Hand-check of G2 regressions:** equation-form answers like `f(n) = 0`
(with space) where strict's `normalize_final_answer` strips spaces and
matches but mathd/sympy fail to parse the equation form cleanly. Most
regressions are real (strict was correctly matching).

Real net lift after FP correction ≈ **+2.5 to +3 pp** on non-zero rate.

### Finding 3: math_verify is worse than strict under verl prompt

G3 went −5.63 pp. Cause: math_verify chokes on **word answers** (`google
pixel 6`, `No`, `ranch`) — it tries to symbolically parse them, fails,
returns 0. Strict just compares strings, gets the match.

Also G3 (Rank-2 extract + math_verify) regresses on plain integer cases
where the parsed string is `1` and gold is `1`, but math_verify's `parse()`
returns an empty list or unparseable result. Strict
`normalize_final_answer == normalize_final_answer` handles these trivially.

**Don't use math_verify** for our training reward. Wrong tool for the
Polaris answer-type distribution (lots of words, equations, multiple choice).

### Updated picture

- **Step-0 baseline gap: closed.** We match paper at step 0 under the
  actual training prompt.
- **The remaining gap is SLOPE only.** Paper goes 43% → 55% over 145
  steps (+12 pp); we go 36% → 41% on poly_epo (+5 pp), 38% → 44% on GRPO
  (+6 pp). Slope ≈ half of paper's. That's the real story.
- **mathd∨sympy is worth landing as the training reward.** +3 pp non-zero
  rate from step 0 means more prompts contribute gradient — directly
  unlocks faster slope. Hand-check confirms ~96% of rescues are
  legitimately correct answers that strict is wrongly punishing.

### Recommendation (revised) — before relaunching

1. **Switch verl training reward from strict `math.py` to mathd∨sympy
   equivalent.** Three implementation paths:
   - Use upstream verl's `math_dapo.py` reward (mathd extraction + SymPy
     fallback) — closest to DeepScaleR/rLLM rule.
   - Patch maxrl `math.py` to add a SymPy fallback when string equality
     fails (smallest blast radius; matches `grade_parsed_answer` in
     `main/train/reward.py`).
   - Wire `main/train/reward.py:compute_reward` in as a verl
     `custom_reward_function` (loses fork-locked benefits but trivially
     correct).
2. **Ping Ifdita to confirm her training reward stack** — paper text
   doesn't say. If she also used Hendrycks strict, we'll be a slight
   overcorrection from paper; if she used mathd∨sympy or math_verify
   (more common in 2025–26 math-RL papers), we're now matched.
3. **The 10-item relaunch list still ships** — those are correctness
   debt independent of this finding.

Artifacts:
- `main/configs/probe_verl_prompt_4b_n800.yaml`
- `main/data/probes/05-31/verl_prompt_4b_n800/{phase1_rollouts.jsonl, four_grader_rescore.json}`
- `main/docs/probes/four_grader_rescore_verl_prompt.md`
- W&B run `w57tqx8a` (group `probe-verl-prompt-4b-n800-05-31`)
- New prompt variant `verl_polaris_maxrl` in `main/train/prompts.py`

Diagnostic-table update: `Reward strict-string vs sympy` moves to
**OPEN — mathd∨sympy gives ≈+3 pp lift on canonical prompt; consider
switching before relaunch**.

---

## 2026-05-31 — Parked speedup option: sequence packing on actor + log-prob

**Status: NOT TAKEN for the relaunch.** Recording the option so it's
ready if a future run needs it.

### Context

poly_epo step time on `B200:4` is ~240s, broken down (`timing_s/*`):

- `gen` 55s (eager-locked on Blackwell — `enforce_eager: true` is
  required, see `main-verl/docs/verl-reference.md` §6.2)
- `update_actor` 75s
- `adv` 42s (dominated by judge calls — left alone for cost reasons)
- `old_log_prob` 22s
- weight sync / val / save: ~46s residual

`update_actor` and the two log-prob passes run at
`ppo_micro_batch_size_per_gpu: 4` with `max_response_length: 4096`. Under
the current static-batch path, every forward pads to the longest
sequence in the micro-batch, so a large fraction of FLOPs are padding
when response lengths are uneven.

### Proposed change

Enable verl's dynamic batching on actor + both log-prob passes (rollout
config unchanged — rollout doesn't use this knob):

```yaml
actor_rollout_ref.actor.use_dynamic_bsz: true
actor_rollout_ref.actor.ppo_max_token_len_per_gpu: 16384   # 4× cap, safer first try
actor_rollout_ref.ref.log_prob_use_dynamic_bsz: true
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu: 16384
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz: true
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu: 16384
```

`ppo_micro_batch_size_per_gpu: 4` becomes a no-op when
`use_dynamic_bsz: true` — leave it set, it's just ignored.

### Expected speedup

At `ppo_max_token_len_per_gpu: 16384`:
- `update_actor` 75s → ~55s
- `old_log_prob` 22s → ~16s
- Step total: 240s → ~215s (~10% faster)

At `ppo_max_token_len_per_gpu: 24576` (6× cap, more aggressive):
- `update_actor` 75s → ~45s
- `old_log_prob` 22s → ~14s
- Step total: 240s → ~200s (~17% faster)

### Why we're not taking it now

The risk is a real one, specific to our `loss_agg_mode:
seq-mean-token-sum-norm`. Sequence packing only changes which samples
share a forward pass; it does NOT change sample composition, optimizer
step count, advantage computation, or the rollout itself. But the loss
math is only partition-invariant under `token-mean`. Under `seq-mean`
the framework has to reweight each micro-batch's gradient by its
sequence count (not just sum-then-divide-by-N_microbatches), and
verl's `seq-mean` packing path is less battle-tested than its
`token-mean` path — historically bug-prone in this exact reweighting.

If the reweighting is off, the effective per-sequence loss weighting
shifts in a length-distribution-dependent way — equivalent to a small
hidden LR drift. We already pinned LR=1e-6 as too low in the Stage 8
diagnostic; we don't want a second LR perturbation on top, mid-relaunch.

### Verification recipe if we do enable it

5–10 step smoke at matched seed against the static-batch baseline,
compare:

- `actor/grad_norm`
- `actor/ppo_kl`
- `actor/entropy_loss`
- `actor/pg_loss`
- mean response length

Pass if all match within ~1–2%. Drift >5% on `grad_norm` or `ppo_kl`
indicates verl's seq-mean packing path is mishandling reweighting —
fall back to static and don't ship.

### Related knobs

- `gpu_memory_utilization: 0.85 → 0.90` is a zero-math-risk win worth
  ~5s/step from larger vLLM KV cache; can ship independently of
  packing.
- B200 `enforce_eager` is locked on for the rollout engine; do not
  propose flipping it.
- Judge container scale-out (containers 2→4) would save ~17s/step on
  `adv` but is too expensive for the current budget.


## 2026-06-01 00:30 PT — Stage 8 v2 lr3e6 relaunch (3rd relaunch attempt of v2)

After 3 false starts on the v2 relaunch tonight (host topology, data parquet missing on stonedpinecones, extra_info schema mismatch on aime val), the three runs got past trainer init and logged a few steps before the team pulled them down for a hyperparameter change.

**Observed magnitudes from the brief v2 runs:**

| arm | grad_norm | adv/max | pg_loss | ppo_kl |
|---|---|---|---|---|
| GRPO (anastasia, 2 steps) | 0.26 | 0.875 | 1e-5 | 4e-5 |
| poly_epo (stonedpinecones, 1 step) | 0.0026 | 0.116 | 4e-4 | 1.5e-5 |
| minority (emma, 9 steps in v1 schema before kill) | 0.011 | 0.50 | 2e-3 | -5e-5 |

Poly-EPO v2 grad_norm is ~2.6x smaller than v1's `edna0184` (0.0068). Most-likely cause: cluster-100 fix removed the spurious diversity-reward inflation from degenerate clusters. The shrinkage is the right direction algorithmically, but combined with LR=1e-6 it puts us near the Adam ε-floor (per-param |g| ≈ 4.7e-8 vs ε=1e-8; denominator ratio ≈ 0.82, so eps shaves ~20% off the step).

**Decision: bump LR 1e-6 → 3e-6 symmetrically across all 3 arms.**

Reasoning:
- v1 poly_epo at grad_norm 0.0068 demonstrably learned (pass@8 0.36 → 0.41, entropy 0.85 → 0.55, bin[0,0] 0.64 → 0.59 over 145 steps). The signal at v1's magnitude WAS sufficient.
- v2 set arms at grad_norm 0.0026 (2.6x smaller, post cluster-100 fix) need either patience or LR compensation. We don't have patience (budget tight).
- Asymmetric LR (only set arms bumped) was rejected as hard-to-defend in writeup.
- Std-norm deviation for set arms was rejected: paper explicitly criticizes it in §7 as "a significant deviation from our general set RL recipe", and v1 evidence shows set arms learn without it.
- 3e-6 (not 2e-6) chosen to give set arms maximum signal headroom. GRPO at 0.26 grad_norm + 3e-6 LR is the main risk — DAPO asymmetric clipping (low=0.20, high=0.28) provides safety margin; if GRPO destabilizes we have time to revert since it trains faster.

**New ckpt dirs:** `*_lr3e6` suffix on experiment_name + default_local_dir for all three. Prevents resume from any in-flight v2/v3 checkpoint.

**Launch tags:** `verl,production,{arm},4b,stage-08,lr3e6,judge-fewshot-fix,aime-val,schema-fix`. WandB filter: `lr3e6 AND production`.

**Mapping (unchanged):**
- GRPO → anastasia (LR=3e-6)
- minority → emma (LR=3e-6, judge URL `https://stonedpinecones--v1-chat-completions.modal.run`)
- poly_epo → stonedpinecones (LR=3e-6, intra-account judge)
- judge → stonedpinecones (deployed, B200×2)

**Watch (per v1 post-mortem at timeline.md:1638–1664):**
- `critic/rewards/mean` slope (not grad_norm, not ppo_kl)
- `actor/entropy` direction (should fall like v1 did: 0.85 → 0.55)
- For minority: `bin[0.0,0.0]` should FALL (cluster-100 fix kicking in; v1 had it rising)
- For GRPO: watch `actor/pg_clipfrac` for clip storms from the 3x LR bump
- If after ~30 steps the trends are wrong, intervene with LR cut or arm-specific tuning
