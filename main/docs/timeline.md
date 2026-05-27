# Timeline

Chronological narrative of major work, decisions, and pivots on the main experiment. Pilot timeline lives in `pre-milestone/nancy_explore/narrative/timeline.md`.

This doc records the **journey** — what we tried, what we learned, what we decided. For the static rules of the project, see `[STANDARDS.md](./STANDARDS.md)`; for the strategic plan, `[PLAN.md](./PLAN.md)`.

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

`**contains_show_that(problem) -> bool*`* — inner-arm keyword

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

## 2026-05-27 (Wednesday) — GRPO checkpoint slice eval: flat wandb ≠ flat learning

**Context.** ~275 training steps in (nancy `8qesa78k` + anastasia `pcas3emd`); live `train/mean_reward` and pass@k histogram looked roughly constant. Raised concern about wiring bugs or skipping an LR sweep.

**Method.** Offline fixed-slice rollout eval on Modal (H200): **same 128 prompts** every time (seed 42, 2×64 from `polaris_train.jsonl`, arm C, 8 rollouts/prompt, train grader). Compared **base** `Qwen3-1.7B-Base` vs `/vol/checkpoints/train_real/step_{49,99,149}.pt` (HF load → vLLM weight sync, one shared engine). Harness: `[main/probes/checkpoint_rollout_eval.py](../probes/checkpoint_rollout_eval.py)`; launch: `bash main/scripts/launch_checkpoint_eval.sh`. Modal app `ap-iV80927zwKrFQ4Yer60vZp`.

**Results (128 prompts, identical slice):**


| Checkpoint | mean_reward | pass@8   | frac 0/8 correct |
| ---------- | ----------- | -------- | ---------------- |
| Base       | 0.059       | 0.21     | 0.79             |
| step 49    | 0.080       | 0.29     | 0.71             |
| step 99    | 0.059       | 0.27     | 0.73             |
| step 149   | 0.070       | **0.31** | **0.69**         |


Δ step 149 vs base: pass@8 **+0.10**, frac₀ **−0.10**; mean_reward +0.012 (noisy on this n).

**Verdict.**

- **Not a wiring failure** — stability metrics on the live run were healthy (`ratio_max` < 3, clipping < 0.1%, grad norms ~0.3–0.6).
- **Training is moving the policy** on a fixed probe; flat in-loop curves are largely **batch noise** (random 64-prompt slice each step, σ ≈ 0.02 on reward) plus **sparse signal** (~68% prompts filtered per step).
- **Not a substitute for held-out eval** — 128 training-distribution prompts; directional only. Scale up or run AIME/HMMT harness before paper claims.

**Artifacts.** Volume: `/vol/probes/checkpoint_eval/20260527T041910Z/results.json`. Local copy: `[main/data/probes/checkpoint_eval/results_20260527T041910Z.json](../data/probes/checkpoint_eval/results_20260527T041910Z.json)`.