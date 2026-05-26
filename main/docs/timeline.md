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

- **Group B rerun OOM'd at batch_size=64 on H100** in `_completion_logprobs_hf` (vLLM holds ~35 GB KV after rollout; one-shot logprob forward at n_kept~110 won't fit in remaining 45 GB). Switched probe GPU to H200 across `group_b_step_probe.py:350`, `trainer.py:691`, and `configs/probe_step_b_05-25.yaml`. Modal app `ap-Uo8iajUVI3CHaxlycxFwNv` (H100 failure); rerun `ap-L7YjvrKS6ICh3rOKz9OAE9` (H200 success, wandb `66g5uyt6`).
- **Probe-side code fixes landed before the rerun** (agent-led): (a) microbatch sweep clamp visibility — break out when requested mb ≥ `n_kept`, persist `sweep_limited_by` + `sweep_n_kept` to `phase1_done.json`; (b) Phase 1b warmup — one untimed full `run_one_grpo_step` before the timed step so kernels are warm; (c) probe `batch_size: 32 → 64` and `toy_batch.problem_ids` extended to 0–63. Smoke config untouched.
- **H200 readout at batch_size=64 (Phase 1, warm step at mb=1):**

  | Metric                  | H100 bs=32 (prior)   | H200 bs=64 (new)        | Read                                       |
  | ----------------------- | -------------------- | ----------------------- | ------------------------------------------ |
  | Step time               | 90s                  | 118.7s                  | +32% raw — but 2× the prompts              |
  | $/step                  | $0.099               | $0.150                  | +52% raw                                   |
  | **$/prompt**            | **$0.0031**          | **$0.0023**             | **−25% per useful unit**                   |
  | **wall/prompt**         | **2.81s**            | **1.85s**               | **−34% per useful unit**                   |
  | VRAM peak               | 70 / 80 GB (88%)     | 105 / 140 GB (75%)      | Real headroom on H200 (room for bs ~80–96) |
  | Rollout share           | 60%                  | 73%                     | Rollout dominates again at bs=64           |
  | Backward time           | 29.4s @ n_kept=56    | 29.5s @ n_kept=72 (~0.41 s/kept seq) | Use **s / n_kept** for planning — not Phase 1b (different mb + fresh rollouts) |
  | n_kept (group-survival) | 56 (22% group-kept)  | 72 (14% group-kept)     | Lower survival on problem_ids 32–63 — batch variance, not a code issue |

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


| Metric | DAPO pilot (human `cleaned_answers.parquet`, 500×8) | Polaris n800 **arm C** |
| --- | --- | --- |
| pass@1 | 9.03% | **8.45%** |
| pass@8 | 34.40% | **33.12%** |
| mixed_reward | ~34% | **33.0%** |
| all_wrong | ~65.6% | **66.9%** |


Polaris with arm C is **~1 pp below** the DAPO pilot on both pass@1 and pass@8 — not the ~8 pp gap from arm A. Canonical pilot analysis remains `pre-milestone/nancy_explore/run0_analysis/` (dashboard + `minority_metrics.md`); see also `[probes/dapo_vs_polaris_rollout_comparison.md](./probes/dapo_vs_polaris_rollout_comparison.md)` (note: that doc's Polaris row is arm A only — do not use it for train-data decisions).

**Decision — train on Polaris.** Mentor recommendation + **`difficulty` bands** (8-way mirror-J labels) are worth the small baseline gap vs DAPO. PLAN §2 freeze (`polaris_train.jsonl`) is still the binding blocker; target ~16k stratified rows, **full gold types** (not integer-only), arm C prompt + Rank-2 + mathd∨sympy reward at train time.

**Arm C implementation check (same day):**

| Layer | Status |
| --- | --- |
| Prompt template `hybrid_answer_boxed` | **Done** — `main/train/prompts.py`; Group B + probe C yaml |
| Rank-2 parser hybrid path | **Done** — `extract_rank2()` in `main/train/reward.py` |
| Trainer rollouts | **Done** — `format_problem(..., variant=cfg.prompt_variant)` in `trainer.py` |
| Train yaml default | **Fixed** — `configs/train_grpo_05-25.yaml` → `hybrid_answer_boxed` (was still `dapo_answer_v1`) |
| Train-time reward | **Fixed** — `compute_reward()` → Rank-2 + `prompt_variant`; grading **mathd OR sympy** via `grade_parsed_answer()` |

Fallback if convergence issues: swap yaml to `dapo_answer_v1` or `verl_math_boxed` (both kept in `prompts.py`).

### Evening — train grader: mathd OR sympy (DeepScaleR / rLLM)

**Decision:** Train reward = Rank-2 extract + **`grade_answer_mathd ∨ grade_answer_sympy`** on `parsed_answer` (same rule as rLLM `grade_answer_verl`; vendored in `math_grade_deepscaler.py`). See [`decisions.md`](./decisions.md) §2026-05-26.

**Why:** Matches DeepScaleR/Polaris upstream; SymPy rescues strict/format false negatives on probes (`01`/`1`, commas) → better GRPO signal. On n800 parsed rollouts mathd added 0 passes beyond SymPy, but OR is cheap and covers rare Hendrycks-only extractions.

**Shipped:** `grade_parsed_answer()` in `reward.py`; wired through `extract_rank2` / `compute_reward` / trainer / Group A judge.

### Late evening — `batch_size: 128` probes (OOM); lock **bs=64**

**Motivation.** Poly-EPO Table 1 uses **128 prompts / batch 64** on **4× H200** (4B, VeRL). We asked whether single-GPU collocated train+vLLM could match **128 prompts/step** on H200 to improve utilization and `n_kept` (more surviving GRPO groups per step). No dynamic sampling planned — larger batch was the lever under consideration.

**Canonical bs=64 on H200 (volume `probes/05-25/group_b/phase1_done.json`, wandb `g0hrklub`).** Same Group B pipeline as early-morning readout; stochastic variance vs other wandb ids on the same config. Measured: **n_kept=96** (12/64 prompts × 8 rollouts), **VRAM peak ~115 GB / 140**, rollout **~67%** of step, `max_microbatch_ok=96` (limited by `n_kept`).

**bs=128 @ `gpu_memory_utilization: 0.45`** — `configs/probe_step_b_05-26_bs128.yaml`, artifacts `probes/05-26/group_b_bs128/` (no `phase1_done` — failed). Modal `ap-SwkHA9fDlPgLCYugbI9YZL`, wandb `1burtfuq`.

| Stage | Result |
| --- | --- |
| Rollout (128×8 = 1024) | ✅ ~3 min |
| Phase 1 train (`logprob_fwd`) | ❌ OOM in warmup — **139.7 / 139.8 GB** used, needed **+176 MiB** in `_completion_logprobs_hf` |

**bs=128 util sweep (same stack; lowering vLLM cap did not free post-rollout memory).**

| `rollout.gpu_memory_utilization` | Modal | Wandb | OOM after rollout |
| --- | --- | --- | --- |
| **0.38** | `ap-zS1o9oJTat5ZWMmDee8wdV` | `6zhrsrc3` | ✅ 1024/1024 rollouts → ❌ **139.4 GB** used, needed **+394 MiB** |
| **0.40** | `ap-xOSMb7WRVxnVMVK5pphPKa` | `4wkoecge` | Same |

**Readout.** Collocated single H200: doubling prompts fills GPU during/after rollout (KV + dual model copies); HF `logprob_fwd` has no headroom. **`gpu_memory_utilization` is a vLLM pool ceiling**, not a train/rollout split — lowering 0.45→0.38/0.40 did not materially change ~139 GB footprint after rollout. Fitting 128 would need **structural** changes (vLLM sleep/KV release before train, microbatched `logprob_fwd`, or 2-GPU), not yaml util tweaks alone.

**Infra fix (same session):** `pylatexenc` added to `main/infra/modal_image.py` — fresh Modal image builds failed reward import until fixed.

**Decision:** Lock **`train.batch_size: 64`**, **`gpu_memory_utilization: 0.45`**, H200 — see [`decisions.md`](./decisions.md) §2026-05-26 batch size.

### Evening — GRPO train smoke + pylatexenc log noise

**Run.** `launch_train.sh --mode smoke` → Modal `ap-SPj5QSem9RFgU9602NthEF`, wandb `yfmhev1g`. Step 0 completed (W&B logged); step 1 OOM in `_completion_logprobs_hf` (~139.7 GB) — same collocated VRAM story as Group B bs=128.

**Weird log.** Bursts of `macro '\frac' failed its substitution` (~3% of grades per rollout batch) right after each 512-completion vLLM batch — pylatexenc during sympy grading on **malformed frac LaTeX in policy extractions**, not gold or infra failure.

**Decision.** Silence pylatexenc warnings; policy garbage is expected early — see [`decisions.md`](./decisions.md) §2026-05-26 pylatexenc.

### Late night — random full-gold n800 vs integer stratified (train-data sanity check)

**Question.** For PLAN §2 `polaris_train.jsonl`, should we drop non-integer gold (like Group A probes) or keep all Polaris answers? Would a random 800 with full gold look harder/easier than the stratified integer n800?

**Experiment.** Built `scripts/build_polaris_random_manifest.py` (relaxed clean: non-empty problem + gold only; seed 42). Phase-1 rollouts on Modal: `configs/probe_random_fullgold_n800.yaml`, arm C, 800×8 → `probes/05-27/random_fullgold_n800/`. Analysis: `scripts/analyze_random_fullgold_rollouts.py`.

**Unified grader (both runs).** Offline re-score saved completions with **`extract_rank2(..., hybrid_answer_boxed)` + `grade_parsed_answer`** (mathd OR sympy) — not the jsonl `reward` column on the May 25 integer run.

| Run | Sample | pass@1 | pass@8 (any) | parse_ok_rank2 |
| --- | --- | --- | --- | --- |
| Integer stratified n800 (arm C) | 100/band, integer gold | **8.50%** | **33.25%** | 88.0% |
| Random full-gold n800 | uniform random, all gold types | ~9.4% (partial) / matches at 60% | ~33.1% | ~86% |

**Readout.** Headline difficulty is **the same** whether we integer-filter at sample time or include LaTeX/fraction/string gold — sampling pool choice is not moving baseline pass rates much for 1.7B + arm C. Safe to drop integer-only filter for the 16k freeze unless we want probe parity with Group A manifests.

**Gotcha logged.** `05-25/prompt_c/phase1_rollouts.jsonl` stored `reward` at **2.77%** pass@1 (old probe grading); offline unified regrade is **8.50%**. Always re-score completions for train-aligned metrics. Write-up: [`probes/integer_vs_random_fullgold_unified_grade.md`](./probes/integer_vs_random_fullgold_unified_grade.md).

**Frozen** full clean Polaris-53K (53,291 rows) → [`source/polaris_train_full.jsonl`](../data/source/polaris_train_full.jsonl) + meta via `preprocess_polaris.py --n 53291` (seed 42, full gold). **Canonical train manifest:** [`polaris_train.jsonl`](../data/polaris_train.jsonl) (51,139 rows after prompt filter) — see [`data/README.md`](../data/README.md).

---

## 2026-05-27 (Wednesday) — Polaris prompt filter (proof / gold-leak)

**Motivation.** The full pool (`source/polaris_train_full.jsonl`) includes proof-style prompts (“Prove that …”) and cases where the HF gold string appears verbatim in the problem. Train stack is arm C (`hybrid_answer_boxed`) + Rank-2 + **mathd∨sympy** on a **parsed final answer** — not proof grading. Bad rows add noise (model writes proofs, boxed extract fails) or fake reward (model copies gold from the stem).

### Heuristic labeling (full 53,291 rows)

Shipped `main/data/prompt_heuristics.py` + `main/scripts/label_polaris_prompts.py` → `main/data/polaris_train_labeled.jsonl`, `polaris_train_heuristic_summary.json`.

| Flag | Count | % | Notes |
| --- | ---: | ---: | --- |
| `last_starts_prove` | 1,507 | 2.8% | Last sentence matches `^prove\b` after split on `.!?` / newlines |
| `last_contains_prove` | 1,854 | 3.5% | `prove` in last sentence (includes starts) |
| `contains_show_that` | 720 | 1.4% | `\bshow\s+that\b` anywhere |
| `gold_in_prompt` | 10,826 | 20.3% | Stripped gold substring of problem (case-insensitive) |
| Any of four flags | 12,466 | 23.4% | — |

**Manual spot check (n=80):** [`probes/prove_prompt_spotcheck_80.md`](./probes/prove_prompt_spotcheck_80.md). Pools: 40 with `prove` anywhere, 40 with last sentence starting `Prove`. ~70% of “contains prove” sample are genuine proof tasks; ~88% of “last starts Prove” are proof / show-equality. “Contains prove” is **broader** than “ends with Prove” (~35% of A-only sample are find-all + prove or split-sentence artifacts). Many `Prove`-ending rows have gold **not** in the prompt (938 / 1,507) — still proof-style, not leakage.

### Predicate definitions (frozen spec)

Locked 2026-05-27. `main/data/prompt_heuristics.py` is the current source-of-truth code, but **these definitions take precedence** if the module is refactored — re-deriving the 2,152-drop / 51,139-keep counts requires this exact semantics.

**Field provenance.** Both inputs come from frozen `main/data/source/polaris_train_full.jsonl` (post-`clean_rows`, pre-template-wrap):
- `problem` — raw HF `problem` string.
- `gold` — `normalize_train_gold(answer)` = `str(answer).strip()`. **Whitespace-only normalization** — no `\boxed{}` strip, no LaTeX canonicalization, no comma removal.

**`last_sentence(problem) -> str`**
- `problem.strip()` first.
- Split by regex `(?<=[.!?])\s+|\n+` — whitespace following `.!?`, OR one-or-more newlines.
- Each chunk stripped; empty chunks dropped.
- Return last chunk, or `""` if none.
- Does **not** strip leading `$`, parentheses, list markers (`(a)`), or other punctuation from the returned chunk. (Deferred relaxation noted in `decisions.md` §2026-05-27.)

**`last_starts_prove(problem) -> bool`** — outer-arm trigger
- `re.match(r"^prove\b", last_sentence(problem), re.IGNORECASE)`.
- Anchored at start of last sentence; word-boundary after; case-insensitive.
- Will NOT fire on `"$\\,$ Prove …"`, `"(b) Prove …"`, or any last sentence that doesn't *begin* with the literal token `prove`.

**`contains_show_that(problem) -> bool`** — inner-arm keyword
- `re.search(r"\bshow\s+that\b", problem, re.IGNORECASE)`.
- Anywhere in the full problem; word-boundaries on both sides; `\s+` between tokens (matches `show  that`, `show\nthat`); case-insensitive.

**`gold_in_prompt(problem, gold) -> bool`** — inner-arm gate
- Let `g = str(gold).strip()`. If `g == ""`, return `False`.
- Otherwise return `g.lower() in problem.lower()`.
- Case-insensitive substring. **No length floor. No word boundary. No `\boxed{}` strip.** `gold='1'` substring-matches any `"1"` in the problem.

**`"prove" in problem.lower()`** — inner-arm keyword (inline; **not** a labeled function in `prompt_heuristics.py`)
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

| Policy | Drop | Keep | Verdict |
| --- | ---: | ---: | --- |
| `gold_in_prompt` alone | 10,826 | 42,465 | **Reject** — removes ~9.9k MCQs / logic puzzles with answer text in clues |
| Any of four flags | 12,466 | 40,825 | **Reject** — same MCQ problem |
| `prove` anywhere **OR** `(gold ∧ show_that)` | 2,988 | 50,303 | **Reject** on outside arm — +836 mostly find-all / multi-part with numeric gold |
| `last_contains_prove` **OR** `(gold ∧ prove/show)` | 2,388 | 50,303 | **Reject** vs `last_starts` outside — +236 rows; mostly “Given …, prove”, `(b) Prove`, find-all completeness; still valid math, not leaks |
| `last_starts` **OR** `(gold ∧ prove anywhere)` | 2,152 | 51,139 | **Adopted** |
| `last_starts` only | 1,507 | 51,784 | Too narrow for mid-body “Prove \(X\)” with gold \(X\) in stem (+292 leaky rows beyond starts-only) |

**Gold-leak branch nuance.** `gold_in_prompt ∧ prove anywhere` catches mid-body “Prove that \(a^2+b^2\)…” with gold `a^{2}+b^{2}` (+292 vs `gold ∧ prove/show in last sentence` only). We kept **prove anywhere** in the **inner** branch (not only last sentence) so leaks in part (a) still drop when gold appears in the stem. We did **not** require gold leak for the outer `last_starts` arm — most proof endings (62%) have no substring leak but are still poor boxed-answer targets.

**`last_starts` vs `last_contains` on the outside.** `last_contains` adds 236 rows where the last sentence has `prove` mid-sentence (`Given …, prove`, `$$  Prove`, “Provide all answers and prove no others”). Spot check + samples showed these are often still real proof tasks or formatting variants — not worth the extra cut vs tightening `^prove` to strip `$`/whitespace (deferred).

**`show that` without `prove`.** ~453 rows kept (no gold leak). ~232 with gold leak dropped via inner branch.

### Decision — locked filter

See [`decisions.md`](./decisions.md) §2026-05-27. Predicate:

```text
DROP  last_starts_prove
   OR (gold_in_prompt AND ("prove" in problem OR contains_show_that))
```

**Result:** **2,152 dropped (4.0%)**, **51,139 kept (96.0%)** on the frozen 53,291 pool.

**Materialized (frozen):** [`polaris_train.jsonl`](../data/polaris_train.jsonl) (51,139 rows) + [`polaris_train.meta.json`](../data/polaris_train.meta.json) via `filter_polaris_train.py` from full pool; dropped audit [`polaris_train_dropped.jsonl`](../data/polaris_train_dropped.jsonl) (2,152 rows). `train_real.yaml` → `/vol/data/polaris_train.jsonl`. Upload train jsonl to Modal `main-artifacts` before full train.
