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

