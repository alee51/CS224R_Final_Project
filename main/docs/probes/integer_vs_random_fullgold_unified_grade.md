# Integer stratified n800 vs random full-gold n800 — unified offline grade

**Generated:** 2026-05-26 (local re-score of saved completions).

## Canonical grader (both runs)

| Piece | Value |
|-------|--------|
| Prompt | `hybrid_answer_boxed` (arm C) |
| Extract | `extract_rank2` (`main/train/reward.py`) |
| Correctness | `grade_parsed_answer` → mathd **OR** sympy (`math_grade_deepscaler.py`) |
| pass@8 | Chen unbiased, k=8, n=8; **pass@8 (any)** = % prompts with ≥1 correct |

Do **not** use the `reward` field in `05-25/prompt_c/phase1_rollouts.jsonl` without regrading — that file was written when probe `compute_reward` still under-counted (see below).

## Artifacts

| Run | Manifest | Rollouts |
|-----|----------|----------|
| Integer stratified (100/band, integer-gold filter at sample time) | `main/data/probes/05-25/group_a_n800/manifest.jsonl` | `main/data/probes/05-25/prompt_c/phase1_rollouts.jsonl` (6400 lines) |
| Random full-gold (relaxed clean, seed 42) | `main/data/probes/05-27/random_fullgold_n800/manifest.jsonl` | `main/data/probes/05-27/random_fullgold_n800/phase1_rollouts.jsonl` |

## Headline comparison (offline unified grader)

| Run | Rollouts | pass@1 | pass@8 (mean) | pass@8 (any) | parse_ok_rank2 | mixed_reward | all_wrong |
|-----|--------:|-------:|--------------:|-------------:|---------------:|-------------:|----------:|
| **Integer stratified n800** | 6400 | **8.50%** | **33.25%** | **33.25%** | 87.98% | 33.12% | 66.75% |
| **Random full-gold** (partial) | 3840 / 6400 | 9.40% | 33.12% | 33.12% | 86.25% | 32.92% | 66.88% |

**Readout:** At ~60% of the random run, headline metrics match the integer stratified arm-C snapshot within **~1 pp** on pass@1 and **&lt;0.2 pp** on pass@8. Sampling pool (integer-only vs all gold types) is **not** moving baseline difficulty much in this probe.

## Stored `reward` vs offline regrade

| Run | `reward` column at write time | Offline `grade_parsed_answer` | Agreement |
|-----|------------------------------|------------------------------|-----------|
| Integer `prompt_c` (May 25) | **2.77%** pass@1 | **8.50%** pass@1 | 94.2% |
| Random full-gold (May 26 partial) | 9.40% | 9.40% | **100%** |

The May 25 probe job logged rewards with an older grading path; completions are still valid — always re-run `extract_rank2` + `grade_parsed_answer` for train-aligned metrics.

## Historical mixups (do not compare to this table)

| Source | Why it differs |
|--------|----------------|
| `dapo_vs_polaris_rollout_comparison.md` Polaris row | Arm **A** (`dapo_answer_v1`), not arm C |
| `answer_matching_probe.md` “6.0% strict” arm A | Arm A + same train grader |
| `timeline.md` arm C row “8.45% / 33.12%” | Same manifest; recomputed with “strict normalize” wording — within rounding of this table |
| Scary “26.6% pass@8” | Arm A unified strict string match |

## Reproduce

```bash
PYTHONPATH=main python3 main/scripts/analyze_random_fullgold_rollouts.py \
  --manifest main/data/probes/05-25/group_a_n800/manifest.jsonl \
  --rollouts main/data/probes/05-25/prompt_c/phase1_rollouts.jsonl \
  --out-md /tmp/integer_arm_c_unified.md --out-json /tmp/integer_arm_c_unified.json

PYTHONPATH=main python3 main/scripts/analyze_random_fullgold_rollouts.py
# random run when 6400 lines exist
```
