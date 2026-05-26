# DeepScaleR mathd OR sympy rescore (Group A n800)

**Dataset:** `main/data/probes/05-25/group_a_n800/` — 800 prompts × 8 rollouts = 6400.
**Prompt arm:** `dapo_answer_v1` (arm A). Manifest gold is integer-only (Polaris probe).

## Matchers

| ID | Extraction | Grading |
|----|------------|---------|
| `old_strict` | Rank-2 `parsed_answer` when `parse_ok_rank2` | `normalize_final_answer` string equality (legacy) |
| `train` | same | `grade_parsed_answer` = mathd OR sympy (**current train reward**) |
| `mathd_only` | same | `grade_answer_mathd` (Hendrycks mathd normalize) |
| `sympy_only` | same | `grade_answer_sympy` (integer gold → strict int match) |
| `mathd_or_sympy` | same | mathd OR sympy (DeepScaleR train rule) |

**Variant B (boxed):** last `\boxed{}` via `extract_answer` on full completion; same graders. Parse failures count as fail for pass rates.

**Rescued** = `parse_ok_rank2`, old_strict fail, mathd_or_sympy pass. **Regressed** = old_strict pass, mathd_or_sympy fail.

- Manifest: `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-25/group_a_n800/manifest.jsonl`
- Rollouts: `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-25/group_a_n800/phase1_rollouts.jsonl`

## Overall (variant A: Rank-2 extraction)

| Metric | Value |
|--------|-------|
| Rollouts | 6400 |
| parse_ok_rank2 | 84.8% (5428) |
| parse_ok_boxed (full completion) | 33.6% (2149) |
| pass old_strict | 6.0% |
| pass mathd_only | 6.0% |
| pass sympy_only | 6.5% |
| pass mathd_or_sympy | 6.5% |
| lift (mathd_or_sympy − strict) | +0.52 pp |
| rescued (old fail → new pass) | 33 |
| regressed (old pass → new fail) | 0 |

## Overall (variant B: boxed extract, mathd_or_sympy)

| pass old_strict (boxed) | 3.1% |
| pass mathd_or_sympy (boxed) | 3.1% |
| lift | +0.03 pp |
| rescued | 2 |
| regressed | 0 |

## Per difficulty band (100 prompts × 8 rollouts = 800/band)

| Band | n | parse_ok R2 | parse_ok boxed | strict | mathd∨sympy | rescued | regressed | lift pp |
|------|---|-------------|----------------|--------|-------------|---------|-----------|---------|
| 0/8 | 800 | 82.0% | 31.6% | 6.0% | 6.8% | 6 | 0 | +0.75 |
| 1/8 | 800 | 86.0% | 31.4% | 4.6% | 5.2% | 5 | 0 | +0.62 |
| 2/8 | 800 | 82.9% | 35.8% | 8.2% | 9.0% | 6 | 0 | +0.75 |
| 3/8 | 800 | 83.8% | 36.1% | 7.0% | 7.8% | 6 | 0 | +0.75 |
| 4/8 | 800 | 82.8% | 33.5% | 4.5% | 4.8% | 2 | 0 | +0.25 |
| 5/8 | 800 | 85.1% | 29.5% | 5.0% | 5.1% | 1 | 0 | +0.12 |
| 6/8 | 800 | 88.4% | 36.2% | 5.0% | 5.8% | 6 | 0 | +0.75 |
| 7/8 | 800 | 87.6% | 34.5% | 7.9% | 8.0% | 1 | 0 | +0.12 |

## Key findings

- **mathd adds nothing** beyond `normalize_final_answer` strict match on this slice (`mathd_only` = `old_strict` at 6.0% overall). All **33 rescues** come from the **sympy** path (`100^4` vs `100000000`, etc.).
- **Integer gold:** `grade_answer_sympy` requires strict int match when GT is int — rescues are small (+0.52 pp overall) vs lightweight `sympy_equiv` in the prior probe.
- **No regressions** (0 old-correct → new-wrong) on Rank-2 extraction.
- **Boxed extract (variant B)** parses only **33.6%** of rollouts (full completion, last `\boxed{}`) vs **84.8%** Rank-2 tail window — not recommended for Polaris arm A without aligning extract window.

## Provenance

Graders in `main/train/math_grade_deepscaler.py` (rLLM / DeepScaleR).
Script: `main/scripts/rescore_mathd_sympy.py`.
