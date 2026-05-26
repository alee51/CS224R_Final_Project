# Answer-matching probe (Group A n800)

**Data:** Modal volume `main-artifacts` — `probes/05-25/group_a_n800/manifest.jsonl`, `group_a_n800/phase1_rollouts.jsonl` (arm A), `prompt_c/phase1_rollouts.jsonl` (arm C). Analysis script: `main/scripts/analyze_answer_matching.py`. JSON cache: `main/data/probes/05-25/answer_matching_results.json`.

## Polaris difficulty semantics

HF field `difficulty` is documented as the **pass rate of the problem** estimated by DeepSeek-R1-distill-Qwen-7B (8 rollouts), encoded as `k/8` for `k` successes out of 8.

- **`7/8` = easiest** (model solved 7/8 times; highest reference pass rate).
- **`0/8` = hardest** (0/8 successes).

Our manifest uses bands `0/8` … `7/8` (100 prompts each, 800 total). Note: `PLAN.md` once stated `1/8` easiest → `7/8` hardest; that disagrees with the dataset README and the `0/8` band present in HF — **treat HF + `k/8` pass-count semantics as source of truth.**

## Methods

For each rollout we run Rank-2 extraction (`extract_rank2` in `main/train/reward.py`) with the arm’s `prompt_variant`, then grade `parsed_answer` vs manifest `gold` under:

| Matcher | Rule |
|---------|------|
| `strict_rank2` | `grade_parsed_answer` — mathd OR sympy (**train reward**; name legacy) |
| `string_strip` | Strip + case-insensitive string equality |
| `int_equiv` | Both parse as integers (commas allowed); compare ints |
| `float_tol` | Both parse as float; `|a−b| < 1e-6` |
| `sympy_equiv` | `sympify` after `normalize_final_answer` + `\frac{a}{b}` → `(a)/(b)`; `simplify(pred−gold)==0` |
| `math_verify` | `math_verify.parse` + `verify` on pred and gold |

**Pass rate** = fraction of all rollouts labeled correct (parse failures count as fail). **Lift (pp)** = loose pass rate − strict pass rate (percentage points). **Rescued** = rollouts with `parse_ok_rank2` and strict fail but loose pass.

`sympy_equiv` skips unsafe/long strings (e.g. `int(input())`) after one hang was found in rollouts.

## Failure decomposition (both arms)

| Bucket | Arm A | Arm C |
|--------|-------|-------|
| **No extract** (`parse_ok_rank2` false) | 972 (15.2%) | 769 (12.0%) |
| **Parsed, model wrong** (extract ok, strict fail) | 5042 (78.8%) | 5090 (79.5%) |
| **Strict pass** | 386 (6.0%) | 541 (8.5%) |
| **Conditional pass \| parse ok** | **7.11%** | **9.61%** |

Among parsed-but-wrong rollouts, rescues vs **strict** (same extraction):

| Matcher | Arm A rescued | Arm C rescued |
|---------|---------------|---------------|
| `int_equiv` / `string_strip` | 2 (0.04%) | 2 (0.04%) |
| `sympy_equiv` | 8 (0.16%) | 2 (0.04%) |
| `float_tol` | 10 (0.20%) | 2 (0.04%) |
| `math_verify` | 69 (1.37%) | 28 (0.55%) |

**Interpretation:** Low headline pass rates are dominated by **wrong answers after successful Rank-2 extract**, not by strict integer normalization. Looser train-time matchers (`int_equiv`, `sympy_equiv`) recover **&lt;0.2%** of parsed-wrong mass. `math_verify` (OOD-eval style) adds **~1 pp** on arm A but is **not** the training reward and may admit equivalences strict integer match should reject.

**Per-band pass (strict)** is **not monotone** in Polaris band (e.g. arm A: `2/8` 8.2% &gt; `1/8` 4.6%; `7/8` 7.9% is not the maximum). Base Qwen3-1.7B is weak on all bands at n=100/band.

## Arm A (`dapo_answer_v1`)

### Overall

| Metric | Value |
| --- | --- |
| Rollouts | 6400 |
| parse_ok_rank2 | 84.8% |
| pass (strict_rank2) | 6.0% |
| pass (string_strip) | 6.0% |
| pass (int_equiv) | 6.1% |
| pass (float_tol) | 6.2% |
| pass (sympy_equiv) | 6.2% |
| pass (math_verify) | 7.1% |

**Strict-failure rescue (overall):**

- `string_strip`: 0 rollouts rescued (0.0% of strict-fails), lift **+0.00 pp**
- `int_equiv`: 2 rollouts rescued (0.0% of strict-fails), lift **+0.03 pp**
- `float_tol`: 10 rollouts rescued (0.2% of strict-fails), lift **+0.16 pp**
- `sympy_equiv`: 8 rollouts rescued (0.1% of strict-fails), lift **+0.12 pp**
- `math_verify`: 69 rollouts rescued (1.1% of strict-fails), lift **+1.08 pp**

Mixed-reward prompts (strict): 26.5%

### Per difficulty band (100 prompts × 8 rollouts = 800 rollouts/band)

| Band | parse_ok | strict | strip | int | float | sympy | mv | lift sympy pp | rescued sympy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0/8 | 82.0% | 6.0% | 6.0% | 6.1% | 6.2% | 6.1% | 7.4% | +0.12 | 1 |
| 1/8 | 86.0% | 4.6% | 4.6% | 4.6% | 4.9% | 4.9% | 5.8% | +0.25 | 2 |
| 2/8 | 82.9% | 8.2% | 8.2% | 8.2% | 8.5% | 8.5% | 9.2% | +0.25 | 2 |
| 3/8 | 83.8% | 7.0% | 7.0% | 7.1% | 7.1% | 7.0% | 8.8% | +0.00 | 0 |
| 4/8 | 82.8% | 4.5% | 4.5% | 4.5% | 4.6% | 4.6% | 4.8% | +0.12 | 1 |
| 5/8 | 85.1% | 5.0% | 5.0% | 5.0% | 5.1% | 5.1% | 6.1% | +0.12 | 1 |
| 6/8 | 88.4% | 5.0% | 5.0% | 5.0% | 5.1% | 5.1% | 6.4% | +0.12 | 1 |
| 7/8 | 87.6% | 7.9% | 7.9% | 7.9% | 7.9% | 7.9% | 8.5% | +0.00 | 0 |

### Mixed-reward prompt fraction by matcher

| Band | strict rank2 | string strip | int equiv | float tol | sympy equiv | math verify |
| --- | --- | --- | --- | --- | --- | --- |
| 0/8 | 24.0% | 24.0% | 25.0% | 25.0% | 24.0% | 26.0% |
| 1/8 | 23.0% | 23.0% | 23.0% | 24.0% | 24.0% | 27.0% |
| 2/8 | 29.0% | 29.0% | 29.0% | 30.0% | 30.0% | 32.0% |
| 3/8 | 31.0% | 31.0% | 32.0% | 32.0% | 31.0% | 35.0% |
| 4/8 | 19.0% | 19.0% | 19.0% | 19.0% | 19.0% | 19.0% |
| 5/8 | 26.0% | 26.0% | 26.0% | 26.0% | 26.0% | 30.0% |
| 6/8 | 27.0% | 27.0% | 27.0% | 27.0% | 27.0% | 33.0% |
| 7/8 | 33.0% | 33.0% | 33.0% | 33.0% | 33.0% | 34.0% |

## Arm C (`hybrid_answer_boxed`)

### Overall

| Metric | Value |
| --- | --- |
| Rollouts | 6400 |
| parse_ok_rank2 | 88.0% |
| pass (strict_rank2) | 8.5% |
| pass (string_strip) | 8.4% |
| pass (int_equiv) | 8.5% |
| pass (float_tol) | 8.5% |
| pass (sympy_equiv) | 8.5% |
| pass (math_verify) | 8.9% |

**Strict-failure rescue (overall):**

- `string_strip`: 0 rollouts rescued (0.0% of strict-fails), lift **+-0.03 pp**
- `int_equiv`: 2 rollouts rescued (0.0% of strict-fails), lift **+0.03 pp**
- `float_tol`: 2 rollouts rescued (0.0% of strict-fails), lift **+0.03 pp**
- `sympy_equiv`: 2 rollouts rescued (0.0% of strict-fails), lift **+0.03 pp**
- `math_verify`: 28 rollouts rescued (0.5% of strict-fails), lift **+0.44 pp**

Mixed-reward prompts (strict): 33.0%

### Per difficulty band (100 prompts × 8 rollouts = 800 rollouts/band)

| Band | parse_ok | strict | strip | int | float | sympy | mv | lift sympy pp | rescued sympy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0/8 | 84.0% | 9.5% | 9.5% | 9.8% | 9.8% | 9.5% | 9.9% | +0.00 | 0 |
| 1/8 | 88.4% | 5.9% | 5.9% | 5.9% | 5.9% | 5.9% | 6.1% | +0.00 | 0 |
| 2/8 | 89.9% | 8.2% | 8.2% | 8.2% | 8.2% | 8.2% | 9.1% | +0.00 | 0 |
| 3/8 | 88.2% | 10.0% | 10.0% | 10.0% | 10.0% | 10.0% | 10.8% | +0.00 | 0 |
| 4/8 | 86.1% | 7.6% | 7.4% | 7.6% | 7.6% | 7.6% | 8.0% | +0.00 | 0 |
| 5/8 | 87.4% | 8.2% | 8.2% | 8.2% | 8.2% | 8.2% | 8.6% | +0.00 | 0 |
| 6/8 | 89.8% | 9.2% | 9.2% | 9.2% | 9.2% | 9.2% | 9.2% | +0.00 | 0 |
| 7/8 | 90.1% | 8.9% | 8.9% | 8.9% | 8.9% | 9.1% | 9.4% | +0.25 | 2 |

### Mixed-reward prompt fraction by matcher

| Band | strict rank2 | string strip | int equiv | float tol | sympy equiv | math verify |
| --- | --- | --- | --- | --- | --- | --- |
| 0/8 | 33.0% | 33.0% | 33.0% | 33.0% | 33.0% | 33.0% |
| 1/8 | 30.0% | 30.0% | 30.0% | 30.0% | 30.0% | 31.0% |
| 2/8 | 27.0% | 27.0% | 27.0% | 27.0% | 27.0% | 28.0% |
| 3/8 | 37.0% | 37.0% | 37.0% | 37.0% | 37.0% | 39.0% |
| 4/8 | 27.0% | 26.0% | 27.0% | 27.0% | 27.0% | 29.0% |
| 5/8 | 41.0% | 41.0% | 41.0% | 41.0% | 41.0% | 43.0% |
| 6/8 | 36.0% | 36.0% | 36.0% | 36.0% | 36.0% | 36.0% |
| 7/8 | 33.0% | 33.0% | 33.0% | 33.0% | 32.0% | 33.0% |

## Key findings

1. **Strict matching is not the bottleneck.** ~79% of rollouts fail because the extracted answer ≠ gold under any practical train matcher; only ~15% (A) / ~12% (C) fail extraction.
2. **Looser train matchers barely move the needle.** `sympy_equiv` lifts arm A pass by **+0.13 pp** (8 rollouts); arm C **+0.03 pp** (2). `int_equiv` ≡ strict for practical purposes on integer Polaris gold.
3. **Hybrid prompt (C) helps extraction and accuracy, not grading:** parse_ok **88.0%** vs **84.8%**; strict pass **8.5%** vs **6.0%** (+2.5 pp absolute).
4. **`math_verify` is not a substitute for fixing the reward:** +1.08 pp (A) / +0.44 pp (C) vs strict, still ≪ the ~79% parsed-wrong slice — use for OOD eval only per `STANDARDS.md`.
5. **Polaris bands:** `k/8` = k successes out of 8 on the reference 7B model → **`7/8` easiest, `0/8` hardest**; our measured pass rates do not track that ordering cleanly at 100 prompts/band.

## Spot checks (easy band `7/8`, arm C)

Manual read of bucket **B** = `parse_ok_rank2` but strict fail (650 rollouts on `7/8`).

| Verdict | Share (approx.) |
|---------|-----------------|
| Real model error (wrong math, off-by-one, placeholder `\boxed{...}`) | ~95% |
| Matching bug (parsed answer ≡ gold under sympy) | ~0–1% |
| Borderline off-by-one (still count as model error) | ~4% |
| Ambiguous gold | ~0% in sample |

**Executive summary:** On the easiest band, ~90% of rollouts extract cleanly but only ~9% pass strict — yet almost all strict failures are **wrong answers**, not grading. Looser train matchers would recover on the order of **1–2 rollouts per 650** B cases, not tens of percentage points.

### Illustrative examples

| problem_id | gold | parsed | Verdict |
|------------|------|--------|---------|
| 759 | `100000000` | `100^4` (boxed) | **Matcher bug** — sympy-equivalent; strict string fails |
| 714 | `39` | `41` | Model error — wrong count |
| 715 | `-1` | `0` | Model error |
| 752 | `139968` | `16384` | Model error |
| 711 | (numeric) | `\boxed{...}` literal | Model error — no real answer |

**Sanity (bucket A on `7/8`):** when parsed matches gold (e.g. pids 715, 714, 700, 759 with correct form), strict pass works.

**Contrast `0/8`:** similar pattern — low pass is not explained by easy-band gold messiness alone.
