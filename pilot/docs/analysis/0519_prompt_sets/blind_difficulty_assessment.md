# Blind difficulty assessment: Batch X vs Batch Y

**Date:** 2026-05-19  
**Analyst protocol:** Blinded labels only (X/Y). Seed mapping stored in `blinding_key.json` and revealed only in [Unblinding](#unblinding).  
**Data:** 32 prompts per batch, resolved from `pilot/data/dapo_slice_3k.jsonl` (see `manifest.json`; zero overlap between batches).

> ## Erratum (methodology invalid for difficulty claims)
>
> **The primary verdict in this document is not valid.** Difficulty was inferred from prompt text (regex, length, contest-style keywords) without independently solving the problems. That measures *apparent* difficulty, not *actual* difficulty for any solver.
>
> **Valid difficulty assessment requires attempts:** work each problem to a claimed answer, verify against gold (same `canonicalize_answer` / `is_correct` path as the pilot), and aggregate solve rate, partial progress, or effort — per solver (human expert, analyst model, or fixed base model with reported pass@k).
>
> **What remains usable here:** blinding key, prompt manifests, and the **secondary** rollout pass@8 table (base-model *attempts*, not text heuristics). The intrinsic-proxy sections are exploratory only; do not cite the “Batch Y slightly harder” primary verdict.
>
> **Replacement protocol:** see `attempt_based_difficulty_protocol.md` (to be filled by a solve-and-verify pass).

---

## Methods

### Blinding

1. Loaded full prompt records from `set_a_seed43_run1b.json` and `set_b_seed42_run2.json` (32 prompts each, all resolved).
2. Assigned **Batch X** and **Batch Y** via deterministic coin flip:
  `sha256("2026-05-19-blind-prompt-difficulty") mod 2 == 0` → set A → X, set B → Y.  
   Mapping recorded in `blinding_key.json`.
3. All scoring and narrative below use **X/Y only**.

### Primary proxies (intrinsic; no rollout rewards)


| Proxy                             | Description                                                                                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Domain tags**                   | Regex keyword tagging: number theory, geometry, algebra, combinatorics, probability, calculus, linear algebra (prompts may get multiple tags). |
| **Benchmark heuristics**          | Flags: olympiad-style (`find all`, `prove`), AIME-style (`compute`, `100m+n`, `gcd(m,n)`), contest-integer phrasing, Asymptote diagrams.       |
| **Structural complexity**         | Prompt length, multi-part cues, proof/find-all language, diagram presence, equation-system cues (0–6+ scale).                                  |
| **Reasoning steps (qualitative)** | Estimated steps from conditionals, multiple questions, find-all/proof cues (capped at 8).                                                      |
| **Answer form**                   | Gold answer type: integer / rational / expression / multi-value (all 32 per batch are integers).                                               |
| **Composite difficulty score**    | Weighted sum of structure, steps, markers, answer complexity, length (not calibrated to a benchmark).                                          |
| **Manual tier rubric**            | Independent 1–5 rubric: baseline contest math, ± for length, find-all, proof, diagrams, short mod arithmetic, large answers.                   |


**Excluded from primary verdict:** step-1 rollout `pass@8`, mean reward, or any training-run metrics (circular with seed-driven batching).

### Secondary proxy (labeled separately)

- **Empirical rollout:** Base-model `pass@8` from `raw_predictions.jsonl` (8 completions per prompt), reported only in [Empirical rollout (secondary)](#empirical-rollout-secondary-not-used-for-primary-verdict).

---

## Per-proxy results: Batch X vs Batch Y

### 1. Domain / problem type


| Domain (tag count) | Batch X | Batch Y |
| ------------------ | ------- | ------- |
| uncategorized      | 17      | 17      |
| number_theory      | 9       | 8       |
| geometry           | 8       | 8       |
| algebra            | 2       | 2       |
| combinatorics      | 2       | 2       |
| probability        | 1       | 0       |


**Read:** Nearly identical topical mix. Both are DAPO-Math (`open-r1/DAPO-Math-17k-Processed:en`), mostly single-integer answers. Regex tagging leaves half “uncategorized” because many prompts are short contest statements without explicit domain keywords.

**Edge:** Neutral (tie).

---

### 2. Benchmark / contest heuristics


| Marker          | Batch X | Batch Y |
| --------------- | ------- | ------- |
| aime_style      | 9       | 6       |
| olympiad_style  | 1       | 2       |
| find_all        | 0       | 2       |
| contest_integer | 3       | 1       |
| asy_diagram     | 2       | 2       |
| proof_required  | 0       | 0       |


**Read:** X has more AIME-flavored “compute / contest integer” surface form. Y has more **find-all** and olympiad-style statements (e.g. prime-power Diophantine, divisor-table characterization). No explicit proof-only items in either batch.

**Edge:** **Y** on conceptual depth markers (find-all, olympiad); **X** on contest-compute phrasing density.

---

### 3. Structural complexity & reasoning length


| Metric                          | Batch X | Batch Y |
| ------------------------------- | ------- | ------- |
| Mean prompt length (chars)      | 260     | 289     |
| Mean structural score           | 0.84    | 1.00    |
| Mean estimated steps            | 1.88    | 2.22    |
| High structural (≥4)            | 1       | 0       |
| Manual “hard” tier (≥3 signals) | 3       | 5       |
| Manual “easy” tier              | 1       | 1       |


**Read:** Y prompts are slightly longer on average and more often flagged as multi-constraint or diagram-heavy. X has one extreme geometry+diagram outlier (longest single prompt in X at 647 chars; Y max 1094 chars on unit-cube dissection).

**Edge:** **Y** (modest).

---

### 4. Answer form complexity


|                            | Batch X   | Batch Y |
| -------------------------- | --------- | ------- |
| All integer answers        | 32/32     | 32/32   |
| Expression / rational gold | 0         | 0       |
| Max |answer|               | 2,017,036 | 7,960   |
| Mean log₁₀(|answer|)       | 2.15      | 1.32    |


**Read:** Both batches use simple verifier-friendly integers. X includes a few **large-magnitude** answers (e.g. binomial-sum evaluation, nested arithmetic), which raises arithmetic error risk but not necessarily mathematical depth.

**Edge:** **X** for computational heaviness of targets; **neutral** on formal answer type.

---

### 5. Composite heuristic score (primary numeric aggregate)


| Statistic                | Batch X     | Batch Y     |
| ------------------------ | ----------- | ----------- |
| Mean                     | 2.95        | 3.30        |
| Median                   | 2.68        | 2.87        |
| Min / max                | 0.92 / 9.59 | 0.92 / 8.20 |
| Manual rubric mean (1–5) | 1.94        | 1.97        |


**Read:** Composite score gives Y a **~12%** higher mean; manual rubric is **effectively tied** (Δ ≈ 0.03). X’s max is driven by one hard geometry configuration problem; Y’s tail is broader (several scores in 6–8 range).

**Edge:** **Y** on composite mean; **tie** on rubric.

---

### Qualitative spot-check (blinded)

**Batch X — notably harder-looking (by text):**

- Right-triangle + altitude configuration with Asymptote figure (contest `m+n` form).
- Equilateral-triangle configuration with shared side (long setup).
- Polygon-diagonal sequence with diagram.

**Batch X — notably easier-looking:**

- Short modular congruence ($6n \equiv 7 \pmod{13}$).
- One-line arithmetic (`99(99^2+3) + 3·99²`).
- Perfect-square divisor of 2800.

**Batch Y — notably harder-looking:**

- “Find all positive integers $n$ …” (prime power minus power; divisors in rectangular table).
- Unit-cube dissection with figure (very long).
- Shaded-area ratio with scaled figure.

**Batch Y — notably easier-looking:**

- Single-variable exponential equation (real solutions).
- Simple divisibility / inequality count problems under 100 chars.

---

## Aggregate comparison


| Dimension                        | Favors             |
| -------------------------------- | ------------------ |
| Domain mix                       | Tie                |
| Olympiad / find-all depth        | Y                  |
| AIME-style compute density       | X                  |
| Prompt length & multi-step setup | Y                  |
| Answer formal complexity         | Tie (both integer) |
| Arithmetic magnitude of answers  | X                  |
| Composite heuristic mean         | Y                  |
| Manual rubric mean               | Tie                |


**Overall intrinsic picture:** Both batches are **homogeneous slices** of DAPO-Math training data: mostly short–medium contest problems with integer gold. Neither looks like a dedicated “hard olympiad” or “easy drill” slice. **Batch Y is slightly harder** on structure and olympiad-style statements; **Batch X** has slightly more compute-style items and larger numeric answers.

---

## Verdict


|                                               |                                                                                         |
| --------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Harder batch (primary, intrinsic proxies)** | **Batch Y**                                                                             |
| **Confidence**                                | **Low**                                                                                 |
| **Magnitude**                                 | Small — means differ by ~~0.35 on composite scale (~~12%); rubric means differ by 0.03. |


### Rationale

Multiple independent **text-based** proxies point weakly to Y (longer prompts, more find-all/olympiad flags, more manual “hard” tier counts, higher composite mean). None of the differences are large enough to treat as a strong difficulty gap. Distributions overlap heavily (shared median band ~2.7–2.9, both all-integer, 17/32 uncategorized by domain regex).

### Caveats

1. **Heuristic noise:** Regex domain and difficulty markers are imperfect; half of prompts lack specific domain tags.
2. **n = 32:** High variance; a few outliers move batch means (X max 9.59 vs Y max 8.20).
3. **No human expert rating:** Assessment is automated + light qualitative spot-check, not full expert review.
4. **Model difficulty ≠ intrinsic difficulty:** See secondary section — empirical rollout favors the opposite direction.
5. **Same source distribution:** Both from the same 3k DAPO slice; seed only shuffles which 32 appear — expect similarity unless sampling is biased.

---

## Empirical rollout (secondary; NOT used for primary verdict)

From step-1 `raw_predictions.jsonl` (8 rollouts per prompt, same base model):


| Metric                   | Batch X | Batch Y |
| ------------------------ | ------- | ------- |
| Mean pass@8              | 0.0625  | 0.1719  |
| Median pass@8            | 0.0     | 0.0     |
| Prompts with 0/8 correct | 24      | 17      |
| Prompts with any correct | 8       | 15      |
| Max per-prompt pass@8    | 0.375   | 0.75    |


**Read:** Under this base model, **Batch X was solved less often** (lower pass@8). That **contradicts** the weak intrinsic edge toward Y and may reflect format sensitivity, computation errors on large answers, or luck on 8 samples — not used to overturn the intrinsic verdict because it confounds prompt identity with rollout outcomes.

---

## Artifacts


| File                                                | Purpose                          |
| --------------------------------------------------- | -------------------------------- |
| `set_a_seed43_run1b.json`, `set_b_seed42_run2.json` | Resolved prompt payloads         |
| `manifest.json`                                     | Provenance, overlap check        |
| `blinding_key.json`                                 | X/Y ↔ seed mapping               |
| `analysis_data.json`                                | Per-prompt scores and aggregates |


---

## Unblinding

See `**blinding_key.json`**:

```json
{
  "mapping": {
    "X": "seed43_run1b",
    "Y": "seed42_run2"
  }
}
```

- **Batch X** = seed **43**, `run1b_grpo` final-pull rollouts.  
- **Batch Y** = seed **42**, `run2_inverse_freq` final-pull rollouts.

Assignment method: `sha256("2026-05-19-blind-prompt-difficulty") mod 2 == 0`.