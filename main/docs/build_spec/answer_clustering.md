# Answer-Hash Clustering Substrate — Current State & Open Questions

**Status:** **v1.5 shipped as production default (2026-05-26)** — hardened canonicalize + sympy union-find with **expanded allowlist** gate. Wired via `clustering.sympy_mode: allowlist` in `main/configs/train_real.yaml`. See `docs/timeline.md` §2026-05-26 "Production decision" for the decision, expansion list, uplift numbers, and known limitations.

**Used by:** arms `minority_answer` and `poly_epo_answer` (`main/train/clustering.py::answer_hash_clusters`). Arm `minority_cot` uses a separate substrate (LLM judge `cluster_id`) and is not affected by anything in this doc.

---

## v1 algorithm (currently in `main/train/clustering.py`)

Two-pass, fully deterministic:

```
for each rollout:
    if parse_failed:
        assign unique negative id  (never collides, never forms minority cluster)
    else:
        key = canonicalize_answer(parsed)

group rollouts by key                   # pass 1: textual identity
for each pair of distinct keys (sorted):
    if sympy_equiv(rep_a, rep_b):       # pass 2: math equivalence
        union them via union-find       # union-by-string-order for determinism

emit cluster ids
```

### Pass 1: `canonicalize_answer`

Single straight-line transform of the parsed-answer string:

1. strip whitespace, drop commas
2. strip `\(...\)` LaTeX inline-math wrappers
3. strip leading/trailing `$` (display math), `[`, `(`, `{`
4. unwrap `\boxed{...}` (one level)
5. strip trailing `.`, `}`, `]`, `\)`, whitespace
6. if `int(s)` parses → `str(int(s))`
7. if `float(s)` parses, is finite, and rounds to itself → `str(int(round(f)))` (catches `60.0` → `60`, `3160.0000000000002` → `3160`)
8. else `lower()`

### Pass 2: `sympy_equiv`

Calls `grade_answer_mathd_or_sympy(a, b) or grade_answer_mathd_or_sympy(b, a)` (symmetric OR over the grader from `train/math_grade_deepscaler.py`). Wrapped by a blocklist that skips sympy entirely if either input contains LaTeX set-operator / text-wrapper commands (`\in`, `\notin`, `\subset`, `\text`, `\mathrm`, etc.) — see `_SYMPY_UNSAFE` regex.

**Per-prompt cost:** ~6 ms (28 pairwise sympy calls per N=8 prompt). Negligible vs ~180 s/step.

---

## Why we kept the grader's asymmetric branches (decision: 2026-05-26)

`grade_answer_sympy` has two intentional asymmetric branches that we do **not** bypass:

```python
if _is_frac(a) and _is_frac(b):
    is_correct = (a == b)         # both frac → string compare, no sympy
elif _str_is_int(a) != _str_is_int(b):
    is_correct = False            # int strictness
else:
    is_correct = are_equal_under_sympy(a, b)
```

Rationale: in our prompts the model is asked for a **simplified** answer. A model writing `2/4` is producing a different (lazy/wrong-form) output than one writing `1/2` — they should land in different clusters. Bypassing the branch would merge `1/2 ≡ 2/4`, which dilutes the minority-cluster signal.

Symmetric OR is fine because `grader(1/2, 2/4)` and `grader(2/4, 1/2)` both return False under the asymmetric branch — string compare doesn't care about order.

### Known inconsistency

The unreduced-fraction branch only fires when **both** sides are fracs. So:

| comparison | result | reason |
|---|---|---|
| `1/2` vs `2/4` | not merged | both frac → string compare |
| `0.5` vs `2/4` | merged | only one frac → falls through to `simplify` |

Accepted for v1 because unreduced fractions are likely rare in trained model output. If observed empirically, fix is to pre-screen unreduced fractions and lock them into their own bucket regardless of comparator.

---

## Findings that drove v1

Scanned **2,580 prompts × 8 rollouts** from local probes (`05-25/prompt_c`, `05-25/group_a_n800`, `05-27/random_fullgold_n800`) + Run 0 (`predictions_reparsed.jsonl`).

| metric | value |
|---|---|
| prompts where sympy-union-find disagrees with old string-canonicalize | 5.81% |
| prompts where the minority-cluster identity flips (real training-signal change) | 5.78% |
| distinct sympy-merged pairs observed | 142 |
| ~ fraction that are textual noise (hardened canonicalize alone catches) | ~80% |
| ~ fraction that are genuine LaTeX equivalences (need sympy) | ~20% |

**Examples that hardened canonicalize catches without sympy:**
```
'\(36\)' ~~ '36'       '60.0' ~~ '60'        '\boxed{11}' ~~ '11'
'$10$'   ~~ '10'       '115.' ~~ '115'       '\(150\).'   ~~ '150'
```

**Examples only sympy catches (real LaTeX equivalence):**
```
'8/5'              ~~ '\frac{8}{5}'
'(0,\frac{1}{e})'  ~~ '\left(0,\dfrac{1}{e}\right)'
```

**Examples sympy got wrong (motivated the blocklist):**
```
'13824\inA' ~~ '13824\notinA'    ← FALSE POSITIVE
```
Root cause: sympy's LaTeX parser strips `\in` / `\notin`, reducing both to `13824 * A` → `simplify(diff) = 0`. Blocklist regex prevents this by refusing to call sympy on either side.

---

## Open question: blocklist → allowlist

The current blocklist is the right-shaped fix but has the wrong **failure mode**:

- **Blocklist** (current): sympy active by default; skip if input contains one of ~25 dangerous commands. Failure mode = **too permissive** (forget to block a command → silent wrong merge → wrong training signal).
- **Allowlist** (proposed): sympy inactive by default; allow only if input consists entirely of known-safe primitives (digits, arithmetic, `\frac`, `\sqrt`, `\pi`, `\dfrac`, `\cdot`, `\times`, `\div`, parens, braces). Failure mode = **too strict** (forget to allow a command → miss a merge → fall back to canonicalize, never produces wrong cluster).

The cleanest source for the allowlist is sympy's own LaTeX parser supported-commands set. That maps almost 1-to-1 to what we trust. The blocklist would always be incomplete because LaTeX has thousands of commands.

**Decision: switch to allowlist before the first long `minority_answer` production run** (after we analyze smoke rollouts). The 10-step arm smoke uses **v1 + blocklist** (`clustering.use_sympy: true` in `train_real.yaml`) — same as current code default.

**Resolved (2026-05-26):** allowlist shipped as production default after the 10-step smoke ablation. Config key replaced: `clustering.use_sympy: true` → `clustering.sympy_mode: allowlist`. Allowlist expanded with trig, log, infty, Greek, and `^/_/[/]` chars; bare letters intentionally excluded (false-merge risk on geometry vertex labels). Full reasoning + limitations: `docs/timeline.md` §2026-05-26 "Production decision — hardened canon + expanded allowlist as default".

---

## Clustering comparison (from arm smoke rollouts)

Use completions + parsed answers from the 10-step smoke jsonl on the volume (`smoke_probes.rollouts_jsonl_path` → `/vol/probes/05-26/{arm}_smoke/train_rollouts.jsonl`), not a separate checkpoint probe.

**Offline:** `main/scripts/compare_clustering_methods.py` on smoke `train_rollouts.jsonl` — writes `clustering_compare_detail.jsonl` (per-rollout canon + `cluster_*` columns) and `clustering_compare_summary.json` (partition agreement rates).

Re-run these on saved `completion` / `parsed_answer` fields:
- `old_canon` (pre-v1 string canon)
- `hardened_canon` (v1 pass 1 only)
- `hardened_canon + sympy(blocklist)` (current v1 — smoke default)
- `hardened_canon + sympy(allowlist)` (candidate for full run)

**Decision criteria:** same as prior plan (allowlist vs blocklist false-positive rate; sympy minority-flip rate on smoke distribution). See `main/scripts/analyze_answer_matching.py` / answer-matching probe doc for grading context.

---

## Files touched

| file | role |
|---|---|
| `main/train/clustering.py` | implementation: `canonicalize_answer`, `sympy_equiv`, `answer_hash_clusters`, `cot_clusters_from_judge` |
| `main/tests/test_clustering.py` | tests for both passes + union-find determinism + safety guards |
| `main/train/math_grade_deepscaler.py:434` | underlying grader primitive (`grade_answer_sympy`) |
| `main/docs/build_spec/remaining_arms.md` | parent spec — notes that this doc supersedes the "clustering policy note" section |

## Decisions log

- 2026-05-26: v1 design (this doc) — canonicalize + sympy(grader symmetric OR + blocklist), default on, behind no config flag.
- 2026-05-26: keep grader asymmetric branches (don't merge `1/2 ≡ 2/4`).
- 2026-05-26: defer blocklist → allowlist switch; defer Modal probe; pause until infra up.
- 2026-05-26 (later): allowlist shipped as production default with expanded command list + `^/_/[/]` chars. Bare letters held out (false-merge risk). Config key: `clustering.sympy_mode`. Decision detail + limitations in `docs/timeline.md`.
