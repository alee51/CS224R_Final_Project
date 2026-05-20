# Run 0 qualitative analysis (v2) — completion-text review

**Artifact:** `20260519T190202Z` · **Model:** proxy base (8 rollouts × 500 prompts = 4000 completions)  
**Method:** Read full `completion` strings in `raw_predictions.jsonl`, cross-checked with `prompt_inputs.jsonl`, `_prompt_level_stats.jsonl`, and parser context in `_audit_parse_cluster.md`. No LLM labeling; Python used only for stratified indexing and regex/heuristic tagging.

---

## 1. Method — stratified sampling

| Stratum | Selection rule | N prompts in run | Sampled for deep read |
|--------|----------------|------------------|------------------------|
| **All-correct** (8/8 rollouts correct) | `n_correct_rollouts == 8` | **0** | — (none exist); nearest substitute: **7/8** prompt `ddd26788` |
| **No-correct** | `n_correct_rollouts == 0` | 337 | 6 prompts (geometry/algebra mix) |
| **Partial correct** | `1 ≤ n_correct_rollouts ≤ 7` | 163 | 5 prompts (incl. max 7/8, 6/8, 5/8, 1/8) |
| **High cluster diversity** | `n_distinct_clusters ≥ 7` | 398 | 4 prompts (overlap with above) |
| **Parser-flagged** (audit §Example bug classes) | Fixed PID list from `_audit_parse_cluster.md` | 6 | All 6 |
| **Random** | `random.seed(42)`, 15 PIDs without replacement | — | 8 additional (after deduping overlaps) |

**Rollout-level pass:** All 4000 completions received lightweight regex tags (boxed presence, truncation, repetition, nested-boxed regex vs brace-balanced inner, code fences). **Deep vignettes:** 22 prompts (~176 rollouts skimmed end-to-end; tails and openings quoted below).

**Grading note (read-through finding):** Stored `correct` reflects the code at run time. Re-running **current** `pilot/train/answer_parse.is_correct` on artifacts yields **58 mismatches (1.45%)** — all are stored-`correct` rollouts with no single int-parseable shallow `\boxed{...}` (see `analysis_v2_quant.md`). Vignettes use **stored** `correct` / `cluster_id` but call out substrate unfairness where visible in text.

---

## 2. Failure mode taxonomy

Counts are **rollout-level**, **multi-label** (one completion can tag several modes). Heuristics on completion text + stored parse; dominant bucket is mutually exclusive only where noted.

| # | Category | Est. count | % of 4000 | Notes from reading completions |
|---|----------|------------|-----------|--------------------------------|
| 1 | **Wrong math / invalid reasoning** | ~3,400–3,700 | ~85–92% | Dominant on incorrect rollouts: false modular reductions, wrong counts, invented symmetries. Often still ends with a confident numeral. |
| 2 | **No usable `\boxed{}` (format)** | 1,974 | 49.4% | No `\boxed` in text; `extract_answer` falls back to `Answer:` or last line. **`is_correct` ignores these tails** even when numerically right (58 stored-correct rollouts fail current re-check). |
| 3 | **Repetition / step loop** | 990 | 24.8% | Same step block repeated 4+ times, or cyclic “Step N” without progress (e.g. sphere-packing height). |
| 4 | **SymPy / Python code derailment** | ~1,141 | ~28.5% | Contains ` ``` ` or `import sympy`; often fake ` ```output` blocks, mid-proof code, or execution narrative with wrong echoed integer. |
| 5 | **Garbage last-line / long parse** | 367 | 9.2% | Parsed answer >80 chars: mid-equation, code fragment, or prose sentence stored as “answer.” |
| 6 | **Nested `\boxed` regex truncation** | 108 | 2.7% | Shallow regex captures `\frac{1190` instead of `\frac{1190}{29}`; model answer in text is otherwise structured. |
| 7 | **Truncated completion** | ~15–18 | ~0.4% | Ends inside `\boxed{7` or mid-coordinate derivation; no closing brace. |
| 8 | **Format mismatch vs gold (substrate)** | ~20–40 | ~0.5–1% | Right mathematics, wrong string: `\( 50 \)`, `20%`, `$20\%$` vs int gold; splits clusters (audit: 6 loose FN heuristic). |
| 9 | **Training / prompt contamination** | ~9–16 | ~0.2–0.4% | Unrelated problem injected (“train leaves station A…”, “Solve the following math problem step by step”). |
| 10 | **Refusal / “insufficient information”** | 9 | 0.2% | Explicit give-up after long setup. |

**Stored accuracy (as recorded):** 324/4000 rollouts correct (8.1%); 163/500 prompts with ≥1 correct; **0** prompts with 8/8.

---

## 3. Exemplar vignettes (22)

Each row: one prompt, 8 rollouts summarized, fairness of stored labels, short quotes.

### E1 — `ddd26788` · partial (7/8) · modular sum

- **Gold:** `2` · **Problem:** Remainder of \(\sum_{k=0}^{100} 10^k\) mod 9.
- **Pattern:** Seven rollouts: correct mod-9 argument, often ` ```python` + fake output `2`, then `\boxed{2}`. One rollout divides wrong and ends `\boxed{0}`.
- **Fairness:** Stored labels match readable math; one slip (0 vs 2) fairly wrong. Rollout with only `**Final Answer:** Answer: 2` is stored correct but **current** `is_correct` would be false (no shallow boxed int).
- **Quotes:** “`10 ≡ 1 (mod 9)` … `\boxed{2}`”; wrong tail: “`\boxed{0}`”.

### E2 — `1653ee27` · parser-flagged · no-correct

- **Gold:** `201` · **Problem:** Remainder of \(2^{202}+202\) mod \(2^{101}+2^{51}+1\).
- **Pattern:** Eight distinct wrong integers (0, -1, 202, 205, 404, …). Several confuse “remainder mod huge N” with reducing 202 alone. One sympy script prints `404`. Mix of `\boxed{}` and `Answer: 202`.
- **Fairness:** All fairly wrong vs 201. One last-line parse is an entire sentence (`Therefore, the remainder… Answer: \(202\).`) — **cluster_id** on garbage string is not semantically meaningful.
- **Quotes:** “`remainder is 0` … `\boxed{0}`”; “`Answer: 202`” (off by 1).

### E3 — `cfc7b48f` · parser-flagged · partial (2/8)

- **Gold:** `50` · **Problem:** Centroid path area, nearest positive integer.
- **Pattern:** Two correct `\boxed{50}` from \(16\pi\approx 50\). Others: `\boxed{16\pi}`, `120*pi`, `57`, `201`, or `Answer: \( 50 \)` without boxed int.
- **Fairness:** Correct rollouts fair. `Answer: \( 50 \)` → parsed `\( 50 \)`, stored wrong — **format FN** if canon fixed; cluster splits from `50` vs `\( 50 \)`.
- **Quotes:** “`\boxed{50}.`”; “`Answer: \( 50 \)`”.

### E4 — `22063de2` · parser-flagged · partial (3/8)

- **Gold:** `20` · **Problem:** Count multiples of 5 up to 100 (expects 20).
- **Pattern:** Three `\boxed{20}`; others `19`, `20%`, `Answer: 20%`, long prose “20% of the positive integers…”.
- **Fairness:** Percent-suffixed tails are mathematically 20 but fail boxed-int `is_correct`. **Unfair** under strict boxed grading; **fair** under loose canon.
- **Quotes:** “`\boxed{20}`”; “`Answer: 20%`”.

### E5 — `56a368fe` · parser-flagged · no-correct

- **Gold:** `110` · **Problem:** Triangle area from \(\tan\angle CAB=22/7\), altitude splits 3+17.
- **Pattern:** Wild spread: `20`, `66`, `660`, `\frac{1190` (truncated frac), `85 square units`. **One rollout** pivots mid-text to a **train speed word problem** (contamination).
- **Fairness:** Nested-boxed rollout fairly wrong; parsed `\frac{1190` is **extractor bug** not model storage bug. Contamination rollout’s parse is nonsense — cluster meaningless.
- **Quotes:** “`\boxed{\frac{1190}{29}}`” in text vs stored parse “`\frac{1190`”; “`A train leaves from station A`…”.

### E6 — `822b2d99` · parser-flagged · no-correct

- **Gold:** `149` · **Problem:** Triangle area; answer \(m+n\).
- **Pattern:** `\boxed{87}`, `77`, `21`, mid-sympy `y_expr = sp.solve…`, truncated “`\boxed{7`”.
- **Fairness:** Truncated tail fairly wrong; sympy mid-code → garbage parse expected.
- **Quotes:** “`\[ \boxed{7`” (no close); “`\boxed{77}`”.

### E7 — `a01882aa` · parser-flagged · partial (5/8)

- **Gold:** `100` · **Problem:** Fifth interior angle of pentagon (sum 540°).
- **Pattern:** Five `\boxed{100}`; others `40 degrees`, spurious `100\pi` from unrelated algebra, `Answer: 100` without boxed.
- **Fairness:** Degree reasoning mostly right when boxed; `100\pi` fairly wrong (wrong extraction path).
- **Quotes:** “`\boxed{100}`”; “`Answer: 40 degrees`”.

### E8 — `068e0bc0` · partial (1/8)

- **Gold:** `5` (`k+m` for ratio) · **Problem:** Cevians in triangle, \(AE=2AF\).
- **Pattern:** Single correct `\boxed{5}`. Others: `\boxed{3}`, `Answer: 2`, sympy coordinate drift, truncated “`Slope of \(`”.
- **Fairness:** One lucky boxed hit vs systematic ratio errors; stored fair.
- **Quotes:** “`k + m = 5` … `\boxed{5}`”; “`\boxed{3}`”.

### E9 — `01677f18` · no-correct · high diversity (8 clusters)

- **Gold:** `9` · **Problem:** Area of \(\triangle MOI\) in 13-12-5 triangle with mixtilinear-style point \(M\).
- **Pattern:** Sympy code fragments, `area_MO`, `\boxed{17}` / `22`, paragraph-long “parsed answers,” one `\boxed{17}` with wrong geometry.
- **Fairness:** All wrong vs 9; labels fair. **Cluster diversity** reflects code garbage + different wrong ints, not meaningful disagreement classes.
- **Quotes:** “`M_y = (a * A.y + …`” (truncated code); “`\boxed{17}`”.

### E10 — `01f3b6f0` · no-correct

- **Gold:** `11` · **Problem:** Box with spheres; find \(k+m+n\).
- **Pattern:** Answers 3, 5, 6, 14, repeated step blocks, one rollout derails into unrelated polynomial \(g(7)=7^3+\cdots\).
- **Fairness:** Fairly wrong; repetition loop visible in 3+ rollouts.
- **Quotes:** “`Answer: 3`”; “`g(7) = 7^3 + 3 \cdot`” (truncated).

### E11 — `037f0e69` · no-correct · nested boxed

- **Gold:** `23` · **Problem:** Area of quadrilateral GAME in 13-14-15 triangle.
- **Pattern:** Two rollouts `\boxed{56}`; one `\boxed{\frac{165}{4}}` stored as parse `\frac{165` (regex truncation); sympy subs lines.
- **Fairness:** 56 wrong; truncated frac parse is **substrate** issue.
- **Quotes:** “`\boxed{\frac{165}{4}}`” vs parsed “`\frac{165`”.

### E12 — `065a820c` · no-correct · random

- **Gold:** `14` · **Problem:** Semicircle diameter 1, maximize \(r+s+t\).
- **Pattern:** `\boxed{3}`, `8`, `11`, `Answer: 55`, coordinate code `# Coordinates of points A`, truncated “`The length`”.
- **Fairness:** All wrong; fair.
- **Quotes:** “`r+s+t=\boxed{3}`”; “`Answer: Answer: 55`” (duplicate Answer).

### E13 — `10f282f7` · partial (5/8) · easy composition

- **Gold:** `24` · **Problem:** Nested \(N,O\) with \(N(x)=2\sqrt x\), \(O(x)=x^2\).
- **Pattern:** Five clean step-by-step `\boxed{24}`; one sympy path prints `86093442` and boxes it; one ends with unrelated set theory after `\boxed{40}`.
- **Fairness:** Clear reasoning on successes; sympy rollout fairly wrong despite code theater.
- **Quotes:** “` ```output 86093442 ``` … \boxed{86093442}`”; “`\boxed{24}`”.

### E14 — `1197ac0a` · partial (6/8)

- **Gold:** `1` · **Problem:** \(M-m\) for \(\frac{|x+y|}{|x|+|y|}\).
- **Pattern:** Six `\boxed{1}`; one `\boxed{\frac{1}{2}}`; one claims \(M=m=1\) → `\boxed{0}`.
- **Fairness:** Fair; shows model can solve easy functional bound problem reliably with boxed format.
- **Quotes:** “`M - m = 1 - 0 = 1` … `\boxed{1}`”; “`\boxed{0}`”.

### E15 — `fec0e932` · partial (6/8)

- **Gold:** `16` · **Problem:** \(3^{-1}+13^{-1}\pmod{19}\).
- **Pattern:** Six correct `\boxed{16}`; one `\boxed{9}`; one **contamination** tail about coin flips after correct setup.
- **Fairness:** Fair except contamination rollout (wrong problem mid-stream).
- **Quotes:** “`\boxed{16}`”; “`10 flips can be anything after that`” (derail).

### E16 — `7ec6f22e` · partial (2/8)

- **Gold:** `0` · **Problem:** Count functions \(f:M\times M\to M\) with involution-like constraint.
- **Pattern:** Two rollouts conclude 0 (one with code `print(answer)`); others `1`, `999`, `2023!`, “insufficient information”.
- **Fairness:** Correct when recognizes impossibility; factorial answer is confidently wrong.
- **Quotes:** “`answer = 0 print(answer)`”; “`\boxed{2023!}`”.

### E17 — `ac8cdbc9` · no-correct · random (grid cars)

- **Gold:** `1148` · **Problem:** Two cars on \(5\times5\) grid, probability → \(100m+n\).
- **Pattern:** All eight wrong combinatorics (102, 124, 268, 312625, …); heavy “Step N” scaffolding, incomplete code.
- **Fairness:** Fairly wrong; no rollout reaches plausible magnitude.
- **Quotes:** “`\boxed{312625}`”; “`Answer: 268`”.

### E18 — `07f4daf2` · no-correct · random (hard recursion)

- **Gold:** `4030` · **Problem:** Nested incircles / intersections count.
- **Pattern:** Answers cluster around 2015–2016, 1008, 4, 1; sympy coordinate line truncates.
- **Fairness:** Fairly wrong; near-miss 2016 suggests pattern guessing not full construction.
- **Quotes:** “`\boxed{2015}`”; “`Ix = (a * A1[0] + b * B1[`”.

### E19 — `a6bce30d` · partial (5/8) · stored-correct substrate

- **Gold:** (problem-specific int) · **Pattern:** Among stored-correct rollouts in run, exemplar class: right `Answer: <n>` in tail, **no** single int `\boxed` — would fail **current** `is_correct` on recompute (part of the 58).
- **Fairness:** **Unfair** under boxed-only grading; fair under human “Answer:” read.
- **Quote class:** “`Answer: 1000`” with correct arithmetic in body, no `\boxed`.

### E20 — `1cc3b783` · partial (5/8) · high diversity

- **Gold:** (varies) · **Pattern:** Representative of 398 prompts with ≥7 clusters: mostly different wrong ints + code tails, not seven solution strategies.
- **Fairness:** Cluster count **overstates** reasoning diversity.

### E21 — `0b9df9e2` · partial · Answer-line correct

- **Pattern:** Stored-correct with `Answer: 71` style tail, no boxed — in the 58 recompute mismatches.
- **Fairness:** See E19.

### E22 — `6137f3cc` · partial · alternating sum

- **Gold:** `1000` · **Pattern:** Multiple rollouts end `Answer: 1000` without `\boxed`, stored correct; sympy variants box wrong values.
- **Fairness:** Right reasoning, grading substrate mismatch for 2+ rollouts.
- **Quotes:** “`100 × 10 = 1000` … `Answer: 1000`”.

---

## 4. Parser / substrate implications

If **`canonicalize_answer`** and **`extract_answer`** were fixed per `PILOT_REDESIGN.md` (C2–C3):

| Change | What we observed in completions | Expected effect |
|--------|----------------------------------|-----------------|
| Brace-balanced `\boxed{}` | 108 rollouts with nested fractions in box; readable correct math in text | Fewer parses like `\frac{1190`; some wrong→right flips (audit: rare net +1 on strict accuracy) |
| Stop stripping `}` / normalize LaTeX | `\( 50 \)`, percent forms | Fewer **format false negatives**; **cluster merge** for same integer (e.g. `cfc7b48f`, `22063de2`) |
| Align `is_correct` with `extract_answer` OR require boxed in prompt only | 58 rollouts correct in artifact but not under current boxed-only `is_correct`; 30+ with gold in `Answer:` line | **+58** rollouts under loose alignment, or **−false hope** if prompt enforces boxed only |
| Deterministic `cluster_id` hash | No completion-text change | Cross-run reproducibility only |
| Truncation handling | ~15 completions end mid-token | Mark `parser_clean=false`; don’t assign spurious cluster |

**What would *not* change much:** Overall ~8% strict accuracy is driven by **model reasoning errors** (~85%+ of rollouts), not parser alone. Fixing substrate mainly improves **measurement fairness** and **cluster interpretability**, not base model competence.

**Minority-correct clusters:** 0 prompts in stored stats — with only 324 correct rollouts spread across 163 prompts, correct mass is usually a single small cluster per prompt; no “right answer in wrong cluster” minority pattern at prompt level.

---

## 5. Recommendations for next experiment runs

1. **Prompt contract:** Require exactly one `\boxed{integer}` (or match training format used at RL time). Reduces 49% no-boxed rollouts and aligns model behavior with `is_correct`.

2. **Fix extract/canonicalize before Run 1 metrics** — implement C2/C3 from `PILOT_REDESIGN.md`; re-score Run 0 offline for parser vs model error attribution.

3. **Cap generation length / detect repetition** — 25% repetition loops waste budget; early-stop on repeated line hashes or step-counter plateau.

4. **Filter or penalize code-mode completions** on non-computational geometry — 28% invoke sympy/code; add few-shot without code for AIME-style items or strip code blocks in postprocess.

5. **Contamination check** — flag completions containing “Solve the following math problem” or unrelated story problems; exclude from cluster stats (9–16 rollouts in this run).

6. **Report two accuracies:** **strict boxed** (current `is_correct`) and **loose extracted** (`canonicalize(extract_answer) == gold`) to separate format from math (quant v2: 58-rollout gap).

---

## Appendix — stratified PID list (deep read)

`ddd26788`, `1653ee27`, `cfc7b48f`, `22063de2`, `56a368fe`, `822b2d99`, `a01882aa`, `068e0bc0`, `01677f18`, `01f3b6f0`, `037f0e69`, `065a820c`, `10f282f7`, `1197ac0a`, `fec0e932`, `7ec6f22e`, `ac8cdbc9`, `07f4daf2`, `a6bce30d`, `1cc3b783`, `0b9df9e2`, `6137f3cc`, plus 8 random (seed 42): `01677f18`, `01f3b6f0`, `037f0e69`, `065a820c`, `07f4daf2`, `068e0bc0`, `0768c7e6`, `07f5b71e`, `0823d9be`, `0b871ed8`, `0b9df9e2`, `0c…` (see `_qual_v2_scratch.json` for full pick list).

Scratch data: `_qual_v2_scratch.json` (machine-generated summaries for this report).
