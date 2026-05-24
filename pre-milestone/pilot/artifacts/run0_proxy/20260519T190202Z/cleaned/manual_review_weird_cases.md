# Manual review: weird cases (Run 0 clean labels)

Human judgment on samples flagged by `clean_run0_artifacts.py`. Completions are immutable; only labels were re-derived.

## 1. Run-on rejected (`extract_path_clean=runon_rejected`)
### `1653ee27-05d2-49ea-b9fb-3cdd58a05730` rollout 5 (line ~6)
- **Gold:** 201
- **Raw parsed:** `Therefore, the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\) is …`
- **Clean parsed:** _(empty)_
- **Completion tail:** "ivide the expression by the divisor \(2^{101} + 2^{51} + 1\) is \(202\).  Therefore, the remainder when \(2^{202} + 202\…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `ac8cdbc9-816e-4c77-aa3c-7ef9b3170d8b` rollout 0 (line ~9)
- **Gold:** 1148
- **Raw parsed:** `Since the problem is quite complex to solve manually, we will assume that the probability …`
- **Clean parsed:** _(empty)_
- **Completion tail:** " that the probability is given in the form $\frac{m}{n}$ where $m$ and $n$ are relatively prime. We can use the fact tha…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `7ec6f22e-5008-43cf-8218-ea0c4ce775ac` rollout 4 (line ~29)
- **Gold:** 0
- **Raw parsed:** `To determine the number of possible functions \( f \) modulo 1000, we can use the fact tha…`
- **Clean parsed:** _(empty)_
- **Completion tail:** "we need to consider the properties of factorials and modular arithmetic more carefully.  To determine the number of poss…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `09a07fe7-fbd8-45ee-9e87-63d37c831153` rollout 2 (line ~35)
- **Gold:** 3
- **Raw parsed:** `Step 23:  Given that \(abc = 3b + 2b + 2 + \frac{2}{b}\), we can simplify this expression …`
- **Clean parsed:** _(empty)_
- **Completion tail:** "ubstituting and simplifying, we find that \(abc = 3b + 2b + 2 + \frac{2}{b}\). Step 23:  Given that \(abc = 3b + 2b + 2 …"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `09a07fe7-fbd8-45ee-9e87-63d37c831153` rollout 5 (line ~38)
- **Gold:** 3
- **Raw parsed:** `So, \(abc = a \cdot \frac{a}{a - 1} \cdot \frac{a(2a - 1)}{a - 1} = \frac{a^3(2a - 1)}{(a …`
- **Clean parsed:** _(empty)_
- **Completion tail:** " a + a^2}{a - 1} = \frac{2a^2 - a}{a - 1} = \frac{a(2a - 1)}{a - 1} \]  So, \(abc = a \cdot \frac{a}{a - 1} \cdot \frac{…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `87125f5a-b1d7-47d0-a222-a85846c2f856` rollout 6 (line ~55)
- **Gold:** 49
- **Raw parsed:** `However, the sum of the squares of the roots must be a non-negative number, so we need to …`
- **Clean parsed:** _(empty)_
- **Completion tail:** "sure we have the maximum possible value. By checking all permutations, we find that the maximum value for the sum of the…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `14647322-e8d8-41a1-a3e5-ac3e865663bb` rollout 6 (line ~79)
- **Gold:** 47
- **Raw parsed:** `Thus, the integral evaluates to $\frac{9! \cdot 8! \cdot 4!}{10!}$. Here, $a = 9$, $b = 8$…`
- **Clean parsed:** _(empty)_
- **Completion tail:** "49875584}{3628800} = \frac{787498755840}{36288000} = \frac{196875}{1} \]  Thus, the integral evaluates to $\frac{9! \cdo…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `5097fb46-8292-4218-b289-6140223ca22f` rollout 2 (line ~91)
- **Gold:** 505
- **Raw parsed:** `However, this approach is not rigorous, and we need to carefully consider the overlaps bet…`
- **Clean parsed:** _(empty)_
- **Completion tail:** "ic progressions for different prime numbers. A more accurate estimation would require a detailed analysis of the overlap…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `ce3091eb-d231-4776-8e7c-1765ff579257` rollout 4 (line ~109)
- **Gold:** 160
- **Raw parsed:** `d = \frac{120 \pm \sqrt{120^2 - 4(1)(-31300)}}{2(1)} = \frac{120 \pm \sqrt{14400 + 125200}…`
- **Clean parsed:** _(empty)_
- **Completion tail:** " = -120 \), and \( c = -31300 \):    \[    d = \frac{120 \pm \sqrt{120^2 - 4(1)(-31300)}}{2(1)} = \frac{120 \pm \sqrt{14…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `57844481-891b-489d-97b3-4f73498383da` rollout 0 (line ~153)
- **Gold:** 638
- **Raw parsed:** `The GCD of 1, 1, 2, 3, 5, and 8 is 1. Therefore, the largest \( d \) such that \( d \) div…`
- **Clean parsed:** _(empty)_
- **Completion tail:** " the first 6 terms of this sequence, which are 1, 1, 2, 3, 5, and 8.  The GCD of 1, 1, 2, 3, 5, and 8 is 1. Therefore, t…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `57844481-891b-489d-97b3-4f73498383da` rollout 5 (line ~158)
- **Gold:** 638
- **Raw parsed:** `Since the GCD of the first two terms is 1, it is clear that the sequence has no common div…`
- **Clean parsed:** _(empty)_
- **Completion tail:** "ear that the sequence has no common divisors other than 1. Therefore, the largest positive integer $d$ such that $d$ div…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `3624db0e-a7fc-4fc6-884b-b71efc624eaa` rollout 1 (line ~162)
- **Gold:** 1980
- **Raw parsed:** `There are no valid solutions for the given problem.`
- **Clean parsed:** _(empty)_
- **Completion tail:** "utions. Step 11:  Given the constraints and the equations, it's possible that no solution exists for the given problem. …"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `3624db0e-a7fc-4fc6-884b-b71efc624eaa` rollout 7 (line ~168)
- **Gold:** 1980
- **Raw parsed:** `Therefore, there are no distinct prime numbers $p$, $q$, and $r$ that satisfy the given eq…`
- **Clean parsed:** _(empty)_
- **Completion tail:** "\frac{55}{4}$$  Again, this solution is not valid, as $qr$ cannot be a fraction.   Therefore, there are no distinct prim…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `c3d5b6b5-eac9-4a7a-8908-e924d74d46b1` rollout 5 (line ~174)
- **Gold:** 6
- **Raw parsed:** `- However, \( n^n \) is not equal to \( n^{2n} \) unless \( n = 1 \), which is not`
- **Clean parsed:** _(empty)_
- **Completion tail:** "^n n^{2n} \), we can see that:      - \( \frac{a^n}{b^n} = (-1)^n \)      - \( n^n = n^{2n} \)    - However, \( n^n \) i…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

### `7215180b-759e-41e9-a371-81c0c16a3ce2` rollout 3 (line ~180)
- **Gold:** 164
- **Raw parsed:** `However, since \(m\) and \(n\) are integers, \(\frac{200}{3}\) must be an integer, which i…`
- **Clean parsed:** _(empty)_
- **Completion tail:** "s in the form \(10^k\). Since we were unable to find any positive integer solutions for \(a\) and \(b\) that satisfy the…"
- **Judgment:** **Fair** — last line is prose/run-on; rejecting avoids garbage cluster. Raw last-line parse was not meaningful.

## 2. Truncated / cut-off `\boxed{`
### `ee4283e3-709c-4a71-88b9-08c98b029a71` rollout 7
- **Gold:** 5
- **Raw parsed:** `\boxed{8`
- **Clean:** path=`last_line`, parsed=`\boxed{8`
- **Tail:** "\] The products are: \(0, 2, 0, 0, 0, 4, 0, 2\).  The sum of these products is: \[ 0 + 2 + 0 + 0 + 0…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `c2c7f62c-9f1f-4fe5-8582-35f8fa16a5c1` rollout 3
- **Gold:** 15
- **Raw parsed:** `15`
- **Clean:** path=`boxed_balanced`, parsed=`15`
- **Tail:** "ers except \(x = 0\). In interval notation, the domain is: \[(-\infty, 0) \cup (0, \infty)\]  The do…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `307a95f6-9495-4cf2-938a-f8d468847f3d` rollout 0
- **Gold:** 214
- **Raw parsed:** `Thus, the smallest possible value of \(a^2 + b^2 + c^2 + d^2\) is \(\boxed{\frac{`
- **Clean:** path=`runon_rejected`, parsed=``
- **Tail:** "s: \[ a^2 + b^2 + c^2 + d^2 = \frac{579 + 1}{3} = \frac{580}{3}. \]  Thus, the smallest possible val…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `29e2fd56-33b5-480a-8959-456b9dc6836c` rollout 0
- **Gold:** 20
- **Raw parsed:** `The radius of \(\Omega_1\) is \(\boxed{4`
- **Clean:** path=`last_line`, parsed=`The radius of \(\Omega_1\) is \(\boxed{4`
- **Tail:** "equation2), (x, R))  # Extract the radius R radius = solution[0][1] print(radius) ``` ```output 4*sq…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `718b2859-ffd7-4201-9e96-c899a29b9b6f` rollout 2
- **Gold:** 16
- **Raw parsed:** `\[ |AC| = \boxed{13 +`
- **Clean:** path=`last_line`, parsed=`\[ |AC| = \boxed{13 +`
- **Tail:** "} - 9\sqrt{6})^2 - 2 \cdot 10 \cdot \sqrt{236 + 18\sqrt{3}} \cdot \frac{\sqrt{2} - \sqrt{6}}{4} \]  …"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `822b2d99-412d-4d82-855c-1f3a313b0b1f` rollout 3
- **Gold:** 149
- **Raw parsed:** `\boxed{7`
- **Clean:** path=`last_line`, parsed=`\boxed{7`
- **Tail:** "the correct interpretation from the symmetry and the conditions given should lead us to a simpler ra…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `29874818-92ad-485c-b651-2893f0b6c588` rollout 4
- **Gold:** 417
- **Raw parsed:** `\boxed{45`
- **Clean:** path=`last_line`, parsed=`\boxed{45`
- **Tail:** ".2428, 0.2429, 0.2430.  This gives us a total of 45 possible values for \( r \). Therefore, the numb…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

### `6ea68b67-aeb7-4673-980a-ae4f8054e7eb` rollout 0
- **Gold:** 1000
- **Raw parsed:** `\boxed{6, 11, 16`
- **Clean:** path=`last_line`, parsed=`\boxed{6, 11, 16`
- **Tail:** "\[ n = 6, 11, 16, \ldots, 996 \]  These are the integers from 6 to 996 with a common difference of 5…"
- **Judgment:** **Still wrong** — generation cut off mid-answer; cleaner cannot recover gold.

## 3. `correct_clean != correct` (all 6 rollouts)
### `65da7224-5f07-48e3-9b01-3c9ea1dfb036` rollout 2
- **Gold:** 87
- **Stored correct:** False → **clean:** True
- **Parsed raw/clean:** `\( 87 \)` / `\( 87 \)`
- **Path clean:** answer_line
- **Tail:** "- \( z = 3 \)  ### Step 5: Compute \( x + y + z \) \[ x + y + z = 48 + 36 + 3 = 87 \]  ###…"
- **Judgment:** **Fair** — stored `correct` used production boxed-only check; clean canon matches gold on same extracted tail.

### `2e690d58-de84-4003-a33f-fbebdb71dae5` rollout 4
- **Gold:** 100
- **Stored correct:** False → **clean:** True
- **Parsed raw/clean:** `\(100\)` / `\(100\)`
- **Path clean:** answer_line
- **Tail:** ".  Thus, the minimum number of dominoes that are entirely inside some \(2 \times 2\) squar…"
- **Judgment:** **Fair** — stored `correct` used production boxed-only check; clean canon matches gold on same extracted tail.

### `70aabfd8-5728-4d08-8363-94e175fc0632` rollout 0
- **Gold:** 1250
- **Stored correct:** False → **clean:** True
- **Parsed raw/clean:** `1250\%` / `1250\%`
- **Path clean:** boxed_balanced
- **Tail:** "e.**    - \( 12.5 \times 100 = 1250\% \).  Therefore, a quarter of a million is \( \boxed{…"
- **Judgment:** **Fair** — stored `correct` used production boxed-only check; clean canon matches gold on same extracted tail.

### `22063de2-a7a2-4214-895f-e015e0b78f87` rollout 2
- **Gold:** 20
- **Stored correct:** False → **clean:** True
- **Parsed raw/clean:** `20%` / `20%`
- **Path clean:** answer_line
- **Tail:** " 100  Plugging in the values we found:  Percent = ($20 / 100$) * 100  Percent = $0.2$ * 10…"
- **Judgment:** **Fair** — stored `correct` used production boxed-only check; clean canon matches gold on same extracted tail.

### `22063de2-a7a2-4214-895f-e015e0b78f87` rollout 7
- **Gold:** 20
- **Stored correct:** False → **clean:** True
- **Parsed raw/clean:** `$20\%$` / `$20\%$`
- **Path clean:** answer_line
- **Tail:** "$ positive integers less than or equal to $100$, the percentage is:  $\frac{20}{100} \cdot…"
- **Judgment:** **Fair** — stored `correct` used production boxed-only check; clean canon matches gold on same extracted tail.

### `cfc7b48f-94bf-429f-b1c9-a7ac15e86b80` rollout 0
- **Gold:** 50
- **Stored correct:** False → **clean:** True
- **Parsed raw/clean:** `\( 50 \)` / `\( 50 \)`
- **Path clean:** answer_line
- **Tail:** "i r^2 = \pi (4)^2 = 16\pi \] To the nearest positive integer, this is \( 50 \).  ---  ### …"
- **Judgment:** **Fair** — stored `correct` used production boxed-only check; clean canon matches gold on same extracted tail.

## 4. Parsed changed, correctness unchanged (sample)
### `1653ee27-05d2-49ea-b9fb-3cdd58a05730` rollout 5
- **Gold:** 201 | correct: False (both labels)
- **Raw:** `Therefore, the remainder when \(2^{202} + 202\) is divided by \(2^{101…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `ac8cdbc9-816e-4c77-aa3c-7ef9b3170d8b` rollout 0
- **Gold:** 1148 | correct: False (both labels)
- **Raw:** `Since the problem is quite complex to solve manually, we will assume t…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `a8c414e5-c522-49d6-a1af-2afcb37e3ddc` rollout 7
- **Gold:** 3 | correct: False (both labels)
- **Raw:** `\text{No such`
- **Clean:** `\text{No such } k \text{ exists.}` (boxed_balanced)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `7ec6f22e-5008-43cf-8218-ea0c4ce775ac` rollout 4
- **Gold:** 0 | correct: False (both labels)
- **Raw:** `To determine the number of possible functions \( f \) modulo 1000, we …`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `09a07fe7-fbd8-45ee-9e87-63d37c831153` rollout 2
- **Gold:** 3 | correct: False (both labels)
- **Raw:** `Step 23:  Given that \(abc = 3b + 2b + 2 + \frac{2}{b}\), we can simpl…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `09a07fe7-fbd8-45ee-9e87-63d37c831153` rollout 5
- **Gold:** 3 | correct: False (both labels)
- **Raw:** `So, \(abc = a \cdot \frac{a}{a - 1} \cdot \frac{a(2a - 1)}{a - 1} = \f…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `56a368fe-51a8-4879-9b96-053ea9485fea` rollout 1
- **Gold:** 110 | correct: False (both labels)
- **Raw:** `\frac{1190`
- **Clean:** `\frac{1190}{29}` (boxed_balanced)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `56a368fe-51a8-4879-9b96-053ea9485fea` rollout 7
- **Gold:** 110 | correct: False (both labels)
- **Raw:** `\frac{660`
- **Clean:** `\frac{660}{7}` (boxed_balanced)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `87125f5a-b1d7-47d0-a222-a85846c2f856` rollout 6
- **Gold:** 49 | correct: False (both labels)
- **Raw:** `However, the sum of the squares of the roots must be a non-negative nu…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `a6bce30d-9781-402b-95ae-882c43e72b79` rollout 6
- **Gold:** 296 | correct: False (both labels)
- **Raw:** `\frac{19448`
- **Clean:** `\frac{19448}{9}` (boxed_balanced)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `14647322-e8d8-41a1-a3e5-ac3e865663bb` rollout 6
- **Gold:** 47 | correct: False (both labels)
- **Raw:** `Thus, the integral evaluates to $\frac{9! \cdot 8! \cdot 4!}{10!}$. He…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `5097fb46-8292-4218-b289-6140223ca22f` rollout 2
- **Gold:** 505 | correct: False (both labels)
- **Raw:** `However, this approach is not rigorous, and we need to carefully consi…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `ce3091eb-d231-4776-8e7c-1765ff579257` rollout 4
- **Gold:** 160 | correct: False (both labels)
- **Raw:** `d = \frac{120 \pm \sqrt{120^2 - 4(1)(-31300)}}{2(1)} = \frac{120 \pm \…`
- **Clean:** `` (runon_rejected)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

### `ce3091eb-d231-4776-8e7c-1765ff579257` rollout 6
- **Gold:** 160 | correct: False (both labels)
- **Raw:** `40\sqrt{1045`
- **Clean:** `40\sqrt{1045}` (boxed_balanced)
- **Judgment:** **Fair** — brace-balanced boxed or run-on fix; still wrong/right as before.

## 5. Nested boxed mismatch (sample)
### `a8c414e5-c522-49d6-a1af-2afcb37e3ddc` rollout 7
- **Balanced inner:** `\text{No such } k \text{ exists.}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\text{No such` → `\text{No such } k \text{ exists.}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `56a368fe-51a8-4879-9b96-053ea9485fea` rollout 1
- **Balanced inner:** `\frac{1190}{29}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\frac{1190` → `\frac{1190}{29}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `56a368fe-51a8-4879-9b96-053ea9485fea` rollout 7
- **Balanced inner:** `\frac{660}{7}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\frac{660` → `\frac{660}{7}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `a6bce30d-9781-402b-95ae-882c43e72b79` rollout 6
- **Balanced inner:** `\frac{19448}{9}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\frac{19448` → `\frac{19448}{9}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `ce3091eb-d231-4776-8e7c-1765ff579257` rollout 6
- **Balanced inner:** `40\sqrt{1045}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `40\sqrt{1045` → `40\sqrt{1045}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `cfecb90b-3f7d-4493-af64-ff306ba84d0f` rollout 4
- **Balanced inner:** `\frac{3}{2}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\frac{3` → `\frac{3}{2}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `cfecb90b-3f7d-4493-af64-ff306ba84d0f` rollout 6
- **Balanced inner:** `\frac{2}{1}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\frac{2` → `\frac{2}{1}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

### `cfecb90b-3f7d-4493-af64-ff306ba84d0f` rollout 7
- **Balanced inner:** `\frac{1}{2}`
- **Shallow inner:** ``
- **Raw/clean parsed:** `\frac{1` → `\frac{1}{2}`
- **Judgment:** **Fair** — clean prefers balanced last boxed.

## 6. Cleaner bugs found

- **Leading-integer peel (fixed):** Early draft mapped `3, 4, 5` and `3.19` to `3` via aggressive prefix rule; removed before final metrics.
- **No remaining false correct flips** in the 6 rollout gains; each is LaTeX wrapper / percent normalization vs stored boxed-only `correct`.
- **Run-on reject** may be aggressive on long but valid `Answer:` lines with inline math; manual spot-check shows most are genuinely non-answers.
