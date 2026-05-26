# Prove-prompt wording spot check (n=80)

## Method

- **Source:** `main/data/source/polaris_train_full.jsonl` (53,291 rows; unfiltered pool).
- **Pool A (`contains_prove`):** `"prove" in problem.lower()` → **2,756** rows.
- **Pool B (`last_starts_prove`):** last sentence (see below) matches `^prove\b` (case-insensitive) → **1,507** rows.
- **Last-sentence split:** `re.split(r'(?<=[.!?])\s+|\n+', text.strip())`, take final non-empty segment. Caveat: parentheticals, display math `\]`, and multi-part prompts `(a)/(b)` can make the “last sentence” a non-instruction fragment (e.g. A08, A11, A20, A34, A40).
- **Sampling:** Pool A — `random.Random(42).sample(..., 40)`; Pool B — `random.Random(99).sample(..., 40)`. Labels assigned by reading full `problem` text (not gold-only).
- **Overlap:** row in both pools (B ⊆ A by construction).

## Summary — label counts

| Label | Pool A (n=40) | Pool B (n=40) |
|-------|---------------|---------------|
| `proof` | 28 | 32 |
| `single_answer` | 6 | 5 |
| `show_equality` | 5 | 3 |
| `find_compute` | 1 | 0 |
| `other` | 0 | 0 |

## Overlap (Pool B sample)

- All **40/40** Pool B samples are also in Pool A (`overlap=True`).
- Of B's 40: **`proof` + `show_equality` = 35** (pure proof-style prompts); **`single_answer` = 5** (prove + extract constant/count/coordinates).

## Pool A only (not in B)

- **14/40** Pool A samples are **A-only** (last sentence does not start with “Prove”).

### A-only label breakdown

| Label | Count |
|-------|-------|
| `proof` | 5 |
| `single_answer` | 5 |
| `show_equality` | 3 |
| `find_compute` | 1 |

A-only patterns: “Find all … and prove no others” (A01); formula with trailing parenthetical (A08); mid-body “Prove that …” with non-prove last line (A10, A31); equality in prose not starting last sentence (A11); multi-part limits (A13, A39); counting with “prove” only in part (a) (A34); fill-in count with awkward “to prove” (A37); mixed Show/What row (A40).

## Key finding

Among problems that **contain** “prove” (Pool A sample), the majority (**28/40, 70%**) are genuine **proof** requests (existence, divisibility, inequalities, concurrency, etc.), with **5** **`show_equality`** identity-style items (**33/40, 83%** proof-like overall). **Single-answer** tasks that use prove language for completeness or as part of a multi-step prompt account for **6/40 (15%)** — e.g. find all solutions then prove none remain (A01), find limit + prove convergence (A13, A39), identify a closed form (A32), or count solutions (A34, A37).

Restricting to problems whose **last sentence starts with “Prove”** (Pool B) sharpens the distribution further: **35/40 (88%)** are `proof` or `show_equality`; only **5/40 (12%)** are `single_answer`, typically “prove constant and find its value” (B04) or “prove uniqueness / solve then verify” (B17, B22, B38).

**Takeaway:** “Contains prove” is **not** dominated by single-answer drills, but it is **broader** than “ends with Prove”: ~35% of the A sample either does not end with Prove (14/40) or couples prove with find/compute/count (6/40 single_answer + 1 find_compute). For reward matching, last-sentence “Prove …” items align more cleanly with proof-style gold (often symbolic sentences), while A-only rows are where compute/count language and split-sentence artifacts cluster.

## Appendix — all 80 rows

| problem_id | pool | label | overlap | gold (40c) | note |
|------------|------|-------|---------|------------|------|
| 49542 | A | `single_answer` | N | `14` | Find all originals; prove completeness; gold=14 |
| 8589 | A | `proof` | Y | `Thesequence{a_n}isanarithmeticsequence.` | Prove sequence is arithmetic |
| 1805 | A | `proof` | Y | `360` | Prove dihedral-angle sum is 360° |
| 21382 | A | `proof` | Y | `V \leq \sqrt{S_1 S_2 S_3}` | Volume projection inequality |
| 18940 | A | `proof` | Y | `1+\sqrt{2}` | Projection ratio upper bound |
| 17415 | A | `proof` | Y | `|f(z_0)| \geq |C_0| + |C_n|` | Polynomial value at |z|≤1 |
| 10711 | A | `proof` | Y | `\frac{1}{2^{n-1}}\lea_n\le1-\sin^n2\the…` | Sequence bounds in θ |
| 7871 | A | `proof` | N | `t_{12}=\frac{}{r}\sqrt{(r\r_1)(r\r_2)}` | Prove tangent-length formula (note after) |
| 40837 | A | `proof` | Y | `\sum_{i=1}^{n}(-1)^{i} a_{i} \text{ is …` | Alternating coeff sum is real |
| 6684 | A | `proof` | N | `4qx\lep^2` | Cubic roots satisfy 4qx≤p² |
| 44661 | A | `show_equality` | N | `5` | Chord product squared equals 5 |
| 30711 | A | `proof` | Y | `0` | Sum powers divisible by p² |
| 2114 | A | `single_answer` | N | `1` | (a) limit=1; (b) eventual monotonicity |
| 2030 | A | `proof` | Y | `17.5` | Area bound S≤17.5 |
| 7289 | A | `proof` | Y | `\frac{17}{10} \sum_{i=1}^{n} a_{i}^{2}` | Inequality + equality conditions |
| 17031 | A | `proof` | Y | `2` | Grid line bichromatic nodes |
| 18057 | A | `proof` | Y | `3` | Some line through exactly two points |
| 37436 | A | `proof` | Y | `R \text{ lies on } BD` | Point R on diagonal BD |
| 45382 | A | `proof` | Y | `\frac{b}{2}` | Incircle tangency iff c=(a+b)/2 |
| 1844 | A | `show_equality` | N | `d_{\mathrm{}}d_{\mathrm{}}d_{\mathrm{ca…` | Product of distances equality |
| 42251 | A | `proof` | Y | `(1,2)\equiv(2,1)\pmod{3}` | m(1,2)≡m(2,1) mod 3 |
| 15558 | A | `show_equality` | N | `\frac{1}{R_{1}}+\frac{1}{R_{2}}=\frac{2…` | 1/R₁+1/R₂=2/d |
| 50555 | A | `proof` | Y | `\lfloor^{2014}\rfloor+\lfloorb^{2014}\r…` | Floor identity at exponent 2014 |
| 40802 | A | `proof` | Y | `F(p)\ge(p+k)^4` | F(p)≥(p+k)⁴ |
| 30611 | A | `proof` | Y | `f(n)=n` | Functional equation forces f(n)=n |
| 17208 | A | `show_equality` | Y | `\frac{b+}{}` | AE/ED=(b+c)/a |
| 32469 | A | `proof` | N | `\geb` | Prove c≥b divisibility setup |
| 44577 | A | `proof` | Y | `\text{All numbers in } N \text{ must be…` | Two-coloring forces one color |
| 21728 | A | `proof` | Y | `10` | Divisibility by 10 (gold is modulus) |
| 401 | A | `proof` | Y | `\text{The lines } Q_1P_2, Q_2P_1, \text…` | Concurrency of three lines |
| 12494 | A | `proof` | N | `\sum_{k=1}^{n}a_{k}<1` | Partial sum <1 (prove in body, not last sent.) |
| 30770 | A | `single_answer` | Y | `1994-n` | Identify floor(a_n)=1994−n |
| 25335 | A | `proof` | Y | `NP=PQ=QG` | Area ratio + NP=PQ=QG |
| 21708 | A | `single_answer` | N | `24` | Count config automorphisms (24 for 6-pt) |
| 12162 | A | `show_equality` | Y | `2\cos(\alpha-\varphi)=7\cos(\beta-\gamm…` | Trig identity from system |
| 16842 | A | `proof` | N | `4` | Area relations (IMO-style) |
| 25104 | A | `single_answer` | N | `336005` | Count (x,y,z) with x<y<z; gold=336005 |
| 7868 | A | `proof` | Y | `Thesequence{x_n}isgeometricprogression.` | Sequence is geometric progression |
| 7225 | A | `single_answer` | N | `e^2` | Limit of b_n^n is e² |
| 28028 | A | `find_compute` | N | `12` | Row index for 2004; part 1 is Show not Prove |
| 27158 | B | `proof` | Y | `\frac{3}{2\sqrt{2}}` | Tetrahedron in sphere radius 3/(2√2) |
| 25778 | B | `proof` | Y | `k` | k is a perfect square |
| 13838 | B | `show_equality` | Y | `\frac{x}{^{n-1}}+\frac{y}{b^{n-1}}+\fra…` | Signed segment sum identity |
| 40682 | B | `single_answer` | Y | `\sqrt{2}` | Constant ratio; find √2 |
| 12470 | B | `proof` | Y | `2` | Plane through KL bisects volume |
| 16004 | B | `proof` | Y | `1` | Some of BD, CD non-integer |
| 17237 | B | `show_equality` | Y | `\frac{\sqrt{p}}{2}` | Circumradius equals √p/2 |
| 9179 | B | `proof` | Y | `2` | Median triangle similar to original |
| 5424 | B | `proof` | Y | `45` | Verify tangent sum equals 45 |
| 17446 | B | `proof` | Y | `2` | Infinitely many triple sums of two squares |
| 52264 | B | `proof` | Y | `2S` | Hexagon area 2S |
| 25891 | B | `single_answer` | Y | `1994-n` | Floor(a_n)=1994−n (duplicate theme A32) |
| 35025 | B | `proof` | Y | `1000` | Existence of far pair (gold=1000 cm) |
| 47270 | B | `show_equality` | Y | `4r^2` | OP²+OQ²+OR²+OS²=4r² |
| 48996 | B | `proof` | Y | `^3+b^3+3abc>^3` | Triangle side cubic inequality |
| 35383 | B | `proof` | Y | `\text{The circumcircle of } \triangle P…` | Circumcircle tangent to BP, BR |
| 5755 | B | `single_answer` | Y | `(8, 14, 18)` | Unique GCD triplet (8,14,18) |
| 42558 | B | `proof` | Y | `\frac{p(1-p)}{2}` | Translation intersection length bound |
| 32008 | B | `proof` | Y | `\dfrac{1}{3} \leqslant \dfrac{V_{1}}{V}…` | Volume ratio in tetrahedron slice |
| 13824 | B | `proof` | Y | `\varphi(n)` | ∃m: a^m−1 divisible by n (gold φ(n)) |
| 28262 | B | `proof` | Y | `2^{2n}` | Polynomial bound on [-n,n] |
| 51791 | B | `single_answer` | Y | `-\frac{}{2}` | Solve cubic + root integrality |
| 41672 | B | `proof` | Y | `^2+b^2=2c^2` | Iff a²+b²=2c² for Euler line |
| 39718 | B | `proof` | Y | `Q` | Q is midpoint of arc BAC |
| 14920 | B | `proof` | Y | `\dfrac{x^2}{4} - y^2 = 1` | PQ through fixed point (part 2) |
| 25356 | B | `proof` | Y | `p-r=1` | Euler p−q+r=1 for dissection |
| 26175 | B | `proof` | Y | `a_n` | Sequence strictly increasing |
| 46972 | B | `proof` | Y | `243` | 243 ones divisible by 243 |
| 14429 | B | `proof` | Y | `T_{2023}` | T_2023 odd |
| 10364 | B | `proof` | Y | `n` | Snail returns only after integer hours |
| 30598 | B | `proof` | Y | `\cot(\frac{\pi}{2^{2018}})` | ∃a giving exactly 2018 distinct terms |
| 45256 | B | `proof` | Y | `a_{100}>14` | a_100>14 |
| 37995 | B | `proof` | Y | `-7 \leq f(x) \leq 7` | Quadratic bound on [-2,2] |
| 23422 | B | `proof` | Y | `X\ge\max{3a,3b,3c}` | X bound + sqrt relation |
| 5136 | B | `proof` | Y | `\frac{1}{\alpha\beta}+\frac{1}{\beta\ga…` | Area reciprocal inequality |
| 50945 | B | `proof` | Y | `10` | Independent 10-set in hat graph |
| 34811 | B | `proof` | Y | `1996` | Hotel revenue cap 1996 |
| 46638 | B | `single_answer` | Y | `(\sqrt{2}-\frac{1}{2},1)` | Fixed point coords on C₂ |
| 5472 | B | `proof` | Y | `\text{The projections of } M \text{ for…` | Projections form equilateral triangle |
| 30943 | B | `proof` | Y | `0 \leq a_{k} - a_{k+1} < \frac{2}{k^2}` | Gap a_k−a_{k+1} bound |
