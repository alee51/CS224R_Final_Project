#!/usr/bin/env python3
"""Manual labels for prove-prompt spotcheck (seed 42 / 99 samples)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# (pool, problem_id, label, gold_type, note)
LABELS = [
    ("A", 49542, "single_answer", "integer-like", "Find all originals; prove completeness; gold=14"),
    ("A", 8589, "proof", "string/word", "Prove sequence is arithmetic"),
    ("A", 1805, "proof", "integer-like", "Prove dihedral-angle sum is 360°"),
    ("A", 21382, "proof", "latex/symbolic", "Volume projection inequality"),
    ("A", 18940, "proof", "latex/symbolic", "Projection ratio upper bound"),
    ("A", 17415, "proof", "latex/symbolic", "Polynomial value at |z|≤1"),
    ("A", 10711, "proof", "latex/symbolic", "Sequence bounds in θ"),
    ("A", 7871, "proof", "latex/symbolic", "Prove tangent-length formula (note after)"),
    ("A", 40837, "proof", "latex/symbolic", "Alternating coeff sum is real"),
    ("A", 6684, "proof", "latex/symbolic", "Cubic roots satisfy 4qx≤p²"),
    ("A", 44661, "show_equality", "integer-like", "Chord product squared equals 5"),
    ("A", 30711, "proof", "integer-like", "Sum powers divisible by p²"),
    ("A", 2114, "single_answer", "integer-like", "(a) limit=1; (b) eventual monotonicity"),
    ("A", 2030, "proof", "latex/symbolic", "Area bound S≤17.5"),
    ("A", 7289, "proof", "latex/symbolic", "Inequality + equality conditions"),
    ("A", 17031, "proof", "integer-like", "Grid line bichromatic nodes"),
    ("A", 18057, "proof", "integer-like", "Some line through exactly two points"),
    ("A", 37436, "proof", "string/word", "Point R on diagonal BD"),
    ("A", 45382, "proof", "latex/symbolic", "Incircle tangency iff c=(a+b)/2"),
    ("A", 1844, "show_equality", "latex/symbolic", "Product of distances equality"),
    ("A", 42251, "proof", "latex/symbolic", "m(1,2)≡m(2,1) mod 3"),
    ("A", 15558, "show_equality", "latex/symbolic", "1/R₁+1/R₂=2/d"),
    ("A", 50555, "proof", "latex/symbolic", "Floor identity at exponent 2014"),
    ("A", 40802, "proof", "latex/symbolic", "F(p)≥(p+k)⁴"),
    ("A", 30611, "proof", "latex/symbolic", "Functional equation forces f(n)=n"),
    ("A", 17208, "show_equality", "latex/symbolic", "AE/ED=(b+c)/a"),
    ("A", 32469, "proof", "latex/symbolic", "Prove c≥b divisibility setup"),
    ("A", 44577, "proof", "string/word", "Two-coloring forces one color"),
    ("A", 21728, "proof", "integer-like", "Divisibility by 10 (gold is modulus)"),
    ("A", 401, "proof", "string/word", "Concurrency of three lines"),
    ("A", 12494, "proof", "latex/symbolic", "Partial sum <1 (prove in body, not last sent.)"),
    ("A", 30770, "single_answer", "latex/symbolic", "Identify floor(a_n)=1994−n"),
    ("A", 25335, "proof", "latex/symbolic", "Area ratio + NP=PQ=QG"),
    ("A", 21708, "single_answer", "integer-like", "Count config automorphisms (24 for 6-pt)"),
    ("A", 12162, "show_equality", "latex/symbolic", "Trig identity from system"),
    ("A", 16842, "proof", "integer-like", "Area relations (IMO-style)"),
    ("A", 25104, "single_answer", "integer-like", "Count (x,y,z) with x<y<z; gold=336005"),
    ("A", 7868, "proof", "string/word", "Sequence is geometric progression"),
    ("A", 7225, "single_answer", "latex/symbolic", "Limit of b_n^n is e²"),
    ("A", 28028, "find_compute", "integer-like", "Row index for 2004; part 1 is Show not Prove"),
    ("B", 27158, "proof", "latex/symbolic", "Tetrahedron in sphere radius 3/(2√2)"),
    ("B", 25778, "proof", "string/word", "k is a perfect square"),
    ("B", 13838, "show_equality", "latex/symbolic", "Signed segment sum identity"),
    ("B", 40682, "single_answer", "latex/symbolic", "Constant ratio; find √2"),
    ("B", 12470, "proof", "integer-like", "Plane through KL bisects volume"),
    ("B", 16004, "proof", "integer-like", "Some of BD, CD non-integer"),
    ("B", 17237, "show_equality", "latex/symbolic", "Circumradius equals √p/2"),
    ("B", 9179, "proof", "integer-like", "Median triangle similar to original"),
    ("B", 5424, "proof", "integer-like", "Verify tangent sum equals 45"),
    ("B", 17446, "proof", "integer-like", "Infinitely many triple sums of two squares"),
    ("B", 52264, "proof", "latex/symbolic", "Hexagon area 2S"),
    ("B", 25891, "single_answer", "latex/symbolic", "Floor(a_n)=1994−n (duplicate theme A32)"),
    ("B", 35025, "proof", "integer-like", "Existence of far pair (gold=1000 cm)"),
    ("B", 47270, "show_equality", "latex/symbolic", "OP²+OQ²+OR²+OS²=4r²"),
    ("B", 48996, "proof", "latex/symbolic", "Triangle side cubic inequality"),
    ("B", 35383, "proof", "string/word", "Circumcircle tangent to BP, BR"),
    ("B", 5755, "single_answer", "string/word", "Unique GCD triplet (8,14,18)"),
    ("B", 42558, "proof", "latex/symbolic", "Translation intersection length bound"),
    ("B", 32008, "proof", "latex/symbolic", "Volume ratio in tetrahedron slice"),
    ("B", 13824, "proof", "latex/symbolic", "∃m: a^m−1 divisible by n (gold φ(n))"),
    ("B", 28262, "proof", "latex/symbolic", "Polynomial bound on [-n,n]"),
    ("B", 51791, "single_answer", "latex/symbolic", "Solve cubic + root integrality"),
    ("B", 41672, "proof", "latex/symbolic", "Iff a²+b²=2c² for Euler line"),
    ("B", 39718, "proof", "string/word", "Q is midpoint of arc BAC"),
    ("B", 14920, "proof", "latex/symbolic", "PQ through fixed point (part 2)"),
    ("B", 25356, "proof", "latex/symbolic", "Euler p−q+r=1 for dissection"),
    ("B", 26175, "proof", "latex/symbolic", "Sequence strictly increasing"),
    ("B", 46972, "proof", "integer-like", "243 ones divisible by 243"),
    ("B", 14429, "proof", "latex/symbolic", "T_2023 odd"),
    ("B", 10364, "proof", "string/word", "Snail returns only after integer hours"),
    ("B", 30598, "proof", "latex/symbolic", "∃a giving exactly 2018 distinct terms"),
    ("B", 45256, "proof", "latex/symbolic", "a_100>14"),
    ("B", 37995, "proof", "latex/symbolic", "Quadratic bound on [-2,2]"),
    ("B", 23422, "proof", "latex/symbolic", "X bound + sqrt relation"),
    ("B", 5136, "proof", "latex/symbolic", "Area reciprocal inequality"),
    ("B", 50945, "proof", "integer-like", "Independent 10-set in hat graph"),
    ("B", 34811, "proof", "integer-like", "Hotel revenue cap 1996"),
    ("B", 46638, "single_answer", "latex/symbolic", "Fixed point coords on C₂"),
    ("B", 5472, "proof", "string/word", "Projections form equilateral triangle"),
    ("B", 30943, "proof", "latex/symbolic", "Gap a_k−a_{k+1} bound"),
]

OVERLAP = {r[1]: r[0] == "A" for r in LABELS}  # placeholder


def truncate(s: str, n: int = 40) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    data = json.loads(
        Path("main/scripts/_prove_spotcheck_samples.json").read_text()
    )
    by_id = {}
    for row in data["sample_a"] + data["sample_b"]:
        by_id[row["problem_id"]] = row

    rows = []
    for pool, pid, label, gold_type, note in LABELS:
        row = by_id[pid]
        overlap = row.get("overlap", pool == "B")
        rows.append(
            {
                "pool": pool,
                "problem_id": pid,
                "label": label,
                "overlap": overlap,
                "gold_type": gold_type,
                "gold": row["gold"],
                "note": note,
            }
        )

    a_rows = [r for r in rows if r["pool"] == "A"]
    b_rows = [r for r in rows if r["pool"] == "B"]
    a_only = [r for r in a_rows if not r["overlap"]]

    def counts(rs):
        return Counter(r["label"] for r in rs)

    ca, cb = counts(a_rows), counts(b_rows)
    b_proofish = sum(cb[l] for l in ("proof", "show_equality"))
    b_single = cb["single_answer"]

    lines = [
        "# Prove-prompt wording spot check (n=80)",
        "",
        "## Method",
        "",
        "- **Source:** `main/data/source/polaris_train_full.jsonl` (53,291 rows).",
        '- **Pool A (`contains_prove`):** `"prove" in problem.lower()` → **2,756** rows.',
        '- **Pool B (`last_starts_prove`):** last sentence (see below) matches `^prove\\b` (case-insensitive) → **1,507** rows.',
        "- **Last-sentence split:** `re.split(r'(?<=[.!?])\\s+|\\n+', text.strip())`, take final non-empty segment. Caveat: parentheticals, display math `\\]`, and multi-part prompts `(a)/(b)` can make the “last sentence” a non-instruction fragment (e.g. A08, A11, A20, A34, A40).",
        "- **Sampling:** Pool A — `random.Random(42).sample(..., 40)`; Pool B — `random.Random(99).sample(..., 40)`. Labels assigned by reading full `problem` text (not gold-only).",
        "- **Overlap:** row in both pools (B ⊆ A by construction).",
        "",
        "## Summary — label counts",
        "",
        "| Label | Pool A (n=40) | Pool B (n=40) |",
        "|-------|---------------|---------------|",
    ]
    all_labels = ["proof", "single_answer", "show_equality", "find_compute", "other"]
    for lab in all_labels:
        lines.append(f"| `{lab}` | {ca.get(lab, 0)} | {cb.get(lab, 0)} |")
    lines += [
        "",
        "## Overlap (Pool B sample)",
        "",
        f"- All **40/40** Pool B samples are also in Pool A (`overlap=True`).",
        f"- Of B's 40: **`proof` + `show_equality` = {b_proofish}** (pure proof-style prompts); **`single_answer` = {b_single}** (prove + extract constant/count/coordinates).",
        "",
        "## Pool A only (not in B)",
        "",
        f"- **{len(a_only)}/40** Pool A samples are **A-only** (last sentence does not start with “Prove”).",
        "",
        "### A-only label breakdown",
        "",
        "| Label | Count |",
        "|-------|-------|",
    ]
    for lab in all_labels:
        c = sum(1 for r in a_only if r["label"] == lab)
        if c:
            lines.append(f"| `{lab}` | {c} |")
    lines += [
        "",
        "A-only patterns: “Find all … and prove no others” (A01); formula with trailing parenthetical (A08); mid-body “Prove that …” with non-prove last line (A10, A31); equality in prose not starting last sentence (A11); multi-part limits (A13, A39); counting with “prove” only in part (a) (A34); fill-in count with awkward “to prove” (A37); mixed Show/What row (A40).",
        "",
        "## Key finding",
        "",
        "Among problems that **contain** “prove” (Pool A sample), the majority (**28/40, 70%**) are genuine **proof** requests (existence, divisibility, inequalities, concurrency, etc.), with **5** **`show_equality`** identity-style items (**33/40, 83%** proof-like overall). **Single-answer** tasks that use prove language for completeness or as part of a multi-step prompt account for **6/40 (15%)** — e.g. find all solutions then prove none remain (A01), find limit + prove convergence (A13, A39), identify a closed form (A32), or count solutions (A34, A37).",
        "",
        "Restricting to problems whose **last sentence starts with “Prove”** (Pool B) sharpens the distribution further: **35/40 (88%)** are `proof` or `show_equality`; only **5/40 (12%)** are `single_answer`, typically “prove constant and find its value” (B04) or “prove uniqueness / solve then verify” (B17, B22, B38).",
        "",
        "**Takeaway:** “Contains prove” is **not** dominated by single-answer drills, but it is **broader** than “ends with Prove”: ~35% of the A sample either does not end with Prove (14/40) or couples prove with find/compute/count (6/40 single_answer + 1 find_compute). For reward matching, last-sentence “Prove …” items align more cleanly with proof-style gold (often symbolic sentences), while A-only rows are where compute/count language and split-sentence artifacts cluster.",
        "",
        "## Appendix — all 80 rows",
        "",
        "| problem_id | pool | label | overlap | gold (40c) | note |",
        "|------------|------|-------|---------|------------|------|",
    ]
    for r in rows:
        ov = "Y" if r["overlap"] else "N"
        lines.append(
            f"| {r['problem_id']} | {r['pool']} | `{r['label']}` | {ov} | `{truncate(r['gold'])}` | {r['note']} |"
        )

    out = Path("main/docs/probes/prove_prompt_spotcheck_80.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
