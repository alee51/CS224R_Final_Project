"""Clustering substrates for set-based arms.

Two substrates share the same per-prompt shape (length-N list of int cluster ids):
  - answer_hash_clusters: parsed-answer identity (arms 2 + 4). Two-pass:
      1. canonicalize string normalization (textual noise)
      2. symmetric sympy union-find (genuine math equivalence)
  - cot_clusters_from_judge: Poly-EPO LLM judge `cluster_id` (arm 3).

Parse-failed rollouts get unique negative ids so they never collide and never
create artificial minority clusters.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable

from train.math_grade_deepscaler import grade_answer_mathd_or_sympy


# --- Pass 1: string canonicalize -------------------------------------------------

# LaTeX inline-math wrappers / common textual noise the Rank-2 extractor lets
# through. Stripped before integer-parse so '\\(36\\).' canonicalizes to '36'.
_TRAILING_NOISE = re.compile(r"[.}\]\\)\s]+$")
_LEADING_NOISE = re.compile(r"^[\\\s({\[\$]+")
_INLINE_PAREN = re.compile(r"\\\(|\\\)")
_DOLLAR_MATH = re.compile(r"^\$+|\$+$")


def _strip_boxed(s: str) -> str:
    """Unwrap \\boxed{...} once (no recursion; outer wrapper is what leaks)."""
    m = re.match(r"^\\boxed\{(.*)\}$", s)
    return m.group(1) if m else s


def canonicalize_answer(text: str | None) -> str:
    """Aggressive textual normalization for clustering.

    Steps (order matters):
      1. strip surrounding whitespace, drop commas
      2. strip \\( \\) inline-math wrappers
      3. strip leading/trailing $ (display math), `[`, `(`, `{`
      4. unwrap \\boxed{...}
      5. strip trailing punctuation: `.`, `}`, `]`, `\\)`, whitespace
      6. if int-parseable -> str(int)
      7. if float-parseable and float == int(float) -> str(int(float))  (catches '60.0' -> '60', '3160.0000000000002' -> '3160')
      8. else lowercase
    """
    if text is None:
        return ""
    s = text.strip().replace(",", "")
    s = _INLINE_PAREN.sub("", s)
    s = _DOLLAR_MATH.sub("", s)
    s = _strip_boxed(s)
    s = _TRAILING_NOISE.sub("", s)
    s = _LEADING_NOISE.sub("", s)
    s = s.strip()
    if not s:
        return ""
    try:
        return str(int(s))
    except ValueError:
        pass
    try:
        f = float(s)
        # Normalize int-valued floats (handles '60.0', '3160.0000000000002').
        # Guard inf/nan: float('inf') parses but round(inf) raises OverflowError.
        if math.isfinite(f) and abs(f - round(f)) < 1e-9 and abs(f) < 1e15:
            return str(int(round(f)))
    except ValueError:
        pass
    return s.lower()


def canonicalize_answer_old(text: str | None) -> str:
    """Pre-v1 pilot canonicalize (strip, commas, int else lower). For clustering ablations."""
    if text is None:
        return ""
    s = text.strip().replace(",", "")
    if not s:
        return ""
    try:
        return str(int(s))
    except ValueError:
        return s.lower()


def _answer_cluster_id(text: str) -> int:
    return hash(canonicalize_answer(text)) % (2**31)


# --- Pass 2: symmetric sympy union-find ------------------------------------------

# Skip sympy on any answer containing these markers — sympy's LaTeX parser
# strips them lossily, producing false-positive merges. Concrete failure case
# we observed: '13824\\inA' vs '13824\\notinA' returns True because both parse
# to '13824 * A'. Anything with set membership / inequality / text wrappers is
# off-limits.
#
# Lookahead `(?![a-z])` prevents matching inside longer LaTeX commands:
#   '\\inA'    -> matches '\\in' (block: A is uppercase)        OK
#   '\\in '    -> matches '\\in' (block: space)                 OK
#   '\\infty'  -> no match ('\\in' followed by 'f' lowercase)   correct: \infty is safe
#   '\\notin'  -> matches '\\notin' (longest alt first)         OK
_SYMPY_UNSAFE = re.compile(
    r"\\(?:notin|subseteq|supseteq|subset|supset|neq|geq|leq|not|in|"
    r"text|mathrm|mathbf|mathit|operatorname|to|rightarrow|leftarrow|"
    r"forall|exists|implies|iff|land|lor|cap|cup|emptyset)(?![a-z])"
)


def _sympy_safe(s: str) -> bool:
    """False if `s` contains LaTeX that sympy parses lossily."""
    return _SYMPY_UNSAFE.search(s) is None


def sympy_equiv(a: str, b: str) -> bool:
    """Symmetric grader-based equivalence for clustering.

    Calls `grade_answer_mathd_or_sympy(a,b) or grade_answer_mathd_or_sympy(b,a)`.
    The grader's intentional asymmetric branches (unreduced-fraction rejection,
    int-strictness) are PRESERVED — that's the desired clustering behavior since
    prompts ask for simplified answers, so unreduced forms (`2/4`) are distinct
    rollout outputs from reduced ones (`1/2`). See `docs/build_spec/answer_clustering.md`.

    Returns False on any input where sympy's LaTeX parser is known to leak
    semantic content (set operators, \\text, \\mathrm, etc.) — blocklist defense
    against the `13824\\inA` vs `13824\\notinA` false-positive class.
    """
    if not _sympy_safe(a) or not _sympy_safe(b):
        return False
    try:
        return grade_answer_mathd_or_sympy(a, b) or grade_answer_mathd_or_sympy(b, a)
    except Exception:
        return False


# Allowlist: sympy only when the string uses known-safe LaTeX (see answer_clustering.md).
# Each command was verified to round-trip losslessly through sympy's LaTeX parser
# and to not cross-merge distinct values (e.g. \alpha vs \beta stay distinct).
_ALLOWED_LATEX_CMD = re.compile(
    r"\\(?:"
    # arithmetic / structural
    r"frac|dfrac|tfrac|sqrt|cdot|times|div|left|right"
    # constants
    r"|pi|infty"
    # trig + hyp + inverse
    r"|sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan|sinh|cosh|tanh"
    # logs / exp
    r"|log|ln|exp"
    # Greek lowercase
    r"|alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta"
    r"|iota|kappa|lambda|mu|nu|xi|rho|varrho|sigma|varsigma|tau"
    r"|upsilon|phi|varphi|chi|psi|omega"
    # Greek uppercase
    r"|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega"
    # Negative lookahead on letters only (mirrors `_SYMPY_UNSAFE`). Plain `\b`
    # would refuse to match `\log_2` because `_` is a word-char; using
    # `(?![a-zA-Z])` lets `_`, `(`, `{`, digits etc. follow while still
    # protecting against `\logarithm` being chopped to `\log`.
    r")(?![a-zA-Z])"
)
# Bare characters permitted after stripping allowed commands.
# Adds `^`, `_` (powers/subscripts on numeric expressions) and `[`, `]` (closed intervals)
# beyond the v1 set. Bare letters are intentionally excluded: sympy applies
# commutativity to juxtaposed letters (`xy` → `x*y`, `ABC` ≡ `BCA`), which would
# false-merge multi-letter labels (vertex names, sequence labels) on geometry-style
# answers. See docs/build_spec/answer_clustering.md.
# `*` (zero or more) so strings reducible entirely to allowed commands — e.g. bare
# `\pi`, `\infty`, `\alpha` — pass the residual check.
_ALLOWED_SAFE_CHARS = re.compile(r"^[\d\s+\-*/().,{}\[\]^_]*$")


def _sympy_allowlist_safe(s: str) -> bool:
    if _SYMPY_UNSAFE.search(s):
        return False
    t = _ALLOWED_LATEX_CMD.sub("", s)
    if "\\" in t:
        return False
    return bool(_ALLOWED_SAFE_CHARS.match(t))


def sympy_equiv_allowlist(a: str, b: str) -> bool:
    """Symmetric grader equivalence with allowlist guard (proposed v1 variant)."""
    if not _sympy_allowlist_safe(a) or not _sympy_allowlist_safe(b):
        return False
    try:
        return grade_answer_mathd_or_sympy(a, b) or grade_answer_mathd_or_sympy(b, a)
    except Exception:
        return False


def answer_hash_clusters(
    parsed_answers: list[str | None],
    parse_ok: list[bool],
    *,
    use_sympy: bool = True,
    canonicalize_fn: Callable[[str | None], str] = canonicalize_answer,
    sympy_equiv_fn: Callable[[str, str], bool] | None = None,
) -> list[int]:
    """Return per-rollout cluster ids.

    Two-pass algorithm for reproducibility:
      1. Group rollouts by canonicalize_answer string identity.
      2. Union-find merge canonicalize buckets whose representatives are
         sympy-equivalent. Bucket iteration is sorted lexicographically and
         union-by-string-order so output is fully deterministic.

    Parse-fail rollouts each get a unique negative id (`-1 - rollout_idx`).
    `use_sympy=False` skips pass 2 (pure canonicalize) — kept for ablation.
    `sympy_equiv_fn` defaults to `sympy_equiv` (blocklist); use `sympy_equiv_allowlist`
    for the allowlist variant.
    """
    if len(parsed_answers) != len(parse_ok):
        raise ValueError("parsed_answers and parse_ok must have equal length")
    equiv = sympy_equiv_fn if sympy_equiv_fn is not None else sympy_equiv

    canon: list[str] = []
    for i, (ans, ok) in enumerate(zip(parsed_answers, parse_ok)):
        if ok and ans is not None:
            canon.append(canonicalize_fn(ans))
        else:
            canon.append(f"__fail_{i}__")

    # Group by canonicalize key. Failed-parse buckets are size-1 and never merged.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(canon):
        groups[k].append(i)

    real_keys = sorted(k for k in groups if not k.startswith("__fail_"))

    if use_sympy and len(real_keys) > 1:
        # Representative = first parsed_answer in each group (sort-stable input order).
        reps = {k: parsed_answers[groups[k][0]] for k in real_keys}
        parent = {k: k for k in real_keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # Deterministic root choice via string order.
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        for i in range(len(real_keys)):
            for j in range(i + 1, len(real_keys)):
                a, b = real_keys[i], real_keys[j]
                if find(a) == find(b):
                    continue
                if equiv(reps[a], reps[b]):
                    union(a, b)

        canon = [
            (find(k) if k in parent else k) for k in canon
        ]

    # Assign stable negative ids to parse-fails so they never collide.
    out: list[int] = []
    root_to_id: dict[str, int] = {}
    for i, k in enumerate(canon):
        if k.startswith("__fail_"):
            out.append(-1 - i)
        else:
            if k not in root_to_id:
                root_to_id[k] = hash(k) % (2**31)
            out.append(root_to_id[k])
    return out


def cot_clusters_from_judge(
    assignment: dict[int, int],
    n_rollouts: int,
) -> list[int]:
    """Map judge assignment (rollout_idx -> cluster_id) to dense list[int].

    Caller is expected to pass already-normalized ids (Poly-EPO 100 -> -1 done
    by `judge/format.py::_normalize_cluster_id`).
    """
    if set(assignment.keys()) != set(range(n_rollouts)):
        raise ValueError(
            f"assignment must cover 0..{n_rollouts - 1}, got {sorted(assignment)}"
        )
    return [assignment[i] for i in range(n_rollouts)]
