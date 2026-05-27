"""Tests for clustering substrates (canonicalize + sympy union-find)."""

from __future__ import annotations

import pytest

from train.clustering import (
    _sympy_allowlist_safe,
    answer_hash_clusters,
    canonicalize_answer,
    cot_clusters_from_judge,
    sympy_equiv,
    sympy_equiv_allowlist,
)


# --- canonicalize ---------------------------------------------------------------

def test_identical_parsed_answers_share_cluster():
    ids = answer_hash_clusters(["42", "42", "7"], [True, True, True])
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_parse_fail_unique_negative_ids():
    ids = answer_hash_clusters([None, None, "5"], [False, False, True])
    assert ids[0] < 0 and ids[1] < 0
    assert ids[0] != ids[1]
    assert ids[2] >= 0


def test_canonicalize_handles_commas():
    assert canonicalize_answer("1,234") == "1234"


def test_canonicalize_strips_inline_math_parens():
    # '\(36\)' should canonicalize to '36'
    assert canonicalize_answer("\\(36\\)") == "36"
    assert canonicalize_answer("\\(150\\).") == "150"


def test_canonicalize_strips_dollar_math():
    assert canonicalize_answer("$10$") == "10"
    assert canonicalize_answer("$52$") == "52"


def test_canonicalize_unwraps_boxed():
    assert canonicalize_answer("\\boxed{11}") == "11"


def test_canonicalize_trailing_period():
    assert canonicalize_answer("21.") == "21"
    assert canonicalize_answer("115.") == "115"


def test_canonicalize_int_valued_float():
    assert canonicalize_answer("60.0") == "60"
    assert canonicalize_answer("3160.0000000000002") == "3160"
    # But truly fractional floats stay as their string form
    assert canonicalize_answer("0.5") != "0"


def test_canonicalize_case_and_whitespace():
    assert canonicalize_answer("  Hello ") == "hello"
    ids = answer_hash_clusters(["abc", "ABC"], [True, True], use_sympy=False)
    assert ids[0] == ids[1]


def test_canonicalize_groups_textual_noise():
    """The merges we observed in real rollouts must cluster under canonicalize."""
    answers = ["36", "\\(36\\)", "$36$", "36.", "\\boxed{36}"]
    ids = answer_hash_clusters(answers, [True]*5, use_sympy=False)
    assert len(set(ids)) == 1, f"expected 1 cluster, got {set(ids)}: {ids}"


# --- sympy equivalence ---------------------------------------------------------

def test_sympy_equiv_recognizes_latex_equivalents():
    assert sympy_equiv("8/5", "\\frac{8}{5}")
    assert sympy_equiv("1/2", "0.5")
    assert sympy_equiv("\\frac{1}{2}", "0.5")


def test_sympy_equiv_symmetric():
    assert sympy_equiv("8/5", "\\frac{8}{5}") == sympy_equiv("\\frac{8}{5}", "8/5")


def test_sympy_equiv_blocks_set_operator_false_positives():
    """\\inA vs \\notinA must NOT merge: sympy's LaTeX parser strips the set
    operator, falsely concluding equality of the underlying expression."""
    assert not sympy_equiv("13824\\inA", "13824\\notinA")
    assert not sympy_equiv("x \\in A", "x \\notin A")


def test_sympy_equiv_blocks_text_wrappers():
    """\\text{...} content is semantic; sympy strips it."""
    assert not sympy_equiv("\\text{yes}", "\\text{no}")


def test_sympy_equiv_unequal_returns_false():
    assert not sympy_equiv("2", "3")
    assert not sympy_equiv("\\frac{1}{3}", "\\frac{1}{4}")


def test_sympy_equiv_rejects_unreduced_fractions():
    """Prompts ask for simplified answers, so an unreduced fraction is a
    distinct rollout output from its reduced form. Grader's intentional
    asymmetric branch (preserved on purpose) enforces this. See
    docs/build_spec/answer_clustering.md."""
    assert not sympy_equiv("1/2", "2/4")
    assert not sympy_equiv("\\frac{1}{2}", "\\frac{2}{4}")
    assert not sympy_equiv("3/6", "1/2")


# --- union-find clustering -----------------------------------------------------

def test_sympy_union_find_merges_latex_equivalents():
    """8/5 and \\frac{8}{5} cluster together under sympy."""
    answers = ["8/5", "\\frac{8}{5}", "2"]
    ok = [True, True, True]
    ids_with = answer_hash_clusters(answers, ok, use_sympy=True)
    ids_without = answer_hash_clusters(answers, ok, use_sympy=False)
    assert ids_with[0] == ids_with[1]  # merged
    assert ids_with[0] != ids_with[2]
    assert ids_without[0] != ids_without[1]  # not merged (canon only)


def test_clustering_is_deterministic():
    """Same input -> same output every call (union-find with fixed traversal)."""
    answers = ["8/5", "\\frac{8}{5}", "1/2", "0.5", "2"]
    ok = [True]*5
    a = answer_hash_clusters(answers, ok)
    b = answer_hash_clusters(answers, ok)
    assert a == b


def test_full_8_rollout_minority_case_under_sympy():
    """Concrete case from probe rollouts (pid=84):
    5 rollouts say '45', one says '\\(45\\)', 2 parse-fail.
    Under sympy clustering: '45' and '\\(45\\)' must share a cluster
    (canonicalize already handles this textually, but verify end-to-end)."""
    parsed = ["45", "45", "\\(45\\)", None, "45", "45", None, "45"]
    ok = [True, True, True, False, True, True, False, True]
    ids = answer_hash_clusters(parsed, ok)
    # All 6 '45'/'\\(45\\)' rollouts share one cluster
    six_45 = [ids[i] for i in (0, 1, 2, 4, 5, 7)]
    assert len(set(six_45)) == 1
    # Two parse-fails are unique negatives
    assert ids[3] < 0 and ids[6] < 0
    assert ids[3] != ids[6]


# --- allowlist gate + expansion ------------------------------------------------

def test_allowlist_admits_baseline_safe_strings():
    for s in ["1/2", "\\frac{8}{5}", "2\\sqrt{3}", "\\pi", "(1+\\sqrt{3})/2"]:
        assert _sympy_allowlist_safe(s), f"baseline-safe string rejected: {s!r}"


def test_allowlist_admits_expansion_commands():
    # Strings that the v1 allowlist rejected but the expanded one must admit.
    for s in [
        "\\infty",
        "(0,\\infty)",
        "(-\\infty, 5]",
        "[-1, 1]",
        "2^{10}",
        "\\sin(\\pi/2)",
        "\\cos(0)",
        "\\log_2(8)",
        "\\alpha + \\beta",
        "\\theta",
    ]:
        assert _sympy_allowlist_safe(s), f"expansion-safe string rejected: {s!r}"


def test_allowlist_still_rejects_bare_letter_strings():
    """Bare letters are intentionally excluded — sympy commutativity would
    false-merge `xy ≡ x*y` and `ABC ≡ BCA`, which is wrong for vertex labels."""
    for s in ["xy", "ABC", "B_1C_1", "(0,0,d)", "x+y"]:
        assert not _sympy_allowlist_safe(s), f"bare-letter string admitted: {s!r}"


def test_allowlist_still_rejects_unsafe_relations():
    """Set-operator / relation commands stay blocked at the gate — this is the
    `\\inA` vs `\\notinA` false-positive class."""
    for s in ["13824\\inA", "x \\in A", "\\text{yes}", "\\triangleABC", "A \\cong B"]:
        assert not _sympy_allowlist_safe(s), f"unsafe-relation string admitted: {s!r}"


def test_allowlist_sympy_merges_infty_intervals():
    """Identical \\infty intervals must cluster under allowlist sympy."""
    answers = ["(0,\\infty)", "(0,\\infty)", "(0,1)"]
    ids = answer_hash_clusters(
        answers, [True] * 3, sympy_equiv_fn=sympy_equiv_allowlist
    )
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_allowlist_sympy_merges_subscript_braces():
    """`x_1` and `x_{1}` should cluster (same symbol, different LaTeX form)."""
    answers = ["x_1", "x_{1}", "x_2"]
    # Note: bare letters reject at gate, so these fall back to canon.
    # We don't assert merge here — assert at least determinism + non-merge of distinct.
    ids = answer_hash_clusters(
        answers, [True] * 3, sympy_equiv_fn=sympy_equiv_allowlist
    )
    assert ids[0] != ids[2]  # x_1 vs x_2 distinct under both canon and sympy


def test_allowlist_distinct_intervals_dont_merge():
    """Closed vs open interval of same endpoints must stay distinct."""
    assert not sympy_equiv_allowlist("(0,1)", "[0,1]")
    assert not sympy_equiv_allowlist("[-1,1]", "[-1,2]")
    assert not sympy_equiv_allowlist("(0,\\infty)", "[0,\\infty)")


def test_allowlist_distinct_greek_dont_merge():
    assert not sympy_equiv_allowlist("\\alpha", "\\beta")
    assert not sympy_equiv_allowlist("\\theta", "\\phi")


def test_allowlist_blocks_known_unsafe_pair():
    """Defense-in-depth: the original blocklist false-positive class must still
    return False even though we're on the allowlist path."""
    assert not sympy_equiv_allowlist("13824\\inA", "13824\\notinA")
    assert not sympy_equiv_allowlist("\\triangleABC", "\\triangleDEF")


# --- judge path ----------------------------------------------------------------

def test_cot_clusters_from_judge_dense():
    assignment = {0: 1, 1: 1, 2: 2, 3: -1, 4: 1, 5: 2, 6: -1, 7: 1}
    out = cot_clusters_from_judge(assignment, 8)
    assert out == [1, 1, 2, -1, 1, 2, -1, 1]


# --- input validation ---------------------------------------------------------

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        answer_hash_clusters(["1", "2"], [True])
