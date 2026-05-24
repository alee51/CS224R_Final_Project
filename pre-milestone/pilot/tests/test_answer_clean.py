"""Unit tests for pilot.train.answer_clean."""

from __future__ import annotations

from pilot.train.answer_clean import (
    cluster_id_clean,
    extract_answer_clean,
    is_correct_clean,
    is_runon,
    last_brace_balanced_boxed_inner,
    normalize_answer_clean,
)


def test_brace_balanced_nested_boxed() -> None:
    text = r"Thus \(\boxed{\frac{1}{2}}\) is the answer."
    assert last_brace_balanced_boxed_inner(text) == r"\frac{1}{2}"
    parsed, path = extract_answer_clean(text)
    assert path == "boxed_balanced"
    assert parsed == r"\frac{1}{2}"
    assert normalize_answer_clean(parsed) == "1/2"


def test_paren_dollar_vs_integer_gold() -> None:
    assert normalize_answer_clean(r"\(50\)") == "50"
    assert normalize_answer_clean("50") == "50"
    assert is_correct_clean(r"\(50\)", "50")
    assert is_correct_clean("50", "50")


def test_runon_rejection() -> None:
    prose = (
        "Therefore, the remainder when dividing is zero and we conclude "
        "that the answer must be forty-two in the limit."
    )
    assert is_runon(prose)
    completion = f"Some work.\n\nAnswer: {prose}"
    parsed, path = extract_answer_clean(completion)
    assert path == "runon_rejected"
    assert parsed == ""


def test_cluster_id_deterministic() -> None:
    a = cluster_id_clean("42")
    b = cluster_id_clean("42")
    c = cluster_id_clean("43")
    assert a == b
    assert a != c
    assert 0 <= a < 2**32


def test_answer_line_short() -> None:
    completion = "Work shown.\n\nAnswer: 201"
    parsed, path = extract_answer_clean(completion)
    assert path == "answer_line"
    assert parsed == "201"
    assert is_correct_clean(parsed, "201")
