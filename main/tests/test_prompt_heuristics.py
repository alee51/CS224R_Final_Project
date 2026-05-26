"""Unit tests for prompt heuristic labels."""

from data.prompt_heuristics import (
    label_contains_show_that,
    label_gold_in_prompt,
    label_last_contains_prove,
    label_last_starts_prove,
    label_prompt_heuristics,
    should_drop_train_prompt_filter,
)


def test_last_starts_prove():
    assert label_last_starts_prove("Foo. Prove that x > 5.")
    assert not label_last_starts_prove("Prove that x > 5. Find y.")


def test_last_contains_prove_not_only_start():
    assert label_last_contains_prove("The sum is 45. Prove this.")
    assert not label_last_contains_prove("Prove that x > 5. Find y.")


def test_contains_show_that():
    assert label_contains_show_that("Please show that n is even.")
    assert not label_contains_show_that("Prove that n is even.")


def test_gold_in_prompt():
    assert label_gold_in_prompt("The answer is 42.", "42")
    assert label_gold_in_prompt(r"Prove $a_{100}>14$.", r"a_{100}>14")
    assert not label_gold_in_prompt("Find x.", "42")


def test_label_bundle():
    flags = label_prompt_heuristics("Show that x=1.", "1")
    assert flags["contains_show_that"]
    assert flags["gold_in_prompt"]


def test_should_drop_train_filter():
    assert should_drop_train_prompt_filter("Foo. Prove that x > 5.", "x > 5")
    assert not should_drop_train_prompt_filter("The answer is 42. Pick one.", "42")
    assert should_drop_train_prompt_filter(
        "Part (a) Prove that n=2. Find the sum.",
        "2",
    )
    assert not should_drop_train_prompt_filter("Show that x=1.", "99")
