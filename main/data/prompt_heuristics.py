"""Heuristic flags for Polaris prompt quality / train filtering."""

from __future__ import annotations

import re
from typing import Any

_LAST_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LAST_STARTS_PROVE = re.compile(r"^prove\b", re.IGNORECASE)
_CONTAINS_SHOW_THAT = re.compile(r"\bshow\s+that\b", re.IGNORECASE)


def last_sentence(problem: str) -> str:
    parts = [p.strip() for p in _LAST_SENTENCE_SPLIT.split(problem.strip()) if p.strip()]
    return parts[-1] if parts else ""


def label_last_starts_prove(problem: str) -> bool:
    return bool(_LAST_STARTS_PROVE.match(last_sentence(problem)))


def label_last_contains_prove(problem: str) -> bool:
    return "prove" in last_sentence(problem).lower()


def label_contains_show_that(problem: str) -> bool:
    return bool(_CONTAINS_SHOW_THAT.search(problem))


def label_gold_in_prompt(problem: str, gold: str) -> bool:
    """True if stripped gold appears as a substring of problem (case-insensitive)."""
    g = str(gold).strip()
    if not g:
        return False
    return g.lower() in problem.lower()


def label_prompt_heuristics(problem: str, gold: str) -> dict[str, bool]:
    return {
        "last_starts_prove": label_last_starts_prove(problem),
        "last_contains_prove": label_last_contains_prove(problem),
        "contains_show_that": label_contains_show_that(problem),
        "gold_in_prompt": label_gold_in_prompt(problem, gold),
    }


def any_heuristic(flags: dict[str, bool]) -> bool:
    return any(flags.values())


def contains_prove_anywhere(problem: str) -> bool:
    return "prove" in problem.lower()


def should_drop_train_prompt_filter(problem: str, gold: str) -> bool:
    """Locked train filter (decisions.md §2026-05-27).

    Drop if last sentence starts with Prove, or gold leaks in the stem with prove/show wording.
    """
    if label_last_starts_prove(problem):
        return True
    if not label_gold_in_prompt(problem, gold):
        return False
    return contains_prove_anywhere(problem) or label_contains_show_that(problem)
