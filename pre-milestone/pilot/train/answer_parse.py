"""Extract final answers from model completions for RLVR verification."""

from __future__ import annotations

import re

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_LINE = re.compile(r"^\s*answer\s*:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_boxed_answer(text: str) -> tuple[str | None, bool]:
    """Returns (raw_extracted, parser_clean).

    parser_clean is True iff exactly one \\boxed{...} is found and the contents
    parse as an int.
    """
    matches = BOXED_RE.findall(text)
    if len(matches) != 1:
        return (None, False)
    raw = matches[0].strip()
    try:
        return (str(int(raw.replace(",", ""))), True)
    except ValueError:
        return (raw, False)


def extract_answer(completion: str) -> str:
    """Legacy extractor: prefer boxed, else Answer: line, else last line."""
    raw, _ = extract_boxed_answer(completion)
    if raw is not None:
        return raw
    text = completion.strip()
    line_matches = list(_ANSWER_LINE.finditer(text))
    if line_matches:
        return line_matches[-1].group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


def is_correct(completion: str, gold: str) -> bool:
    from pilot.train.canonicalize import canonicalize_answer

    raw, _ = extract_boxed_answer(completion)
    if raw is None:
        return False
    return canonicalize_answer(raw) == canonicalize_answer(str(gold))
