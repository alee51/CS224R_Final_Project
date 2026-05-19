"""Extract final answers from model completions for RLVR verification."""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
_ANSWER_LINE = re.compile(r"^\s*answer\s*:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(completion: str) -> str:
    text = completion.strip()
    matches = list(_BOXED.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    line_matches = list(_ANSWER_LINE.finditer(text))
    if line_matches:
        return line_matches[-1].group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


def is_correct(completion: str, gold: str) -> bool:
    from pilot.train.canonicalize import canonicalize_answer

    return canonicalize_answer(extract_answer(completion)) == canonicalize_answer(str(gold))
