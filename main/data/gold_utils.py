"""Gold helpers for Polaris train freeze and probe diagnostics."""

from __future__ import annotations

from typing import Any


def normalize_train_gold(answer: Any) -> str:
    """Train freeze gold: verbatim HF string, strip whitespace only."""
    if answer is None:
        return ""
    return str(answer).strip()


def is_nonempty_gold(answer: Any) -> bool:
    return bool(normalize_train_gold(answer))


def is_integer_gold(answer: Any) -> bool:
    """Probe/diagnostic only — Group A manifests still filter integer gold."""
    s = str(answer).strip().replace(",", "")
    if not s:
        return False
    if s.startswith("-") and len(s) > 1:
        return s[1:].isdigit()
    return s.isdigit()


def canonicalize_gold(answer: Any) -> str:
    """Strip, remove commas, return str(int(...)). Caller must verify is_integer_gold first."""
    normalized = str(answer).strip().replace(",", "")
    return str(int(normalized))
