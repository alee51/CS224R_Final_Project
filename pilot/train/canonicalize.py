"""Exact-answer canonicalization for cluster IDs."""

from __future__ import annotations

import re


def canonicalize_answer(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("$", "").replace("\\boxed{", "").replace("}", "")
    return s


def cluster_id(answer: str) -> int:
    return hash(canonicalize_answer(answer)) % (2**31)
