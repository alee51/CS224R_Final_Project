"""Exact-answer canonicalization for cluster IDs."""

from __future__ import annotations


def canonicalize_answer(text: str) -> str:
    s = (text or "").strip().replace(",", "")
    try:
        return str(int(s))
    except ValueError:
        return s.lower()


def cluster_id(answer: str) -> int:
    return hash(canonicalize_answer(answer)) % (2**31)
