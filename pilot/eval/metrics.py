"""Frozen pilot metrics — Pass@k, Cover@tau, worst-subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PromptRollouts:
    """Per-prompt rollout outcomes."""

    prompt_id: str
    correct: Sequence[bool]
    cluster_ids: Sequence[int]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator for n samples, c correct."""
    if n - c < k:
        return 1.0
    num = 1.0
    den = 1.0
    for i in range(k):
        num *= (n - c - i)
        den *= (n - i)
    return 1.0 - num / den


def pass_at_1(prompt: PromptRollouts) -> float:
    n = len(prompt.correct)
    c = sum(prompt.correct)
    return pass_at_k(n, c, 1)


def pass_at_k_metric(prompt: PromptRollouts, k: int) -> float:
    n = len(prompt.correct)
    c = sum(prompt.correct)
    return pass_at_k(n, c, min(k, n))


def cover_at_tau(prompt: PromptRollouts, tau: float) -> float:
    """
    Fraction of correct clusters with empirical support >= tau.
    Returns 0 if no correct rollout.
    """
    correct_clusters: dict[int, int] = {}
    total = len(prompt.cluster_ids)
    if total == 0:
        return 0.0
    for ok, cid in zip(prompt.correct, prompt.cluster_ids):
        if ok:
            correct_clusters[cid] = correct_clusters.get(cid, 0) + 1
    if not correct_clusters:
        return 0.0
    covered = sum(1 for cnt in correct_clusters.values() if cnt / total >= tau)
    return covered / len(correct_clusters)


def prompt_accuracy(prompt: PromptRollouts) -> float:
    if not prompt.correct:
        return 0.0
    return sum(prompt.correct) / len(prompt.correct)


def worst_subset_accuracy(
    prompts: Iterable[PromptRollouts], quantile: float = 0.25
) -> float:
    """Mean per-prompt accuracy over bottom `quantile` fraction of prompts."""
    accs = sorted(prompt_accuracy(p) for p in prompts)
    if not accs:
        return 0.0
    n = max(1, int(len(accs) * quantile))
    return sum(accs[:n]) / n


def aggregate_metrics(
    prompts: Sequence[PromptRollouts],
    *,
    k: int = 8,
    tau: float = 0.15,
    worst_q: float = 0.25,
) -> dict[str, float]:
    if not prompts:
        return {
            "pass_at_1": 0.0,
            f"pass_at_{k}": 0.0,
            "cover_at_tau": 0.0,
            "worst_subset_accuracy": 0.0,
            "n_prompts": 0,
        }
    return {
        "pass_at_1": sum(pass_at_1(p) for p in prompts) / len(prompts),
        f"pass_at_{k}": sum(pass_at_k_metric(p, k) for p in prompts) / len(prompts),
        "cover_at_tau": sum(cover_at_tau(p, tau) for p in prompts) / len(prompts),
        "worst_subset_accuracy": worst_subset_accuracy(prompts, worst_q),
        "n_prompts": float(len(prompts)),
    }
