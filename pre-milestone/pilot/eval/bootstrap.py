"""Prompt-bootstrap CIs for pilot metrics."""

from __future__ import annotations

import random
from typing import Callable, Sequence

from pilot.eval.metrics import PromptRollouts, aggregate_metrics


def bootstrap_ci(
    prompts: Sequence[PromptRollouts],
    metric_key: str,
    *,
    n_samples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
    k: int = 8,
    tau: float = 0.15,
    worst_q: float = 0.25,
) -> dict[str, float]:
    """Return point estimate, CI low/high, and std for one metric."""
    if not prompts:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}

    rng = random.Random(seed)
    full = aggregate_metrics(prompts, k=k, tau=tau, worst_q=worst_q)
    point = full[metric_key]

    samples: list[float] = []
    n = len(prompts)
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot = [prompts[i] for i in idx]
        samples.append(aggregate_metrics(boot, k=k, tau=tau, worst_q=worst_q)[metric_key])

    samples.sort()
    alpha = (1 - ci_level) / 2
    lo = samples[int(alpha * n_samples)]
    hi = samples[int((1 - alpha) * n_samples) - 1]
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    return {"point": point, "ci_low": lo, "ci_high": hi, "std": var**0.5}


def bootstrap_all(
    prompts: Sequence[PromptRollouts],
    metric_keys: Sequence[str],
    **kwargs,
) -> dict[str, dict[str, float]]:
    return {key: bootstrap_ci(prompts, key, **kwargs) for key in metric_keys}
