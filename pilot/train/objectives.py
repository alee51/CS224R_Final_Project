"""
Advantage weighting hooks for pilot objectives.
Trainer must call `weighted_advantages` after computing base GRPO advantages.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

ObjectiveName = Literal["grpo", "inverse_freq", "f_grpo"]


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Per-trajectory advantages A_i = r_i - mean(r)."""
    n = len(rewards)
    if n == 0:
        return []
    baseline = sum(rewards) / n
    return [r - baseline for r in rewards]


def per_trajectory_cluster_counts(cluster_ids: list[int]) -> list[int]:
    """For each rollout i, return n_{x, c_i} (cluster size within the prompt)."""
    counts = Counter(cluster_ids)
    return [counts[cid] for cid in cluster_ids]


def inverse_freq_weights(cluster_counts: list[int], gamma: float = 1.0, w_max: float = 8.0) -> list[float]:
    """Per-trajectory weights; sum to N after normalization (before cap may bind)."""
    n = len(cluster_counts)
    raw = [(max(c, 1)) ** (-gamma) for c in cluster_counts]
    s = sum(raw) or 1.0
    w = [n * r / s for r in raw]
    return [min(x, w_max) for x in w]


def apply_weights(advantages: list[float], weights: list[float]) -> list[float]:
    return [a * w for a, w in zip(advantages, weights)]


def f_grpo_prompt_scale(mean_reward: float, focal_gamma: float = 2.0) -> float:
    """Prompt-level focal scaling from mean rollout reward."""
    p = max(min(mean_reward, 1.0), 0.0)
    return (1.0 - p) ** focal_gamma


def weighted_advantages(
    objective: ObjectiveName,
    rewards: list[float],
    cluster_ids: list[int] | None = None,
    *,
    inverse_gamma: float = 1.0,
    w_max: float = 8.0,
    focal_gamma: float = 2.0,
) -> list[float]:
    """Base GRPO advantages with objective-specific scaling."""
    advantages = grpo_advantages(rewards)
    if objective == "grpo":
        return advantages
    if objective == "inverse_freq":
        if cluster_ids is None:
            raise ValueError("cluster_ids required for inverse_freq")
        counts = per_trajectory_cluster_counts(cluster_ids)
        weights = inverse_freq_weights(counts, gamma=inverse_gamma, w_max=w_max)
        return apply_weights(advantages, weights)
    if objective == "f_grpo":
        scale = f_grpo_prompt_scale(sum(rewards) / max(len(rewards), 1), focal_gamma=focal_gamma)
        return [a * scale for a in advantages]
    raise ValueError(f"unknown objective: {objective!r}")
