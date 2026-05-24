"""
Advantage weighting hooks for pilot objectives.
Trainer must call `weighted_advantages` after computing base GRPO advantages.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

import torch

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


def _clip_surrogate_scalar(
    logprobs: list[float],
    old_logprobs: list[float],
    advantages: list[float],
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> tuple[float, float]:
    """Mean clipped policy-gradient surrogate and clip fraction (asymmetric PPO clip)."""
    if not logprobs:
        return 0.0, 0.0
    losses: list[float] = []
    clipped = 0
    for lp, old_lp, adv in zip(logprobs, old_logprobs, advantages):
        ratio = math.exp(lp - old_lp)
        unclipped = ratio * adv
        lo = 1.0 - clip_ratio_low
        hi = 1.0 + clip_ratio_high
        clipped_ratio = min(max(ratio, lo), hi) * adv
        losses.append(-min(unclipped, clipped_ratio))
        if ratio != min(max(ratio, lo), hi):
            clipped += 1
    return sum(losses) / len(losses), clipped / len(losses)


def _clip_surrogate_tensor(
    logprob: torch.Tensor,
    old_logprob: float,
    advantage: float,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> torch.Tensor:
    old_t = torch.tensor(old_logprob, device=logprob.device, dtype=logprob.dtype)
    ratio = torch.exp(logprob - old_t)
    adv_t = torch.tensor(advantage, device=logprob.device, dtype=logprob.dtype)
    unclipped = ratio * adv_t
    clipped_ratio = (
        torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * adv_t
    )
    return -torch.minimum(unclipped, clipped_ratio)


def advantage_l2(advantages: list[float]) -> float:
    return math.sqrt(sum(a * a for a in advantages))

