"""GRPO / set-based advantage computation (GRPO path implemented)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class AdvantageOut:
    advantages: torch.Tensor
    keep_mask: torch.Tensor
    diagnostics: dict[str, Any]


def _grpo_advantages(rewards: torch.Tensor) -> AdvantageOut:
    """A_i = r_i - mean(r) per prompt group; shape [n_prompts, n_rollouts]."""
    group_mean = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - group_mean
    keep_mask = advantages.abs().sum(dim=1) > 0
    n_filtered = int((~keep_mask).sum().item())
    diagnostics = {
        "fraction_filtered": n_filtered / max(rewards.shape[0], 1),
        "n_filtered_prompts": n_filtered,
    }
    return AdvantageOut(
        advantages=advantages,
        keep_mask=keep_mask,
        diagnostics=diagnostics,
    )


def compute_advantages(
    arm: str,
    rewards: torch.Tensor,
    clusters: torch.Tensor | None = None,
) -> AdvantageOut:
    """
    Per-rollout advantages and prompt-level keep_mask.

    Set-based arms (minority_*, poly_epo_answer) are reserved; not implemented.
    """
    if arm == "grpo":
        return _grpo_advantages(rewards)

    raise NotImplementedError(
        f"advantage arm {arm!r} not implemented in skeleton "
        "(plug in at objective.py per PLAN §3)"
    )
