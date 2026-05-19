"""
Minimal GRPO training loop scaffold for the CS224R pilot.

Integration: launch scripts in `pilot/infra/` load YAML configs and call
`GRPOTrainer.train_step` with the run's `objective` field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pilot.train.objectives import ObjectiveName, weighted_advantages


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.001
    rollouts_per_prompt: int = 8
    inverse_gamma: float = 1.0
    w_max: float = 8.0
    focal_gamma: float = 2.0


@dataclass
class PromptRolloutGroup:
    """One prompt with N sampled trajectories."""

    prompt_id: str
    rewards: list[float]
    cluster_ids: list[int]
    logprobs: list[float]
    old_logprobs: list[float]
    ref_logprobs: list[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.rewards)
        for name, seq in (
            ("cluster_ids", self.cluster_ids),
            ("logprobs", self.logprobs),
            ("old_logprobs", self.old_logprobs),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} length {len(seq)} != rewards length {n}")
        if self.ref_logprobs is not None and len(self.ref_logprobs) != n:
            raise ValueError("ref_logprobs length mismatch")


@dataclass
class TrainStepOutput:
    loss: float
    policy_loss: float
    kl_penalty: float
    clip_fraction: float
    mean_advantage: float
    n_prompts: int
    n_rollouts: int


class PolicyModel(Protocol):
    """Optional real model hook; tests use `MockPolicyModel`."""

    def logprobs_for_rollouts(self, groups: list[PromptRolloutGroup]) -> list[list[float]]:
        ...


@dataclass
class MockPolicyModel:
    """CPU-only stand-in: returns stored old logprobs (no forward pass)."""

    def logprobs_for_rollouts(self, groups: list[PromptRolloutGroup]) -> list[list[float]]:
        return [g.old_logprobs for g in groups]


def _clip_surrogate(
    logprobs: list[float],
    old_logprobs: list[float],
    advantages: list[float],
    clip_eps: float,
) -> tuple[float, float]:
    """Mean clipped policy-gradient surrogate and clip fraction."""
    if not logprobs:
        return 0.0, 0.0
    losses: list[float] = []
    clipped = 0
    for lp, old_lp, adv in zip(logprobs, old_logprobs, advantages):
        ratio = math.exp(lp - old_lp)
        unclipped = ratio * adv
        clipped_ratio = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps) * adv
        losses.append(-min(unclipped, clipped_ratio))
        if ratio != min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps):
            clipped += 1
    return sum(losses) / len(losses), clipped / len(losses)


def _kl_penalty(
    logprobs: list[float],
    ref_logprobs: list[float],
) -> float:
    if not logprobs:
        return 0.0
    return sum(lp - ref for lp, ref in zip(logprobs, ref_logprobs)) / len(logprobs)


class GRPOTrainer:
    def __init__(
        self,
        cfg: GRPOConfig | None = None,
        model: PolicyModel | MockPolicyModel | None = None,
    ) -> None:
        self.cfg = cfg or GRPOConfig()
        self.model = model or MockPolicyModel()

    def train_step(
        self,
        groups: list[PromptRolloutGroup],
        objective: ObjectiveName,
        *,
        cfg: GRPOConfig | None = None,
        objective_overrides: dict[str, Any] | None = None,
    ) -> TrainStepOutput:
        """
        One optimizer step over a batch of prompts (each with N rollouts).

        Args:
            groups: batch of prompt rollout groups.
            objective: `grpo` | `inverse_freq` | `f_grpo`.
            cfg: optional per-step config override.
            objective_overrides: extra kwargs for `weighted_advantages`.
        """
        step_cfg = cfg or self.cfg
        overrides = objective_overrides or {}
        logprobs_batch = self.model.logprobs_for_rollouts(groups)

        policy_losses: list[float] = []
        clip_fracs: list[float] = []
        kl_terms: list[float] = []
        all_advantages: list[float] = []
        n_rollouts = 0

        for group, logprobs in zip(groups, logprobs_batch):
            adv = weighted_advantages(
                objective,
                group.rewards,
                group.cluster_ids,
                inverse_gamma=overrides.get("inverse_gamma", step_cfg.inverse_gamma),
                w_max=overrides.get("w_max", step_cfg.w_max),
                focal_gamma=overrides.get("focal_gamma", step_cfg.focal_gamma),
            )
            all_advantages.extend(adv)
            pg_loss, clip_frac = _clip_surrogate(
                logprobs, group.old_logprobs, adv, step_cfg.clip_eps
            )
            policy_losses.append(pg_loss)
            clip_fracs.append(clip_frac)
            n_rollouts += len(group.rewards)

            if group.ref_logprobs is not None:
                kl_terms.append(_kl_penalty(logprobs, group.ref_logprobs))

        policy_loss = sum(policy_losses) / max(len(policy_losses), 1)
        kl_penalty = sum(kl_terms) / max(len(kl_terms), 1) if kl_terms else 0.0
        loss = policy_loss + step_cfg.kl_coef * kl_penalty
        clip_fraction = sum(clip_fracs) / max(len(clip_fracs), 1)
        mean_adv = sum(all_advantages) / max(len(all_advantages), 1)

        return TrainStepOutput(
            loss=loss,
            policy_loss=policy_loss,
            kl_penalty=kl_penalty,
            clip_fraction=clip_fraction,
            mean_advantage=mean_adv,
            n_prompts=len(groups),
            n_rollouts=n_rollouts,
        )


def make_mock_batch(
    n_prompts: int = 2,
    n_rollouts: int = 8,
    *,
    seed: int = 0,
) -> list[PromptRolloutGroup]:
    """Synthetic batch for dry-run / unit tests (no GPU)."""
    import random

    rng = random.Random(seed)
    groups: list[PromptRolloutGroup] = []
    for p in range(n_prompts):
        rewards = [float(rng.random() > 0.6) for _ in range(n_rollouts)]
        cluster_ids = [rng.randint(0, 3) for _ in range(n_rollouts)]
        old_logprobs = [-rng.random() * 2 for _ in range(n_rollouts)]
        logprobs = [lp - 0.01 for lp in old_logprobs]
        ref_logprobs = [lp - 0.005 for lp in old_logprobs]
        groups.append(
            PromptRolloutGroup(
                prompt_id=f"mock_{p}",
                rewards=rewards,
                cluster_ids=cluster_ids,
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
            )
        )
    return groups
