"""GRPO and set-based advantage computation.

Set-RL math is ported verbatim from
`pre-milestone/nancy_explore/run0_analysis/analysis_c/set_score_simulation.py`
(`SUBSETS`, `INCL`, `minority_f`, `f_poly_score`, `marginal_from_fG`).

PLAN locks N=8, k=4 -> 70 size-4 subsets per prompt, 35 containing each rollout.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


N_ROLLOUTS = 8
SUBSET_SIZE = 4
_SIZE4_SUBSETS: tuple[tuple[int, ...], ...] = tuple(
    itertools.combinations(range(N_ROLLOUTS), SUBSET_SIZE)
)
_SUBSET_ARR = np.array(_SIZE4_SUBSETS, dtype=np.int64)  # [70, 4]
# INCL[i] = subset indices containing rollout i; each length 35.
_INCL: tuple[np.ndarray, ...] = tuple(
    np.where((_SUBSET_ARR == i).any(axis=1))[0] for i in range(N_ROLLOUTS)
)

SET_ARMS = frozenset({"minority_answer", "minority_cot", "poly_epo_answer"})


@dataclass
class AdvantageOut:
    advantages: torch.Tensor
    keep_mask: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


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


def _marginal_from_fG(fG: np.ndarray) -> np.ndarray:
    """Per-rollout marginal set advantage (length N)."""
    baseline = fG.mean()
    set_adv = fG - baseline
    return np.array([set_adv[_INCL[i]].mean() for i in range(N_ROLLOUTS)])


def _minority_subset_score(
    rewards4: np.ndarray,
    clusters4: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """f(G) = mean reward of rollouts in the rarest cluster (random tiebreak)."""
    counts = Counter(clusters4.tolist())
    min_count = min(counts.values())
    rarest = [c for c, cnt in counts.items() if cnt == min_count]
    pick = rng.choice(rarest) if len(rarest) > 1 else rarest[0]
    mask = clusters4 == pick
    return float(rewards4[mask].mean())


def _poly_epo_subset_score(rewards4: np.ndarray, clusters4: np.ndarray) -> float:
    """f(G) = mean(r in G) * (distinct cluster ids in G) / k."""
    return float(rewards4.mean() * len(set(clusters4.tolist())) / SUBSET_SIZE)


def set_based_marginal_advantages(
    rewards: torch.Tensor,
    clusters: torch.Tensor,
    subset_score_fn: Callable[..., float],
    *,
    needs_rng: bool,
    global_seed: int | None,
    problem_ids: list[int] | None,
) -> AdvantageOut:
    """Marginal advantages over 70 size-4 subsets per prompt.

    Per prompt p:
      1. Compute f(G) for each of 70 subsets.
      2. Baseline = mean of those 70 scores.
      3. Set advantage for G = f(G) - baseline.
      4. Rollout i advantage = mean set-advantage over the 35 subsets containing i.

    keep_mask[p] = False when all N rollouts share one cluster id (collapsed mode
    -> zero marginal by construction). This is *not* GRPO's "all rewards equal"
    rule: a prompt with all-correct identical answers must be filtered for set
    arms (no diversity signal) even though GRPO would also drop it.
    """
    if rewards.shape != clusters.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)} and clusters {tuple(clusters.shape)} "
            "must have matching shape"
        )
    n_prompts, n_roll = rewards.shape
    if n_roll != N_ROLLOUTS:
        raise ValueError(
            f"set-based arms require n_rollouts={N_ROLLOUTS}, got {n_roll}"
        )
    if needs_rng and (global_seed is None or problem_ids is None):
        raise ValueError("minority arms require global_seed and problem_ids for rng")
    if problem_ids is not None and len(problem_ids) != n_prompts:
        raise ValueError(
            f"problem_ids length {len(problem_ids)} != n_prompts {n_prompts}"
        )

    r_np = rewards.detach().cpu().numpy().astype(np.float64)
    c_np = clusters.detach().cpu().numpy().astype(np.int64)

    advantages = np.zeros((n_prompts, n_roll), dtype=np.float32)
    keep = np.zeros(n_prompts, dtype=bool)
    marg_for_diag: list[np.ndarray] = []

    for p in range(n_prompts):
        r = r_np[p]
        c = c_np[p]
        if len(set(c.tolist())) <= 1:
            # Collapsed: every subset has one cluster -> minority/poly-epo all
            # produce the same f(G); marginals are exactly zero. Skip the prompt.
            continue
        if needs_rng:
            rng = np.random.default_rng(int(global_seed) + int(problem_ids[p]))
        else:
            rng = None
        fG = np.zeros(len(_SIZE4_SUBSETS), dtype=np.float64)
        for k, idxs in enumerate(_SIZE4_SUBSETS):
            r4 = r[list(idxs)]
            c4 = c[list(idxs)]
            if needs_rng:
                fG[k] = subset_score_fn(r4, c4, rng)
            else:
                fG[k] = subset_score_fn(r4, c4)
        marg = _marginal_from_fG(fG)
        advantages[p] = marg.astype(np.float32)
        keep[p] = True
        marg_for_diag.append(marg)

    n_filtered = int((~keep).sum())
    diagnostics: dict[str, Any] = {
        "fraction_filtered": n_filtered / max(n_prompts, 1),
        "n_filtered_prompts": n_filtered,
    }
    if marg_for_diag:
        flat = np.concatenate(marg_for_diag)
        diagnostics["adv_marginal_p05"] = float(np.quantile(flat, 0.05))
        diagnostics["adv_marginal_p50"] = float(np.quantile(flat, 0.50))
        diagnostics["adv_marginal_p95"] = float(np.quantile(flat, 0.95))

    return AdvantageOut(
        advantages=torch.from_numpy(advantages),
        keep_mask=torch.from_numpy(keep),
        diagnostics=diagnostics,
    )


def _minority_advantages(
    rewards: torch.Tensor,
    clusters: torch.Tensor,
    *,
    global_seed: int,
    problem_ids: list[int],
) -> AdvantageOut:
    return set_based_marginal_advantages(
        rewards,
        clusters,
        _minority_subset_score,
        needs_rng=True,
        global_seed=global_seed,
        problem_ids=problem_ids,
    )


def _poly_epo_answer_advantages(
    rewards: torch.Tensor,
    clusters: torch.Tensor,
) -> AdvantageOut:
    return set_based_marginal_advantages(
        rewards,
        clusters,
        _poly_epo_subset_score,
        needs_rng=False,
        global_seed=None,
        problem_ids=None,
    )


def compute_advantages(
    arm: str,
    rewards: torch.Tensor,
    clusters: torch.Tensor | None = None,
    *,
    global_seed: int | None = None,
    problem_ids: list[int] | None = None,
) -> AdvantageOut:
    """Per-rollout advantages and prompt-level keep_mask."""
    if arm == "grpo":
        return _grpo_advantages(rewards)
    if arm in ("minority_answer", "minority_cot"):
        if clusters is None:
            raise ValueError(f"arm {arm!r} requires clusters tensor")
        if global_seed is None or problem_ids is None:
            raise ValueError(
                f"arm {arm!r} requires global_seed and problem_ids for tiebreak rng"
            )
        return _minority_advantages(
            rewards,
            clusters,
            global_seed=global_seed,
            problem_ids=problem_ids,
        )
    if arm == "poly_epo_answer":
        if clusters is None:
            raise ValueError(f"arm {arm!r} requires clusters tensor")
        return _poly_epo_answer_advantages(rewards, clusters)
    raise ValueError(f"unknown arm: {arm!r}")
