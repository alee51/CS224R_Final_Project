"""Poly-EPO-CoT advantage kernel.

Math ported verbatim from ``main/train/objective.py`` (the only allowed read
from main/).  Only the ``poly_epo_cot`` arm is included here; ``poly_epo_answer``
is out of scope per migration plan §1.

Why a separate file from ``objective_minority``:

* ``poly_epo_cot`` uses a **different subset-score function**
  (``_poly_epo_subset_score``) and does **not need an rng** for tiebreak —
  the score is deterministic.
* Shared kernel ``set_based_marginal_advantages`` lives in
  ``objective_minority`` and is imported here unchanged.
* Shared verl adapters ``_group_rewards_by_index`` /
  ``_scatter_advantages_to_tokens`` are also imported unchanged.
* Cluster-source routing (mock vs judge) is shared via
  ``assign_clusters_from_arm_config`` in ``objective_minority`` — this hook
  only selects the ``algorithm.poly_epo_cot`` Hydra block.

No imports from ``main.train.*``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from train.objective_minority import (
    SUBSET_SIZE,
    AdvantageOut,
    assign_clusters_from_arm_config,
    set_based_marginal_advantages,
)


# ---------------------------------------------------------------------------
# Subset-score primitive (verbatim from main/train/objective.py:83-85)
# ---------------------------------------------------------------------------

def _poly_epo_subset_score(rewards4: np.ndarray, clusters4: np.ndarray) -> float:
    """f(G) = mean(r in G) * (distinct cluster ids in G) / k.

    Deterministic — no rng tiebreak needed (unlike minority_cot).
    """
    return float(rewards4.mean() * len(set(clusters4.tolist())) / SUBSET_SIZE)


def _poly_epo_cot_advantages(
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


# ---------------------------------------------------------------------------
# Cluster-source routing hook
# ---------------------------------------------------------------------------

def assign_clusters_for_poly_epo_cot_hook(
    *,
    problem_ids: list[Any],
    n_rollouts: int,
    config: Any,
    data: Any,
) -> Any:
    """Route between mock and judge cluster sources for poly_epo_cot."""
    from train.objective_minority import arm_block_from_adv_config

    pe = arm_block_from_adv_config(config, "poly_epo_cot")
    return assign_clusters_from_arm_config(
        problem_ids=problem_ids,
        n_rollouts=n_rollouts,
        arm_config=pe,
        data=data,
        arm_name="poly_epo_cot",
    )


# ---------------------------------------------------------------------------
# Unit-test shim
# ---------------------------------------------------------------------------

def compute_advantages_poly_epo_cot(
    rewards: torch.Tensor,
    clusters: torch.Tensor,
) -> AdvantageOut:
    """Test-only convenience wrapper. Production code goes via the registered hook."""
    return _poly_epo_cot_advantages(rewards, clusters)
