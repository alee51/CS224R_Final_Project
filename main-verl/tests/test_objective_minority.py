"""Set-based advantage kernel tests (minority_cot arm).

Ported from ``main/tests/test_objective_minority.py`` with the following changes:

* Import path: ``train.objective`` → ``train.objective_minority`` (new module).
* Arm string: ``"minority_answer"`` → ``"minority_cot"`` throughout.  Both arms share
  the exact same ``_minority_advantages`` body; migration plan §1 scopes the verl port
  to ``minority_cot`` only.  ``minority_answer`` is intentionally not accepted by the
  ``compute_advantages`` shim in ``objective_minority.py``.

Tests intentionally NOT ported (see plan S3a.3):
* ``test_poly_epo_subset_score_hand``, ``test_poly_epo_advantages_zero_sum_and_diversity_signal``
  — Stage 5 (``poly_epo_cot``) scope.
* ``test_grpo_unchanged`` — verl owns GRPO; no custom GRPO path in main-verl.

Run locally (pure CPU math, no GPU needed):

    cd /path/to/cs224r_finalproject
    PYTHONPATH=main-verl python3 -m pytest main-verl/tests/test_objective_minority.py -v
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest
import torch

from train.objective_minority import (
    N_ROLLOUTS,
    SUBSET_SIZE,
    _minority_subset_score,
    compute_advantages,
    set_based_marginal_advantages,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ind_marginal(fG: np.ndarray, n_rollouts: int = 8, subset_size: int = 4):
    """Independent reference marginal computation."""
    subsets = list(itertools.combinations(range(n_rollouts), subset_size))
    arr = np.array(subsets, dtype=np.int64)
    incl = [np.where((arr == i).any(axis=1))[0] for i in range(n_rollouts)]
    baseline = fG.mean()
    set_adv = fG - baseline
    return np.array([set_adv[incl[i]].mean() for i in range(n_rollouts)])


# ---------------------------------------------------------------------------
# Ported fixture tests (arm = "minority_cot")
# ---------------------------------------------------------------------------

def test_collapsed_cluster_filtered():
    """All 8 rollouts in the same cluster -> keep_mask False, zero advantages."""
    rewards = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    clusters = torch.zeros((1, 8), dtype=torch.long)
    out = compute_advantages(
        "minority_cot",
        rewards,
        clusters,
        global_seed=42,
        problem_ids=[0],
    )
    assert not bool(out.keep_mask[0].item())
    assert torch.allclose(out.advantages[0], torch.zeros(8))
    assert out.diagnostics["fraction_filtered"] == 1.0


def test_minority_seven_one_split_signs():
    """7 in cluster A (reward 0), 1 in cluster B (reward 1).

    Minority cluster B appears in C(7,3)=35 subsets — every subset containing
    rollout 0 (the minority). f(G) for those = 1, otherwise = 0 (mean reward
    of cluster A in G). So marginal advantage of rollout 0 must be positive
    and marginals of the other 7 negative.
    """
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    clusters = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    out = compute_advantages(
        "minority_cot",
        rewards,
        clusters,
        global_seed=0,
        problem_ids=[0],
    )
    assert bool(out.keep_mask[0].item())
    adv = out.advantages[0].numpy()
    assert adv[0] > 0
    assert (adv[1:] < 0).all()
    # Sum of advantages must be zero by construction (marginals around mean).
    assert abs(adv.sum()) < 1e-5


def test_minority_marginals_match_reference():
    """Kernel output matches an independent numpy reference."""
    rewards = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    clusters = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long)
    # Compute reference f(G) over all 70 subsets via the published scorer.
    rng = np.random.default_rng(123 + 7)
    fG = []
    for idxs in itertools.combinations(range(8), 4):
        r4 = rewards[0].numpy()[list(idxs)]
        c4 = clusters[0].numpy()[list(idxs)]
        fG.append(_minority_subset_score(r4, c4, rng))
    ref = _ind_marginal(np.array(fG))
    out = compute_advantages(
        "minority_cot",
        rewards,
        clusters,
        global_seed=123,
        problem_ids=[7],
    )
    np.testing.assert_allclose(out.advantages[0].numpy(), ref, atol=1e-6)


def test_tiebreak_reproducible_same_seed():
    """Same (global_seed, problem_id) -> identical advantages across calls."""
    rewards = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    # Two clusters tied at count 4 in many subsets -> tiebreak matters.
    clusters = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.long)
    a = compute_advantages(
        "minority_cot", rewards, clusters,
        global_seed=99, problem_ids=[3],
    )
    b = compute_advantages(
        "minority_cot", rewards, clusters,
        global_seed=99, problem_ids=[3],
    )
    np.testing.assert_array_equal(a.advantages.numpy(), b.advantages.numpy())


def test_minority_arm_requires_clusters_and_seed():
    rewards = torch.zeros((1, 8))
    with pytest.raises(ValueError):
        compute_advantages("minority_cot", rewards)
    with pytest.raises(ValueError):
        compute_advantages(
            "minority_cot",
            rewards,
            torch.zeros((1, 8), dtype=torch.long),
        )


def test_subset_constants():
    assert N_ROLLOUTS == 8
    assert SUBSET_SIZE == 4


# ---------------------------------------------------------------------------
# New test: mock-cluster reproducibility (S3a.1 surface)
# ---------------------------------------------------------------------------

def test_mock_cluster_reproducibility():
    """assign_mock_clusters(seed, problem_ids, n_rollouts, n_clusters) is
    deterministic across calls with the same seed."""
    from train.clusters_mock import assign_mock_clusters
    a = assign_mock_clusters([0, 1, 2], n_rollouts=8, n_clusters=4, seed=42)
    b = assign_mock_clusters([0, 1, 2], n_rollouts=8, n_clusters=4, seed=42)
    assert torch.equal(a.cluster_ids, b.cluster_ids)
    # Different seed should produce different IDs (with high probability).
    c = assign_mock_clusters([0, 1, 2], n_rollouts=8, n_clusters=4, seed=43)
    assert not torch.equal(a.cluster_ids, c.cluster_ids)


# ---------------------------------------------------------------------------
# New bridge test: mock → objective → marginal advantage path
# ---------------------------------------------------------------------------

def test_mock_clusters_drive_minority_advantages_end_to_end():
    """8 rollouts × 4 mock clusters -> keep_mask should be True for at least
    one prompt in a small batch, and sum-of-advantages ≈ 0 per prompt."""
    from train.clusters_mock import assign_mock_clusters
    from train.objective_minority import _minority_advantages
    pids = list(range(16))
    rewards = torch.rand((16, 8))
    asg = assign_mock_clusters(pids, n_rollouts=8, n_clusters=4, seed=0)
    out = _minority_advantages(rewards, asg.cluster_ids, global_seed=0, problem_ids=pids)
    assert out.keep_mask.any().item()
    kept_adv = out.advantages[out.keep_mask]
    assert torch.allclose(kept_adv.sum(dim=1), torch.zeros(kept_adv.shape[0]), atol=1e-5)
