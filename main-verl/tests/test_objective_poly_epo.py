"""Poly-EPO-CoT advantage kernel tests (Stage 5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from train.objective_poly_epo import (
    _poly_epo_subset_score,
    compute_advantages_poly_epo_cot,
)


def test_poly_epo_subset_score_hand():
    """rewards=[1,1,0,0], clusters=[0,0,1,1] -> 0.5 * 2/4 = 0.25."""
    r4 = np.array([1.0, 1.0, 0.0, 0.0])
    c4 = np.array([0, 0, 1, 1])
    assert _poly_epo_subset_score(r4, c4) == pytest.approx(0.25)


def test_poly_epo_advantages_zero_sum_and_diversity_signal():
    """Higher cluster diversity within a subset boosts f -> positive marginal
    for rollouts that bring new clusters."""
    rewards = torch.ones((1, 8), dtype=torch.float32)
    clusters = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.long)
    out = compute_advantages_poly_epo_cot(rewards, clusters)
    adv = out.advantages[0].numpy()
    assert bool(out.keep_mask[0].item())
    assert adv[7] > 0
    assert (adv[:7] < 0).all()
    assert abs(adv.sum()) < 1e-5


def test_poly_epo_differs_from_minority_on_same_fixture():
    """Same (rewards, clusters) -> different advantages (migration plan row-5 gate)."""
    rewards = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)
    clusters = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long)
    from train.objective_minority import compute_advantages as minority_adv
    from train.objective_poly_epo import compute_advantages_poly_epo_cot

    m = minority_adv("minority_cot", rewards, clusters, global_seed=0, problem_ids=[0])
    p = compute_advantages_poly_epo_cot(rewards, clusters)
    assert not torch.allclose(m.advantages, p.advantages, atol=1e-6)


def test_poly_epo_mock_cluster_end_to_end():
    """Mock clusters -> poly_epo kernel -> keep_mask + zero-sum per prompt."""
    from train.clusters_mock import assign_mock_clusters
    from train.objective_poly_epo import compute_advantages_poly_epo_cot

    pids = list(range(16))
    rewards = torch.rand((16, 8))
    asg = assign_mock_clusters(pids, n_rollouts=8, n_clusters=4, seed=0)
    out = compute_advantages_poly_epo_cot(rewards, asg.cluster_ids)
    assert out.keep_mask.any().item()
    kept = out.advantages[out.keep_mask]
    assert torch.allclose(kept.sum(dim=1), torch.zeros(kept.shape[0]), atol=1e-5)
