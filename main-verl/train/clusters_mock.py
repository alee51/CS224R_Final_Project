"""Deterministic mock cluster-ID source for Stage 3a smoke.

Stage 3b → judge swap contract
-------------------------------
This module implements the **mock** side of the cluster-ID interface.  Stage 3b
will drop-in replace it with ``clusters_judge.py`` that calls the real judge
service, **without changing the objective hook or any downstream consumer**.

Fields that remain identical in Stage 3b:

* ``ClusterAssignment.cluster_ids`` — ``torch.int64`` shape
  ``[n_prompts, n_rollouts]``.  Every consumer (``_minority_advantages``,
  ``set_based_marginal_advantages``) only touches this field.
* ``ClusterAssignment.diagnostics`` dict structure — keys
  ``distinct_clusters_mean`` (float) and ``degenerate_rollouts`` (int).

Fields that change in Stage 3b:

* The ``degenerate_rollouts`` value: always ``0`` here because the mock hash
  never fails; the real judge may emit a sentinel for prompts it cannot cluster
  ("degenerate bucket" per migration plan §4).
* The function signature of the generator: Stage 3b's
  ``assign_judge_clusters`` adds ``rollout_texts`` and ``judge_client``
  arguments.  The objective hook must be written to call via a thin adapter
  rather than calling this function directly.

Why this mock is faithful for bring-up
---------------------------------------
1. **Same shape contract.** ``cluster_ids[p, r] ∈ [0, K)`` with K configurable
   via Hydra (``algorithm.minority_cot.n_clusters``).  The marginal-advantage
   kernel never looks at where the IDs came from.

2. **Same value range.** K=4 matches the upper end of the Stage 4 forced-k
   candidate range (TA OH 2026-05-28: k=2..4 under discussion).

3. **Same failure mode reachable.** When the hash maps all 8 rollouts of a
   prompt to the same cluster, ``set_based_marginal_advantages`` sets
   ``keep_mask[p]=False`` and emits zero advantages — exactly the code path the
   real judge triggers on "degenerate cluster" prompts.

4. **No information about CoT content.** That is intentional: Stage 3a tests
   plumbing correctness, not science.  Stage 3b is where real clustering signal
   arrives.

<!-- TODO: confirm n_clusters=4 is the right Stage 3a default once Stage 4
     forced-k decision lands.  If Stage 4 picks forced-k=2, bump to 2 here
     so smoke stats match the production distribution. -->
<!-- TODO: add a pytest.mark.parametrize smoke test for _mock_cluster (see
     test_objective_minority.py test_mock_cluster_reproducibility). -->
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Public contract (unchanged by Stage 3b judge swap)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterAssignment:
    """Cluster assignments for a batch of prompts × rollouts.

    Attributes
    ----------
    cluster_ids:
        ``torch.int64`` tensor of shape ``[n_prompts, n_rollouts]``.
        ``cluster_ids[p, r] ∈ [0, K)`` where K = ``n_clusters``.
    diagnostics:
        Scalar diagnostics dict.  Keys:

        * ``distinct_clusters_mean`` (float) — mean over prompts of
          ``len(set(cluster_ids[p]))``.  Logged as ``train/distinct_clusters``.
        * ``degenerate_rollouts`` (int) — count of rollouts the judge could not
          cluster.  Always ``0`` in the mock; real judge fills this.
    """

    cluster_ids: torch.Tensor   # int64, shape [n_prompts, n_rollouts]
    diagnostics: dict           # keys: distinct_clusters_mean, degenerate_rollouts


def assign_mock_clusters(
    problem_ids: list[int],
    n_rollouts: int,
    n_clusters: int,
    *,
    seed: int,
) -> ClusterAssignment:
    """Deterministic mock cluster IDs for Stage 3a smoke.

    Contract (must match Stage 3b real-judge contract):

    * ``cluster_ids[p, r]`` = stable hash of ``(seed, problem_ids[p], r)``
      modulo ``n_clusters``.
    * ``distinct_clusters_mean`` = mean over p of
      ``len(set(cluster_ids[p]))``.
    * ``degenerate_rollouts`` is always 0 in the mock (real judge fills this).

    Parameters
    ----------
    problem_ids:
        List of integer problem IDs; length = n_prompts.
    n_rollouts:
        Number of rollouts per prompt.  Must be 8 for the minority-CoT arm
        (``objective_minority.py`` constant ``N_ROLLOUTS = 8``).
    n_clusters:
        Cluster count K; IDs are in ``[0, K)``.
    seed:
        Hash seed.  Controls reproducibility across runs.  Default in Hydra:
        ``algorithm.minority_cot.seed = 0``.
    """
    n_prompts = len(problem_ids)
    ids = torch.zeros((n_prompts, n_rollouts), dtype=torch.int64)
    for p, pid in enumerate(problem_ids):
        for r in range(n_rollouts):
            ids[p, r] = _mock_cluster(seed, pid, r, n_clusters)

    distinct = float(
        sum(len(set(ids[p].tolist())) for p in range(n_prompts)) / max(n_prompts, 1)
    )
    diagnostics = {
        "distinct_clusters_mean": distinct,
        "degenerate_rollouts": 0,
    }
    return ClusterAssignment(cluster_ids=ids, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Internal hash primitive (locked per stage-03a-agent-plan.md §S3a.1)
# ---------------------------------------------------------------------------

def _mock_cluster(seed: int, problem_id: int, rollout_idx: int, K: int) -> int:
    """Stable hash of (seed, problem_id, rollout_idx) mod K.

    Uses ``hashlib.blake2b`` (process-stable, cryptographically strong).
    Python's built-in ``hash()`` is explicitly forbidden here: it is
    PYTHONHASHSEED-salted and produces different values across Modal container
    starts, breaking reproducibility.
    """
    h = hashlib.blake2b(
        f"{seed}|{problem_id}|{rollout_idx}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "big") % K
