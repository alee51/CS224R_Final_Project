"""Minority-CoT advantage kernel + verl adapter helpers.

Math ported verbatim from ``main/train/objective.py`` (the only allowed read from
main/).  Only the ``minority_cot`` arm is included here; GRPO is verl's built-in
(``AdvantageEstimator.GRPO``), and ``poly_epo_cot`` is Stage 5 scope.

Public surface consumed by the verl hook in
``infra/patches/maxrl_minority_cot_adv_est.patch``:

* ``_minority_advantages(rewards, clusters, *, global_seed, problem_ids)``
* ``_group_rewards_by_index(token_level_rewards, response_mask, index, n_rollouts)``
* ``_scatter_advantages_to_tokens(per_rollout_adv, index, response_mask)``
* ``compute_advantages(arm, rewards, clusters, *, global_seed, problem_ids)``
  — thin shim for unit tests; only accepts ``"minority_cot"``.

Public surface shared with Stage 5 (``poly_epo_cot``):

* ``set_based_marginal_advantages`` — Stage 5 imports this with a different
  ``subset_score_fn`` (``_poly_epo_subset_score``).
* ``assign_clusters_from_arm_config`` — mock/judge routing shared by both arms.
* ``_group_rewards_by_index`` / ``_scatter_advantages_to_tokens`` / ``_group_rollouts_for_judge``
  — adapter helpers reused unchanged.

No imports from ``main.train.*``.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (verbatim from main/train/objective.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data class (verbatim from main/train/objective.py)
# ---------------------------------------------------------------------------

@dataclass
class AdvantageOut:
    advantages: torch.Tensor
    keep_mask: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Math primitives (verbatim from main/train/objective.py)
# ---------------------------------------------------------------------------

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
            # verl problem_ids are UUID strings (e.g. 'f8745e4e-b10b-...'); main/'s
            # objective.py assumed integer indices. Hash deterministically (blake2b,
            # 8-byte digest) so the same problem_id seeds the same rng across runs.
            pid = problem_ids[p]
            if isinstance(pid, (int, np.integer)):
                pid_int = int(pid)
            else:
                pid_int = int.from_bytes(
                    hashlib.blake2b(str(pid).encode("utf-8"), digest_size=8).digest(),
                    "big",
                )
            rng = np.random.default_rng((int(global_seed) + pid_int) % (2**63))
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


# ---------------------------------------------------------------------------
# Unit-test shim (only "minority_cot" accepted — "minority_answer" is out of
# scope per migration plan §1; poly_epo_* is Stage 5)
# ---------------------------------------------------------------------------

def compute_advantages(
    arm: str,
    rewards: torch.Tensor,
    clusters: torch.Tensor | None = None,
    *,
    global_seed: int | None = None,
    problem_ids: list[int] | None = None,
) -> AdvantageOut:
    """Per-rollout advantages and prompt-level keep_mask.

    Only ``arm="minority_cot"`` is accepted.  Calling with any other arm raises
    ``ValueError``.  This is intentionally narrower than ``main/train/objective.py``
    which accepts ``grpo``, ``minority_answer``, ``poly_epo_answer``.
    """
    if arm == "minority_cot":
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
    raise ValueError(
        f"unknown arm: {arm!r}.  Only 'minority_cot' is implemented in "
        "main-verl/train/objective_minority.py.  GRPO is verl built-in; "
        "poly_epo_cot is Stage 5 scope."
    )


# ---------------------------------------------------------------------------
# Verl adapter helpers (isolated so Stage 5 reuses unchanged)
# ---------------------------------------------------------------------------

def _group_rewards_by_index(
    token_level_rewards: torch.Tensor,   # [batch, response_length]
    response_mask: torch.Tensor,          # [batch, response_length]
    index: "np.ndarray",                  # [batch] of prompt uids
    n_rollouts: int,
) -> tuple[torch.Tensor, list[int]]:
    """Group per-token rewards into per-rollout scalars.

    Returns
    -------
    rewards : torch.Tensor
        Shape ``[n_prompts, n_rollouts]``.  Each entry is the sum of
        ``token_level_rewards * response_mask`` over the response tokens for
        that rollout — matching the outcome-reward convention used by verl's
        GRPO and MaxRL estimators.
    problem_ids : list[int]
        Ordered list of unique prompt IDs extracted from ``index``.

    Notes
    -----
    ``index`` may be a ``torch.Tensor`` on some maxrl versions.  We call
    ``.cpu().numpy()`` defensively so the caller doesn't need to branch.
    """
    if isinstance(index, torch.Tensor):
        index_np = index.cpu().numpy()
    else:
        index_np = np.asarray(index)

    # Per-rollout scalar reward: sum over response tokens.
    scores = (token_level_rewards * response_mask).sum(dim=-1)  # [batch]

    # Collect unique prompt IDs in stable order of first appearance.
    seen: dict[Any, int] = {}
    for uid in index_np:
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in seen:
            seen[key] = len(seen)

    n_prompts = len(seen)
    # Group rollout scores by prompt ID.
    grouped = [[] for _ in range(n_prompts)]
    for i, uid in enumerate(index_np):
        key = uid.item() if hasattr(uid, "item") else uid
        grouped[seen[key]].append(scores[i])

    # Validate uniform rollout count.
    for p, grp in enumerate(grouped):
        if len(grp) != n_rollouts:
            raise ValueError(
                f"prompt index {p}: expected {n_rollouts} rollouts, "
                f"got {len(grp)}.  Check actor_rollout_ref.rollout.n "
                f"and train_batch_size."
            )

    rewards_grouped = torch.stack([torch.stack(grp) for grp in grouped])  # [n_prompts, n_rollouts]
    problem_ids = list(seen.keys())
    return rewards_grouped, problem_ids


def _group_rollouts_for_judge(
    response_ids: torch.Tensor,           # [batch, response_length] int64
    prompt_ids: torch.Tensor,             # [batch, prompt_length] int64
    index: "np.ndarray",                  # [batch] of prompt uids
    n_rollouts: int,
) -> tuple[list[list[list[int]]], list[list[int]], list[Any]]:
    """Group per-batch rollout token IDs + prompt token IDs by prompt uid.

    Stage 3b (2026-05-30): switched from reading data.non_tensor_batch["raw_prompt"]
    to decoding data.batch["prompts"] in clusters_judge. Reason: verl's rollout-n
    expansion does not interleave non-tensor fields to match batch_size, causing
    an AssertionError ("raw_prompt length 32 != batch size 256") at step 1.
    data.batch["prompts"] IS n-interleaved correctly by verl's standard
    repeat-interleave path.

    Mirrors the grouping convention of ``_group_rewards_by_index``.

    Returns
    -------
    rollout_token_ids : list[n_prompts][n_rollouts][response_length] of int
    prompt_token_ids : list[n_prompts][prompt_length] of int
        Tokenized prompts, one per unique uid (we take the first occurrence —
        all rollouts of a uid share the same prompt).
    problem_ids : list[n_prompts]
    """
    if isinstance(index, torch.Tensor):
        index_np = index.cpu().numpy()
    else:
        index_np = np.asarray(index)

    seen: dict[Any, int] = {}
    for uid in index_np:
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in seen:
            seen[key] = len(seen)

    n_prompts = len(seen)
    rollouts_by_prompt: list[list[list[int]]] = [[] for _ in range(n_prompts)]
    prompts_by_prompt: list[list[int] | None] = [None] * n_prompts
    problem_ids = list(seen.keys())

    for i, uid in enumerate(index_np):
        key = uid.item() if hasattr(uid, "item") else uid
        p = seen[key]
        rollouts_by_prompt[p].append(response_ids[i].cpu().tolist())
        if prompts_by_prompt[p] is None:
            prompts_by_prompt[p] = prompt_ids[i].cpu().tolist()

    for p, rolls in enumerate(rollouts_by_prompt):
        if len(rolls) != n_rollouts:
            raise ValueError(
                f"prompt index {p}: expected {n_rollouts} rollouts, got {len(rolls)}"
            )

    return rollouts_by_prompt, [pi or [] for pi in prompts_by_prompt], problem_ids


def assign_clusters_from_arm_config(
    *,
    problem_ids: list[Any],
    n_rollouts: int,
    arm_config: Any,
    data: Any,
    arm_name: str,
) -> Any:
    """Shared mock/judge cluster routing for set-based arms (minority_cot, poly_epo_cot).

    Reads ``cluster_source`` and judge knobs from a Hydra arm block
    (``algorithm.minority_cot`` or ``algorithm.poly_epo_cot``). Keeps judge
    wiring in one place so fixes (prompt decode, pad strip, client cache) land
    for every arm at once.
    """
    cluster_source = "mock"
    if arm_config is not None:
        cluster_source = str(getattr(arm_config, "cluster_source", "mock"))

    if cluster_source == "mock":
        from train.clusters_mock import assign_mock_clusters
        n_clusters = int(getattr(arm_config, "n_clusters", 4)) if arm_config is not None else 4
        seed = int(getattr(arm_config, "seed", 0)) if arm_config is not None else 0
        return assign_mock_clusters(
            problem_ids, n_rollouts=n_rollouts, n_clusters=n_clusters, seed=seed
        )

    if cluster_source == "judge":
        if data is None:
            raise RuntimeError(
                f"cluster_source=judge for {arm_name} requires the ray_trainer patch "
                "maxrl_expose_data_to_adv_est.patch to expose `data` to the "
                "registered adv_estimator hook. Got data=None."
            )
        from train.clusters_judge import assign_judge_clusters, build_judge_client_from_env

        decoder_path = str(getattr(arm_config, "tokenizer_path"))
        judge_model = str(
            getattr(arm_config, "judge_model", "Qwen/Qwen3-4B-Instruct-2507")
        )
        judge_max_input_tokens = int(
            getattr(arm_config, "judge_max_input_tokens", 36864)
        )

        response_ids = data.batch["responses"]
        prompt_ids = data.batch["prompts"]
        index = data.non_tensor_batch["uid"]

        rollout_token_ids, prompt_token_ids, regrouped_pids = _group_rollouts_for_judge(
            response_ids, prompt_ids, index, n_rollouts=n_rollouts
        )
        if regrouped_pids != problem_ids:
            raise RuntimeError(
                "problem_id ordering mismatch between _group_rewards_by_index "
                "and _group_rollouts_for_judge — refusing to send mis-aligned "
                "data to judge."
            )

        judge_client = build_judge_client_from_env(judge_model=judge_model)
        return assign_judge_clusters(
            problem_ids=problem_ids,
            n_rollouts=n_rollouts,
            rollout_token_ids=rollout_token_ids,
            prompt_token_ids=prompt_token_ids,
            decoder_tokenizer_path=decoder_path,
            judge_tokenizer_path=judge_model,
            judge_client=judge_client,
            judge_max_input_tokens=judge_max_input_tokens,
        )

    raise ValueError(
        f"unknown cluster_source for {arm_name}: {cluster_source!r}. "
        "Expected 'mock' or 'judge'."
    )


def assign_clusters_for_minority_cot_hook(
    *,
    problem_ids: list[Any],
    n_rollouts: int,
    config: Any,
    data: Any,
) -> Any:
    """Route between mock and judge cluster sources based on Hydra config.

    Called from the patched ``compute_minority_cot_outcome_advantage`` hook in
    core_algos.py. Keeps the patch surface in core_algos.py minimal — all
    routing logic lives here so we can change it without re-patching maxrl.

    ``config.algorithm.minority_cot.cluster_source`` values:

    * ``mock`` (default) — Stage 3a behavior: deterministic hash via
      ``clusters_mock.assign_mock_clusters``. No judge call.
    * ``judge`` — Stage 3b behavior: call the deployed judge service via
      ``clusters_judge.assign_judge_clusters``. Requires ``data`` (DataProto
      from the ray_trainer dispatch — see ``maxrl_expose_data_to_adv_est.patch``)
      and a JUDGE_BASE_URL env var.
    """
    mc = getattr(getattr(config, "algorithm", None), "minority_cot", None) if config else None
    return assign_clusters_from_arm_config(
        problem_ids=problem_ids,
        n_rollouts=n_rollouts,
        arm_config=mc,
        data=data,
        arm_name="minority_cot",
    )


def _scatter_advantages_to_tokens(
    per_rollout_adv: torch.Tensor,        # [n_prompts, n_rollouts]
    index: "np.ndarray",                  # [batch]
    response_mask: torch.Tensor,          # [batch, response_length]
) -> torch.Tensor:                        # [batch, response_length]
    """Broadcast per-rollout scalar advantage to every response token.

    Each response token gets the same per-rollout advantage value — matching
    verl's GRPO / MaxRL output convention.  Tokens outside the response mask
    are zero.
    """
    if isinstance(index, torch.Tensor):
        index_np = index.cpu().numpy()
    else:
        index_np = np.asarray(index)

    # Build the same prompt-ID → position mapping as _group_rewards_by_index.
    seen: dict[Any, int] = {}
    for uid in index_np:
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in seen:
            seen[key] = len(seen)

    # Rollout counter per prompt: tracks which rollout index within the prompt
    # each batch row corresponds to.
    rollout_counter: dict[Any, int] = {}

    batch_size, resp_len = response_mask.shape
    token_adv = torch.zeros(batch_size, resp_len, dtype=per_rollout_adv.dtype,
                            device=per_rollout_adv.device)

    for i, uid in enumerate(index_np):
        key = uid.item() if hasattr(uid, "item") else uid
        p = seen[key]
        r = rollout_counter.get(key, 0)
        rollout_counter[key] = r + 1
        adv_scalar = per_rollout_adv[p, r]
        token_adv[i] = adv_scalar * response_mask[i]

    return token_adv
