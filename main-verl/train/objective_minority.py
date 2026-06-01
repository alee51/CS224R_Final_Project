"""Minority-CoT advantage kernel + verl adapter helpers.

Math ported verbatim from ``main/train/objective.py`` (the only allowed read from
main/).  Only the ``minority_cot`` arm is included here; GRPO is verl's built-in
(``AdvantageEstimator.GRPO``), and ``poly_epo_cot`` is Stage 5 scope.

Public surface consumed by the verl hook on the maxrl fork (fork commit
``e047d0e cs224r: add MINORITY_COT advantage estimator to core_algos`` on
branch ``cs224r-patches``):

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
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from judge.types import DEGENERATE_CLUSTER_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-relaunch #2c: shared `\boxed{...}` extractor
# ---------------------------------------------------------------------------
#
# The Hendrycks-style reward fn at ``verl/utils/reward_score/math.py``
# (``compute_score``) only returns a scalar 0/1 — it parses ``\boxed{...}``
# internally via ``last_boxed_only_string``/``remove_boxed`` but does not
# expose the parsed answer. Threading a side-channel through verl's reward
# manager + DataProto would touch ``naive.py`` / ``batch.py`` and the reward
# dispatch shim. Cheaper to mirror the parse trainer-side and call it from
# both the set-arm path (``clusters_judge.assign_judge_clusters``, which
# already decodes responses) and the GRPO path (``_build_grpo_step_metrics``,
# which decodes via a tokenizer stashed on ``data.meta_info``).
#
# Used by both arms via the per-rollout JSONL writer in ``_build_step_metrics``
# so the schema's ``parsed_answer`` column is populated symmetrically.
def _extract_boxed_answer(text: str) -> str:
    """Return the contents of the last ``\\boxed{...}`` in ``text``, or empty string.

    Matches ``last_boxed_only_string`` / ``remove_boxed`` in
    ``verl/utils/reward_score/math.py`` semantically but uses a balanced-brace
    scan so nested braces in the answer (``\\frac{1}{2}``, etc.) round-trip.
    We do NOT normalize / strip — offline analysis decides equivalence.
    """
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    # Unbalanced — best-effort: return the tail.
    return text[start:]

# ---------------------------------------------------------------------------
# Stage 7: per-step W&B metrics state
# ---------------------------------------------------------------------------

# Accumulates problem_ids of prompts where ≥1 rollout was correct across the run.
# Lives in the driver process (compute_advantage runs on the head node), so it
# persists across training steps and resets on container restart (one run = one container).
_SOLVED_PROBLEM_IDS: set = set()


# ---------------------------------------------------------------------------
# Pre-relaunch #2: per-rollout JSONL detail logging
# ---------------------------------------------------------------------------
#
# Sink: Modal volume ``main-artifacts`` (constants in ``infra/modal_volume.py``)
# mounted at ``/vol`` inside the training container. We do **not** import
# infra/modal_volume here — `objective_minority.py` is imported by unit tests
# that have no Modal dependency. Instead, we read the root from the env var
# ``CS224R_PER_ROLLOUT_ROOT`` (set by the Modal probe), with a fallback default
# of ``/vol/per_rollout`` that matches the production mount path.
#
# Layout: ``<root>/<run_id>/step_<global_step>.jsonl``. ``run_id`` comes from
# ``WANDB_RUN_ID`` (verl/wandb sets this) or ``CS224R_RUN_ID`` for unit/probe
# overrides; falls back to ``unknown_run``. One row per
# ``(prompt × rollout_idx)`` per step — append-only, survives resume.
#
# If the root path is not writable (e.g. unit tests, smokes without volume),
# the writer is a no-op and emits a single warning per process.
_PER_ROLLOUT_WARNED: set = set()


def _per_rollout_root() -> Path:
    return Path(os.environ.get("CS224R_PER_ROLLOUT_ROOT", "/vol/per_rollout"))


def _per_rollout_run_id() -> str:
    return (
        os.environ.get("CS224R_RUN_ID")
        or os.environ.get("WANDB_RUN_ID")
        or "unknown_run"
    )


def _per_rollout_path(global_step: int) -> Path:
    return _per_rollout_root() / _per_rollout_run_id() / f"step_{global_step}.jsonl"


def _write_per_rollout_rows(rows: list[dict[str, Any]], global_step: int) -> Path | None:
    """Append one JSONL row per ``(prompt × rollout)`` to the step-partitioned file.

    Returns the path written, or None if the sink was unavailable (logged once).
    """
    if not rows:
        return None
    path = _per_rollout_path(global_step)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        return path
    except OSError as exc:
        if str(path.parent) not in _PER_ROLLOUT_WARNED:
            _PER_ROLLOUT_WARNED.add(str(path.parent))
            logger.warning(
                "per-rollout JSONL sink unavailable at %s (%s); skipping further writes.",
                path,
                exc,
            )
        return None


def _finish_reasons_for_rollouts(
    data: Any,
    index_np: np.ndarray,
    problem_ids: list,
    n_rollouts: int,
) -> list[list[str | None]] | None:
    """Group ``data.non_tensor_batch['finish_reasons']`` by prompt.

    Returns shape ``[n_prompts][n_rollouts]`` of finish-reason strings, or None
    if the field is not present (older fork, mock smoke).
    """
    if data is None:
        return None
    fr = data.non_tensor_batch.get("finish_reasons") if hasattr(data, "non_tensor_batch") else None
    if fr is None:
        return None
    # Mirror the grouping in _group_rewards_by_index so rows align to (prompt, rollout_idx).
    seen: dict[Any, int] = {}
    for uid in index_np:
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in seen:
            seen[key] = len(seen)
    grouped: list[list[str | None]] = [[] for _ in range(len(seen))]
    for i, uid in enumerate(index_np):
        key = uid.item() if hasattr(uid, "item") else uid
        grouped[seen[key]].append(fr[i] if i < len(fr) else None)
    return grouped


def _response_lengths_for_rollouts(
    data: Any,
    index_np: np.ndarray,
    n_rollouts: int,
) -> list[list[int]] | None:
    """Per-rollout response length (count of unmasked response tokens), grouped by prompt."""
    if data is None or not hasattr(data, "batch"):
        return None
    rm = data.batch.get("response_mask") if "response_mask" in data.batch.keys() else None
    if rm is None:
        return None
    lengths = rm.sum(dim=-1).long().cpu().tolist()  # [batch]
    seen: dict[Any, int] = {}
    for uid in index_np:
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in seen:
            seen[key] = len(seen)
    grouped: list[list[int]] = [[] for _ in range(len(seen))]
    for i, uid in enumerate(index_np):
        key = uid.item() if hasattr(uid, "item") else uid
        grouped[seen[key]].append(int(lengths[i]))
    return grouped


def _finish_reason_counts(finish_reasons_flat: list[str | None] | None) -> dict[str, int]:
    """Per-step aggregate counts for #3 ``train/finish_*`` W&B metrics.

    Categories:
      - ``length`` — vLLM ``finish_reason == "length"`` (response cap hit).
      - ``eos``    — ``"stop"`` or ``"eos"``.
      - ``stop``   — any other non-empty reason (vLLM ``"abort"``, custom stop strings, etc.).
    """
    out = {"length": 0, "eos": 0, "stop": 0}
    if not finish_reasons_flat:
        return out
    for fr in finish_reasons_flat:
        if fr is None:
            continue
        s = str(fr).lower()
        if s == "length":
            out["length"] += 1
        elif s in ("stop", "eos"):
            out["eos"] += 1
        else:
            out["stop"] += 1
    return out


def _build_step_metrics(
    rewards_grouped: torch.Tensor,   # [n_prompts, n_rollouts]
    problem_ids: list,
    adv_diagnostics: dict,
    cluster_diagnostics: dict,
    data: Any = None,
) -> dict:
    """Metrics dict written to batch.meta_info['cs224r_metrics'] each step.

    Read by the ray_trainer hook on the maxrl fork (fork commit ``096fae1
    cs224r: forward cs224r_metrics from adv hooks to W&B``) and merged into
    the W&B metrics dict alongside verl's native ``compute_data_metrics()``.

    Pre-relaunch #2: when ``data`` is provided, also writes one JSONL row per
    ``(prompt × rollout)`` to the Modal volume. Per-rollout payloads
    (``cluster_ids``, ``parsed_answers``) come in via ``cluster_diagnostics``
    from ``clusters_judge.assign_judge_clusters`` — they are popped before
    return so the forwarder doesn't try to log tensors/lists to W&B.

    Pre-relaunch #3: when ``finish_reasons`` is present in
    ``data.non_tensor_batch`` (vLLM rollout adapter populates it), emit
    ``train/finish_length`` / ``train/finish_eos`` / ``train/finish_stop``.
    """
    global _SOLVED_PROBLEM_IDS

    any_correct = (rewards_grouped > 0).any(dim=1)          # [n_prompts] bool
    pass_at_8 = any_correct.float().mean().item()

    solved_this_step = [
        pid for pid, ok in zip(problem_ids, any_correct.tolist()) if ok
    ]
    _SOLVED_PROBLEM_IDS.update(solved_this_step)

    metrics: dict[str, Any] = {
        "train/pass_at_8": pass_at_8,
        "train/prompts_unlocked": len(_SOLVED_PROBLEM_IDS),
        "train/fraction_filtered": adv_diagnostics.get("fraction_filtered", 0.0),
    }

    # ---- Pop per-rollout payloads (non-scalar) BEFORE we copy scalars to W&B.
    cluster_ids_t = cluster_diagnostics.pop("cluster_ids", None) if cluster_diagnostics else None
    parsed_answers = cluster_diagnostics.pop("parsed_answers", None) if cluster_diagnostics else None

    # Judge/cluster diagnostics — only present when cluster_source=judge.
    if cluster_diagnostics:
        for key in ("distinct_clusters_mean", "degenerate_rollouts",
                    "judge_parse_ok_rate", "judge_overflow_skipped"):
            if key in cluster_diagnostics:
                metrics[f"train/{key}"] = cluster_diagnostics[key]

    # ---- Per-rollout JSONL detail logging (#2) + finish_reason aggregates (#3).
    if data is not None:
        index = data.non_tensor_batch.get("uid") if hasattr(data, "non_tensor_batch") else None
        index_np = np.asarray(index) if index is not None else None
        finish_grouped = (
            _finish_reasons_for_rollouts(data, index_np, problem_ids, rewards_grouped.shape[1])
            if index_np is not None else None
        )
        length_grouped = (
            _response_lengths_for_rollouts(data, index_np, rewards_grouped.shape[1])
            if index_np is not None else None
        )

        # #3 aggregates — count across all rollouts in this batch.
        if finish_grouped is not None:
            flat = [fr for row in finish_grouped for fr in row]
            counts = _finish_reason_counts(flat)
            metrics["train/finish_length"] = counts["length"]
            metrics["train/finish_eos"] = counts["eos"]
            metrics["train/finish_stop"] = counts["stop"]

        # #2 per-rollout JSONL.
        global_step = int(data.meta_info.get("global_steps", -1)) if hasattr(data, "meta_info") else -1
        rows = _per_rollout_rows(
            global_step=global_step,
            problem_ids=problem_ids,
            rewards_grouped=rewards_grouped,
            cluster_ids=cluster_ids_t,
            parsed_answers=parsed_answers,
            finish_grouped=finish_grouped,
            length_grouped=length_grouped,
        )
        _write_per_rollout_rows(rows, global_step)

    return metrics


def _per_rollout_rows(
    *,
    global_step: int,
    problem_ids: list,
    rewards_grouped: torch.Tensor,
    cluster_ids: torch.Tensor | None,
    parsed_answers: list[list[str]] | None,
    finish_grouped: list[list[str | None]] | None,
    length_grouped: list[list[int]] | None,
) -> list[dict[str, Any]]:
    """Build one dict per (prompt, rollout_idx) for the JSONL sink.

    Schema (per the relaunch doc):
      global_step, prompt_id, rollout_idx, parsed_answer, reward,
      cluster_id, finish_reason, response_length.

    Set-arm rows carry a non-null ``cluster_id``; GRPO rows pass ``cluster_ids=None``
    and get ``cluster_id=null`` — same schema across all three arms so offline
    joins are trivial.
    """
    rows: list[dict[str, Any]] = []
    rewards = rewards_grouped.detach().cpu().tolist()
    cids = cluster_ids.detach().cpu().tolist() if cluster_ids is not None else None
    n_prompts = len(problem_ids)
    n_rollouts = rewards_grouped.shape[1]
    for p in range(n_prompts):
        pid = problem_ids[p]
        pid_out = pid.item() if hasattr(pid, "item") else pid
        for r in range(n_rollouts):
            row = {
                "global_step": global_step,
                "prompt_id": pid_out,
                "rollout_idx": r,
                "parsed_answer": (parsed_answers[p][r] if parsed_answers else None),
                "reward": float(rewards[p][r]),
                "cluster_id": (int(cids[p][r]) if cids is not None else None),
                "finish_reason": (finish_grouped[p][r] if finish_grouped else None),
                "response_length": (int(length_grouped[p][r]) if length_grouped else None),
            }
            rows.append(row)
    return rows


def _parsed_answers_for_grpo(
    data: Any,
    index_np: np.ndarray,
    n_rollouts: int,
) -> list[list[str]] | None:
    """Per-rollout boxed-answer strings on the GRPO path, grouped by prompt.

    Pre-relaunch #2c: symmetric with the set arms, which get
    ``parsed_answers`` from ``clusters_judge.assign_judge_clusters`` (it
    decodes responses for the judge call and reuses the decode). On the GRPO
    path there's no judge, so we decode ourselves with the tokenizer stashed
    on ``data.meta_info["cs224r_tokenizer"]`` (set in ``fit()`` next to the
    ``global_steps`` stash, fork-side).

    Returns ``None`` if the tokenizer is missing (smoke/older fork) — the JSONL
    writer then emits ``parsed_answer=null`` for that step, same as before.
    """
    if data is None or not hasattr(data, "batch"):
        return None
    tok = data.meta_info.get("cs224r_tokenizer") if hasattr(data, "meta_info") else None
    if tok is None:
        return None
    responses = data.batch.get("responses")
    if responses is None:
        return None
    # Decode batch once, then group by uid like _group_rewards_by_index.
    texts = tok.batch_decode(responses, skip_special_tokens=True)
    seen: dict[Any, int] = {}
    for uid in index_np:
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in seen:
            seen[key] = len(seen)
    grouped: list[list[str]] = [[] for _ in range(len(seen))]
    for i, uid in enumerate(index_np):
        key = uid.item() if hasattr(uid, "item") else uid
        grouped[seen[key]].append(_extract_boxed_answer(texts[i]))
    return grouped


def _build_grpo_step_metrics(
    token_level_rewards: torch.Tensor,   # [batch, response_length]
    response_mask: torch.Tensor,         # [batch, response_length]
    index: np.ndarray,                   # [batch] uids
    data: Any,
) -> dict:
    """GRPO W&B parity (#9) + GRPO-side per-rollout JSONL (#2).

    Verl's stock GRPO advantage path does not go through ``_build_step_metrics``
    today — this is the entry point the fork-side patch
    (``infra/patches/maxrl_relaunch_2_3_9.patch``) calls from inside the
    GRPO branch of ``compute_advantage`` in ``ray_trainer.py``.

    Populates only ``train/pass_at_8``, ``train/prompts_unlocked``,
    ``train/fraction_filtered`` (and finish_reason aggregates). No judge keys.

    Pre-relaunch #2c: also extracts ``\\boxed{...}`` per-rollout via the
    tokenizer stash, so the GRPO JSONL rows carry the same ``parsed_answer``
    field the set arms do — joins across the three arms stay trivial.

    Pre-relaunch #d: ``fraction_filtered`` is computed from the same
    ``keep_mask = all-rollouts-equal-reward`` predicate verl's GRPO path
    implicitly applies (id2mean − rewards = 0 when all rewards in a group
    match, so the resulting advantage is zero regardless of std-norm). We do
    NOT additionally zero anything trainer-side: the gradient contribution is
    already zero by construction; this just reports the rate.
    """
    # Group rewards prompt-major to compute pass_at_8 / fraction_filtered.
    rewards_grouped, problem_ids = _group_rewards_by_index(
        token_level_rewards, response_mask, index, n_rollouts=N_ROLLOUTS,
    )
    # zero-grad trigger on GRPO: all rollouts in a prompt share one reward →
    # the GRPO baseline (rewards − id2mean) is identically zero → no gradient
    # signal from that prompt. Mirrors the set-arm `keep_mask = False` rule at
    # objective_minority.set_based_marginal_advantages (single-cluster
    # collapse). Both arms emit `train/fraction_filtered` with consistent
    # semantics: "fraction of prompts where the kernel produces zero
    # gradient for this prompt".
    keep_mask = ~((rewards_grouped == rewards_grouped[:, :1]).all(dim=1))
    n_prompts = rewards_grouped.shape[0]
    n_filtered = int((~keep_mask).sum().item())
    adv_diagnostics = {
        "fraction_filtered": n_filtered / max(n_prompts, 1),
        "n_filtered_prompts": n_filtered,
    }

    # #2c: build parsed_answers via the stashed tokenizer + boxed extractor;
    # plumb through cluster_diagnostics so the JSONL writer in
    # _build_step_metrics picks it up like it does on the judge path.
    index_np = np.asarray(index) if not isinstance(index, np.ndarray) else index
    parsed_answers = _parsed_answers_for_grpo(data, index_np, N_ROLLOUTS)
    cluster_diagnostics: dict[str, Any] = {}
    if parsed_answers is not None:
        cluster_diagnostics["parsed_answers"] = parsed_answers

    return _build_step_metrics(
        rewards_grouped, problem_ids, adv_diagnostics, cluster_diagnostics, data=data
    )


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
    """f(G) = mean reward of rollouts in the rarest cluster (random tiebreak).

    Degenerate-rollout handling (cluster_id = DEGENERATE_CLUSTER_ID = -1, paper's
    "cluster 100" for code/gibberish/non-math responses) is an OPEN POLICY
    DECISION — see `main/docs/timeline.md` (2026-05-31 "Open policy question").
    Current behavior: -1 is treated like any other cluster ID, so it can be
    selected as "rarest" and a degenerate rollout that happened to box the right
    answer can become the subset score.
    """
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
            # Pre-relaunch #d: this is the set-arm zero-grad trigger reported
            # via `train/fraction_filtered`. Mirror on the GRPO path is
            # all-rollouts-equal-reward (see `_build_grpo_step_metrics`).
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


def arm_block_from_adv_config(config: Any, block_name: str) -> Any:
    """Resolve ``algorithm.<block_name>`` from the config VeRL passes to adv hooks.

    ``ray_trainer`` calls ``compute_advantage(..., config=self.config.algorithm)`` —
    the hook receives the **algorithm** subtree, not the full trainer config. Older
    code used ``config.algorithm.minority_cot``, which is always ``None`` here and
    silently forced ``cluster_source=mock`` even when Hydra set ``judge``.
    """
    if config is None:
        return None
    direct = getattr(config, block_name, None)
    if direct is not None:
        return direct
    nested = getattr(getattr(config, "algorithm", None), block_name, None)
    return nested


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

    print(
        f"[cluster_route] arm={arm_name} source={cluster_source} "
        f"arm_config_set={arm_config is not None} data_set={data is not None}",
        flush=True,
    )

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
                f"cluster_source=judge for {arm_name} requires the ray_trainer "
                "hook from maxrl fork commit 572a592 (cs224r-patches branch), "
                "which exposes `data` to registered adv_estimator hooks. Got "
                "data=None — check MAXRL_BRANCH_COMMIT in infra/modal_image.py."
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

        judge_client = build_judge_client_from_env(
            judge_model=judge_model,
            arm_config=arm_config,
        )
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
      from the ray_trainer dispatch — exposed by maxrl fork commit 572a592
      on cs224r-patches) and a ``JUDGE_BASE_URL`` env var.
    """
    mc = arm_block_from_adv_config(config, "minority_cot")
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
