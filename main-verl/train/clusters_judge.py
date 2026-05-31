"""Real-judge cluster-ID source for Stage 3b (and Stage 5 poly_epo_cot).

Stage 3b → swap from clusters_mock
-----------------------------------
Drop-in replacement for ``clusters_mock.assign_mock_clusters``. The
``ClusterAssignment`` dataclass and shape contract are identical so the
``compute_minority_cot_outcome_advantage`` kernel does not change.

What changes vs the mock:

* Source of cluster IDs: calls the Modal-hosted judge service
  (``judge.client.JudgeClient``) instead of hashing problem_id × rollout_idx.
* ``diagnostics`` adds:
  * ``judge_parse_ok_rate`` — fraction of prompts whose JSON parsed cleanly.
  * ``judge_overflow_skipped`` — prompts dropped because total token budget
    exceeded the judge's ``max_model_len``.
  * ``judge_wall_s`` — async batch wall time.
* ``degenerate_rollouts`` is now populated from real judge output (no longer
  always zero like the mock).

Judge input overflow policy (matches main/ Group A Phase 2)
----------------------------------------------------------
For each prompt we tokenize ``system + problem + 8 × rollout`` with the **judge
tokenizer** and check against ``judge_max_input_tokens`` (config:
``algorithm.minority_cot.judge_max_input_tokens``; default 36864 = 40960 −
4096 output reserve).

If the prompt overflows: every rollout receives ``DEGENERATE_CLUSTER_ID`` and
the prompt counts in ``judge_overflow_skipped``. The downstream marginal kernel
treats DEGENERATE the same as any other cluster ID (it cares about distinctness
within a prompt, not the specific value). With all rollouts marked DEGENERATE,
``len(set(cluster_ids[p])) == 1`` triggers ``keep_mask[p]=False`` → zero
advantage contribution for that prompt — the same fallback the mock-collapsed
case uses.

Why not truncate rollouts: truncation hides the tail of the CoT, which is
exactly where divergent reasoning shows up. Better to drop a rare overflow
prompt and log it than ship a clustering signal computed on chopped text.

Tokenizer plumbing
------------------
Two tokenizers are involved:

1. **Trainer tokenizer** (Qwen3-1.7B-Base for Stage 3b smoke): decodes
   ``data.batch["responses"]`` (token IDs) → strings before sending to judge.
   Path read from ``algorithm.minority_cot.tokenizer_path``.
2. **Judge tokenizer** (Qwen3-4B-Instruct-2507): counts tokens for the
   overflow check above. Loaded from ``algorithm.minority_cot.judge_model``.

Both tokenizers are loaded lazily and cached at module level — Modal containers
are long-lived, so the one-time HF cache hit per container start is acceptable.
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache
from typing import Any

import torch

from judge.client import JudgeClient, JudgeClientConfig
from judge.prompt import build_judge_messages
from judge.types import DEGENERATE_CLUSTER_ID, JudgeClusterResult, JudgeTask
from train.clusters_mock import ClusterAssignment


# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_tokenizer(model_path: str):
    """Lazy-load a HuggingFace tokenizer, cached per (model_path) within process."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _strip_left_pad(token_ids: list[int], pad_id: int | None) -> list[int]:
    """Drop leading pad tokens from verl's left-padded ``data.batch["prompts"]`` rows."""
    if pad_id is None:
        return token_ids
    i = 0
    n = len(token_ids)
    while i < n and token_ids[i] == pad_id:
        i += 1
    return token_ids[i:] if i < n else token_ids


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_judge_clusters(
    problem_ids: list[Any],
    n_rollouts: int,
    *,
    rollout_token_ids: list[list[list[int]]],
    prompt_token_ids: list[list[int]] | None = None,
    raw_prompts: list[str] | None = None,
    decoder_tokenizer_path: str,
    judge_tokenizer_path: str,
    judge_client: JudgeClient,
    judge_max_input_tokens: int,
) -> ClusterAssignment:
    """Real-judge cluster IDs for Stage 3b (and Stage 5).

    Parameters
    ----------
    problem_ids:
        Length ``n_prompts``. Used only for ``diagnostics`` ordering.
    n_rollouts:
        Rollouts per prompt (must equal ``N_ROLLOUTS=8`` for set arms).
    rollout_token_ids:
        Shape ``[n_prompts][n_rollouts][response_length]``. Token IDs from
        ``data.batch["responses"]`` — caller groups by prompt before passing.
    prompt_token_ids:
        Length ``n_prompts``, each a list of token IDs from
        ``data.batch["prompts"]``. Decoded with the trainer tokenizer to get
        the problem string sent to the judge. **Preferred Stage 3b path.**
    raw_prompts:
        Legacy path — pre-decoded prompt strings. Used only if
        ``prompt_token_ids`` is None. Kept for unit-test convenience.
    decoder_tokenizer_path:
        HF path of the trainer's tokenizer (e.g. ``Qwen/Qwen3-1.7B-Base``).
        Used to decode both rollout AND prompt token IDs back to strings.
    judge_tokenizer_path:
        HF path of the judge's tokenizer (e.g. ``Qwen/Qwen3-4B-Instruct-2507``).
        Used to count tokens for the overflow check.
    judge_client:
        Pre-constructed ``JudgeClient`` (caller owns lifecycle).
    judge_max_input_tokens:
        Per-call input budget. Prompts whose tokenized
        ``system + problem + 8 × rollout`` exceeds this are skipped (all
        rollouts → DEGENERATE_CLUSTER_ID).
    """
    n_prompts = len(problem_ids)
    if n_prompts == 0:
        return ClusterAssignment(
            cluster_ids=torch.zeros((0, n_rollouts), dtype=torch.int64),
            diagnostics={
                "distinct_clusters_mean": 0.0,
                "degenerate_rollouts": 0,
                "judge_parse_ok_rate": 0.0,
                "judge_overflow_skipped": 0,
                "judge_wall_s": 0.0,
                "judge_n_tasks": 0,
            },
        )
    if prompt_token_ids is None and raw_prompts is None:
        raise ValueError("must pass either prompt_token_ids or raw_prompts")
    if prompt_token_ids is not None and len(prompt_token_ids) != n_prompts:
        raise ValueError(
            f"prompt_token_ids length {len(prompt_token_ids)} != n_prompts {n_prompts}"
        )
    if raw_prompts is not None and len(raw_prompts) != n_prompts:
        raise ValueError(
            f"raw_prompts length {len(raw_prompts)} != n_prompts {n_prompts}"
        )
    if len(rollout_token_ids) != n_prompts:
        raise ValueError(
            f"rollout_token_ids length {len(rollout_token_ids)} != n_prompts {n_prompts}"
        )

    decoder_tok = _get_tokenizer(decoder_tokenizer_path)
    judge_tok = _get_tokenizer(judge_tokenizer_path)

    # Decode prompts to text if we got token IDs (Stage 3b path).
    pad_id = decoder_tok.pad_token_id
    if prompt_token_ids is not None:
        prompt_texts = [
            decoder_tok.decode(
                _strip_left_pad(pid, pad_id), skip_special_tokens=True
            )
            for pid in prompt_token_ids
        ]
    else:
        prompt_texts = list(raw_prompts)  # type: ignore[arg-type]

    # Step 1: decode rollouts to text, build JudgeTask per prompt, mark overflows.
    tasks: list[JudgeTask | None] = []
    overflow_skipped = 0
    for p in range(n_prompts):
        rollouts_p = [
            decoder_tok.decode(ids, skip_special_tokens=True)
            for ids in rollout_token_ids[p]
        ]
        if len(rollouts_p) != n_rollouts:
            raise ValueError(
                f"prompt {p}: expected {n_rollouts} rollouts, got {len(rollouts_p)}"
            )

        # Overflow check: tokenize the full judge prompt envelope.
        system, user = build_judge_messages(prompt_texts[p], rollouts_p)
        envelope_token_ct = (
            len(judge_tok.encode(system, add_special_tokens=False))
            + len(judge_tok.encode(user, add_special_tokens=False))
        )
        if envelope_token_ct > judge_max_input_tokens:
            overflow_skipped += 1
            tasks.append(None)
        else:
            tasks.append(JudgeTask(problem=prompt_texts[p], rollouts=rollouts_p))

    # Step 2: async batch the in-budget tasks to the judge.
    in_budget = [t for t in tasks if t is not None]
    t_start = time.monotonic()
    if in_budget:
        results_in_budget = judge_client.cluster_batch_sync(in_budget)
    else:
        results_in_budget = []
    wall_s = time.monotonic() - t_start

    # Step 3: stitch judge results back to prompt order; degenerate fills overflows.
    results: list[JudgeClusterResult | None] = []
    in_budget_iter = iter(results_in_budget)
    for t in tasks:
        if t is None:
            results.append(None)
        else:
            results.append(next(in_budget_iter))

    # Step 4: build cluster_ids tensor.
    cluster_ids = torch.zeros((n_prompts, n_rollouts), dtype=torch.int64)
    parse_ok_count = 0
    degenerate_rollout_count = 0
    distinct_counts: list[int] = []

    for p, res in enumerate(results):
        if res is None or not res.parse_ok:
            # Overflow OR judge parse failure → mark all rollouts degenerate.
            # Downstream: keep_mask[p]=False (collapsed single-cluster case).
            cluster_ids[p, :] = DEGENERATE_CLUSTER_ID
            degenerate_rollout_count += n_rollouts
            distinct_counts.append(1)
        else:
            parse_ok_count += 1
            for r in range(n_rollouts):
                cid = res.assignment.get(r, DEGENERATE_CLUSTER_ID)
                cluster_ids[p, r] = cid
                if cid == DEGENERATE_CLUSTER_ID:
                    degenerate_rollout_count += 1
            distinct_counts.append(len(set(cluster_ids[p].tolist())))

    in_budget_count = sum(1 for t in tasks if t is not None)
    diagnostics = {
        "distinct_clusters_mean": (
            float(sum(distinct_counts) / max(n_prompts, 1))
        ),
        "degenerate_rollouts": degenerate_rollout_count,
        "judge_parse_ok_rate": (
            parse_ok_count / max(in_budget_count, 1)
        ),
        "judge_overflow_skipped": overflow_skipped,
        "judge_wall_s": wall_s,
        "judge_n_tasks": in_budget_count,
    }
    print(
        "[clusters_judge] "
        f"n_tasks={diagnostics['judge_n_tasks']} "
        f"parse_ok_rate={diagnostics['judge_parse_ok_rate']:.3f} "
        f"distinct_clusters_mean={diagnostics['distinct_clusters_mean']:.3f} "
        f"overflow_skipped={diagnostics['judge_overflow_skipped']} "
        f"degenerate_rollouts={diagnostics['degenerate_rollouts']} "
        f"wall_s={diagnostics['judge_wall_s']:.1f}",
        flush=True,
    )
    return ClusterAssignment(cluster_ids=cluster_ids, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Convenience constructor for the registered hook
# ---------------------------------------------------------------------------

_cached_judge_client: JudgeClient | None = None
_cached_judge_model: str | None = None


def build_judge_client_from_env(
    *,
    judge_model: str,
    concurrency: int | None = None,
) -> JudgeClient:
    """Construct JudgeClient from JUDGE_BASE_URL env + minority_cot config.

    Concurrency is always clamped to JUDGE_CONCURRENCY_CAP=8 inside JudgeClient
    (per Stage 4 S4.6 finding); ``concurrency`` here lets a caller request a
    lower value without bypassing the cap.

    Reuses one client per process (Modal containers are long-lived) to avoid
    reconstructing httpx pools on every training step.
    """
    global _cached_judge_client, _cached_judge_model

    from judge.client import JUDGE_CONCURRENCY_CAP

    if _cached_judge_client is not None and _cached_judge_model == judge_model:
        return _cached_judge_client

    base_url = os.environ.get("JUDGE_BASE_URL", "")
    if not base_url:
        raise RuntimeError(
            "JUDGE_BASE_URL env var is required for cluster_source=judge. "
            "Set it to the deployed judge chat-completions URL (Stage 4 deploy)."
        )
    final_concurrency = min(concurrency or JUDGE_CONCURRENCY_CAP, JUDGE_CONCURRENCY_CAP)
    _cached_judge_client = JudgeClient(
        JudgeClientConfig(
            base_url=base_url,
            auth_token=os.environ.get("JUDGE_AUTH_TOKEN"),
            model=judge_model,
            concurrency=final_concurrency,
        )
    )
    _cached_judge_model = judge_model
    return _cached_judge_client
