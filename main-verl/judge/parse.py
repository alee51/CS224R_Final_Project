"""Parse Poly-EPO judge JSON responses."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from judge.types import (
    DEGENERATE_CLUSTER_ID,
    POLY_EPO_DEGENERATE_RAW,
    JudgeClusterResult,
)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _normalize_cluster_id(cid: int) -> int:
    return DEGENERATE_CLUSTER_ID if cid == POLY_EPO_DEGENERATE_RAW else cid


def _assignment_from_poly_epo_payload(
    payload: dict, n_responses: int
) -> tuple[dict[int, int], list[dict]]:
    assignment: dict[int, int] = {}
    cot_by_idx: dict[int, str] = {}
    for key, val in payload.items():
        if not str(key).isdigit():
            continue
        rollout_1idx = int(key)
        if rollout_1idx < 1 or rollout_1idx > n_responses:
            raise ValueError(f"rollout key out of range: {key}")
        idx = rollout_1idx - 1
        if not isinstance(val, dict):
            raise ValueError(f"response {key} must be an object")
        cid = _normalize_cluster_id(int(val["cluster_id"]))
        assignment[idx] = cid
        cot_by_idx[idx] = str(val.get("chain_of_thought", ""))
    if set(assignment.keys()) != set(range(n_responses)):
        raise ValueError(
            f"Poly-EPO JSON must have keys 1-{n_responses}, got {sorted(assignment)}"
        )
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cid in assignment.items():
        by_cluster[cid].append(idx)
    clusters: list[dict] = []
    for cid, members in sorted(by_cluster.items()):
        macro_micro = "; ".join(cot_by_idx[m][:120] for m in sorted(members)[:2])
        clusters.append(
            {
                "cluster_id": cid,
                "member_rollouts": sorted(members),
                "reasoning_signature": macro_micro or f"cluster_{cid}",
            }
        )
    return assignment, clusters


def parse_judge_response(text: str, *, n_rollouts: int) -> JudgeClusterResult:
    """Parse judge model output into a structured cluster assignment."""
    raw = text
    try:
        payload: Any = json.loads(_strip_json_fences(text))
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON must be an object")
        assignment, clusters = _assignment_from_poly_epo_payload(payload, n_rollouts)
        return JudgeClusterResult(
            assignment=assignment,
            clusters=clusters,
            parse_ok=True,
            raw_response=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return JudgeClusterResult(
            assignment={},
            clusters=[],
            parse_ok=False,
            raw_response=raw,
        )
