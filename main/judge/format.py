"""Judge message formatting — ported from analysis_a_llm_clusters.py."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROMPT_MD = Path(__file__).resolve().parent / "poly_epo_a1.md"
POLY_EPO_DEGENERATE_CLUSTER = 100


def _load_prompt_templates() -> tuple[str, str]:
    """Return (system_template, user_template) from Poly-EPO prompt markdown."""
    if not PROMPT_MD.is_file():
        raise FileNotFoundError(f"missing prompt template: {PROMPT_MD}")

    text = PROMPT_MD.read_text()
    system_m = re.search(
        r"## System\s*\n+(.*?)(?=\n## User|\Z)",
        text,
        re.DOTALL,
    )
    user_m = re.search(r"## User\s*\n+(.*?)\Z", text, re.DOTALL)
    system = (system_m.group(1).strip() if system_m else "").strip()
    user = (user_m.group(1).strip() if user_m else "").strip()
    if not system or not user:
        raise ValueError(f"invalid prompt template sections in {PROMPT_MD}")
    return system, user


def _build_responses_block(rollouts: list[dict]) -> str:
    """Poly-EPO instance format: numbered responses 1..N (1-indexed)."""
    blocks: list[str] = []
    for idx, r in enumerate(rollouts):
        n = idx + 1
        completion = r.get("completion", "")
        blocks.append(f"{n}. {completion}")
    return "\n".join(blocks)


def build_judge_messages(problem: str, rollouts: list[dict]) -> tuple[str, str]:
    n_responses = len(rollouts)
    system_tpl, user_tpl = _load_prompt_templates()
    system = system_tpl.replace("{n_responses}", str(n_responses))
    user = user_tpl.replace("{problem}", problem).replace(
        "{responses_block}", _build_responses_block(rollouts)
    )
    return system, user


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _normalize_cluster_id(cid: int) -> int:
    """Paper uses 100 for degenerate; downstream metrics use -1."""
    return -1 if cid == POLY_EPO_DEGENERATE_CLUSTER else cid


def _normalize_cluster_assignment(
    raw: Any, n_responses: int
) -> dict[int, int]:
    if not isinstance(raw, dict):
        raise ValueError("cluster_assignment must be an object")
    out: dict[int, int] = {}
    for k, v in raw.items():
        idx = int(k)
        if idx < 0 or idx >= n_responses:
            raise ValueError(f"rollout index out of range: {idx}")
        out[idx] = _normalize_cluster_id(int(v))
    if set(out.keys()) != set(range(n_responses)):
        raise ValueError(
            f"cluster_assignment must cover 0-{n_responses - 1}, got {sorted(out)}"
        )
    return out


def _normalize_clusters(
    raw: Any, assignment: dict[int, int]
) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("clusters must be an array")
    clusters: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each cluster must be an object")
        cid = int(item["cluster_id"])
        members_raw = item.get("member_rollouts") or item.get("members") or item.get(
            "rollouts"
        )
        if members_raw is not None:
            members = [int(x) for x in members_raw]
        else:
            members = [i for i, a in assignment.items() if a == cid]
        sig = str(
            item.get("reasoning_signature")
            or item.get("signature")
            or item.get("description")
            or ""
        )
        clusters.append(
            {
                "cluster_id": cid,
                "member_rollouts": members,
                "reasoning_signature": sig,
            }
        )
    return clusters


def _assignment_from_poly_epo_payload(
    payload: dict, n_responses: int
) -> tuple[dict[int, int], list[dict]]:
    """Paper §A.1 format: keys \"1\"..\"N\" with chain_of_thought + cluster_id."""
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
