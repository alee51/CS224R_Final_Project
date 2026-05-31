"""Poly-EPO judge prompt construction for Stage 4."""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_MD = Path(__file__).resolve().parent / "prompts" / "poly_epo_a1.md"


def _load_prompt_templates() -> tuple[str, str]:
    if not PROMPT_MD.is_file():
        raise FileNotFoundError(f"missing prompt template: {PROMPT_MD}")

    text = PROMPT_MD.read_text()
    system_m = re.search(r"## System\s*\n+(.*?)(?=\n## User|\Z)", text, re.DOTALL)
    user_m = re.search(r"## User\s*\n+(.*?)\Z", text, re.DOTALL)
    system = (system_m.group(1).strip() if system_m else "").strip()
    user = (user_m.group(1).strip() if user_m else "").strip()
    if not system or not user:
        raise ValueError(f"invalid prompt template sections in {PROMPT_MD}")
    return system, user


def _build_responses_block(rollouts: list[str]) -> str:
    blocks: list[str] = []
    for idx, completion in enumerate(rollouts):
        blocks.append(f"{idx + 1}. {completion}")
    return "\n".join(blocks)


def build_judge_messages(problem: str, rollouts: list[str]) -> tuple[str, str]:
    """Return (system, user) strings for the cluster-assignment judge."""
    n_responses = len(rollouts)
    system_tpl, user_tpl = _load_prompt_templates()
    system = system_tpl.replace("{n_responses}", str(n_responses))
    user = user_tpl.replace("{problem}", problem).replace(
        "{responses_block}", _build_responses_block(rollouts)
    )
    return system, user


def build_poly_epo_schema(n_responses: int) -> dict:
    """JSON schema for vLLM guided decoding — pins the Poly-EPO output shape.

    Eliminates the three observed parse-failure modes:
      (A) stray quote after ``cluster_id: 100``,
      (B) unescaped LaTeX backslashes inside ``chain_of_thought`` strings,
      (C) judge emitting ``{"error": "..."}`` instead of cluster keys.
    """
    response_schema = {
        "type": "object",
        "properties": {
            "chain_of_thought": {"type": "string"},
            "cluster_id": {"type": "integer", "minimum": 0},
        },
        "required": ["chain_of_thought", "cluster_id"],
        "additionalProperties": False,
    }
    keys = [str(i) for i in range(1, n_responses + 1)]
    return {
        "type": "object",
        "properties": {k: response_schema for k in keys},
        "required": keys,
        "additionalProperties": False,
    }
