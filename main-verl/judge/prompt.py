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
