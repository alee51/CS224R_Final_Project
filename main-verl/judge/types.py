"""Structured types for the Stage 4 judge service."""

from __future__ import annotations

from dataclasses import dataclass

DEGENERATE_CLUSTER_ID = -1
POLY_EPO_DEGENERATE_RAW = 100


@dataclass(frozen=True)
class JudgeTask:
    """One judge call: problem text + n rollout completions."""

    problem: str
    rollouts: list[str]
    problem_id: int | None = None


@dataclass(frozen=True)
class JudgeClusterResult:
    """One prompt × n_rollouts judge output (pre-tensor)."""

    assignment: dict[int, int]
    clusters: list[dict]
    parse_ok: bool
    raw_response: str | None = None

    @property
    def degenerate_count(self) -> int:
        return sum(1 for cid in self.assignment.values() if cid == DEGENERATE_CLUSTER_ID)
