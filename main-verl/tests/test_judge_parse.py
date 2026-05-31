"""Unit tests for Stage 4 judge prompt + parser (local, no GPU)."""

from __future__ import annotations

import json

import pytest

from judge.parse import parse_judge_response
from judge.prompt import build_judge_messages
from judge.types import DEGENERATE_CLUSTER_ID


def _valid_payload(n: int = 8) -> dict:
    payload = {}
    for i in range(1, n + 1):
        payload[str(i)] = {
            "chain_of_thought": f"Macro: m{i}. Micro: u{i}.",
            "cluster_id": 0 if i <= 4 else 1,
        }
    return payload


def test_build_judge_messages_eight_rollouts():
    problem = "Find the sum of 1+1."
    rollouts = [f"Solution {i}" for i in range(8)]
    system, user = build_judge_messages(problem, rollouts)
    assert "8" in system
    assert problem in user
    assert "1. Solution 0" in user
    assert "8. Solution 7" in user


def test_parse_valid_poly_epo_json():
    text = json.dumps(_valid_payload())
    result = parse_judge_response(text, n_rollouts=8)
    assert result.parse_ok
    assert result.assignment == {i: (0 if i < 4 else 1) for i in range(8)}
    assert len(result.clusters) == 2


def test_parse_one_indexed_keys():
    payload = _valid_payload(8)
    text = json.dumps(payload)
    result = parse_judge_response(text, n_rollouts=8)
    assert set(result.assignment.keys()) == set(range(8))


def test_parse_degenerate_cluster_100():
    payload = _valid_payload()
    payload["3"]["cluster_id"] = 100
    result = parse_judge_response(json.dumps(payload), n_rollouts=8)
    assert result.parse_ok
    assert result.assignment[2] == DEGENERATE_CLUSTER_ID
    assert result.degenerate_count == 1


def test_parse_markdown_json_fences():
    inner = json.dumps(_valid_payload())
    text = f"```json\n{inner}\n```"
    result = parse_judge_response(text, n_rollouts=8)
    assert result.parse_ok


def test_parse_missing_rollout_key():
    payload = _valid_payload()
    del payload["8"]
    result = parse_judge_response(json.dumps(payload), n_rollouts=8)
    assert not result.parse_ok
    assert result.assignment == {}


def test_parse_invalid_json():
    result = parse_judge_response("not json at all", n_rollouts=8)
    assert not result.parse_ok
