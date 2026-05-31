"""Unit tests for judge.client batch HTTP helpers."""

from __future__ import annotations

import pytest

from judge.client import JudgeClient, JudgeClientConfig
from judge.types import JudgeClusterResult, JudgeTask


def _task(problem: str = "p", n: int = 8) -> JudgeTask:
    return JudgeTask(problem=problem, rollouts=[f"r{i}" for i in range(n)])


def test_chunk_tasks():
    tasks = [_task(str(i)) for i in range(10)]
    chunks = JudgeClient._chunk_tasks(tasks, 4)
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [4, 4, 2]


def test_parse_batch_payload_happy():
    tasks = [_task("a"), _task("b")]
    payload = {
        "results": [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"1":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"2":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"3":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"4":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"5":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"6":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"7":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."},'
                                '"8":{"cluster_id":1,"chain_of_thought":"Macro: a. Micro: b."}}'
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"1":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"2":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"3":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"4":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"5":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"6":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"7":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."},'
                                '"8":{"cluster_id":2,"chain_of_thought":"Macro: c. Micro: d."}}'
                            )
                        }
                    }
                ]
            },
        ]
    }
    results = JudgeClient._parse_batch_payload(payload, tasks)
    assert len(results) == 2
    assert all(r.parse_ok for r in results)
    assert results[0].assignment[0] == 1
    assert results[1].assignment[0] == 2


def test_parse_batch_payload_length_mismatch():
    with pytest.raises(ValueError, match="batch results length"):
        JudgeClient._parse_batch_payload({"results": []}, [_task()])


def test_cluster_batch_sync_preserves_order(monkeypatch):
    tasks = [_task(str(i)) for i in range(5)]
    seen_batches: list[int] = []

    async def fake_post_http_batch(_self, _client, _sem, batch):
        seen_batches.append(len(batch))
        return [
            JudgeClusterResult(assignment={0: 1}, clusters=[], parse_ok=True, raw_response="ok")
            for _ in batch
        ]

    monkeypatch.setattr(JudgeClient, "_post_http_batch", fake_post_http_batch)

    client = JudgeClient(
        JudgeClientConfig(
            base_url="http://example.test/v1",
            auth_token=None,
            model="test-model",
            http_batch_size=2,
            concurrency=2,
        )
    )
    results = client.cluster_batch_sync(tasks)
    assert len(results) == 5
    assert all(r.parse_ok for r in results)
    assert seen_batches == [2, 2, 1]
