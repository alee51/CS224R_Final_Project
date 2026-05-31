"""Unit tests for train.clusters_judge.

These run locally without a Modal container. The judge HTTP call is mocked via
a fake ``JudgeClient`` so we test the routing + decode + overflow logic without
network. Tokenizer is loaded from HF cache if available; tests skip otherwise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

try:
    from transformers import AutoTokenizer  # noqa: F401
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


pytestmark = pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="transformers is not installed in this venv; clusters_judge tests need a tokenizer",
)


from judge.types import DEGENERATE_CLUSTER_ID, JudgeClusterResult, JudgeTask
from train.clusters_judge import assign_judge_clusters
from train.judge_trace import trace_prompt_index
from train.clusters_mock import ClusterAssignment


# ---------------------------------------------------------------------------
# Fake judge client — records calls + returns canned responses.
# ---------------------------------------------------------------------------

class FakeJudgeClient:
    def __init__(self, results: list[JudgeClusterResult]):
        self._results = list(results)
        self.calls: list[JudgeTask] = []

    def cluster_batch_sync(self, tasks: list[JudgeTask]) -> list[JudgeClusterResult]:
        self.calls.extend(tasks)
        # Return the next N pre-canned results; pad with parse failures if short.
        out: list[JudgeClusterResult] = []
        for _ in tasks:
            if self._results:
                out.append(self._results.pop(0))
            else:
                out.append(
                    JudgeClusterResult(
                        assignment={}, clusters=[], parse_ok=False, raw_response=None
                    )
                )
        return out


def _two_cluster_result() -> JudgeClusterResult:
    """4 rollouts in cluster 1, 4 in cluster 2 — a clean split."""
    return JudgeClusterResult(
        assignment={0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2},
        clusters=[
            {"cluster_id": 1, "member_rollouts": [0, 1, 2, 3], "reasoning_signature": "A"},
            {"cluster_id": 2, "member_rollouts": [4, 5, 6, 7], "reasoning_signature": "B"},
        ],
        parse_ok=True,
        raw_response="(canned)",
    )


def _parse_fail_result() -> JudgeClusterResult:
    return JudgeClusterResult(assignment={}, clusters=[], parse_ok=False, raw_response="bad json")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _token_ids_for(text: str, tokenizer_path: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return tok.encode(text, add_special_tokens=False)


_TINY_TOK = "sshleifer/tiny-gpt2"  # ~5MB; fast download for CI


def test_clusters_judge_contract_matches_mock():
    """ClusterAssignment shape + diagnostics keys match clusters_mock."""
    fake = FakeJudgeClient([_two_cluster_result()])
    rollout_ids = [[_token_ids_for(f"r{r}", _TINY_TOK) for r in range(8)]]
    out = assign_judge_clusters(
        problem_ids=["p0"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        raw_prompts=["what is 2+2?"],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=100_000,
    )
    assert isinstance(out, ClusterAssignment)
    assert out.cluster_ids.shape == (1, 8)
    assert out.cluster_ids.dtype == torch.int64
    # Diagnostics must include both clusters_mock keys + new judge keys.
    for key in (
        "distinct_clusters_mean", "degenerate_rollouts",
        "judge_parse_ok_rate", "judge_overflow_skipped", "judge_wall_s",
        "judge_n_tasks",
    ):
        assert key in out.diagnostics, f"missing diagnostics key: {key}"


def test_clusters_judge_happy_path_assignment():
    """Cluster IDs from the judge are passed through verbatim."""
    fake = FakeJudgeClient([_two_cluster_result()])
    rollout_ids = [[_token_ids_for(f"r{r}", _TINY_TOK) for r in range(8)]]
    out = assign_judge_clusters(
        problem_ids=["p0"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        raw_prompts=["what is 2+2?"],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=100_000,
    )
    assert out.cluster_ids[0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert out.diagnostics["judge_parse_ok_rate"] == 1.0
    assert out.diagnostics["judge_overflow_skipped"] == 0
    assert out.diagnostics["degenerate_rollouts"] == 0
    assert out.diagnostics["distinct_clusters_mean"] == 2.0


def test_clusters_judge_parse_failure_becomes_degenerate():
    fake = FakeJudgeClient([_parse_fail_result()])
    rollout_ids = [[_token_ids_for(f"r{r}", _TINY_TOK) for r in range(8)]]
    out = assign_judge_clusters(
        problem_ids=["p0"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        raw_prompts=["short prompt"],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=100_000,
    )
    assert (out.cluster_ids[0] == DEGENERATE_CLUSTER_ID).all()
    assert out.diagnostics["degenerate_rollouts"] == 8
    assert out.diagnostics["judge_parse_ok_rate"] == 0.0
    # Single-cluster prompt collapses to keep_mask=False downstream.
    assert out.diagnostics["distinct_clusters_mean"] == 1.0


def test_clusters_judge_input_overflow_skips_judge_call():
    """Prompts over the judge input cap never reach the judge — degenerate fallback."""
    fake = FakeJudgeClient([_two_cluster_result()])
    rollout_ids = [[_token_ids_for(f"r{r}", _TINY_TOK) for r in range(8)]]
    # tiny budget forces overflow even on short rollouts.
    out = assign_judge_clusters(
        problem_ids=["p0"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        raw_prompts=["q"],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=1,  # smaller than even the system template
    )
    assert fake.calls == [], "judge should not be called for overflow prompts"
    assert (out.cluster_ids[0] == DEGENERATE_CLUSTER_ID).all()
    assert out.diagnostics["judge_overflow_skipped"] == 1
    assert out.diagnostics["judge_n_tasks"] == 0
    assert out.diagnostics["degenerate_rollouts"] == 8


def test_clusters_judge_mixed_batch_partial_overflow():
    """One overflow prompt + one in-budget prompt: judge sees only the second."""
    fake = FakeJudgeClient([_two_cluster_result()])
    rollout_ids = [
        [_token_ids_for(f"long" * 1000, _TINY_TOK) for _ in range(8)],  # overflow
        [_token_ids_for("ok", _TINY_TOK) for _ in range(8)],            # in budget
    ]
    out = assign_judge_clusters(
        problem_ids=["overflow_pid", "ok_pid"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        raw_prompts=["q1", "q2"],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        # System template alone is ~520 tokens; budget=800 leaves room for "ok"x8
        # (~567 total) but not for "long"x1000 x 8 rollouts (~8000+ tokens).
        judge_max_input_tokens=800,
    )
    assert len(fake.calls) == 1, "judge should be called once (the in-budget prompt)"
    assert (out.cluster_ids[0] == DEGENERATE_CLUSTER_ID).all()
    assert out.cluster_ids[1].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert out.diagnostics["judge_overflow_skipped"] == 1
    assert out.diagnostics["judge_n_tasks"] == 1
    assert out.diagnostics["judge_parse_ok_rate"] == 1.0


def test_clusters_judge_empty_batch_returns_empty_assignment():
    fake = FakeJudgeClient([])
    out = assign_judge_clusters(
        problem_ids=[],
        n_rollouts=8,
        rollout_token_ids=[],
        raw_prompts=[],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=100_000,
    )
    assert out.cluster_ids.shape == (0, 8)
    assert out.diagnostics["judge_n_tasks"] == 0
    assert fake.calls == []


def test_clusters_judge_prompt_token_ids_path():
    """Stage 3b production path: decode prompt token IDs (not raw_prompt strings)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_TINY_TOK)
    prompt_text = "what is 2+2?"
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    fake = FakeJudgeClient([_two_cluster_result()])
    rollout_ids = [[_token_ids_for(f"r{r}", _TINY_TOK) for r in range(8)]]
    out = assign_judge_clusters(
        problem_ids=["p0"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        prompt_token_ids=[prompt_ids],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=100_000,
    )
    assert fake.calls[0].problem == prompt_text
    assert out.diagnostics["judge_parse_ok_rate"] == 1.0


def test_clusters_judge_strips_left_pad_from_prompt_token_ids():
    """Left-padded prompt rows from verl must not prepend garbage to judge input."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_TINY_TOK)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_text = "solve for x"
    core_ids = tok.encode(prompt_text, add_special_tokens=False)
    padded_ids = [pad_id] * 5 + core_ids
    fake = FakeJudgeClient([_two_cluster_result()])
    rollout_ids = [[_token_ids_for("r", _TINY_TOK) for _ in range(8)]]
    out = assign_judge_clusters(
        problem_ids=["p0"],
        n_rollouts=8,
        rollout_token_ids=rollout_ids,
        prompt_token_ids=[padded_ids],
        decoder_tokenizer_path=_TINY_TOK,
        judge_tokenizer_path=_TINY_TOK,
        judge_client=fake,
        judge_max_input_tokens=100_000,
    )
    assert fake.calls[0].problem == prompt_text
    assert out.diagnostics["judge_parse_ok_rate"] == 1.0


def test_trace_prompt_index_env(monkeypatch):
    monkeypatch.delenv("CS224R_JUDGE_TRACE", raising=False)
    assert trace_prompt_index() is None
    monkeypatch.setenv("CS224R_JUDGE_TRACE", "1")
    monkeypatch.setenv("CS224R_JUDGE_TRACE_PROMPT_IDX", "3")
    assert trace_prompt_index() == 3
