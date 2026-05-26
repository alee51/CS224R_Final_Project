"""Tests for rollout helpers that decode vLLM completion outputs.

`_extract_old_logprobs` is the boundary between vLLM's logprob struct and the
trainer's advantage math. If vLLM bumps its API, silent decay to 0.0 logprobs
turns the entire GRPO advantage signal into noise — these tests gate that.
"""

from dataclasses import dataclass
from types import SimpleNamespace

from train.rollout import _extract_old_logprobs


@dataclass
class _FakeOut:
    token_ids: list[int]
    logprobs: list | None


def _entry(lp: float):
    """vLLM's per-token entry exposes `.logprob`; bare float also supported."""
    return SimpleNamespace(logprob=lp)


def test_extract_old_logprobs_object_entries():
    out = _FakeOut(
        token_ids=[10, 20, 30],
        logprobs=[{10: _entry(-0.1)}, {20: _entry(-0.5)}, {30: _entry(-1.0)}],
    )
    assert _extract_old_logprobs(out) == [-0.1, -0.5, -1.0]


def test_extract_old_logprobs_bare_float_fallback():
    out = _FakeOut(
        token_ids=[1, 2],
        logprobs=[{1: -0.25}, {2: -0.75}],
    )
    assert _extract_old_logprobs(out) == [-0.25, -0.75]


def test_extract_old_logprobs_none_per_token_entry():
    out = _FakeOut(
        token_ids=[7, 8, 9],
        logprobs=[{7: _entry(-0.3)}, None, {9: _entry(-0.4)}],
    )
    assert _extract_old_logprobs(out) == [-0.3, 0.0, -0.4]


def test_extract_old_logprobs_missing_chosen_token():
    out = _FakeOut(
        token_ids=[5, 6],
        logprobs=[{99: _entry(-0.2)}, {6: _entry(-0.6)}],
    )
    assert _extract_old_logprobs(out) == [0.0, -0.6]


def test_extract_old_logprobs_empty_logprobs_list():
    out = _FakeOut(token_ids=[1, 2, 3], logprobs=None)
    assert _extract_old_logprobs(out) == [0.0, 0.0, 0.0]

    out = _FakeOut(token_ids=[1, 2, 3], logprobs=[])
    assert _extract_old_logprobs(out) == [0.0, 0.0, 0.0]


def test_extract_old_logprobs_non_dict_lp_entry():
    out = _FakeOut(token_ids=[1, 2], logprobs=[_entry(-0.1), _entry(-0.2)])
    assert _extract_old_logprobs(out) == [0.0, 0.0]
