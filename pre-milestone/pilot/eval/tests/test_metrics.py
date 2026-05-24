"""Metric invariants and fixture regression."""

import json
from pathlib import Path

from pilot.eval.metrics import PromptRollouts, aggregate_metrics, pass_at_k

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hand_computed.json"


def _load_fixture_prompts() -> list[PromptRollouts]:
    # Embedded 10-prompt synthetic set matching hand_computed.json
    rows = [
        ([True, False, False, False, False, False, False, False], [0, 1, 1, 1, 1, 1, 1, 1]),
        ([False] * 8, [0] * 8),
        ([True, True, False, False, False, False, False, False], [0, 0, 1, 1, 1, 1, 1, 1]),
        ([True, False, True, False, False, False, False, False], [0, 1, 0, 1, 1, 1, 1, 1]),
        ([False, False, True, True, False, False, False, False], [0, 0, 0, 0, 1, 1, 1, 1]),
        ([True] * 8, [0] * 8),
        ([False, True, False, False, False, False, False, False], [0, 0, 1, 1, 1, 1, 1, 1]),
        ([True, False, False, True, False, False, False, False], [0, 1, 2, 0, 1, 1, 1, 1]),
        ([False] * 8, [0] * 8),
        ([True, True, True, False, False, False, False, False], [0, 0, 0, 1, 1, 1, 1, 1]),
    ]
    return [
        PromptRollouts(prompt_id=f"p{i}", correct=r[0], cluster_ids=r[1])
        for i, r in enumerate(rows)
    ]


def test_pass_at_k_known():
    assert pass_at_k(8, 1, 8) > 0.99
    assert pass_at_k(8, 0, 8) == 0.0


def test_fixture_matches_hand_computed():
    meta = json.loads(FIXTURE.read_text())
    prompts = _load_fixture_prompts()
    got = aggregate_metrics(
        prompts,
        k=meta["k"],
        tau=meta["tau"],
        worst_q=meta["worst_quantile"],
    )
    exp = meta["expected_aggregate"]
    for key in ("pass_at_1", "cover_at_tau", "worst_subset_accuracy", "n_prompts"):
        assert abs(got[key] - exp[key]) < 1e-6, f"{key}: {got[key]} != {exp[key]}"
    assert abs(got[f"pass_at_{meta['k']}"] - exp["pass_at_8"]) < 1e-6


def test_bootstrap_schema_deterministic():
    from pilot.eval.bootstrap import bootstrap_ci

    prompts = _load_fixture_prompts()
    a = bootstrap_ci(prompts, "pass_at_1", n_samples=500, seed=42)
    b = bootstrap_ci(prompts, "pass_at_1", n_samples=500, seed=42)
    assert a == b
    assert set(a) == {"point", "ci_low", "ci_high", "std"}
