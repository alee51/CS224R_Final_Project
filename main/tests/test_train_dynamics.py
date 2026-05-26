"""Training-step wandb aggregates (Group C / Poly-EPO Fig. 2 right)."""

import pytest

from train.trainer import aggregate_train_step_wandb_metrics


def test_aggregate_correct_count_histogram():
    rewards = [
        [0.0] * 8,
        [1.0] * 8,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    meta = [[{"parse_ok": True, "extract_path": "hybrid"} for _ in range(8)] for _ in range(3)]
    m = aggregate_train_step_wandb_metrics(rewards, meta, [])
    assert m["train/frac_prompts_0_correct"] == 1 / 3
    assert m["train/frac_prompts_8_correct"] == 1 / 3
    assert m["train/frac_prompts_1_correct"] == 1 / 3
    assert sum(m[f"train/frac_prompts_{k}_correct"] for k in range(9)) == pytest.approx(1.0)


def test_aggregate_prompt_coverage_and_mixed_rate():
    rewards = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
    meta = [
        [{"parse_ok": True, "extract_path": "hybrid"} for _ in range(4)],
        [{"parse_ok": False, "extract_path": "none"} for _ in range(4)],
        [{"parse_ok": True, "extract_path": "boxed"} for _ in range(4)],
    ]
    m = aggregate_train_step_wandb_metrics(rewards, meta, [10, 20, 30, 40] * 3)
    assert m["train/prompt_coverage"] == 2 / 3
    assert m["train/mixed_reward_rate"] == 2 / 3


def test_aggregate_parse_and_extract_path_fractions():
    rewards = [[1.0, 0.0]]
    meta = [[{"parse_ok": True, "extract_path": "hybrid"}, {"parse_ok": False, "extract_path": "none"}]]
    m = aggregate_train_step_wandb_metrics(rewards, meta, [5, 15])
    assert m["train/parse_ok_rate"] == 0.5
    assert m["train/extract_path_hybrid"] == 0.5
    assert m["train/extract_path_none"] == 0.5


def test_aggregate_completion_token_stats():
    rewards = [[0.0, 1.0]]
    meta = [[{"parse_ok": True, "extract_path": "hybrid"}] * 2]
    lens = [100, 200]
    m = aggregate_train_step_wandb_metrics(rewards, meta, lens)
    assert m["train/mean_completion_tokens"] == 150.0
    assert m["train/p95_completion_tokens"] == 100.0
