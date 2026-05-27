"""Trainer behavior when every prompt is filtered (critical-pass 6b)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from train.rollout import RolloutResult
from train.trainer import StepBatch, run_one_grpo_step, train_cfg_from_dict


def _minimal_cfg(*, arm: str = "grpo", n_rollouts: int = 4) -> object:
    return train_cfg_from_dict(
        {
            "global_seed": 0,
            "arm": arm,
            "train": {
                "n_rollouts": n_rollouts,
                "microbatch": 2,
            },
            "rollout": {"model": "test/no-vllm"},
            "loss": {"clip_low": 0.2, "clip_high": 0.28},
            "weight_sync": {"every_n_steps": 0},
            "wandb": {"entity": "e", "project": "p"},
            "clustering": {"sympy_mode": "off"},
        }
    )


def _fake_rollouts(
    n_prompts: int,
    n_rollouts: int,
    *,
    completion_text: str = "work\nAnswer: 41",
) -> list[RolloutResult]:
    out: list[RolloutResult] = []
    for p_idx in range(n_prompts):
        for r_idx in range(n_rollouts):
            out.append(
                RolloutResult(
                    prompt_idx=p_idx,
                    rollout_idx=r_idx,
                    completion_text=completion_text,
                    completion_ids=[1, 2, 3],
                    prompt_ids=[10, 11],
                    old_logprobs=[-0.1, -0.2, -0.3],
                    finish_reason="stop",
                )
            )
    return out


def _mock_engine(rollouts: list[RolloutResult]) -> MagicMock:
    engine = MagicMock()
    engine.generate.return_value = rollouts
    return engine


def test_run_one_grpo_step_all_filtered_skips_without_train():
    """Uniform wrong answers → zero reward variance → skip step, no backward."""
    n_prompts, n_rollouts = 2, 4
    cfg = _minimal_cfg()
    batch = StepBatch(
        prompts=["What is 40+2?", "Compute 21+21."],
        golds=["42", "42"],
        problem_ids=[0, 1],
    )
    rollouts = _fake_rollouts(n_prompts, n_rollouts)
    engine = _mock_engine(rollouts)
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = run_one_grpo_step(cfg, engine, model, opt, batch, step=0)

    assert result.skipped is True
    assert result.n_kept_sequences == 0
    assert result.fraction_filtered == 1.0
    assert result.train_wandb.get("train/skipped_no_kept") == 1.0
    engine.generate.assert_called_once()
    engine.sleep_for_train.assert_not_called()
    engine.update_weights.assert_not_called()


def test_run_one_grpo_step_minority_collapsed_cluster_skips():
    """Set arm: one answer cluster per prompt → skip step (no train/sync)."""
    n_prompts, n_rollouts = 1, 8
    cfg = _minimal_cfg(arm="minority_answer", n_rollouts=n_rollouts)
    batch = StepBatch(
        prompts=["What is 40+2?"],
        golds=["42"],
        problem_ids=[7],
    )
    rollouts = _fake_rollouts(n_prompts, n_rollouts, completion_text="step\nAnswer: 41")
    engine = _mock_engine(rollouts)
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = run_one_grpo_step(cfg, engine, model, opt, batch, step=0)

    assert result.skipped is True
    assert result.n_kept_sequences == 0
    engine.sleep_for_train.assert_not_called()
