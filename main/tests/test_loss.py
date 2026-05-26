"""Unit tests for GRPO clipped loss (no GPU)."""

import torch

from train.loss import grpo_loss
from train.objective import compute_advantages


def test_grpo_loss_shape_scalar():
    b, t = 4, 16
    new_lp = torch.zeros(b, t)
    old_lp = torch.zeros(b, t)
    adv = torch.ones(b)
    mask = torch.ones(b, t)
    keep = torch.ones(b, dtype=torch.bool)
    out = grpo_loss(new_lp, old_lp, adv, mask, keep)
    assert out.ndim == 0
    assert abs(out.item() - (-1.0)) < 1e-5


def test_grpo_loss_clip_positive_adv():
    """Large ratio + positive adv → clipped at 1 + clip_high."""
    b, t = 2, 4
    new_lp = torch.full((b, t), 5.0)
    old_lp = torch.zeros(b, t)
    adv = torch.ones(b)
    mask = torch.ones(b, t)
    keep = torch.ones(b, dtype=torch.bool)
    out = grpo_loss(
        new_lp, old_lp, adv, mask, keep, clip_low=0.20, clip_high=0.28
    )
    expected = -(1.0 + 0.28) * 1.0
    assert abs(out.item() - expected) < 1e-4


def test_grpo_loss_length_norm_differs():
    new_lp = torch.zeros(2, 4)
    old_lp = torch.zeros(2, 4)
    adv = torch.ones(2)
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    keep = torch.ones(2, dtype=torch.bool)
    per_seq = grpo_loss(
        new_lp, old_lp, adv, mask, keep, length_norm="per_seq"
    ).item()
    batch_max = grpo_loss(
        new_lp, old_lp, adv, mask, keep, length_norm="batch_max"
    ).item()
    assert per_seq != batch_max


def test_grpo_loss_keep_mask_zeros_row():
    new_lp = torch.zeros(2, 4)
    old_lp = torch.zeros(2, 4)
    adv = torch.tensor([1.0, 10.0])
    mask = torch.ones(2, 4)
    keep = torch.tensor([True, False])
    out_on = grpo_loss(new_lp, old_lp, adv, mask, keep).item()
    out_first = grpo_loss(
        new_lp[:1], old_lp[:1], adv[:1], mask[:1], torch.tensor([True])
    ).item()
    assert abs(out_on - out_first) < 1e-5


def test_grpo_advantages_hand_computed():
    rewards = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    out = compute_advantages("grpo", rewards)
    expected_0 = rewards[0] - rewards[0].mean()
    assert torch.allclose(out.advantages[0], expected_0)
    assert not out.keep_mask[1].item()
    assert out.keep_mask[0].item()


def test_grpo_advantages_all_zero_variance_filters_everything():
    """All-uniform groups → keep_mask all False; trainer's RuntimeError fires when this
    holds for every prompt in a batch. Verifying the precondition gates that branch."""
    rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    out = compute_advantages("grpo", rewards)
    assert torch.all(out.advantages == 0.0)
    assert not out.keep_mask.any().item()
    assert out.diagnostics["fraction_filtered"] == 1.0
    assert out.diagnostics["n_filtered_prompts"] == 3
