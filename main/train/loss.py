"""PPO-clipped GRPO surrogate with per-arm length normalization."""

from __future__ import annotations

import torch


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    keep_mask: torch.Tensor,
    clip_low: float = 0.20,
    clip_high: float = 0.28,
    length_norm: str = "per_seq",
    return_stats: bool = False,
):
    """
    REINFORCE-with-asymmetric clip (DAPO ε_low/ε_high).

    Reduction follows Poly-EPO / Dr.GRPO:
      per_seq:   mean over sequences of (sum_t loss_t * mask_t) / |y_i|
      batch_max: mean over sequences of (sum_t loss_t * mask_t) / T_max

    keep_mask False rows contribute zero before the final mean.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = (
        torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
    )
    per_token_loss = -torch.min(surr1, surr2)

    masked = per_token_loss * mask
    seq_sums = masked.sum(dim=1)
    seq_lens = mask.sum(dim=1).clamp(min=1)

    if length_norm == "per_seq":
        per_seq = seq_sums / seq_lens
    elif length_norm == "batch_max":
        t_max = seq_lens.max().clamp(min=1).float()
        per_seq = seq_sums / t_max
    else:
        raise ValueError(f"unknown length_norm: {length_norm!r}")

    per_seq = per_seq * keep_mask.float()
    denom = keep_mask.float().sum().clamp(min=1)
    loss = per_seq.sum() / denom

    if not return_stats:
        return loss

    # Detach for stats so monitoring tensors never enter the autograd graph.
    with torch.no_grad():
        r = ratio.detach()
        m = mask.detach()
        mask_sum = m.sum().clamp(min=1)
        sel = m > 0
        if sel.any():
            r_sel = r[sel]
            ratio_max = float(r_sel.max().item())
            ratio_p95 = float(r_sel.quantile(0.95).item())
        else:
            ratio_max = 0.0
            ratio_p95 = 0.0
        stats = {
            "ratio_mean": float(((r * m).sum() / mask_sum).item()),
            "ratio_max": ratio_max,
            "ratio_p95": ratio_p95,
            "clipped_low_frac": float(
                (((r < 1.0 - clip_low) & sel).float().sum() / mask_sum).item()
            ),
            "clipped_high_frac": float(
                (((r > 1.0 + clip_high) & sel).float().sum() / mask_sum).item()
            ),
            "n_tokens": int(m.sum().item()),
        }
    return loss, stats
