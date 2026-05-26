"""
Weight-sync spike: HF weight change → vLLM generation shift.

Requires GPU + vLLM + a small HF model. Skipped locally when unavailable.
See trainer_skeleton.md §9.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed (Modal/GPU only)")
@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="CUDA required for weight-sync spike",
)
def test_sync_hf_to_vllm_changes_generation():
    """
    TODO(spike): load ~125M model in HF + vLLM, perturb HF weights, sync, assert
    generation distribution shifts. Blocked until GPU Modal run — API path is
    train.weight_sync.sync_hf_to_vllm (vLLM 0.8.5 load_weights).
    """
    pytest.skip(
        "End-to-end weight-sync spike not run locally; execute on Modal with "
        "e.g. Qwen/Qwen2.5-0.5B or gpt2 after confirming VRAM. "
        "See main/docs/trainer_skeleton.md §9."
    )


def test_sync_stats_dataclass():
    from train.weight_sync import SyncStats

    s = SyncStats(wall_clock_s=1.0, bytes_moved=1024, n_tensors=10)
    assert s.bytes_moved == 1024
    assert s.n_tensors == 10
