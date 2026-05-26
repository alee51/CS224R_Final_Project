"""HF → vLLM weight sync (vLLM 0.8.5 load_weights path)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    wall_clock_s: float
    bytes_moved: int
    n_tensors: int


def _hf_weight_iterator(hf_model: Any) -> Iterator[tuple[str, Any]]:
    for name, param in hf_model.named_parameters():
        yield name, param.data


def _vllm_runner_model(vllm_llm: Any) -> Any:
    engine = getattr(vllm_llm, "llm_engine", vllm_llm)
    executor = engine.model_executor
    worker = executor.driver_worker
    return worker.model_runner.model


def sync_hf_to_vllm(hf_model: Any, vllm_llm: Any) -> SyncStats:
    """
    Push HF weights into a collocated vLLM LLM instance.

    Uses model.load_weights(iterator) on the driver worker (vLLM 0.8.5).
    See trainer_skeleton.md §9 — API may change across vLLM versions.
    """
    t0 = time.monotonic()
    model = _vllm_runner_model(vllm_llm)
    weights = list(_hf_weight_iterator(hf_model))
    bytes_moved = sum(t.numel() * t.element_size() for _, t in weights)
    if not hasattr(model, "load_weights"):
        raise RuntimeError("vLLM model has no load_weights; check vLLM version")
    model.load_weights(weights)
    elapsed = time.monotonic() - t0
    logger.info(
        "Synced %s tensors (%.2f MB) to vLLM in %.3fs",
        len(weights),
        bytes_moved / (1024**2),
        elapsed,
    )
    return SyncStats(
        wall_clock_s=elapsed,
        bytes_moved=bytes_moved,
        n_tensors=len(weights),
    )
