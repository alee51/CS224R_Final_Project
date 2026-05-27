"""HF → vLLM weight sync (vLLM 0.9.x compatible path resolution)."""

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
    """Resolve model object behind LLM across vLLM 0.8/0.9 variants."""
    engine = getattr(vllm_llm, "llm_engine", vllm_llm)
    executor = getattr(engine, "model_executor", None)
    if executor is None:
        raise RuntimeError("vLLM engine has no model_executor; check VLLM_USE_V1 and version")

    worker = getattr(executor, "driver_worker", None)
    if worker is None:
        raise RuntimeError("vLLM executor has no driver_worker")

    # vLLM wrappers may store the worker object in .worker.
    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("vLLM worker has no model_runner")

    model = getattr(model_runner, "model", None)
    if model is None and hasattr(model_runner, "get_model"):
        model = model_runner.get_model()
    if model is None:
        raise RuntimeError("vLLM model_runner has no model/get_model")

    logger.info(
        "Resolved vLLM sync target type=%s runner=%s worker=%s",
        type(model).__name__,
        type(model_runner).__name__,
        type(worker_obj).__name__,
    )
    return model


def sync_hf_to_vllm(hf_model: Any, vllm_llm: Any) -> SyncStats:
    """
    Push HF weights into a collocated vLLM LLM instance.

    Uses model.load_weights(iterator) on the driver worker.
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
