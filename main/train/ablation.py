"""OOM ablation flags (Modal CLI args → env for collocated train)."""

from __future__ import annotations

import os


def apply_ablation_env(
    *,
    ablation: str = "",
    vllm_sleep: int = 0,
    logprob_chunk: int = 0,
) -> None:
    if ablation:
        os.environ["CS224R_ABLATION"] = ablation
        os.environ["CS224R_NO_RESUME"] = "1"
    os.environ["CS224R_VLLM_SLEEP"] = "1" if int(vllm_sleep) else "0"
    os.environ["CS224R_LOGPROB_CHUNK"] = str(int(logprob_chunk))


def ablation_label() -> str | None:
    label = os.environ.get("CS224R_ABLATION", "").strip()
    return label or None


def vllm_sleep_enabled() -> bool:
    return os.environ.get("CS224R_VLLM_SLEEP", "0").strip() in ("1", "true", "yes")


def logprob_chunk_size() -> int:
    raw = os.environ.get("CS224R_LOGPROB_CHUNK", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def prepare_pytorch_alloc_for_vllm_sleep() -> None:
    """vLLM sleep memory pool conflicts with expandable_segments:True."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in conf:
        return
    parts = [p.strip() for p in conf.split(",") if p.strip() and "expandable_segments" not in p]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts) if parts else "max_split_size_mb:128"
