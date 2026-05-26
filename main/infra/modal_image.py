"""Shared Modal image and app naming for main experiment jobs."""

from __future__ import annotations

import os
from pathlib import Path

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MAIN_DIR = _LOCAL_REPO_ROOT / "main"

# vLLM 0.6.x does not support Qwen3ForCausalLM; 0.8.5+ required for Qwen3-1.7B / 4B-Instruct.
# Let vllm own torch/transformers/xformers pins to avoid version skew.
_VLLM_VERSION = "0.8.5"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/main",
            # Canonical package root (train/, configs/, probes/) on Modal workers.
            "CS224R_MAIN_ROOT": "/root/main",
            # vLLM V1 multiprocessing breaks on Modal ("Cannot re-initialize CUDA in forked subprocess").
            "VLLM_USE_V1": "0",
        }
    )
    .pip_install(f"vllm=={_VLLM_VERSION}")
    # vllm 0.8.5 otherwise pulls transformers 5.x; breaks Qwen2Tokenizer in vLLM cache path.
    .pip_install("transformers>=4.55.2,<5.0.0")
    # FlashAttention-2 for the HF train-side forward/backward (build_hf uses
    # attn_implementation="flash_attention_2"). Prebuilt wheel matched to
    # torch 2.6 + cu12 + py311 + cxx11abiFALSE (PyPI default ABI). Source build
    # fails because debian_slim has no nvcc; the wheel ships compiled kernels.
    # See docs/efficiency_wins_2026-05-26.md.
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install(
        "datasets>=2.20",
        "wandb",
        "pyyaml>=6.0",
        "pytest",
        "pylatexenc>=2.10",
    )
    .add_local_dir(
        str(_LOCAL_MAIN_DIR),
        remote_path="/root/main",
        # Include data/*.py (dataset loader). Exclude large frozen jsonl only.
        ignore=[
            "docs",
            "data/*.jsonl",
            "data/**/*.jsonl",
            "*.md",
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
        ],
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-untagged")
