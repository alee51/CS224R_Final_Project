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
        }
    )
    .pip_install(f"vllm=={_VLLM_VERSION}")
    # vllm 0.8.5 otherwise pulls transformers 5.x; breaks Qwen2Tokenizer in vLLM cache path.
    .pip_install("transformers>=4.55.2,<5.0.0")
    .pip_install(
        "datasets>=2.20",
        "wandb",
        "pyyaml>=6.0",
        "pytest",
    )
    .add_local_dir(
        str(_LOCAL_MAIN_DIR),
        remote_path="/root/main",
        ignore=["docs", "data", "*.md", "__pycache__", ".pytest_cache"],
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-untagged")
