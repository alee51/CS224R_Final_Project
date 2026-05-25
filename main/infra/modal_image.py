"""Shared Modal image and app naming for main experiment jobs."""

from __future__ import annotations

import os
from pathlib import Path

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MAIN_DIR = _LOCAL_REPO_ROOT / "main"

_TORCH_VERSION = "2.5.1"
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
_FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/main",
        }
    )
    .pip_install(
        f"torch=={_TORCH_VERSION}",
        extra_index_url=_TORCH_CUDA_INDEX,
    )
    .pip_install(
        "transformers>=4.51,<4.55",
        "accelerate>=0.30",
        "sentencepiece",
        "safetensors",
        "pyyaml>=6.0",
        "huggingface_hub>=0.23",
        "wandb",
        "vllm==0.6.3",
        "datasets>=2.20",
        "pytest",
        _FLASH_ATTN_WHEEL,
    )
    .add_local_dir(
        str(_LOCAL_MAIN_DIR),
        remote_path="/root/main",
        ignore=["docs", "data", "*.md", "__pycache__", ".pytest_cache"],
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-untagged")
