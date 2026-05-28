"""Shared Modal image and app naming for main experiment jobs."""

from __future__ import annotations

import os
from pathlib import Path

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MAIN_DIR = _LOCAL_REPO_ROOT / "main"

# vLLM 0.6.x does not support Qwen3ForCausalLM; 0.8.5+ required for Qwen3-1.7B / 4B-Instruct.
# Keep dependency pins explicit here so image rebuilds are deterministic.
_VLLM_VERSION = "0.9.0"

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
    # vLLM 0.9.0 upgrades to torch 2.7 and CUDA 12.8-compatible wheels by default.
    # Keep an explicit cu128 torch index for Blackwell/B200 compatibility:
    # https://download.pytorch.org/whl/cu128
    # vLLM release: https://pypi.org/project/vllm/0.9.0/
    .pip_install(
        f"vllm=={_VLLM_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # vLLM 0.9.0 currently conflicts with transformers>=4.54
    # (`aimv2` AutoConfig registration collision during import).
    # Temporary compatibility pin for collocated rollout/train workers.
    .pip_install("transformers<4.54.0")
    # FlashAttention wheel for HF train-side forward/backward on Blackwell.
    # We pin a torch2.7 build aligned with vLLM 0.9.0's torch baseline.
    # Wheel source (includes Blackwell-capable kernels in this release line):
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3
    # Exact wheel:
    # flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
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
            "!data/probes/**/manifest.jsonl",
            "*.md",
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
        ],
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-untagged")
