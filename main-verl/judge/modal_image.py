"""Judge-only Modal image — no maxrl/verl (Stage 4)."""

from __future__ import annotations

import os
from pathlib import Path

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MAIN_VERL_DIR = _LOCAL_REPO_ROOT / "main-verl"

_VLLM_VERSION = "0.9.0"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/main-verl",
            "VLLM_USE_V1": "0",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .pip_install(
        f"vllm=={_VLLM_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers<4.54.0", "httpx>=0.27")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(
        str(_LOCAL_MAIN_VERL_DIR / "judge"),
        remote_path="/root/main-verl/judge",
        copy=True,
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-verl-stage04-judge")
