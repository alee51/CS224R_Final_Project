"""Shared Modal image for main-verl Stage 1+ jobs.

MAXRL_COMMIT: 7197bbb46a2ecd866da52f6b401ff20a34fe9390 (maxrl main @ 2026-05-29)
GPU pins: vLLM 0.9.0 + cu128 torch index, transformers<4.54, flash-attn 2.8.3 (Blackwell)
  — aligned with main/infra/modal_image.py; not maxRL README defaults (torch 2.6 / vLLM 0.8.4).
Stage 2 (S2.3): editable install uses maxrl/setup.py deps; GPU wheels re-pinned after install.
Stage 2 (S2.5b): patch maxrl router so `polaris` / `math_reward` → math.py (upstream math_reward.py).
Runtime: ray installed explicitly for smoke; torch/vllm versions come from vLLM 0.9.0 pin.
PYTORCH_CUDA_ALLOC_CONF: intentionally omits expandable_segments:True — vLLM 0.9 CuMemAllocator
  (VeRL colocated rollout) hard-fails if set; see verl-reference §4.3 / stage-02-log S2.5 attempt 3.
  main/ keeps expandable_segments for the custom trainer; fragmentation fallback = micro-batch / util.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MAIN_VERL_DIR = _LOCAL_REPO_ROOT / "main-verl"

MAXRL_COMMIT = "7197bbb46a2ecd866da52f6b401ff20a34fe9390"
_VLLM_VERSION = "0.9.0"
_RAY_VERSION = "2.44.1"  # <!-- TODO: bump if smoke Ray/GPU init fails -->

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        "git clone https://github.com/tajwarfahim/maxrl.git /root/maxrl",
        f"cd /root/maxrl && git checkout {MAXRL_COMMIT}",
    )
    .add_local_dir(
        str(_LOCAL_MAIN_VERL_DIR / "infra" / "patches"),
        remote_path="/root/main-verl/infra/patches",
    )
    .run_commands(
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_polaris_math_reward.patch",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/main-verl",
            "CS224R_MAIN_VERL_ROOT": "/root/main-verl",
            "VLLM_USE_V1": "0",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .pip_install(
        f"vllm=={_VLLM_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers<4.54.0")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install(
        f"ray[default]=={_RAY_VERSION}",
        "wandb",
        "math-verify",
        "datasets>=2.20",
        "pyyaml>=6.0",
    )
    .run_commands("cd /root/maxrl && pip install -e .")
    .pip_install(
        "tensordict",
        "hydra-core",
        "omegaconf",
        "accelerate",
        "codetiming",
        "peft",
        "pyarrow",
        "pandas",
        "dill",
        "torchdata",
    )
    .pip_install(
        f"vllm=={_VLLM_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers<4.54.0")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(
        str(_LOCAL_MAIN_VERL_DIR),
        remote_path="/root/main-verl",
        ignore=[
            "docs",
            "*.md",
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
        ],
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-verl-stage01")
