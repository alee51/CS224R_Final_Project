"""Stage 6 smoke: GRPO trainer on B200:4 with Qwen3-4B-Base."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal

from infra.modal_image import app_name, image
from infra.modal_volume import (
    ARTIFACTS_MOUNT,
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
)

CHECKPOINT_DIR = "/vol/checkpoints/main-verl/grpo_smoke_4b"

app = modal.App(app_name())

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200:4",
    timeout=3 * 3600,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def grpo_smoke_4b() -> None:
    import os

    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"device[{i}]:", torch.cuda.get_device_name(i))

    import verl

    print("verl:", getattr(verl, "__version__", "unknown"))

    from verl.trainer import main_ppo  # noqa: F401

    print("main_ppo import OK")

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            "/root/main-verl/configs",
            "--config-name",
            "grpo_smoke_4b",
        ],
        check=True,
    )

    ckpt = Path(CHECKPOINT_DIR)
    if ckpt.is_dir():
        print("checkpoints:", sorted(p.name for p in ckpt.iterdir()))
    else:
        print("checkpoint dir missing:", CHECKPOINT_DIR)


@app.local_entrypoint()
def main() -> None:
    grpo_smoke_4b.remote()
