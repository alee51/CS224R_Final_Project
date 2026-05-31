"""Stage 8 production: GRPO 4B, Polaris-51K filtered, 1 epoch (~400 steps).

24h Modal timeout, modal.Retries for preempt survival; verl resume_mode=auto
reads the latest ckpt from default_local_dir on the persistent volume after
restart (max save_freq=15 steps of lost work).

WANDB_TAGS is read from the local env at module import and funneled via Modal
Secret (Hydra yaml tags do NOT propagate to W&B — see project memory
2026-05-31_bringup).
"""

from __future__ import annotations

import os
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

_CONFIG_NAME = "grpo_train_4b_1epoch"
_CHECKPOINT_DIR = f"/vol/checkpoints/main-verl/{_CONFIG_NAME}"

_WANDB_TAGS = os.environ.get("WANDB_TAGS", "").strip()

_RUNTIME_SECRET_DICT: dict[str, str] = {}
if _WANDB_TAGS:
    _RUNTIME_SECRET_DICT["WANDB_TAGS"] = _WANDB_TAGS
_RUNTIME_SECRET = modal.Secret.from_dict(_RUNTIME_SECRET_DICT)

app = modal.App(app_name())

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200:4",
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=60.0),
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
        _RUNTIME_SECRET,
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def grpo_train_4b_1epoch() -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"device[{i}]:", torch.cuda.get_device_name(i))

    from verl.trainer import main_ppo  # noqa: F401
    print("main_ppo import OK")

    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    assert "grpo" in ADV_ESTIMATOR_REGISTRY
    print("pre-flight: grpo registered — OK")

    # vLLM 0.9 CuMemAllocator (colocated rollout) hard-fails on expandable_segments.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            "/root/main-verl/configs",
            "--config-name",
            _CONFIG_NAME,
        ],
        check=True,
    )

    ckpt = Path(_CHECKPOINT_DIR)
    if ckpt.is_dir():
        print("checkpoints:", sorted(p.name for p in ckpt.iterdir()))
    else:
        print("checkpoint dir missing:", _CHECKPOINT_DIR)


@app.local_entrypoint()
def main() -> None:
    if not _WANDB_TAGS:
        raise SystemExit(
            "WANDB_TAGS required. "
            "export WANDB_TAGS=verl,production,grpo,4b,stage-08"
        )
    print(f"launch: config={_CONFIG_NAME} tags={_WANDB_TAGS}")
    grpo_train_4b_1epoch.remote()
