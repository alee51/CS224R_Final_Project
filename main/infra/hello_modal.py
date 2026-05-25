"""Phase B verification: list /vol and confirm HF cache mount."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Local `modal run` resolves imports from repo root's `main/` package root.
_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

import modal

from infra.modal_image import image
from infra.modal_volume import (
    ARTIFACTS_MOUNT,
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
)

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-untagged"))

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    timeout=120,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def hello() -> None:
    vol = Path(ARTIFACTS_MOUNT)
    entries = sorted(p.name for p in vol.iterdir()) if vol.is_dir() else []
    print("/vol listing:", entries if entries else "(empty)")
    print("HF cache path:", HF_CACHE_MOUNT)


@app.local_entrypoint()
def main() -> None:
    hello.remote()
