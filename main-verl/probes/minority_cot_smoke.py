"""Stage 3a smoke: minority_cot advantage estimator on B200:4 via verl.trainer.main_ppo.

Mirrors probes/grpo_smoke.py with three changes:
1. Function name: grpo_smoke -> minority_cot_smoke.
2. Config name passed to trainer subprocess: grpo_smoke_1p7b -> minority_cot_smoke_1p7b.
3. Pre-flight registry assertion: verifies 'minority_cot' is in ADV_ESTIMATOR_REGISTRY
   before Ray spins up, failing fast if the patch (infra/patches/maxrl_minority_cot_adv_est.patch)
   did not apply at image build.

Default app name: cs224r-verl-stage03a (set via CS224R_APP_NAME env var in launch script).
"""

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

CHECKPOINT_DIR = "/vol/checkpoints/main-verl/minority_cot_smoke_1p7b"

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
def minority_cot_smoke() -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"device[{i}]:", torch.cuda.get_device_name(i))

    import verl

    print("verl:", getattr(verl, "__version__", "unknown"))

    from verl.trainer import main_ppo  # noqa: F401 — fail-fast dep check

    print("main_ppo import OK")

    # Pre-flight: verify minority_cot estimator was registered by the patch.
    # Fails fast inside the container before Ray spins up, saving ~3 min per
    # failed hook iteration if the patch (infra/patches/maxrl_minority_cot_adv_est.patch)
    # did not apply at image build.
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    assert "minority_cot" in ADV_ESTIMATOR_REGISTRY, (
        "minority_cot estimator not registered — patch did not apply at image build. "
        "Check infra/patches/maxrl_minority_cot_adv_est.patch and modal_image.py."
    )
    print("pre-flight: 'minority_cot' in ADV_ESTIMATOR_REGISTRY — OK")

    import os

    # vLLM allocator pool is incompatible with expandable_segments (set in modal_image).
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            "/root/main-verl/configs",
            "--config-name",
            "minority_cot_smoke_1p7b",
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
    minority_cot_smoke.remote()
