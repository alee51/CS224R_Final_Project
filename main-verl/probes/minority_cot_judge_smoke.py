"""Stage 3b smoke: minority_cot with REAL judge clusters on B200:4.

Mirrors probes/minority_cot_smoke.py (Stage 3a mock) with these differences:
1. Config name: minority_cot_smoke_judge_1p7b (cluster_source: judge).
2. JUDGE_BASE_URL / JUDGE_AUTH_TOKEN env vars passed through to the container
   so train.clusters_judge can reach the deployed judge service.
3. Extra pre-flight assertions:
   - "minority_cot" in ADV_ESTIMATOR_REGISTRY (maxrl fork must include
     cs224r-patches commit e047d0e).
   - JUDGE_BASE_URL non-empty (cluster_source=judge requires it).
   - Judge health endpoint returns 200 within 180s (vLLM cold-start tolerated).
4. Reads JUDGE_BASE_URL from the local environment and forwards it via Modal
   secrets / env so the trainer process inside the container can see it.

Default app name: cs224r-verl-stage03b (set via CS224R_APP_NAME).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal

from infra.modal_image import app_name, image as _base_image
from infra.modal_volume import (
    ARTIFACTS_MOUNT,
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
)

CHECKPOINT_DIR = "/vol/checkpoints/main-verl/minority_cot_smoke_judge_1p7b"

# Local-side JUDGE_BASE_URL is read at deploy time and shipped into the container
# as an env var so the trainer can reach the judge. None → probe will fail
# pre-flight with a clear message. Format: full chat-completions URL, e.g.
# https://chicken602--v1-chat-completions.modal.run
_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
_JUDGE_AUTH_TOKEN = os.environ.get("JUDGE_AUTH_TOKEN", "")
_JUDGE_HEALTH_URL = os.environ.get("JUDGE_HEALTH_URL", "")

app = modal.App(app_name())

image = _base_image.add_local_dir(
    str(_MAIN_VERL_ROOT / "train"),
    remote_path="/root/main-verl/train",
).add_local_dir(
    str(_MAIN_VERL_ROOT / "judge"),
    remote_path="/root/main-verl/judge",
)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

# Stage 3b (2026-05-30): inject judge env via Secret.from_dict at function-call
# time instead of image.env() at build time — image.env() is a build step and
# Modal forbids build steps after add_local_dir(_LOCAL_MAIN_VERL_DIR) without
# copy=True. Secret.from_dict avoids invalidating the image layer.
_JUDGE_RUNTIME_SECRET = modal.Secret.from_dict(
    {
        "JUDGE_BASE_URL": _JUDGE_BASE_URL,
        "JUDGE_AUTH_TOKEN": _JUDGE_AUTH_TOKEN,
    }
)


@app.function(
    image=image,
    gpu="B200:4",
    timeout=3 * 3600,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
        _JUDGE_RUNTIME_SECRET,
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def minority_cot_judge_smoke() -> None:
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

    # Pre-flight 1: minority_cot adv estimator registered (maxrl fork must
    # include cs224r-patches commit e047d0e).
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    assert "minority_cot" in ADV_ESTIMATOR_REGISTRY, (
        "minority_cot estimator not registered — maxrl fork is missing commit "
        "e047d0e (cs224r-patches branch)."
    )
    print("pre-flight 1: 'minority_cot' in ADV_ESTIMATOR_REGISTRY — OK")

    # Pre-flight 2: expose-data hook present (maxrl fork must include
    # cs224r-patches commit 572a592 — passes DataProto to registered adv hooks).
    import inspect
    from verl.trainer.ppo import ray_trainer as _rt
    src = inspect.getsource(_rt.compute_advantage)
    assert '"data": data' in src, (
        "ray_trainer compute_advantage dispatch missing 'data' kwarg — "
        "maxrl fork is missing commit 572a592 (cs224r-patches branch)."
    )
    print("pre-flight 2: ray_trainer 'data' kwarg present — OK")

    # Pre-flight 3: JUDGE_BASE_URL set.
    base_url = os.environ.get("JUDGE_BASE_URL", "")
    assert base_url, (
        "JUDGE_BASE_URL env var not set inside container. "
        "Set it locally before launching the smoke (see launch script)."
    )
    print(f"pre-flight 3: JUDGE_BASE_URL='{base_url}' — OK")

    # Pre-flight 4: judge health probe. Derive health URL from base_url by
    # swapping the chat-completions label for the health label (Modal convention).
    health_url = (
        os.environ.get("JUDGE_HEALTH_URL")
        or base_url.replace("--v1-chat-completions.", "--health.")
    )
    print(f"pre-flight 4: probing judge health at {health_url}")
    _wait_for_judge_health(health_url, timeout_s=180.0)
    print("pre-flight 4: judge health OK")

    # vLLM allocator pool is incompatible with expandable_segments.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            "/root/main-verl/configs",
            "--config-name",
            "minority_cot_smoke_judge_1p7b",
        ],
        check=True,
    )

    ckpt = Path(CHECKPOINT_DIR)
    if ckpt.is_dir():
        print("checkpoints:", sorted(p.name for p in ckpt.iterdir()))
    else:
        print("checkpoint dir missing:", CHECKPOINT_DIR)


def _wait_for_judge_health(url: str, *, timeout_s: float) -> None:
    """Block until GET ``url`` returns 200 or timeout. Tolerates cold-start."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(3)
    raise RuntimeError(
        f"judge health endpoint {url!r} did not return 200 within {timeout_s}s. "
        f"Last error: {last_err}"
    )


@app.local_entrypoint()
def main() -> None:
    if not _JUDGE_BASE_URL:
        raise SystemExit(
            "JUDGE_BASE_URL env var is required to launch the Stage 3b smoke. "
            "Set it to the deployed judge chat-completions URL before invoking "
            "the launch script. See scripts/launch_minority_cot_judge_smoke.sh."
        )
    minority_cot_judge_smoke.remote()
