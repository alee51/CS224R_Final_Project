"""Stage 6: minority_cot + real judge on Qwen3-4B-Base (step-time probe)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
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

_DEFAULT_CONFIG_NAME = "minority_cot_smoke_judge_4b_ladder1e"
# Match trainer.default_local_dir in each yaml (legacy 4b config → ladder1b dir).
_CHECKPOINT_BY_CONFIG = {
    "minority_cot_smoke_judge_4b": "/vol/checkpoints/main-verl/minority_cot_smoke_judge_4b_ladder1b",
    "minority_cot_smoke_judge_4b_ladder1c": "/vol/checkpoints/main-verl/minority_cot_smoke_judge_4b_ladder1c",
    "minority_cot_smoke_judge_4b_ladder1d": "/vol/checkpoints/main-verl/minority_cot_smoke_judge_4b_ladder1d",
    "minority_cot_smoke_judge_4b_ladder1e": "/vol/checkpoints/main-verl/minority_cot_smoke_judge_4b_ladder1e",
    "minority_cot_train_4b_1epoch": "/vol/checkpoints/main-verl/minority_cot_train_4b_1epoch",
    "poly_epo_cot_train_4b_1epoch": "/vol/checkpoints/main-verl/poly_epo_cot_train_4b_1epoch",
    "grpo_train_4b_1epoch": "/vol/checkpoints/main-verl/grpo_train_4b_1epoch",
}

# All env-driven knobs are read LOCALLY at module import (when `modal run` evaluates
# this file on the client). They're funneled into the Modal Secret so the container
# sees them at function-call time — module-level os.environ reads on the container do
# NOT see Secret values yet (Secrets inject post-import).
_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
_JUDGE_AUTH_TOKEN = os.environ.get("JUDGE_AUTH_TOKEN", "")
_LOCAL_CONFIG_NAME = os.environ.get("CS224R_SMOKE_CONFIG", "").strip()
_LOCAL_SMOKE_STEPS = os.environ.get("CS224R_SMOKE_STEPS", "").strip()
_TRACE_ENABLE = os.environ.get("CS224R_TRACE_ENABLE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_WANDB_TAGS = os.environ.get("WANDB_TAGS", "").strip()

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

# Trace env vars only injected when CS224R_TRACE_ENABLE=1 at launch (off for production).
# Path slug derived from config name so trace + step-log artifacts don't stomp across runs.
_EFFECTIVE_CONFIG_NAME = _LOCAL_CONFIG_NAME or _DEFAULT_CONFIG_NAME
_TRACE_SLUG = _EFFECTIVE_CONFIG_NAME.replace("minority_cot_", "").replace("poly_epo_cot_", "")
_RUNTIME_SECRET_DICT = {
    "JUDGE_BASE_URL": _JUDGE_BASE_URL,
    "JUDGE_AUTH_TOKEN": _JUDGE_AUTH_TOKEN,
    # Propagate config/steps via Secret so container sees them at function-call time.
    "CS224R_SMOKE_CONFIG": _EFFECTIVE_CONFIG_NAME,
}
if _LOCAL_SMOKE_STEPS:
    _RUNTIME_SECRET_DICT["CS224R_SMOKE_STEPS"] = _LOCAL_SMOKE_STEPS
if _WANDB_TAGS:
    _RUNTIME_SECRET_DICT["WANDB_TAGS"] = _WANDB_TAGS
if _TRACE_ENABLE:
    _RUNTIME_SECRET_DICT.update(
        {
            "CS224R_JUDGE_TRACE": "1",
            "CS224R_JUDGE_TRACE_PROMPT_IDX": "0",
            "CS224R_JUDGE_TRACE_PATH": f"/vol/judge_trace_{_TRACE_SLUG}_step.json",
            "CS224R_JUDGE_STEP_LOG": f"/vol/judge_step_log_{_TRACE_SLUG}.jsonl",
            "CS224R_JUDGE_TRACE_MAX_CHARS": os.environ.get(
                "CS224R_JUDGE_TRACE_MAX_CHARS", "0"
            ),
        }
    )

_JUDGE_RUNTIME_SECRET = modal.Secret.from_dict(_RUNTIME_SECRET_DICT)


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
def minority_cot_judge_smoke_4b() -> None:
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

    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY

    assert "minority_cot" in ADV_ESTIMATOR_REGISTRY
    print("pre-flight: minority_cot registered — OK")

    base_url = os.environ.get("JUDGE_BASE_URL", "")
    assert base_url, "JUDGE_BASE_URL required"
    print(f"pre-flight: JUDGE_BASE_URL='{base_url}' — OK")

    health_url = os.environ.get("JUDGE_HEALTH_URL") or base_url.replace(
        "--v1-chat-completions.", "--health."
    )
    print(f"pre-flight: probing judge health at {health_url}")
    _wait_for_judge_health(health_url, timeout_s=180.0)
    print("pre-flight: judge health OK")

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    # Read config/steps from env at function call (Modal Secret has been injected by now —
    # module-level reads on the container happen too early to see Secret values).
    config_name = os.environ.get("CS224R_SMOKE_CONFIG", "").strip() or _DEFAULT_CONFIG_NAME
    smoke_steps = os.environ.get("CS224R_SMOKE_STEPS", "").strip()
    checkpoint_dir = _CHECKPOINT_BY_CONFIG.get(
        config_name, f"/vol/checkpoints/main-verl/{config_name}"
    )

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        "/root/main-verl/configs",
        "--config-name",
        config_name,
    ]
    if smoke_steps:
        cmd.append(f"trainer.total_training_steps={smoke_steps}")
    print(f"launch: config={config_name} steps={smoke_steps or '(yaml default)'}")
    subprocess.run(cmd, check=True)

    ckpt = Path(checkpoint_dir)
    if ckpt.is_dir():
        print("checkpoints:", sorted(p.name for p in ckpt.iterdir()))
    else:
        print("checkpoint dir missing:", checkpoint_dir)


def _wait_for_judge_health(url: str, *, timeout_s: float) -> None:
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
            "JUDGE_BASE_URL required. "
            "export JUDGE_BASE_URL=https://chicken602--v1-chat-completions.modal.run"
        )
    minority_cot_judge_smoke_4b.remote()
