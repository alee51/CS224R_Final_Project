"""10-step smoke of poly_epo_cot 4B with judge prompt few-shot fix landed.

Identical to poly_epo_cot_train_4b_1epoch.py except for _CONFIG_NAME. Used
once before relaunch to confirm:
  - judge picks up the new prompt (5400-char system block, two few-shot examples)
  - distinct_clusters_mean rises vs prior baseline (~1.5 → expected ~2+)
  - judge_parse_ok_rate stays at 1.000 with the new prompt
  - all 10 steps complete cleanly; no Hydra/config drift
"""

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

_CONFIG_NAME = "poly_epo_cot_smoke_judgefix_4b"
_CHECKPOINT_DIR = f"/vol/checkpoints/main-verl/{_CONFIG_NAME}"

_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
_JUDGE_AUTH_TOKEN = os.environ.get("JUDGE_AUTH_TOKEN", "")
_WANDB_TAGS = os.environ.get("WANDB_TAGS", "").strip()

_RUNTIME_SECRET_DICT = {
    "JUDGE_BASE_URL": _JUDGE_BASE_URL,
    "JUDGE_AUTH_TOKEN": _JUDGE_AUTH_TOKEN,
    "CS224R_JUDGE_PARSE_FAIL_LOG": f"/vol/judge_parse_failures_{_CONFIG_NAME}.jsonl",
}
if _WANDB_TAGS:
    _RUNTIME_SECRET_DICT["WANDB_TAGS"] = _WANDB_TAGS
_RUNTIME_SECRET = modal.Secret.from_dict(_RUNTIME_SECRET_DICT)

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


@app.function(
    image=image,
    gpu="B200:4",
    timeout=2 * 3600,
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
def poly_epo_cot_smoke_judgefix_4b() -> None:
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
    assert "poly_epo_cot" in ADV_ESTIMATOR_REGISTRY
    print("pre-flight: poly_epo_cot registered — OK")

    # Pre-flight: confirm the judge prompt on this image contains the few-shot examples.
    from judge.prompt import build_judge_messages
    system, _ = build_judge_messages("smoke test", ["x"] * 8)
    assert "Few-Shot Example 1" in system, "judge prompt missing FS Example 1"
    assert "Few-Shot Example 2" in system, "judge prompt missing FS Example 2"
    assert '"cluster_id": 100' in system, "judge prompt missing FS cluster_id 100 example"
    assert "{{" not in system and "}}" not in system, "judge prompt has stray double braces"
    print(f"pre-flight: judge prompt OK ({len(system)} chars, FS examples present)")

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
            "export JUDGE_BASE_URL=https://chicken602--<judge-app>-v1-chat-completions.modal.run"
        )
    if not _WANDB_TAGS:
        raise SystemExit(
            "WANDB_TAGS required. "
            "export WANDB_TAGS=verl,smoke,poly_epo_cot,4b,judge-fewshot-fix"
        )
    print(f"launch: config={_CONFIG_NAME} tags={_WANDB_TAGS}")
    poly_epo_cot_smoke_judgefix_4b.remote()
