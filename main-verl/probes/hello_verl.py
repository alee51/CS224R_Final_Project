"""Stage 1 smoke: Ray + verl import + one vLLM rollout on B200."""

from __future__ import annotations

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

SMOKE_MODEL = "Qwen/Qwen3-1.7B-Base"
SMOKE_PROMPT = (
    "Solve: What is 2 + 2? Put your final answer in \\boxed{}."
)

app = modal.App(app_name())

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=1800,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def hello_verl() -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    import ray

    ray.init(num_gpus=1, ignore_reinit_error=True)
    try:
        import verl

        print("verl:", getattr(verl, "__version__", "unknown"))

        # Path A: direct vLLM rollout (lightest smoke).
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=SMOKE_MODEL,
            enforce_eager=True,
            trust_remote_code=True,
        )
        outputs = llm.generate(
            [SMOKE_PROMPT],
            SamplingParams(max_tokens=64, temperature=0.0),
        )
        text = outputs[0].outputs[0].text
        print("rollout (first 500 chars):", text[:500])
    finally:
        ray.shutdown()


@app.local_entrypoint()
def main() -> None:
    hello_verl.remote()
