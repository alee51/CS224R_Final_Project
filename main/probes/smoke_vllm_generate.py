"""Minimal vLLM generate smoke on Modal B200 (gate before FA/train smokes)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

import modal

from infra.modal_image import image
from infra.modal_volume import HF_CACHE_MOUNT, HF_CACHE_VOLUME_NAME

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-smoke-vllm-generate"))
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _smoke_vllm_generate_impl(model_id: str = DEFAULT_MODEL) -> dict[str, object]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    os.environ.setdefault("HF_HOME", HF_CACHE_MOUNT)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_MOUNT)

    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        max_num_seqs=8,
    )
    out = llm.generate(
        ["1+1="],
        SamplingParams(temperature=0.0, max_tokens=16, logprobs=1),
    )
    text = out[0].outputs[0].text if out and out[0].outputs else ""
    info = {
        "ok": True,
        "model_id": model_id,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_capability": (
            ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else "none"
        ),
        "text": text,
    }
    print(info, flush=True)
    return info


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
def smoke_vllm_generate_h200(model_id: str = DEFAULT_MODEL) -> dict[str, object]:
    return _smoke_vllm_generate_impl(model_id=model_id)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
def smoke_vllm_generate_b200(model_id: str = DEFAULT_MODEL) -> dict[str, object]:
    return _smoke_vllm_generate_impl(model_id=model_id)


@app.local_entrypoint()
def main(model_id: str = DEFAULT_MODEL, gpu_class: str = "b200") -> None:
    gpu = gpu_class.lower()
    if gpu == "h200":
        out = smoke_vllm_generate_h200.remote(model_id=model_id)
    elif gpu == "b200":
        out = smoke_vllm_generate_b200.remote(model_id=model_id)
    else:
        raise SystemExit("gpu_class must be h200 or b200")
    if not out.get("ok"):
        raise SystemExit(1)
