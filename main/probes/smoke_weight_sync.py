"""Smoke test HF -> vLLM weight sync on Modal B200."""

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
from train.rollout import RolloutCfg, RolloutEngine
from train.trainer import build_hf, train_cfg_from_dict
from train.weight_sync import sync_hf_to_vllm

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-smoke-weight-sync"))
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _probe_logit_sum(hf_model: object) -> float:
    import torch

    model = hf_model  # type: ignore[assignment]
    device = next(model.parameters()).device  # type: ignore[union-attr]
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids)  # type: ignore[operator]
    return float(out.logits[0, -1].sum().item())


def _smoke_weight_sync_impl(model_id: str = DEFAULT_MODEL) -> dict[str, object]:
    import torch

    os.environ.setdefault("HF_HOME", HF_CACHE_MOUNT)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_MOUNT)

    rollout = RolloutEngine(
        RolloutCfg(
            model=model_id,
            gpu_memory_utilization=0.45,
            max_model_len=2048,
            max_num_seqs=16,
        )
    )
    train_cfg = train_cfg_from_dict(
        {
            "global_seed": 42,
            "arm": "grpo",
            "train": {
                "lr": 1.0e-6,
                "weight_decay": 0.0,
                "gradient_checkpointing": False,
            },
            "rollout": {
                "model": model_id,
                "max_prompt_length": 1024,
                "max_response_length": 1024,
                "temperature": 1.0,
                "top_p": 1.0,
                "gpu_memory_utilization": 0.45,
                "max_model_len": 2048,
                "max_num_seqs": 16,
                "enable_prefix_caching": True,
                "logprobs": 1,
            },
            "loss": {"clip_low": 0.20, "clip_high": 0.28, "entropy_coef": 0.0},
            "weight_sync": {"every_n_steps": 1},
            "wandb": {"entity": "noop", "project": "noop"},
        }
    )
    hf, _ = build_hf(train_cfg)
    before = _probe_logit_sum(hf)

    with torch.no_grad():
        first_param = next(hf.parameters())
        first_param.add_(0.001)
    after_perturb = _probe_logit_sum(hf)

    stats = sync_hf_to_vllm(hf, rollout.llm)
    gen = rollout.generate(["2+2="], 1, [1234])
    rollout.shutdown()

    out = {
        "ok": True,
        "model_id": model_id,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hf_probe_before": before,
        "hf_probe_after_perturb": after_perturb,
        "hf_probe_delta": after_perturb - before,
        "sync_wall_clock_s": stats.wall_clock_s,
        "sync_bytes_moved": stats.bytes_moved,
        "sync_n_tensors": stats.n_tensors,
        "vllm_text": gen[0].completion_text if gen else "",
    }
    print(out, flush=True)
    return out


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
def smoke_weight_sync_h200(model_id: str = DEFAULT_MODEL) -> dict[str, object]:
    return _smoke_weight_sync_impl(model_id=model_id)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
def smoke_weight_sync_b200(model_id: str = DEFAULT_MODEL) -> dict[str, object]:
    return _smoke_weight_sync_impl(model_id=model_id)


@app.local_entrypoint()
def main(model_id: str = DEFAULT_MODEL, gpu_class: str = "b200") -> None:
    gpu = gpu_class.lower()
    if gpu == "h200":
        out = smoke_weight_sync_h200.remote(model_id=model_id)
    elif gpu == "b200":
        out = smoke_weight_sync_b200.remote(model_id=model_id)
    else:
        raise SystemExit("gpu_class must be h200 or b200")
    if not out.get("ok"):
        raise SystemExit(1)
