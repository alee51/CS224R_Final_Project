"""Isolated FlashAttention-2 smoke test on Modal (same image as GRPO training).

Surfaces import / load / forward errors without running vLLM or a full train loop.
Launch: bash main/scripts/launch_smoke_flash_attn.sh

Stages (stop on first failure unless --all):
  env         — torch/CUDA/transformers/flash_attn versions
  import      — `import flash_attn`
  load        — HF from_pretrained with attn_implementation=flash_attention_2
  forward     — one short completion logprob-style forward on GPU
  collocated  — vLLM RolloutEngine then HF FA2 (same order as train())
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

import modal

from infra.modal_image import image
from infra.modal_volume import HF_CACHE_MOUNT, HF_CACHE_VOLUME_NAME
from train.rollout import RolloutCfg, RolloutEngine

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-smoke-flash-attn"))
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _stage_env() -> dict[str, str]:
    import torch
    import transformers

    info: dict[str, str] = {
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "transformers": transformers.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        )
    try:
        import flash_attn

        info["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
    except Exception as exc:
        info["flash_attn"] = f"IMPORT FAILED: {exc!r}"
    return info


def _stage_import() -> None:
    import flash_attn  # noqa: F401

    from flash_attn import flash_attn_func  # noqa: F401


def _stage_load(model_id: str, *, train_mode: bool = False) -> tuple[object, str]:
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if train_mode:
        model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
    model.to(device)
    attn = getattr(model.config, "_attn_implementation", None)
    return model, str(attn)


def _stage_collocated(model_id: str) -> dict[str, float]:
    """Mirror train() startup: vLLM first, then HF FA2 (no sleep unless env set)."""
    import torch

    rollout_cfg = RolloutCfg(
        model=model_id,
        max_prompt_length=1024,
        max_response_length=4096,
        gpu_memory_utilization=0.45,
        max_model_len=5120,
        max_num_seqs=128,
    )
    engine = RolloutEngine(rollout_cfg)
    vram_after_vllm = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    model, attn_impl = _stage_load(model_id, train_mode=True)
    vram_after_hf = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    norm = _stage_forward(model)
    engine.shutdown()
    return {
        "vram_gb_after_vllm": vram_after_vllm,
        "vram_gb_after_hf": vram_after_hf,
        "attn_implementation": attn_impl,
        "logit_norm": norm,
    }


def _stage_forward(model: object) -> float:
    import torch

    device = next(model.parameters()).device  # type: ignore[union-attr]
    # Tiny synthetic batch — enough to exercise FA2 kernels.
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids)  # type: ignore[operator]
    return float(out.logits[0, -1].norm().item())


def _smoke_flash_attn_impl(
    model_id: str = DEFAULT_MODEL,
    run_all_stages: bool = False,
) -> dict[str, object]:
    """Run FA2 diagnostics; return structured result (also printed)."""
    os.environ.setdefault("HF_HOME", HF_CACHE_MOUNT)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_MOUNT)

    stages = ("env", "import", "load", "forward", "collocated")
    results: dict[str, object] = {"model_id": model_id, "stages": {}}
    model = None

    for name in stages:
        print(f"\n=== stage: {name} ===", flush=True)
        try:
            if name == "env":
                out = _stage_env()
                results["stages"][name] = {"ok": True, "info": out}
                print(out, flush=True)
            elif name == "import":
                _stage_import()
                results["stages"][name] = {"ok": True}
                print("flash_attn import OK", flush=True)
            elif name == "load":
                model, attn_impl = _stage_load(model_id)
                results["stages"][name] = {
                    "ok": True,
                    "attn_implementation": attn_impl,
                }
                print(f"load OK attn_implementation={attn_impl}", flush=True)
            elif name == "forward":
                if model is None:
                    model, attn_impl = _stage_load(model_id)
                    results.setdefault("stages", {}).setdefault(
                        "load",
                        {"ok": True, "attn_implementation": attn_impl},
                    )
                norm = _stage_forward(model)
                results["stages"][name] = {"ok": True, "logit_norm": norm}
                print(f"forward OK logit_norm={norm:.4f}", flush=True)
            elif name == "collocated":
                import gc

                import torch

                if model is not None:
                    del model
                    model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                stats = _stage_collocated(model_id)
                results["stages"][name] = {"ok": True, **stats}
                print(f"collocated OK {stats}", flush=True)
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            results["stages"][name] = {"ok": False, "traceback": tb}
            results["ok"] = False
            if not run_all_stages:
                break
    else:
        results["ok"] = all(
            s.get("ok") for s in results.get("stages", {}).values()  # type: ignore[union-attr]
        )

    if "ok" not in results:
        results["ok"] = all(
            s.get("ok") for s in results.get("stages", {}).values()  # type: ignore[union-attr]
        )

    print(f"\n=== SUMMARY ok={results['ok']} ===", flush=True)
    print(results, flush=True)
    return results


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
def smoke_flash_attn_h200(
    model_id: str = DEFAULT_MODEL,
    run_all_stages: bool = False,
) -> dict[str, object]:
    return _smoke_flash_attn_impl(model_id=model_id, run_all_stages=run_all_stages)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
def smoke_flash_attn_b200(
    model_id: str = DEFAULT_MODEL,
    run_all_stages: bool = False,
) -> dict[str, object]:
    return _smoke_flash_attn_impl(model_id=model_id, run_all_stages=run_all_stages)


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL,
    gpu_class: str = "h200",
    all_stages: bool = False,
) -> None:
    gpu = gpu_class.lower()
    if gpu == "h200":
        out = smoke_flash_attn_h200.remote(model_id=model_id, run_all_stages=all_stages)
    elif gpu == "b200":
        out = smoke_flash_attn_b200.remote(model_id=model_id, run_all_stages=all_stages)
    else:
        raise SystemExit("gpu_class must be h200 or b200")
    if not out.get("ok"):
        raise SystemExit(1)
