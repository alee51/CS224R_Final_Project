"""Stage 4 judge service — OpenAI-compatible chat completions on B200:1.

Deploy (from repo root):
  export CS224R_APP_NAME=cs224r-verl-stage04-judge
  ./main-verl/scripts/launch_judge_service.sh

POST body: OpenAI chat-completions shape with ``messages`` list.
The server applies the Instruct chat template before vLLM generation.
"""

from __future__ import annotations

import os
from typing import Any

import modal

from judge.modal_image import app_name, image

# Stage 3b (2026-05-30): swap judge model to Qwen3-4B-Instruct-2507 per poly_epo
# paper, bump context to 40960 (YaRN factor 1.25 over 32768 native), bump output
# cap to 4096 to fix the S4.6b truncated-JSON parse failures (22% parse at 2048).
# 40K headroom budget: 800 (system) + 1024 (problem) + 8 × 4096 (worst rollouts) +
# 4096 (output) ≈ 38.7K, ~2K spare. Prompts that overflow are skipped by
# clusters_judge.py (emits DEGENERATE_CLUSTER_ID), matching main/ Group A Phase 2.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
GPU_MEMORY_UTILIZATION = float(os.environ.get("JUDGE_GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN = int(os.environ.get("JUDGE_MAX_MODEL_LEN", "40960"))
DEFAULT_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

HF_CACHE_VOLUME_NAME = "hf-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

app = modal.App(app_name())

hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=image,
    gpu="B200:1",
    timeout=24 * 3600,
    scaledown_window=600,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
    max_containers=1,
)
class JudgeService:
    @modal.enter()
    def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)
        # Qwen3-4B-Instruct-2507 ships with 256K native context (YaRN baked into the
        # HF config), so no rope_scaling override needed for our 40960 budget.
        # Earlier attempt passed rope_scaling=... as a top-level LLM kwarg and crashed
        # with AttributeError in vllm/config.py:480 (vLLM 0.9.0 expects hf_overrides,
        # not a direct kwarg). Dropping the override entirely is simpler + correct.
        #
        # Stage 3b perf fix (2026-05-30): enforce_eager=False enables CUDA graphs,
        # giving ~3-4x decode speedup (S4.5 v2 measured 10.2s/call at eager; expected
        # ~3s/call with graphs). First call after restart pays ~30-90s for one-time
        # CUDA graph compilation; steady-state is much faster.
        self.llm = LLM(
            model=JUDGE_MODEL,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=False,
            trust_remote_code=True,
        )
        self.model_id = JUDGE_MODEL

    @modal.fastapi_endpoint(method="GET", label="health")
    def health(self) -> dict[str, str]:
        return {"status": "ok", "model": self.model_id}

    @modal.fastapi_endpoint(method="POST", label="v1-chat-completions")
    def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        from vllm import SamplingParams

        try:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return {"error": "messages required"}

            temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))
            max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = self.llm.generate([prompt], params)
            text = outputs[0].outputs[0].text

            return {
                "model": body.get("model", self.model_id),
                "choices": [{"message": {"role": "assistant", "content": text}}],
            }
        except Exception as exc:
            return {"error": str(exc), "choices": []}
