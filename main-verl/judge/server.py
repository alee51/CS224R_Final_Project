"""Stage 4 judge service — OpenAI-compatible chat completions on B200:1.

Deploy (from repo root):
  export CS224R_APP_NAME=cs224r-verl-stage04-judge
  ./main-verl/scripts/launch_judge_service.sh

POST body (single prompt, backward compatible):
  {"messages": [...], "temperature": 0, "max_tokens": 4096}

POST body (vLLM batch — multiple independent prompts in one generate() call):
  {"requests": [{"messages": [...]}, ...], "temperature": 0, "max_tokens": 4096}

Batch response:
  {"model": "...", "results": [{"choices": [{"message": {"content": "..."}}]}, ...]}
"""

from __future__ import annotations

import os
from typing import Any

import modal

from judge.modal_image import app_name, image

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
GPU_MEMORY_UTILIZATION = float(os.environ.get("JUDGE_GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN = int(os.environ.get("JUDGE_MAX_MODEL_LEN", "40960"))
DEFAULT_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
MAX_BATCH_SIZE = int(os.environ.get("JUDGE_MAX_BATCH_SIZE", "128"))

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
    max_containers=2,
)
class JudgeService:
    @modal.enter()
    def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)
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

    def _sampling_params(self, body: dict[str, Any]):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))
        max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
        kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        guided_json = body.get("guided_json")
        if guided_json is not None:
            kwargs["guided_decoding"] = GuidedDecodingParams(json=guided_json)
        return SamplingParams(**kwargs)

    def _prompt_from_messages(self, messages: list[dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _single_result(self, text: str) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    @modal.fastapi_endpoint(method="POST", label="v1-chat-completions")
    def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            params = self._sampling_params(body)
            model_name = body.get("model", self.model_id)

            requests = body.get("requests")
            if requests is not None:
                if not isinstance(requests, list) or not requests:
                    return {"error": "requests must be a non-empty list", "results": []}
                if len(requests) > MAX_BATCH_SIZE:
                    return {
                        "error": f"batch size {len(requests)} exceeds max {MAX_BATCH_SIZE}",
                        "results": [],
                    }
                prompts: list[str] = []
                for req in requests:
                    messages = req.get("messages") if isinstance(req, dict) else None
                    if not isinstance(messages, list) or not messages:
                        return {"error": "each request requires messages[]", "results": []}
                    prompts.append(self._prompt_from_messages(messages))
                outputs = self.llm.generate(prompts, params)
                results = [
                    self._single_result(out.outputs[0].text) for out in outputs
                ]
                return {"model": model_name, "results": results}

            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return {"error": "messages required"}

            prompt = self._prompt_from_messages(messages)
            outputs = self.llm.generate([prompt], params)
            text = outputs[0].outputs[0].text
            return {
                "model": model_name,
                "choices": [{"message": {"role": "assistant", "content": text}}],
            }
        except Exception as exc:
            return {"error": str(exc), "choices": [], "results": []}
