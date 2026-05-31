"""Fast judge cluster trace — same ``assign_judge_clusters`` path, no VeRL trainer.

Why this exists: full 1-step minority_cot smoke (~20 min) runs clustering inside Ray
workers; Modal log streaming often drops worker stdout, so ``[judge-trace]`` never
appears even when training succeeds.

This probe (~5–8 min on B200:1):
  1. Load one Polaris row from the same parquet the trainer uses.
  2. Generate 8 rollouts with vLLM (1.7B actor, same as smoke yaml).
  3. Left-pad prompt token IDs like verl (``max_prompt_length=1024``).
  4. Call ``assign_judge_clusters`` in the **main Modal process** (no Ray).
  5. Write ``/vol/artifacts/judge_trace_prompt0.json`` + print ``JUDGE_TRACE_ARTIFACT=``.

Prereqs: judge deployed, ``JUDGE_BASE_URL`` set locally before launch.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
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

POLARIS_TRAIN = f"{ARTIFACTS_MOUNT}/data/main-verl/polaris_train.parquet"
_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
_JUDGE_AUTH_TOKEN = os.environ.get("JUDGE_AUTH_TOKEN", "")
_PROMPT_ROW = int(os.environ.get("CS224R_JUDGE_TRACE_PROMPT_IDX", "0"))
DECODER_MODEL = os.environ.get(
    "CS224R_TRACE_ACTOR_MODEL", "Qwen/Qwen3-1.7B-Base"
)
_ACTOR_SLUG = DECODER_MODEL.rsplit("/", 1)[-1].lower().replace(".", "_")
TRACE_ARTIFACT = (
    f"{ARTIFACTS_MOUNT}/judge_trace_prompt{_PROMPT_ROW}_{_ACTOR_SLUG}.json"
)
JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_PROMPT_LENGTH = 1024
N_ROLLOUTS = 8
# Shorter than training (4096) for speed; increase if you need full-length CoTs.
MAX_ROLLOUT_TOKENS = int(os.environ.get("TRACE_MAX_ROLLOUT_TOKENS", "2048"))

app = modal.App(app_name() + "-judge-trace-fast")

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

_RUNTIME_SECRET = modal.Secret.from_dict(
    {
        "JUDGE_BASE_URL": _JUDGE_BASE_URL,
        "JUDGE_AUTH_TOKEN": _JUDGE_AUTH_TOKEN,
        "CS224R_TRACE_ACTOR_MODEL": DECODER_MODEL,
        "CS224R_JUDGE_TRACE": "1",
        "CS224R_JUDGE_TRACE_PROMPT_IDX": str(_PROMPT_ROW),
        "CS224R_JUDGE_TRACE_PATH": TRACE_ARTIFACT,
        "CS224R_JUDGE_TRACE_MAX_CHARS": os.environ.get("CS224R_JUDGE_TRACE_MAX_CHARS", "0"),
    }
)


def _prompt_text(prompt_field) -> str:
    import numpy as np

    if isinstance(prompt_field, np.ndarray):
        prompt_field = prompt_field.tolist()
    if isinstance(prompt_field, (list, tuple)) and prompt_field:
        first = prompt_field[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(prompt_field)


def _left_pad_token_ids(ids: list[int], *, pad_id: int, width: int) -> list[int]:
    if len(ids) > width:
        return ids[-width:]
    return [pad_id] * (width - len(ids)) + ids


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    secrets=[modal.Secret.from_name("HUGGINGFACE"), _RUNTIME_SECRET],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def judge_cluster_trace_fast() -> None:
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from train.clusters_judge import assign_judge_clusters, build_judge_client_from_env

    base_url = os.environ.get("JUDGE_BASE_URL", "")
    assert base_url, "JUDGE_BASE_URL required"
    health_url = os.environ.get("JUDGE_HEALTH_URL") or base_url.replace(
        "--v1-chat-completions.", "--health."
    )
    _wait_for_judge_health(health_url, timeout_s=180.0)
    print("judge health OK")

    df = pd.read_parquet(POLARIS_TRAIN)
    row_idx = int(os.environ.get("CS224R_JUDGE_TRACE_PROMPT_IDX", "0"))
    row = df.iloc[row_idx]
    prompt = _prompt_text(row["prompt"])
    problem_id = row.get("uid", row_idx)
    print(f"polaris row={row_idx} problem_id={problem_id} prompt_chars={len(prompt)}")

    decoder_model = os.environ.get(
        "CS224R_TRACE_ACTOR_MODEL", "Qwen/Qwen3-1.7B-Base"
    )
    trace_path = os.environ.get("CS224R_JUDGE_TRACE_PATH", TRACE_ARTIFACT)
    print(f"actor={decoder_model} trace_artifact={trace_path}")

    tok = AutoTokenizer.from_pretrained(decoder_model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_ids = _left_pad_token_ids(
        tok.encode(prompt, add_special_tokens=False),
        pad_id=pad_id,
        width=MAX_PROMPT_LENGTH,
    )

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    t0 = time.monotonic()
    llm = LLM(
        model=decoder_model,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_PROMPT_LENGTH + MAX_ROLLOUT_TOKENS,
    )
    print(f"vLLM loaded in {time.monotonic() - t0:.1f}s")

    t1 = time.monotonic()
    outputs = llm.generate(
        [prompt],
        SamplingParams(
            n=N_ROLLOUTS,
            temperature=1.0,
            max_tokens=MAX_ROLLOUT_TOKENS,
        ),
    )
    rollouts = [o.text for o in outputs[0].outputs]
    rollout_token_ids = [
        [tok.encode(text, add_special_tokens=False) for text in rollouts]
    ]
    print(
        f"generated {len(rollouts)} rollouts in {time.monotonic() - t1:.1f}s "
        f"(max_tok={MAX_ROLLOUT_TOKENS})"
    )

    judge_client = build_judge_client_from_env(judge_model=JUDGE_MODEL)
    t2 = time.monotonic()
    out = assign_judge_clusters(
        problem_ids=[problem_id],
        n_rollouts=N_ROLLOUTS,
        rollout_token_ids=rollout_token_ids,
        prompt_token_ids=[prompt_ids],
        decoder_tokenizer_path=decoder_model,
        judge_tokenizer_path=JUDGE_MODEL,
        judge_client=judge_client,
        judge_max_input_tokens=36864,
    )
    print(f"assign_judge_clusters wall {time.monotonic() - t2:.1f}s")
    print("diagnostics:", out.diagnostics)

    artifact = Path(trace_path)
    if artifact.is_file():
        print("--- artifact preview (first 4000 chars) ---")
        text = artifact.read_text(encoding="utf-8")
        print(text[:4000])
        if len(text) > 4000:
            print(f"... [{len(text)} chars total, see {artifact}]")
    else:
        print("WARNING: trace artifact missing:", trace_path)


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
    raise RuntimeError(f"judge health failed: {last_err}")


@app.local_entrypoint()
def main() -> None:
    if not _JUDGE_BASE_URL:
        raise SystemExit("Set JUDGE_BASE_URL before launch (see launch script).")
    judge_cluster_trace_fast.remote()
