"""One-step minority_cot TRAINING smoke with judge artifacts on /vol/.

Uses verl.main_ppo + minority_cot_judge_trace_1p7b (real rollouts, real judge hook).
After training, prints batch diagnostics + prompt-0 trace from the artifacts volume
so you see what the judge actually received during training — not a sidecar probe.
"""

from __future__ import annotations

import json
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

STEP_LOG = f"{ARTIFACTS_MOUNT}/judge_train_step_log.jsonl"
TRACE_ARTIFACT = f"{ARTIFACTS_MOUNT}/judge_trace_training_prompt0.json"

_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
_JUDGE_AUTH_TOKEN = os.environ.get("JUDGE_AUTH_TOKEN", "")
_TRACE_PROMPT_IDX = os.environ.get("CS224R_JUDGE_TRACE_PROMPT_IDX", "0")

app = modal.App(app_name())

# Overlay train/ (and judge/) at deploy time so routing fixes land without a full
# image rebuild. Baked maxrl patches still need rebuild when core_algos changes.
image = (
    _base_image.add_local_dir(
        str(_MAIN_VERL_ROOT / "train"),
        remote_path="/root/main-verl/train",
    ).add_local_dir(
        str(_MAIN_VERL_ROOT / "judge"),
        remote_path="/root/main-verl/judge",
    )
)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

_RUNTIME_SECRET = modal.Secret.from_dict(
    {
        "JUDGE_BASE_URL": _JUDGE_BASE_URL,
        "JUDGE_AUTH_TOKEN": _JUDGE_AUTH_TOKEN,
        "CS224R_JUDGE_TRACE": "1",
        "CS224R_JUDGE_TRACE_PROMPT_IDX": _TRACE_PROMPT_IDX,
        "CS224R_JUDGE_TRACE_PATH": TRACE_ARTIFACT,
        "CS224R_JUDGE_STEP_LOG": STEP_LOG,
        "CS224R_JUDGE_TRACE_MAX_CHARS": os.environ.get(
            "CS224R_JUDGE_TRACE_MAX_CHARS", "8000"
        ),
    }
)


@app.function(
    image=image,
    gpu="B200:4",
    timeout=3 * 3600,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        _RUNTIME_SECRET,
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def minority_cot_judge_trace_smoke() -> None:
    import torch

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())

    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY

    assert "minority_cot" in ADV_ESTIMATOR_REGISTRY

    from omegaconf import OmegaConf
    from train.objective_minority import arm_block_from_adv_config

    algo_cfg = OmegaConf.create(
        {
            "minority_cot": {
                "cluster_source": "judge",
                "tokenizer_path": "Qwen/Qwen3-1.7B-Base",
            }
        }
    )
    mc = arm_block_from_adv_config(algo_cfg, "minority_cot")
    assert mc is not None and str(mc.cluster_source) == "judge", (
        "arm_block_from_adv_config must read minority_cot from algorithm-only config "
        "(verl passes config=self.config.algorithm to adv hooks)"
    )
    print("pre-flight: arm_block_from_adv_config(algorithm subtree) — OK")

    base_url = os.environ.get("JUDGE_BASE_URL", "")
    assert base_url, "JUDGE_BASE_URL required"
    health_url = os.environ.get("JUDGE_HEALTH_URL") or base_url.replace(
        "--v1-chat-completions.", "--health."
    )
    print(f"judge health: {health_url}")
    _wait_for_judge_health(health_url, timeout_s=180.0)

    # Fresh artifacts for this run (append-only log would otherwise mix runs).
    for p in (Path(STEP_LOG), Path(TRACE_ARTIFACT)):
        if p.exists():
            p.unlink()
            print(f"cleared stale {p}")

    print(
        "judge training trace:",
        f"TRACE={os.environ.get('CS224R_JUDGE_TRACE')}",
        f"prompt_idx={os.environ.get('CS224R_JUDGE_TRACE_PROMPT_IDX')}",
        f"STEP_LOG={STEP_LOG}",
        f"TRACE_ARTIFACT={TRACE_ARTIFACT}",
        "resume_mode=disable (v2 checkpoint dir — avoids instant skip)",
    )

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            "/root/main-verl/configs",
            "--config-name",
            "minority_cot_judge_trace_1p7b",
        ],
        check=True,
        env=os.environ.copy(),
    )

    artifacts_volume.commit()
    _print_training_judge_artifacts()


def _print_training_judge_artifacts() -> None:
    """Echo /vol/ artifacts on the driver process (always visible in Modal logs)."""
    log_path = Path(STEP_LOG)
    trace_path = Path(TRACE_ARTIFACT)

    print("\n" + "=" * 72)
    print("[judge-training-artifacts] post-step dump")
    print("=" * 72)

    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"JUDGE_STEP_LOG={log_path} lines={len(lines)}")
        for line in lines:
            rec = json.loads(line)
            print("JUDGE_STEP_RECORD:", json.dumps(rec, indent=2))
    else:
        print(f"MISSING step log: {log_path} — assign_judge_clusters did not run or env not propagated")

    if trace_path.is_file():
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        print(f"JUDGE_TRACE_ARTIFACT={trace_path} bytes={trace_path.stat().st_size}")
        meta = payload.get("meta", {})
        print("JUDGE_TRACE_META:", json.dumps(meta, indent=2))
        problem = payload.get("decoded_problem", "")
        print("JUDGE_TRACE_PROBLEM_PREVIEW:", problem[:500])
        if len(problem) > 500:
            print(f"... ({len(problem)} chars total)")
        print(
            "JUDGE_TRACE_CLUSTERS:",
            json.dumps(payload.get("final_cluster_ids"), indent=2),
        )
        parse = payload.get("judge_parse", {})
        print("JUDGE_TRACE_PARSE:", json.dumps(parse, indent=2))
        raw = payload.get("judge_raw_response") or ""
        print("JUDGE_TRACE_RAW_PREVIEW:", raw[:1500])
        if len(raw) > 1500:
            print(f"... ({len(raw)} chars total)")
    else:
        print(f"MISSING trace artifact: {trace_path}")


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
        raise SystemExit(
            "JUDGE_BASE_URL required. See launch_minority_cot_judge_trace.sh"
        )
    minority_cot_judge_trace_smoke.remote()
