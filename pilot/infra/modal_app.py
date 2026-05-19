"""
Modal GPU entrypoint for pilot runs.

From repo root (with venv active):
  source .venv/bin/activate
  modal run pilot/infra/modal_app.py --run-id run0_proxy
  modal run pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 5
  modal run pilot/infra/modal_app.py --run-ids run1_grpo,run1b_grpo,run2_inverse_freq,run3_f_grpo

Artifacts persist on Modal Volume ``pilot-artifacts`` and are pulled to a new
timestamped folder ``pilot/artifacts/<run_id>/<UTC-timestamp>/`` each run;
``pilot/artifacts/<run_id>/latest`` symlinks to the most recent pull. HF weights
cache on ``hf-cache``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

from pilot.infra.modal_volumes import (
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_VOLUME_NAME,
    REMOTE_ARTIFACTS_ROOT,
    REMOTE_HF_CACHE_ROOT,
    pull_run_artifacts_from_volume,
)

# Local paths (evaluated on your machine when deploying)
_LOCAL_PILOT_DIR = Path(__file__).resolve().parents[1]
_LOCAL_REPO_ROOT = _LOCAL_PILOT_DIR.parent

# Remote paths (inside the container)
_REMOTE_REPO_ROOT = Path("/root")
_REMOTE_PILOT_DIR = Path("/root/pilot")
_REMOTE_ARTIFACTS = Path(REMOTE_ARTIFACTS_ROOT)

app = modal.App("cs224r-pilot")

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.51",
        "accelerate>=0.30",
        "sentencepiece",
        "safetensors",
        "pyyaml>=6.0",
        "huggingface_hub>=0.23",
    )
    # Exclude artifacts/ — mounted separately via pilot-artifacts Volume.
    .add_local_dir(
        str(_LOCAL_PILOT_DIR),
        remote_path="/root/pilot",
        ignore=["artifacts", "artifacts/**"],
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        REMOTE_ARTIFACTS_ROOT: artifacts_volume,
        REMOTE_HF_CACHE_ROOT: hf_cache_volume,
    },
)
def run_pilot_remote(config_json: str) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", REMOTE_HF_CACHE_ROOT)
    sys.path.insert(0, str(_REMOTE_REPO_ROOT))

    from pilot.infra.execute import execute_run

    config = json.loads(config_json)
    out = execute_run(
        config,
        repo_root=_REMOTE_REPO_ROOT,
        artifacts_root=_REMOTE_ARTIFACTS,
    )
    artifacts_volume.commit()

    metrics_path = out / "metrics.json"
    metrics: dict[str, Any] | None = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    return {"artifact_dir": str(out), "run_id": str(config["run_id"]), "metrics": metrics}


def _launch_and_pull_one(
    run_id: str,
    *,
    debug_max_prompts: int | None = None,
    artifacts_root: Path | None = None,
) -> Path:
    """Bootstrap locally, run on Modal, pull volume artifacts into a timestamped dir."""
    from pilot.infra.artifacts import (
        REQUIRED_ARTIFACTS,
        bootstrap_run_artifacts,
        link_latest_run,
        new_timestamped_run_dir,
    )
    from pilot.infra.config_resolver import resolve_run_config

    config = resolve_run_config(run_id)
    if debug_max_prompts is not None:
        config["debug_max_prompts"] = debug_max_prompts

    artifacts = artifacts_root or (_LOCAL_PILOT_DIR / "artifacts")
    local_run_dir = new_timestamped_run_dir(run_id, artifacts_root=artifacts)
    bootstrap_run_artifacts(
        config,
        repo_root=_LOCAL_REPO_ROOT,
        out_dir=local_run_dir,
    )

    print(f"Launching {run_id} on Modal A100...")
    print(f"  local artifact dir: {local_run_dir}")
    print(f"  artifacts volume: {ARTIFACTS_VOLUME_NAME}")
    print(f"  hf cache volume: {HF_CACHE_VOLUME_NAME}")
    result = run_pilot_remote.remote(json.dumps(config))
    print(f"Done. Remote: {result['artifact_dir']}")
    if result.get("metrics"):
        print(json.dumps(result["metrics"], indent=2))

    print(f"Pulling volume:{ARTIFACTS_VOLUME_NAME}/{run_id}/ -> {local_run_dir}")
    pull_run_artifacts_from_volume(run_id, local_run_dir)
    latest = link_latest_run(run_id, local_run_dir, artifacts_root=artifacts)
    print(f"Latest symlink: {latest} -> {local_run_dir.name}")

    missing = [name for name in REQUIRED_ARTIFACTS if not (local_run_dir / name).exists()]
    if missing:
        print(f"WARNING: missing after pull: {missing}")
    else:
        print(f"Local artifacts OK: {local_run_dir}")

    metrics_path = local_run_dir / "metrics.json"
    if metrics_path.exists():
        print(metrics_path.read_text())
    return local_run_dir


def _parse_run_ids(run_id: str, run_ids: str | None) -> list[str]:
    if run_ids:
        ids = [s.strip() for s in run_ids.split(",") if s.strip()]
        if not ids:
            raise ValueError("--run-ids must contain at least one run id")
        return ids
    return [run_id]


@app.local_entrypoint()
def main(
    run_id: str = "run0_proxy",
    run_ids: str | None = None,
    debug_max_prompts: int | None = None,
) -> None:
    ids = _parse_run_ids(run_id, run_ids)
    if len(ids) > 1:
        print(f"Overnight matrix: {len(ids)} runs (sequential in this process)")
        print("  Runs are independent — launch separate modal processes for parallelism.")

    for i, rid in enumerate(ids, start=1):
        if len(ids) > 1:
            print(f"\n=== [{i}/{len(ids)}] {rid} ===")
        _launch_and_pull_one(rid, debug_max_prompts=debug_max_prompts)
