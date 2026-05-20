"""
Modal GPU entrypoint for pilot runs.

From repo root (with venv active):
  source .venv/bin/activate

Production (survives laptop disconnect — requires ``--detach`` + spawn default):
  modal run --detach pilot/infra/modal_app.py --run-id run0_proxy
  modal run --detach pilot/infra/modal_app.py --run-id run0_proxy --debug-max-prompts 5

Interactive (blocks until done, auto-pulls artifacts; laptop must stay on):
  modal run pilot/infra/modal_app.py --run-id run0_proxy --wait

Overnight matrix (one detached spawn per run — use ``launch_pilot_matrix.sh``):
  ./pilot/scripts/launch_pilot_matrix.sh

Artifacts persist on Modal Volume ``pilot-artifacts``. Default spawn mode does not
pull locally; after completion run ``pilot/scripts/pull_run_artifacts.py``.
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

from pilot.infra.artifacts import artifact_dir
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
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .pip_install(
        "torch>=2.2",
        "transformers>=4.51",
        "accelerate>=0.30",
        "sentencepiece",
        "safetensors",
        "pyyaml>=6.0",
        "huggingface_hub>=0.23",
        "wandb",
    )
    .pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")
    # Exclude artifacts/ — mounted separately via pilot-artifacts Volume.
    .add_local_dir(
        str(_LOCAL_PILOT_DIR),
        remote_path="/root/pilot",
        ignore=["artifacts", "artifacts/**"],
    )
)

_PILOT_FUNCTION_KWARGS = dict(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb-api-key"),
    ],
    volumes={
        REMOTE_ARTIFACTS_ROOT: artifacts_volume,
        REMOTE_HF_CACHE_ROOT: hf_cache_volume,
    },
)


@app.cls(**_PILOT_FUNCTION_KWARGS)
class PilotRunner:
    """Modal GPU runner with preempt-time artifact flush via @modal.exit."""

    @modal.method()
    def run_pilot_remote(self, config_json: str) -> dict[str, Any]:
        os.environ.setdefault("HF_HOME", REMOTE_HF_CACHE_ROOT)
        sys.path.insert(0, str(_REMOTE_REPO_ROOT))

        from pilot.infra.execute import execute_run

        config = json.loads(config_json)
        run_id = str(config["run_id"])
        run_dir = artifact_dir(run_id, artifacts_root=_REMOTE_ARTIFACTS)
        state_path = run_dir / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print(
                f"Resuming {run_id} from training_state.json step={state.get('step')} "
                f"-> resume at step {int(state['step']) + 1}",
                flush=True,
            )

        had_error = False
        try:
            out = execute_run(
                config,
                repo_root=_REMOTE_REPO_ROOT,
                artifacts_root=_REMOTE_ARTIFACTS,
            )
        except Exception:
            had_error = True
            raise
        finally:
            try:
                artifacts_volume.commit()
            except Exception as exc:
                if had_error:
                    print(
                        f"WARNING: artifacts volume commit failed after run error: {exc}",
                        file=sys.stderr,
                    )
                else:
                    raise

        metrics_path = out / "metrics.json"
        metrics: dict[str, Any] | None = None
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        return {"artifact_dir": str(out), "run_id": run_id, "metrics": metrics}

    @modal.exit()
    def flush_artifacts_on_exit(self) -> None:
        try:
            artifacts_volume.commit()
        except Exception as exc:
            print(
                f"WARNING: @modal.exit artifacts_volume.commit failed: {exc}",
                file=sys.stderr,
            )


# Backward-compatible remote entry (Modal Method on PilotRunner).
run_pilot_remote = PilotRunner.run_pilot_remote


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
    result: dict[str, Any] | None = None
    remote_exc: Exception | None = None
    pull_exc: Exception | None = None
    try:
        result = PilotRunner().run_pilot_remote.remote(json.dumps(config))
        print(f"Done. Remote: {result['artifact_dir']}")
        if result.get("metrics"):
            print(json.dumps(result["metrics"], indent=2))
    except Exception as exc:
        remote_exc = exc
        print(f"Remote run failed for {run_id}: {exc}")
        print("Attempting artifact recovery pull from Modal volume...")
    finally:
        print(f"Pulling volume:{ARTIFACTS_VOLUME_NAME}/{run_id}/ -> {local_run_dir}")
        try:
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
        except Exception as exc:
            pull_exc = exc
            print(f"WARNING: artifact pull failed for {run_id}: {exc}")

    if remote_exc is not None:
        msg = f"Remote run failed for {run_id} after pull attempt."
        if pull_exc is not None:
            msg += f" Artifact pull also failed: {pull_exc}"
        raise RuntimeError(msg) from remote_exc
    if pull_exc is not None:
        raise RuntimeError(f"Artifact pull failed for successful remote run {run_id}: {pull_exc}") from pull_exc

    return local_run_dir


def _spawn_only_one(
    run_id: str,
    *,
    debug_max_prompts: int | None = None,
    artifacts_root: Path | None = None,
) -> tuple[Path, str]:
    """Bootstrap locally and spawn detached remote execution (no local pull)."""
    from pilot.infra.artifacts import bootstrap_run_artifacts, new_timestamped_run_dir
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

    print(f"Spawning {run_id} on Modal A100 (detached spawn — survives laptop off)...")
    print(f"  local artifact dir: {local_run_dir}")
    print(f"  artifacts volume: {ARTIFACTS_VOLUME_NAME}")
    print(f"  hf cache volume: {HF_CACHE_VOLUME_NAME}")
    print(
        "  NOTE: launch with `modal run --detach ...` so the ephemeral app stays up "
        "after this process exits."
    )
    function_call = PilotRunner().run_pilot_remote.spawn(json.dumps(config))
    call_id = str(getattr(function_call, "object_id", "unknown"))
    print(f"Spawned function call id: {call_id}")
    print("Monitor: `modal app list` and `modal app logs <app-id>` (dashboard URL above).")
    print(
        f"After completion, pull artifacts:\n"
        f"  python pilot/scripts/pull_run_artifacts.py --run-id {run_id} "
        f"--local-dir {local_run_dir}"
    )
    return local_run_dir, call_id


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
    wait: bool = False,
) -> None:
    """Default: detached spawn (use ``modal run --detach``). Pass ``--wait`` to block and pull."""
    ids = _parse_run_ids(run_id, run_ids)
    if len(ids) > 1 and wait:
        print(
            "WARNING: --wait with multiple --run-ids runs sequentially and blocks on .remote(). "
            "For overnight matrix use ./pilot/scripts/launch_pilot_matrix.sh instead."
        )
    if len(ids) > 1:
        mode = "wait+pull" if wait else "detached spawn (one process per run recommended)"
        print(f"Overnight matrix: {len(ids)} runs ({mode})")

    for i, rid in enumerate(ids, start=1):
        if len(ids) > 1:
            print(f"\n=== [{i}/{len(ids)}] {rid} ===")
        if wait:
            _launch_and_pull_one(rid, debug_max_prompts=debug_max_prompts)
        else:
            _spawn_only_one(rid, debug_max_prompts=debug_max_prompts)
