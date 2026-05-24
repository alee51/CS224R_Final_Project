"""
Modal launch — orchestrator entry for GPU training runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from pilot.infra.artifacts import artifact_dir, bootstrap_run_artifacts
from pilot.infra.budget_guard import (
    hard_abort_usd,
    load_cap,
    record_cost,
    simulate_budget_check,
)
from pilot.infra.config_resolver import resolve_run_config

TrainFn = Callable[[dict[str, Any]], Path]

_train_fn: TrainFn | None = None

MODAL_GPU = "A100-80GB"


def register_train_fn(fn: TrainFn) -> None:
    global _train_fn
    _train_fn = fn


def _default_train_fn(config: dict[str, Any]) -> Path:
    from pilot.infra.execute import execute_run

    repo_root = Path(__file__).resolve().parents[2]
    return execute_run(
        config,
        repo_root=repo_root,
        artifacts_root=repo_root / "pilot" / "artifacts",
    )


def train_fn(config: dict[str, Any]) -> Path:
    fn = _train_fn or _default_train_fn
    return fn(config)


def _format_resolved_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def _run_on_modal(config: dict[str, Any], repo_root: Path) -> Path:
    from pilot.infra.modal_app import PilotRunner, app

    with app.run(detach=True):
        out = PilotRunner().run_pilot_remote.spawn(json.dumps(config)).get()
    if isinstance(out, dict):
        return Path(out["artifact_dir"])
    return Path(out)


def launch_run(
    run_id: str,
    *,
    dry_run: bool = False,
    lock_path: Path | None = None,
    configs_dir: Path | None = None,
    artifacts_root: Path | None = None,
    repo_root: Path | None = None,
    use_modal: bool = True,
) -> Path:
    config = resolve_run_config(run_id, configs_dir=configs_dir)
    cap = load_cap(run_id, lock_path)
    hard = hard_abort_usd(run_id, lock_path)
    price_per_sec = float(config.get("modal_price_per_sec", 0.000694))
    root = repo_root or Path(__file__).resolve().parents[2]

    if dry_run:
        print("=== launch_run DRY RUN ===")
        print(f"run_id: {run_id}")
        print(f"budget_cap_usd: {cap}")
        print(f"hard_abort_usd (1.5× cap): {hard}")
        print(f"modal_price_per_sec: {price_per_sec}")
        print(f"modal_gpu: {MODAL_GPU}")
        print("--- resolved config ---")
        print(_format_resolved_config(config))
        sim = simulate_budget_check(run_id, simulated_cost_usd=0.0, lock_path=lock_path)
        print("--- budget guard (simulated) ---")
        print(json.dumps(sim, indent=2))
        print(f"artifact_dir: {artifact_dir(run_id, artifacts_root=artifacts_root)}")
        print("\nTo run on Modal GPU (detached by default; survives laptop off):")
        print(f"  ./pilot/scripts/modal_run_pilot.sh --run-id {run_id}")
        print("\nInteractive (blocks + auto-pull; laptop must stay on):")
        print(f"  ./pilot/scripts/modal_run_pilot.sh --run-id {run_id} --wait")
        return artifact_dir(run_id, artifacts_root=artifacts_root)

    out = bootstrap_run_artifacts(
        config,
        artifacts_root=artifacts_root,
        repo_root=repo_root,
    )

    if use_modal:
        try:
            import modal  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Modal SDK not installed. pip install modal>=0.64 "
                "or pass --no-modal for local GPU (CUDA required)."
            ) from exc
        return _run_on_modal(config, root)

    result = train_fn(config)
    cost_path = result / "cost.json"
    if not cost_path.exists():
        record_cost(
            result,
            gpu_seconds=0.0,
            price_per_sec=price_per_sec,
            run_id=run_id,
            lock_path=lock_path,
        )
    return result
