"""Abort runs that exceed per-run USD cap from preflight_lock.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PILOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = PILOT_ROOT / "preflight_lock.json"
ABORT_MULTIPLIER = 1.5


def load_lock(lock_path: Path | None = None) -> dict[str, Any]:
    lock_path = lock_path or DEFAULT_LOCK
    return json.loads(lock_path.read_text())


def load_cap(run_id: str, lock_path: Path | None = None) -> float:
    lock = load_lock(lock_path)
    caps = lock.get("budget_caps_usd", {})
    if run_id not in caps:
        raise KeyError(f"No budget cap for run_id {run_id!r} in {lock_path or DEFAULT_LOCK}")
    return float(caps[run_id])


def hard_abort_usd(run_id: str, lock_path: Path | None = None) -> float:
    return ABORT_MULTIPLIER * load_cap(run_id, lock_path)


def estimate_usd(gpu_seconds: float, price_per_sec: float) -> float:
    return gpu_seconds * price_per_sec


def check_cost(run_id: str, cost_usd: float, lock_path: Path | None = None) -> None:
    cap = load_cap(run_id, lock_path)
    if cost_usd > ABORT_MULTIPLIER * cap:
        raise RuntimeError(
            f"Budget abort: {run_id} cost ${cost_usd:.2f} exceeds "
            f"{ABORT_MULTIPLIER}× cap ${cap:.2f} (hard limit ${ABORT_MULTIPLIER * cap:.2f})"
        )


def record_cost(
    artifact_dir: Path,
    *,
    gpu_seconds: float,
    price_per_sec: float,
    run_id: str,
    lock_path: Path | None = None,
) -> dict[str, float]:
    """Write cost.json and enforce budget guard."""
    cost_usd = estimate_usd(gpu_seconds, price_per_sec)
    check_cost(run_id, cost_usd, lock_path)
    payload = {"gpu_seconds": gpu_seconds, "estimated_usd": round(cost_usd, 4)}
    (artifact_dir / "cost.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def simulate_budget_check(
    run_id: str,
    *,
    simulated_cost_usd: float = 0.0,
    lock_path: Path | None = None,
) -> dict[str, float]:
    """Dry-run helper: report cap, hard limit, and whether simulated cost would abort."""
    cap = load_cap(run_id, lock_path)
    hard = hard_abort_usd(run_id, lock_path)
    return {
        "cap_usd": cap,
        "hard_abort_usd": hard,
        "simulated_cost_usd": simulated_cost_usd,
        "would_abort": simulated_cost_usd > hard,
    }
