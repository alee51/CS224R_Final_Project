"""Resolve tier-1 / tier-2 eval paths from preflight_lock.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_lock(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "pilot" / "preflight_lock.json"
    return json.loads(path.read_text())


def pilot_eval_paths(lock: dict[str, Any] | None = None) -> dict[str, Path]:
    lock = lock or load_lock()
    pe = lock["pilot_eval"]
    return {
        "primary": ROOT / pe["primary"],
        "secondary": ROOT / pe["secondary"],
        "sanity": ROOT / pe["sanity"],
    }


def pilot_gate_splits(lock: dict[str, Any] | None = None) -> list[str]:
    lock = lock or load_lock()
    return list(lock["gates_frozen"]["gate_eval_splits"])


def paper_eval_paths(lock: dict[str, Any] | None = None) -> dict[str, Path]:
    lock = lock or load_lock()
    pe = lock["paper_eval"]
    return {k: ROOT / v for k, v in pe.items()}
