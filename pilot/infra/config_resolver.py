"""Resolve frozen run configs (merge shared_train.yaml inheritance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PILOT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PILOT_ROOT / "configs"

RUN_CONFIG_FILES: dict[str, str] = {
    "run0_proxy": "run0_proxy.yaml",
    "run1_grpo": "run1_grpo.yaml",
    "run1b_grpo": "run1b_grpo.yaml",
    "run2_inverse_freq": "run2_inverse_freq.yaml",
    "run3_f_grpo": "run3_f_grpo.yaml",
    "smoke": "smoke.yaml",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_run_config(
    run_id: str,
    *,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    configs_dir = configs_dir or CONFIGS_DIR
    if run_id not in RUN_CONFIG_FILES:
        known = ", ".join(sorted(RUN_CONFIG_FILES))
        raise KeyError(f"Unknown run_id {run_id!r}; expected one of: {known}")

    run_path = configs_dir / RUN_CONFIG_FILES[run_id]
    if not run_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_path}")

    run_cfg = _load_yaml(run_path)
    inherits = run_cfg.get("inherits")
    if inherits:
        shared_path = configs_dir / str(inherits)
        if not shared_path.exists():
            raise FileNotFoundError(f"Missing inherited config: {shared_path}")
        shared_cfg = _load_yaml(shared_path)
        merged = _deep_merge(shared_cfg, run_cfg)
    else:
        merged = dict(run_cfg)

    merged["run_id"] = run_id
    return merged
