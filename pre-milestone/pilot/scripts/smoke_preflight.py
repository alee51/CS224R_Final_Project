#!/usr/bin/env python3
"""Local preflight for §6 smoke — no GPU required."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def main() -> int:
    print("=== Smoke preflight ===\n")
    failed = 0

    try:
        from pilot.infra.config_resolver import resolve_run_config
        from pilot.infra.modal_launch import launch_run

        cfg = resolve_run_config("smoke")
        assert cfg["objective"] == "grpo"
        if cfg["max_steps"] != 3:
            raise ValueError(f"max_steps={cfg['max_steps']!r} expected 3")
        assert cfg["budget_cap_usd"] == 10
        assert cfg["debug_max_prompts"] == 32
        assert cfg["max_new_tokens"] == 1536
        _ok("smoke.yaml resolves")
    except Exception as exc:
        _fail(f"config: {exc}")
        failed += 1

    try:
        from pilot.infra.execute import TRAINING_RUN_IDS

        if "smoke" not in TRAINING_RUN_IDS:
            raise ValueError("smoke not in TRAINING_RUN_IDS")
        _ok("execute.py accepts run_id=smoke")
    except Exception as exc:
        _fail(f"execute: {exc}")
        failed += 1

    lock = ROOT / "pilot" / "preflight_lock.json"
    if lock.exists():
        data = json.loads(lock.read_text())
        cap = data.get("budget_caps_usd", {}).get("smoke")
        if cap != 10:
            _fail(f"preflight_lock smoke cap={cap!r} expected 10")
            failed += 1
        else:
            _ok("preflight_lock smoke cap $10")
    else:
        _fail("missing preflight_lock.json")
        failed += 1

    wrapper = ROOT / "pilot" / "scripts" / "modal_run_pilot.sh"
    if wrapper.is_file():
        _ok("modal_run_pilot.sh exists")
    else:
        _fail("missing modal_run_pilot.sh")
        failed += 1

    for mod in (
        "pilot/train/hf_grpo_train.py",
        "pilot/infra/modal_app.py",
        "pilot/infra/execute.py",
    ):
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROOT / mod)],
            check=True,
            capture_output=True,
        )
    _ok("py_compile on core modules")

    try:
        r = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            check=True,
        )
        profile = r.stdout.strip()
        _ok(f"modal profile: {profile}")
    except Exception as exc:
        _fail(f"modal CLI: {exc}")
        failed += 1

    try:
        r = subprocess.run(
            ["modal", "secret", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        for name in ("huggingface", "wandb-api-key"):
            if name not in r.stdout:
                raise ValueError(f"secret {name!r} not listed")
        _ok("modal secrets: huggingface, wandb-api-key")
    except Exception as exc:
        _fail(f"modal secrets: {exc}")
        failed += 1

    data_path = ROOT / "pilot" / "data" / "dapo_slice_3k.jsonl"
    if data_path.is_file():
        _ok(f"train data: {data_path.name}")
    else:
        _fail(f"missing {data_path}")
        failed += 1

    print()
    if failed:
        print(f"Preflight FAILED ({failed} check(s))")
        return 1
    print("Preflight passed. Launch smoke:")
    print("  ./pilot/scripts/modal_run_pilot.sh --run-id smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
