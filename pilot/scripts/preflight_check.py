#!/usr/bin/env python3
"""
Preflight lock verification. Exit 0 only when all checks pass.
Does NOT launch GPU training.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "pilot"


def _fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"PREFLIGHT OK: {msg}")


def check_lock_file() -> dict:
    lock_path = PILOT / "preflight_lock.json"
    if not lock_path.exists():
        _fail("missing preflight_lock.json")
    lock = json.loads(lock_path.read_text())
    for key in ("model_id", "train", "pilot_eval", "paper_eval", "metrics_frozen", "gates_frozen", "budget_caps_usd"):
        if key not in lock:
            _fail(f"preflight_lock.json missing key: {key}")
    if lock["model_id"] != "Qwen/Qwen3-1.7B-Base":
        _fail(f"unexpected model_id: {lock['model_id']}")
    _ok("preflight_lock.json schema + model_id")
    return lock


def check_configs(lock: dict) -> None:
    configs = PILOT / "configs"
    shared = (configs / "shared_train.yaml").read_text()
    if "Qwen/Qwen3-1.7B-Base" not in shared:
        _fail("shared_train.yaml must set Qwen/Qwen3-1.7B-Base")
    for name in (
        "shared_train.yaml",
        "run0_proxy.yaml",
        "run1_grpo.yaml",
        "run1b_grpo.yaml",
        "run2_inverse_freq.yaml",
        "run3_f_grpo.yaml",
    ):
        if not (configs / name).exists():
            _fail(f"missing config {name}")
    _ok("run configs present")


def check_eval_module() -> None:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "pilot/eval/tests", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if tests.returncode != 0:
        _fail(f"eval tests failed:\n{tests.stdout}\n{tests.stderr}")
    _ok("eval metrics + bootstrap tests pass")


def check_data_slices(lock: dict) -> None:
    paths = [
        lock["train"]["path"],
        lock["pilot_eval"]["primary"],
        lock["pilot_eval"]["secondary"],
        lock["pilot_eval"]["sanity"],
    ]
    placeholders = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            _fail(f"dataset slice missing: {rel} — run materialize_data_slices.py")
        text = path.read_text(encoding="utf-8", errors="ignore")[:500]
        if "Placeholder" in text:
            placeholders.append(rel)
    if placeholders:
        _fail(
            "placeholder data detected (run `python pilot/scripts/materialize_data_slices.py` with HF access): "
            + ", ".join(placeholders)
        )
    for section, key in (("train", "sha256"), ("pilot_eval", "primary_sha256")):
        sha = lock[section][key] if section == "train" else lock[section][key]
        if sha.startswith("PLACEHOLDER"):
            _fail(f"{section} {key} not set — materialize real data first")
    _ok("tier-1 data slices exist and are non-placeholder")


def check_trainer_stub() -> None:
    if not (PILOT / "train" / "grpo_trainer.py").exists():
        _fail("missing pilot/train/grpo_trainer.py")
    _ok("trainer module present")


def check_infra_stub() -> None:
    if not (PILOT / "infra" / "modal_launch.py").exists():
        _fail("missing modal_launch.py")
    _ok("infra module present")


def main() -> None:
    import os

    print("=== Pilot preflight ===")
    lock = check_lock_file()
    check_configs(lock)
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    check_eval_module()
    check_data_slices(lock)
    check_trainer_stub()
    check_infra_stub()
    print("=== Preflight PASSED — orchestrator may schedule Run0 ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
