#!/usr/bin/env python3
"""Preflight checks before `launch_train.sh` invokes Modal.

Validates config exists, required keys, and gpu_class matches --gpu-class.
Exit 0 on success, 1 with message on stderr otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from train.trainer import _load_cfg_dict, apply_arm_profile  # noqa: E402

_EXPECTED_GPU = {"h200": "H200", "b200": "B200"}
_REQUIRED_TRAIN = ("data_path", "batch_size", "n_rollouts", "total_steps")
_REQUIRED_ROLLOUT = ("model", "gpu_memory_utilization")


def _fail(msg: str) -> None:
    print(f"preflight_train_launch: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train launch preflight")
    p.add_argument("--gpu-class", required=True, choices=sorted(_EXPECTED_GPU))
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--arm", default="")
    p.add_argument("--mode", choices=("smoke", "full"), default="")
    args = p.parse_args()

    cfg_path = args.config
    if not cfg_path.is_file():
        _fail(f"config not found: {cfg_path}")

    if args.arm:
        os.environ["CS224R_ARM"] = args.arm

    try:
        raw = _load_cfg_dict(cfg_path.resolve())
    except ValueError as e:
        _fail(str(e))

    raw = apply_arm_profile(raw)
    yaml_gpu = str(raw.get("gpu_class", "")).strip()
    want = _EXPECTED_GPU[args.gpu_class.lower()]
    if yaml_gpu != want:
        _fail(
            f"gpu_class mismatch: --gpu-class {args.gpu_class} requires gpu_class: {want} "
            f"in config, but {cfg_path} has gpu_class: {yaml_gpu!r}. "
            f"Use train_real_b200.yaml for b200 or train_real.yaml for h200."
        )

    train = raw.get("train")
    if not isinstance(train, dict):
        _fail(f"missing or invalid train: block in {cfg_path}")

    for key in _REQUIRED_TRAIN:
        if key not in train:
            _fail(f"train.{key} missing in {cfg_path}")

    rollout = raw.get("rollout")
    if not isinstance(rollout, dict):
        _fail(f"missing or invalid rollout: block in {cfg_path}")

    for key in _REQUIRED_ROLLOUT:
        if key not in rollout:
            _fail(f"rollout.{key} missing in {cfg_path}")

    ws = raw.get("weight_sync")
    if not isinstance(ws, dict) or "every_n_steps" not in ws:
        _fail(f"weight_sync.every_n_steps missing in {cfg_path}")

    arm = raw.get("arm", "grpo")
    if args.arm and str(args.arm) != str(arm):
        _fail(f"--arm {args.arm} disagrees with merged config arm={arm}")

    print(
        f"preflight ok: gpu_class={yaml_gpu} arm={arm} "
        f"batch_size={train['batch_size']} weight_sync_every={ws['every_n_steps']}"
    )


if __name__ == "__main__":
    main()
