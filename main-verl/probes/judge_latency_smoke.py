"""Stage 4 latency smoke — concurrent fan-out against live judge."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from probes.judge_smoke_lib import (
    judge_client_from_config,
    judge_tasks_from_config,
    load_smoke_config,
    main_verl_dir,
    parse_failure_diagnostics,
)

_MAIN_VERL = main_verl_dir()

probe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("httpx>=0.27", "pyyaml>=6.0")
    .env({"PYTHONPATH": "/root/main-verl"})
    .add_local_dir(
        str(_MAIN_VERL / "judge"),
        remote_path="/root/main-verl/judge",
        copy=True,
    )
    .add_local_dir(
        str(_MAIN_VERL / "probes"),
        remote_path="/root/main-verl/probes",
        copy=True,
    )
    .add_local_dir(
        str(_MAIN_VERL / "configs"),
        remote_path="/root/main-verl/configs",
        copy=True,
    )
)

app = modal.App(
    os.environ.get("CS224R_APP_NAME", "cs224r-verl-stage04-judge") + "-latency"
)


@app.function(image=probe_image, timeout=3600)
def judge_latency_smoke(config_name: str = "judge_latency_smoke") -> dict:
    cfg = load_smoke_config(config_name)
    client = judge_client_from_config(cfg)
    tasks = judge_tasks_from_config(cfg)
    gates = cfg.get("gates", {})

    t0 = time.perf_counter()
    results = client.cluster_batch_sync(tasks)
    wall_s = time.perf_counter() - t0

    n = len(tasks)
    concurrency = int(cfg["judge"]["concurrency"])
    per_call = wall_s / max(n, 1)
    parse_ok_rate = sum(1 for r in results if r.parse_ok) / n if n else 0.0
    metrics = {
        "config_name": config_name,
        "n_tasks": n,
        "concurrency": concurrency,
        "wall_s": wall_s,
        "mean_wall_per_task_s": per_call,
        "serial_equiv_per_call_s": wall_s / max(n, 1) * concurrency,
        "parse_ok_rate": parse_ok_rate,
        "median_est_s": per_call,
        "p95_est_s": per_call * 1.5,
        "parse_fail": parse_failure_diagnostics(results),
    }
    print(json.dumps(metrics, indent=2))

    latency_max = gates.get("mean_wall_per_task_s_max")
    if latency_max is not None and metrics["mean_wall_per_task_s"] > latency_max:
        raise RuntimeError(
            f"mean wall per task >{latency_max}s: {metrics['mean_wall_per_task_s']:.2f}"
        )
    parse_min = gates.get("parse_ok_rate_min")
    if parse_min is not None and parse_ok_rate < parse_min:
        raise RuntimeError(f"parse_ok_rate below {parse_min:.0%}: {parse_ok_rate:.2%}")
    return metrics


@app.local_entrypoint()
def main(config_name: str = "judge_latency_smoke") -> None:
    judge_latency_smoke.remote(config_name)
