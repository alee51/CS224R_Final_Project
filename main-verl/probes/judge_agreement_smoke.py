"""Stage 4 agreement spot-check — two runs each at temperature=0."""

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
    os.environ.get("CS224R_APP_NAME", "cs224r-verl-stage04-judge") + "-agreement"
)


@app.function(
    image=probe_image,
    timeout=3600,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
)
def judge_agreement_smoke(config_name: str = "judge_agreement_smoke") -> dict:
    cfg = load_smoke_config(config_name)
    client = judge_client_from_config(cfg)
    tasks = judge_tasks_from_config(cfg)
    gates = cfg.get("gates", {})

    t0 = time.perf_counter()
    run1 = client.cluster_batch_sync(tasks)
    run2 = client.cluster_batch_sync(tasks)
    wall_s = time.perf_counter() - t0

    parse_ok_both = 0
    agreement = 0
    cluster_100_hits = 0
    for a, b in zip(run1, run2):
        if a.parse_ok:
            cluster_100_hits += a.degenerate_count
        if a.parse_ok and b.parse_ok:
            parse_ok_both += 1
            if a.assignment == b.assignment:
                agreement += 1

    n = len(tasks)
    metrics = {
        "config_name": config_name,
        "n_tasks": n,
        "concurrency": cfg["judge"]["concurrency"],
        "wall_s": wall_s,
        "mean_wall_per_call_s": wall_s / max(n * 2, 1),
        "parse_ok_both_rate": parse_ok_both / n if n else 0.0,
        "assignment_agreement_rate": agreement / parse_ok_both if parse_ok_both else 0.0,
        "cluster_100_hits_total": cluster_100_hits,
        "parse_ok_both": parse_ok_both,
        "agreement": agreement,
        "parse_fail_run1": parse_failure_diagnostics(run1),
    }
    print(json.dumps(metrics, indent=2))

    parse_min = gates.get("parse_ok_both_rate_min")
    if parse_min is not None and metrics["parse_ok_both_rate"] < parse_min:
        raise RuntimeError(
            f"parse_ok_both_rate below {parse_min:.0%}: {metrics['parse_ok_both_rate']:.2%}"
        )
    agree_min = gates.get("assignment_agreement_rate_min")
    if parse_ok_both and agree_min is not None and metrics["assignment_agreement_rate"] < agree_min:
        raise RuntimeError(
            f"assignment_agreement_rate below {agree_min:.0%}: "
            f"{metrics['assignment_agreement_rate']:.2%}"
        )
    return metrics


@app.local_entrypoint()
def main(config_name: str = "judge_agreement_smoke") -> None:
    judge_agreement_smoke.remote(config_name)
