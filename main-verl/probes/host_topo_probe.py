"""Tiny one-shot probe: request B200:4, dump PCIe bus IDs + SM clocks, exit.

Used to sample what Modal scheduling pool a given account is drawing from
without paying for model load / training startup. Bare-metal HGX shows
single PCIe domain `0000:` with widely-spaced bus numbers. Virtualized
passthrough shows multiple domains (e.g. `0002:` + `0003:`) with sequential
`:01/:02/:03/:04` device numbers.

Run with CS224R_PROBE_TAG to disambiguate parallel attempts:

    CS224R_APP_NAME=host-topo-probe-A \\
    modal run --detach main-verl/probes/host_topo_probe.py

(Repeat with -B, -C app names to spin up multiple in parallel.)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal

from infra.modal_image import app_name, image as _base_image

app = modal.App(app_name())


@app.function(
    image=_base_image,
    gpu="B200:4",
    timeout=600,
)
def topo_probe() -> None:
    print(f"=== topo_probe start app={app_name()} t={time.time()} ===")
    subprocess.run(
        ["nvidia-smi", "-q", "-d", "TEMPERATURE,POWER,CLOCK"],
        check=False,
    )
    print("=== query-gpu ===")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,clocks.sm,clocks.max.sm,"
            "power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active",
            "--format=csv",
        ],
        check=False,
    )
    print("=== nvlink ===")
    subprocess.run(["nvidia-smi", "nvlink", "-s"], check=False)
    print(f"=== topo_probe end app={app_name()} ===")


@app.local_entrypoint()
def main() -> None:
    print(f"launch topo_probe app={app_name()}")
    topo_probe.remote()
