"""Stage 5 S5.1: in-container registry assert for poly_epo_cot (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal

from infra.modal_image import app_name, image

app = modal.App(app_name())


@app.function(image=image, gpu="B200:1", timeout=1800)
def poly_epo_registry_check() -> None:
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, AdvantageEstimator

    assert "poly_epo_cot" in ADV_ESTIMATOR_REGISTRY, (
        "poly_epo_cot not in ADV_ESTIMATOR_REGISTRY — S5.1 patches missing from image"
    )
    assert AdvantageEstimator.POLY_EPO_COT.value == "poly_epo_cot"
    print("pre-flight: poly_epo_cot registered — OK")
    print("registry keys (sample):", sorted(ADV_ESTIMATOR_REGISTRY.keys())[-8:])


@app.local_entrypoint()
def main() -> None:
    poly_epo_registry_check.remote()
