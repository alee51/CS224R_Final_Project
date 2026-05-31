"""S5.1 in-container registry assert: poly_epo_cot registered after image rebuild."""

from __future__ import annotations

import sys
from pathlib import Path

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal

from infra.modal_image import app_name, image

app = modal.App(app_name())


@app.function(image=image, timeout=600)
def poly_epo_cot_registry_assert() -> None:
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, AdvantageEstimator

    assert "poly_epo_cot" in ADV_ESTIMATOR_REGISTRY, (
        "poly_epo_cot estimator not registered — S5.1 patch did not apply."
    )
    assert AdvantageEstimator.POLY_EPO_COT.value == "poly_epo_cot"
    print("registry assert OK: poly_epo_cot in ADV_ESTIMATOR_REGISTRY")


@app.local_entrypoint()
def main() -> None:
    poly_epo_cot_registry_assert.remote()
