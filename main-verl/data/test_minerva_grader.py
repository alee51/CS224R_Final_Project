"""Test Minerva parquet grader compatibility on Modal.

This script runs on Modal with verl available and tests:
1. Schema validation
2. Grader self-equality (gold-through-grader = 1.0)
3. Failure analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal
import pandas as pd
from infra.modal_image import app_name, image as _base_image
from infra.modal_volume import ARTIFACTS_MOUNT, ARTIFACTS_VOLUME_NAME

_PARQUET_PATH = "/vol/data/main-verl/minerva.parquet"
_OUTPUT_PATH = "/vol/probes/minerva_grader_test.json"

app = modal.App(app_name())

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

image = _base_image


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
)
def test_minerva_grader() -> dict:
    """Test Minerva parquet grader compatibility."""
    import json

    print(f"[test] loading {_PARQUET_PATH}")
    df = pd.read_parquet(_PARQUET_PATH)
    print(f"[test] loaded {len(df)} rows")

    # Import grader
    from verl.utils.reward_score.math import compute_score

    # Test: gold-through-grader should yield score=1.0
    print("[test] Testing grader self-equality...")
    failures = []
    passes = 0

    for i, row in df.iterrows():
        gold = row["reward_model"]["ground_truth"]

        try:
            # Gold answer through grader should be a perfect match
            score = compute_score(f"\\boxed{{{gold}}}", gold)
            if score > 0.5:  # Grader returns 1.0 for match, 0.0 for non-match
                passes += 1
            else:
                failures.append({
                    "row": i,
                    "problem_id": row["extra_info"]["problem_id"],
                    "gold_answer": gold[:100],  # Truncate long answers
                    "score": float(score),
                    "reason": "score < 0.5",
                })
        except Exception as e:
            failures.append({
                "row": i,
                "problem_id": row["extra_info"]["problem_id"],
                "gold_answer": gold[:100],
                "score": None,
                "reason": str(e)[:100],
            })

    pass_rate = passes / len(df)
    print(f"[test] Pass rate: {passes}/{len(df)} ({100*pass_rate:.1f}%)")

    if failures:
        print(f"[test] First 10 failures:")
        for f in failures[:10]:
            print(f"  {f}")

    result = {
        "dataset": "minerva",
        "n_total": len(df),
        "n_passes": passes,
        "pass_rate": float(pass_rate),
        "failures": failures,
    }

    # Write result
    output_path = Path(_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[test] wrote {_OUTPUT_PATH}")
    artifacts_volume.commit()

    return result


@app.local_entrypoint()
def main() -> None:
    result = test_minerva_grader.remote()
    print(f"\n=== Result ===")
    print(f"Pass rate: {result['n_passes']}/{result['n_total']} ({100*result['pass_rate']:.1f}%)")
    if result['failures']:
        print(f"Sample failures: {result['failures'][:3]}")
