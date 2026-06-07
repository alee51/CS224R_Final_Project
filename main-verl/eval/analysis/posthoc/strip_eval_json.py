"""Strip large eval JSONs down to only the fields coverage.py needs.

Removes `rollouts` and `logprobs` from each per-prompt entry using streaming
ijson parsing, so multi-GB files never need to fully load into memory.

Usage:
  python main-verl/eval/analysis/posthoc/strip_eval_json.py path/to/eval.json
  # writes path/to/eval.compact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ijson

KEEP_PER_PROMPT = {"problem_id", "ground_truth", "n_correct", "rewards", "preds"}
TOP_SCALARS = {"label", "ckpt_path", "n_rollouts"}


def strip(src: Path, dst: Path) -> None:
    # Pass 1: top-level scalar fields (label, ckpt_path, n_rollouts).
    # These appear before the large `datasets` block so this is fast.
    top: dict = {}
    with src.open("rb") as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix in TOP_SCALARS and event in ("string", "number", "boolean"):
                top[prefix] = value
            if prefix == "datasets" and event == "start_map":
                break

    # Pass 2: stream datasets → per_prompt, dropping rollouts/logprobs.
    datasets_out: dict = {}
    with src.open("rb") as f:
        for ds_name, ds_obj in ijson.kvitems(f, "datasets"):
            pp_stripped = [
                {k: pp[k] for k in KEEP_PER_PROMPT if k in pp}
                for pp in ds_obj.get("per_prompt", [])
            ]
            ds_out = {k: v for k, v in ds_obj.items() if k != "per_prompt"}
            ds_out["per_prompt"] = pp_stripped
            datasets_out[ds_name] = ds_out

    top["datasets"] = datasets_out
    dst.write_text(json.dumps(top, default=float))
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", help="path to large eval_4b JSON")
    args = ap.parse_args()
    src = Path(args.eval_json)
    dst = src.with_suffix(".compact.json")
    if dst.exists():
        print(f"{dst} already exists, skipping")
        return
    print(f"stripping {src} ({src.stat().st_size // 1024 // 1024} MB) → {dst} ...")
    strip(src, dst)


if __name__ == "__main__":
    main()
