"""Potential@k: of problems failing within the first k rollouts, what
fraction are solvable if you keep sampling out to the full n rollouts.

For each problem and each k in {1, 4, 8, 16, 32}:
  - The problem is "failed at k" if all of the first k rewards are 0 (i.e.,
    the model never produced a passing rollout within budget k).
  - It is "ultimately solvable" if `n_correct` over all n rollouts > 0.
  - Potential@k = #(failed_at_k AND ultimately_solvable) / #(failed_at_k).

This measures recoverable upside from increasing the rollout budget — high
Potential@k means failures are budget-bound, low Potential@k means they are
quality-bound. Cross-arm comparison shows which method's failure mode is
"more budget" vs "fundamentally stuck".

Usage:
    python main-verl/eval/analysis/potential_at_k.py /vol/probes/eval_4b/*.json
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_io import collect, collected_from_json, write_markdown  # noqa: E402

K_VALUES = [1, 4, 8, 16, 32]


def potential_at_k(per_prompt, k):
    failed = 0
    recoverable = 0
    for p in per_prompt:
        if k > len(p["rewards"]):
            continue
        first_k = p["rewards"][:k]
        if any(r > 0.5 for r in first_k):
            continue  # already solved within k
        failed += 1
        if p["n_correct"] > 0:
            recoverable += 1
    if failed == 0:
        return 0.0, failed, recoverable
    return recoverable / failed, failed, recoverable


def _render(data: dict) -> str:
    if not data:
        return "# Potential@k\n\nNo input data.\n"
    arms = sorted({a for (a, _) in data})
    datasets = sorted({d for (_, d) in data})
    lines = ["# Potential@k", "",
             "For each (arm, dataset, k): fraction of problems that failed in",
             "the first k rollouts but were solved at least once across all",
             "n rollouts. Higher means more recoverable failures (budget-bound).",
             ""]
    for ds_name in datasets:
        lines.append(f"## {ds_name}")
        lines.append("")
        lines.append("| arm | " + " | ".join(f"pot@{k}" for k in K_VALUES) + " |")
        lines.append("|---|" + "---|" * len(K_VALUES))
        for arm in arms:
            ds = data.get((arm, ds_name))
            if ds is None:
                continue
            row = [arm]
            for k in K_VALUES:
                rate, _failed, _rec = potential_at_k(ds["per_prompt"], k)
                row.append(f"{rate:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def analyze(json_data: dict) -> str:
    return _render(collected_from_json(json_data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="potential_at_k.md")
    args = ap.parse_args()
    data = collect(args.paths)
    if not data:
        print("[potential_at_k] no inputs found")
        return
    md = _render(data)
    out = write_markdown(args.out, md)
    print(md)
    print(f"[potential_at_k] wrote {out}")


if __name__ == "__main__":
    main()
