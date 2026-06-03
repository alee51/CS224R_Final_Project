"""AUC@k for each (arm, dataset).

Computes a single scalar `AUC@k = trapz(pass_at_k_vector, ks)` over the locked
k ladder {1, 2, 4, 8, 16, 32, 64}, skipping k values that exceed n_rollouts for
a given file. The result is a cross-arm × dataset scalar table — useful as a
one-number summary that rewards both pass@1 quality and large-k coverage.

If `pass_at_k` in the JSON is missing some k values, we recompute pass@k from
`per_prompt[i].n_correct` so older files are still comparable.

Usage:
    python main-verl/eval/analysis/auc_at_k.py /vol/probes/eval_4b/*.json
    python main-verl/eval/analysis/auc_at_k.py path/to/single.json
"""

from __future__ import annotations

import argparse
from math import comb

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_io import collect, write_markdown  # noqa: E402

K_LADDER = [1, 2, 4, 8, 16, 32, 64]


def pass_at_k(per_prompt, k, n):
    if k > n:
        return None
    vals = []
    for p in per_prompt:
        c = p["n_correct"]
        if c == 0:
            vals.append(0.0)
        else:
            vals.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(vals))


def auc_for_ds(ds: dict) -> tuple[float, list[tuple[int, float]]]:
    n = max((len(p["rewards"]) for p in ds["per_prompt"]), default=0)
    saved = ds.get("pass_at_k", {})
    points: list[tuple[int, float]] = []
    for k in K_LADDER:
        v = saved.get(f"pass@{k}")
        if v is None:
            v = pass_at_k(ds["per_prompt"], k, n)
        if v is None:
            continue
        points.append((k, float(v)))
    if len(points) < 2:
        return float("nan"), points
    ks = np.array([p[0] for p in points], dtype=float)
    vs = np.array([p[1] for p in points], dtype=float)
    auc = float(np.trapezoid(vs, ks))
    return auc, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one or more eval JSON files / globs")
    ap.add_argument("--out", default="auc_at_k.md")
    args = ap.parse_args()

    data = collect(args.paths)
    if not data:
        print("[auc_at_k] no inputs found")
        return

    arms = sorted({a for (a, _ds) in data})
    datasets = sorted({d for (_a, d) in data})

    lines = ["# AUC@k (locked k ladder {1, 2, 4, 8, 16, 32, 64})", ""]
    lines.append("| arm \\ dataset | " + " | ".join(datasets) + " |")
    lines.append("|---|" + "---|" * len(datasets))
    for arm in arms:
        row = [arm]
        for ds_name in datasets:
            ds = data.get((arm, ds_name))
            if ds is None:
                row.append("—")
                continue
            auc, _pts = auc_for_ds(ds)
            row.append(f"{auc:.3f}" if not np.isnan(auc) else "—")
        lines.append("| " + " | ".join(row) + " |")

    # per-cell pass-at-k breakdown for transparency
    lines += ["", "## Underlying pass@k points", ""]
    for (arm, ds_name), ds in sorted(data.items()):
        _, pts = auc_for_ds(ds)
        pretty = ", ".join(f"pass@{k}={v:.3f}" for k, v in pts)
        lines.append(f"- **{arm} / {ds_name}**: {pretty}")

    md = "\n".join(lines) + "\n"
    out = write_markdown(args.out, md)
    print(md)
    print(f"[auc_at_k] wrote {out}")


if __name__ == "__main__":
    main()
