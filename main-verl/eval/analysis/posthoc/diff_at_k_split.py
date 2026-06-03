"""diff@k split by solved vs unsolved prompts — load-bearing for the
minority-CoT story.

Hypothesis: minority training adds answer diversity, but the extra distinct
answers concentrate on prompts that the model never gets right ("diversity
goes to wrong answers"). Tests that prediction by partitioning prompts into:

  - solved: n_correct > 0 (the policy lands ≥1 correct rollout out of n)
  - unsolved: n_correct == 0

then computing distinct_answers@k on each partition independently for
k ∈ {1, 4, 8, 16, 32, 64}.

If minority's distinct_answers gap over GRPO is concentrated in the unsolved
partition, the hypothesis is supported. If both partitions show similar
gaps, minority's diversity is "real" (useful + transferable).

Usage:
    python main-verl/eval/analysis/diff_at_k_split.py /vol/probes/eval_4b/*.json
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_io import collect, collected_from_json, write_markdown  # noqa: E402

K_VALUES = [1, 4, 8, 16, 32, 64]


def distinct_at_k(per_prompt, k):
    vals = []
    for p in per_prompt:
        first_k = p["preds"][:k]
        distinct = {pred for pred in first_k if pred and pred != "[INVALID]"}
        vals.append(len(distinct))
    return float(np.mean(vals)) if vals else 0.0


def split_solved_unsolved(per_prompt):
    solved = [p for p in per_prompt if p["n_correct"] > 0]
    unsolved = [p for p in per_prompt if p["n_correct"] == 0]
    return solved, unsolved


def _render(data: dict) -> str:
    if not data:
        return "# distinct_answers@k split\n\nNo input data.\n"
    arms = sorted({a for (a, _) in data})
    datasets = sorted({d for (_, d) in data})

    lines = ["# distinct_answers@k split by solved vs unsolved", "",
             "Load-bearing for the minority-CoT diversity story:",
             "if minority's distinct-answers advantage is concentrated in the",
             "unsolved partition, diversity is going to wrong answers.",
             ""]

    for partition_name in ("solved", "unsolved"):
        lines.append(f"## Partition: {partition_name} (n_correct {'>' if partition_name == 'solved' else '=='} 0)")
        lines.append("")
        for ds_name in datasets:
            lines.append(f"### {ds_name}")
            lines.append("")
            lines.append("| arm | n_partition | " + " | ".join(f"diff@{k}" for k in K_VALUES) + " |")
            lines.append("|---|---|" + "---|" * len(K_VALUES))
            for arm in arms:
                ds = data.get((arm, ds_name))
                if ds is None:
                    continue
                solved, unsolved = split_solved_unsolved(ds["per_prompt"])
                pp = solved if partition_name == "solved" else unsolved
                if not pp:
                    cells = [arm, "0"] + ["—"] * len(K_VALUES)
                else:
                    cells = [arm, str(len(pp))]
                    for k in K_VALUES:
                        if any(k > len(p["preds"]) for p in pp):
                            cells.append("—")
                        else:
                            cells.append(f"{distinct_at_k(pp, k):.3f}")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines) + "\n"


def analyze(json_data: dict) -> str:
    return _render(collected_from_json(json_data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="diff_at_k_split.md")
    args = ap.parse_args()
    data = collect(args.paths)
    if not data:
        print("[diff_at_k_split] no inputs found")
        return
    md = _render(data)
    out = write_markdown(args.out, md)
    print(md)
    print(f"[diff_at_k_split] wrote {out}")


if __name__ == "__main__":
    main()
