"""Comparison table across GRPO + poly_epo + minority eval_4b results.

Reads:
  - /tmp/grpo_*.json
  - /tmp/poly_*.json
  - /tmp/min_*.json (when minority eval lands)

Outputs:
  - main-verl/writeup/results/comparison.md (markdown table)
  - main/data/probes/eval_4b/cross_arm_summary.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

K_VALUES = [1, 4, 8, 16]
# NOTE: dataset list and K_VALUES are placeholders — both are open decisions
# tracked in writeup/eval.md §1 and §5. Reconcile with the locked spec before
# trusting any output from this script.
DATASETS: list[str] = []
ARMS = ["grpo", "polyepo", "minority"]


def coverage_at_k(per_prompt, k):
    cov = []
    for p in per_prompt:
        distinct_correct = set()
        for r, pred in zip(p["rewards"][:k], p["preds"][:k]):
            if r > 0.5 and pred and pred != "[INVALID]":
                distinct_correct.add(pred)
        cov.append(len(distinct_correct))
    return sum(cov) / len(cov)


def distinct_answers_at_k(per_prompt, k):
    da = []
    for p in per_prompt:
        distinct = {pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"}
        da.append(len(distinct))
    return sum(da) / len(da)


def entropy_at_k(per_prompt, k):
    ents = []
    for p in per_prompt:
        preds = [pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"]
        if not preds:
            continue
        counts = Counter(preds)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        ents.append(-sum(pr * math.log2(pr) for pr in probs if pr > 0))
    return (sum(ents) / len(ents)) if ents else 0


def majority_at_k(per_prompt, k):
    mvotes = 0
    n = 0
    for p in per_prompt:
        preds_filtered = [pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"]
        if not preds_filtered:
            continue
        n += 1
        most_common, _ = Counter(preds_filtered).most_common(1)[0]
        for pred, r in zip(p["preds"][:k], p["rewards"][:k]):
            if pred == most_common and r > 0.5:
                mvotes += 1
                break
    return mvotes / len(per_prompt) if per_prompt else 0


def load_results(arm_paths: dict[str, list[str]]):
    """For each arm, merge JSON results from possibly multiple files."""
    by_arm = {}
    for arm, paths in arm_paths.items():
        ds_results = {}
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            data = json.load(p.open())
            for ds_name, ds in data["datasets"].items():
                ds_results[ds_name] = ds
        if ds_results:
            by_arm[arm] = ds_results
    return by_arm


def build_markdown(by_arm) -> str:
    out = []
    out.append("# 4B verl run — held-out eval (2026-06-02)")
    out.append("")
    out.append("All step 400 checkpoints. `temperature=1.0`, `top_p=1.0`, `max_tokens=4096`.")
    out.append("Scorer: verl `math.compute_score` (Hendrycks `is_equiv`), same as training (see `writeup/eval.md` §3).")
    out.append("")

    all_datasets = sorted({ds for arm in by_arm for ds in by_arm[arm]})

    for ds_name in all_datasets:
        out.append(f"## {ds_name}")
        n_prompts = next(
            (by_arm[arm][ds_name]["n_prompts"] for arm in by_arm if ds_name in by_arm[arm]), 0
        )
        out.append(f"n={n_prompts} prompts")
        out.append("")

        # pass@k table
        out.append("### pass@k")
        out.append("")
        header = "| arm | " + " | ".join(f"pass@{k}" for k in K_VALUES) + " |"
        sep = "|---|" + "---|" * len(K_VALUES)
        out.append(header)
        out.append(sep)
        for arm in ARMS:
            if arm not in by_arm or ds_name not in by_arm[arm]:
                continue
            ds = by_arm[arm][ds_name]
            cells = [arm]
            for k in K_VALUES:
                v = ds["pass_at_k"].get(f"pass@{k}")
                cells.append(f"{v:.3f}" if v is not None else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

        # Diversity table
        out.append("### Diversity (eval-time)")
        out.append("")
        out.append("| arm | k | coverage | distinct | entropy | majority@k |")
        out.append("|---|---|---|---|---|---|")
        for arm in ARMS:
            if arm not in by_arm or ds_name not in by_arm[arm]:
                continue
            pp = by_arm[arm][ds_name]["per_prompt"]
            for k in K_VALUES:
                out.append(
                    f"| {arm} | {k} | {coverage_at_k(pp, k):.2f} | "
                    f"{distinct_answers_at_k(pp, k):.2f} | "
                    f"{entropy_at_k(pp, k):.2f} | "
                    f"{majority_at_k(pp, k):.3f} |"
                )
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/nancybao/Desktop/dev/cs224r_finalproject")
    args = ap.parse_args()

    # Probe for available result files
    candidates = {
        "grpo": [
            "/tmp/grpo_v3.json",
            "/tmp/grpo_panel.json",
        ],
        "polyepo": [
            "/tmp/poly_v2_result.json",
            "/tmp/poly_panel.json",
        ],
        "minority": [
            "/tmp/min_eval.json",
            "/tmp/min_panel.json",
        ],
    }
    by_arm = load_results(candidates)
    print(f"loaded arms: {list(by_arm.keys())}")
    for arm, dss in by_arm.items():
        print(f"  {arm}: {list(dss.keys())}")

    md = build_markdown(by_arm)
    out_path = Path(args.root) / "main-verl/writeup/results/comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"wrote {out_path}")

    # Brief stdout summary
    print()
    print(md)


if __name__ == "__main__":
    main()
