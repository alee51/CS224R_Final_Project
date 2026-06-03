"""Per-rollout token entropy split by correct vs incorrect — load-bearing for
the minority-CoT story.

For each rollout we have, at every generated token position, a top-K dict of
{token_id: logprob}. We treat that top-K as the (truncated, renormalized)
distribution at that step and compute Shannon entropy in bits. The rollout-
level entropy is the mean over token positions; the (arm, dataset) summary is
the mean over rollouts, partitioned by whether the rollout was correct
(reward > 0.5) or not.

Interpretation:
  - High entropy on incorrect rollouts only → the model is unsure exactly
    when it goes wrong (consistent with "diversity goes to wrong answers").
  - High entropy on both → the policy is broadly less confident overall.
  - High entropy on correct rollouts → broad-but-correct sampling
    (the friendliest version of the minority story).

Requires `per_prompt[i].logprobs` to be present in the eval JSON, which
needs `CS224R_EVAL_LOGPROBS>0` at eval time. Missing field → script skips
that file with a warning.

Usage:
    python main-verl/eval/analysis/token_entropy_split.py /vol/probes/eval_4b/*.json
"""

from __future__ import annotations

import argparse
import math

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_io import collect, collected_from_json, write_markdown  # noqa: E402


def step_entropy_bits(step: dict) -> float | None:
    """Shannon entropy in bits over the top-K dist at one token position.

    Renormalizes over the top-K probs so it's a proper distribution
    (the tail mass is unknown — this is what the spec calls for at top-20
    granularity).
    """
    if not step:
        return None
    probs = [math.exp(lp) for lp in step.values()]
    s = sum(probs)
    if s <= 0:
        return None
    probs = [p / s for p in probs]
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def rollout_mean_entropy(token_steps) -> float | None:
    """Mean per-token entropy over a rollout's saved top-K steps."""
    if not token_steps:
        return None
    es = []
    for step in token_steps:
        h = step_entropy_bits(step)
        if h is not None:
            es.append(h)
    if not es:
        return None
    return float(np.mean(es))


def _render(data: dict) -> str:
    if not data:
        return "# Per-rollout mean token entropy\n\nNo input data.\n"

    rows: dict[tuple[str, str], dict] = {}
    skipped: list[tuple[str, str, str]] = []
    for (arm, ds_name), ds in sorted(data.items()):
        correct_es = []
        incorrect_es = []
        sample = ds["per_prompt"]
        has_logprobs = any("logprobs" in p for p in sample)
        if not has_logprobs:
            skipped.append((arm, ds_name, "no logprobs field"))
            continue
        for p in sample:
            lp_list = p.get("logprobs")
            if not lp_list:
                continue
            for r_i, token_steps in enumerate(lp_list):
                mean_h = rollout_mean_entropy(token_steps)
                if mean_h is None:
                    continue
                rwd = p["rewards"][r_i] if r_i < len(p["rewards"]) else 0.0
                if rwd > 0.5:
                    correct_es.append(mean_h)
                else:
                    incorrect_es.append(mean_h)
        rows[(arm, ds_name)] = {
            "n_correct_roll": len(correct_es),
            "n_incorrect_roll": len(incorrect_es),
            "mean_H_correct": float(np.mean(correct_es)) if correct_es else float("nan"),
            "mean_H_incorrect": float(np.mean(incorrect_es)) if incorrect_es else float("nan"),
        }

    arms = sorted({a for (a, _) in rows})
    datasets = sorted({d for (_, d) in rows})

    lines = ["# Per-rollout mean token entropy (bits) split by correct/incorrect", "",
             "Computed from saved top-K logprobs at every generated token, renormalized",
             "over the top-K. Higher = more uncertain. Requires CS224R_EVAL_LOGPROBS>0",
             "at eval time.",
             ""]
    for ds_name in datasets:
        lines.append(f"## {ds_name}")
        lines.append("")
        lines.append("| arm | n_correct_roll | n_incorrect_roll | H(correct) | H(incorrect) | gap |")
        lines.append("|---|---|---|---|---|---|")
        for arm in arms:
            r = rows.get((arm, ds_name))
            if r is None:
                continue
            hc = r["mean_H_correct"]
            hi = r["mean_H_incorrect"]
            gap = (hi - hc) if (not math.isnan(hc) and not math.isnan(hi)) else float("nan")
            lines.append(
                f"| {arm} | {r['n_correct_roll']} | {r['n_incorrect_roll']} | "
                f"{hc:.4f} | {hi:.4f} | {gap:+.4f} |"
            )
        lines.append("")

    if skipped:
        lines.append("## Skipped (no logprobs field)")
        lines.append("")
        for arm, ds, reason in skipped:
            lines.append(f"- {arm} / {ds}: {reason}")
        lines.append("")

    return "\n".join(lines) + "\n"


def analyze(json_data: dict) -> str:
    return _render(collected_from_json(json_data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="token_entropy_split.md")
    args = ap.parse_args()
    data = collect(args.paths)
    if not data:
        print("[token_entropy_split] no inputs found")
        return
    md = _render(data)
    out = write_markdown(args.out, md)
    print(md)
    print(f"[token_entropy_split] wrote {out}")


if __name__ == "__main__":
    main()
