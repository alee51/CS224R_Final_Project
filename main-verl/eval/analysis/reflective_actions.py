"""Reflective-action frequency in rollout text.

Counts the per-rollout occurrence of a small set of self-monitoring / hedging
phrases:

    wait, however, verify, because, alternatively,
    let me check, let me reconsider

Matched case-insensitively as whole-word regex (`\\b...\\b`). Multi-word phrases
are matched as fixed substrings with word-boundaries on the ends. Averaged per
(arm, dataset) — both per-rollout and per-1000-tokens.

Higher counts are a (rough) proxy for chain-of-thought self-correction
behavior. Not load-bearing for the headline; useful as supporting evidence in
the diversity-vs-process discussion.

Usage:
    python main-verl/eval/analysis/reflective_actions.py /vol/probes/eval_4b/*.json
"""

from __future__ import annotations

import argparse
import re

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_io import collect, write_markdown  # noqa: E402

PATTERNS = [
    r"\bwait\b",
    r"\bhowever\b",
    r"\bverify\b",
    r"\bbecause\b",
    r"\balternatively\b",
    r"\blet me check\b",
    r"\blet me reconsider\b",
]
COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]
LABELS = ["wait", "however", "verify", "because", "alternatively",
          "let_me_check", "let_me_reconsider"]


def count_in_rollout(text: str) -> list[int]:
    return [len(rx.findall(text)) for rx in COMPILED]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="reflective_actions.md")
    args = ap.parse_args()

    data = collect(args.paths)
    if not data:
        print("[reflective_actions] no inputs found")
        return

    rows: dict[tuple[str, str], dict[str, float]] = {}
    for (arm, ds_name), ds in sorted(data.items()):
        per_rollout_counts = [np.zeros(len(PATTERNS), dtype=float)]
        per_rollout_counts.clear()
        tokens_per_rollout = []
        for p in ds["per_prompt"]:
            for r in p["rollouts"]:
                if not r:
                    continue
                per_rollout_counts.append(np.array(count_in_rollout(r), dtype=float))
                tokens_per_rollout.append(max(len(r.split()), 1))
        if not per_rollout_counts:
            rows[(arm, ds_name)] = {"n_rollouts": 0, "total": 0.0}
            continue
        arr = np.stack(per_rollout_counts)  # shape (R, P)
        avg_per_rollout = arr.mean(axis=0)
        total_per_rollout = float(arr.sum(axis=1).mean())
        # per 1k tokens
        per_k_tok = arr.sum(axis=1) / (np.array(tokens_per_rollout) / 1000.0)
        avg_per_k_tok = float(per_k_tok.mean())
        row = {LABELS[i]: float(avg_per_rollout[i]) for i in range(len(LABELS))}
        row["total_per_rollout"] = total_per_rollout
        row["total_per_1k_tokens"] = avg_per_k_tok
        row["n_rollouts"] = int(arr.shape[0])
        rows[(arm, ds_name)] = row

    arms = sorted({a for (a, _) in rows})
    datasets = sorted({d for (_, d) in rows})

    lines = ["# Reflective-action frequency in rollout text", "",
             "Counts per rollout, then averaged. `total_per_1k_tokens` is the",
             "rate normalized by rollout length to control for verbosity.",
             ""]
    for ds_name in datasets:
        lines.append(f"## {ds_name}")
        lines.append("")
        header = ["arm", "n_roll"] + LABELS + ["total/roll", "total/1k_tok"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for arm in arms:
            r = rows.get((arm, ds_name))
            if r is None or r.get("n_rollouts", 0) == 0:
                continue
            cells = [arm, str(r["n_rollouts"])]
            for k in LABELS:
                cells.append(f"{r.get(k, 0):.3f}")
            cells.append(f"{r.get('total_per_rollout', 0):.3f}")
            cells.append(f"{r.get('total_per_1k_tokens', 0):.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    md = "\n".join(lines) + "\n"
    out = write_markdown(args.out, md)
    print(md)
    print(f"[reflective_actions] wrote {out}")


if __name__ == "__main__":
    main()
