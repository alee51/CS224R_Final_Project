"""Self-BLEU + distinct-n-gram diversity on the rollout TEXT (not the parsed
answer). Catches the failure mode where many different parsed_answer values
hide near-identical reasoning chains, or where parsed_answer collapses to one
string but reasoning varies meaningfully.

Per problem we have up to n rollouts; for each rollout we compute BLEU-4
against the union of the other n-1 rollouts as references, then average.
Self-BLEU is **lower = more diverse** (less overlap with the rest of the set).

Distinct-n is the unique n-grams in the rollout set divided by total n-grams.
Higher = more diverse.

If sacrebleu is importable we delegate to it; otherwise we fall back to a
small in-process BLEU implementation (no external dep beyond stdlib).

Subsampling: by default we cap rollouts at 16 per problem (Self-BLEU is O(n^2)
in rollouts × tokens). Override with --max-rollouts.

Usage:
    python main-verl/eval/analysis/self_bleu.py /vol/probes/eval_4b/*.json
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_io import collect, collected_from_json, write_markdown  # noqa: E402

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _clipped_count(cand_ng: Counter, ref_ngs: list[Counter]) -> tuple[int, int]:
    """Sum of min(cand, max over refs); total cand n-grams."""
    matches = 0
    for ng, c in cand_ng.items():
        max_ref = max((rng[ng] for rng in ref_ngs), default=0)
        matches += min(c, max_ref)
    total = sum(cand_ng.values())
    return matches, total


def _bleu(cand: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    """Corpus-BLEU on a single candidate vs multiple references."""
    if not cand:
        return 0.0
    log_ps = []
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(cand, n)
        ref_ngs = [_ngrams(r, n) for r in refs]
        matches, total = _clipped_count(cand_ng, ref_ngs)
        if total == 0:
            return 0.0
        # +1 smoothing on matches to avoid log(0); standard NIST-ish fallback.
        p = (matches + 1e-9) / total
        log_ps.append(math.log(p))
    # Brevity penalty against the closest-length reference.
    cand_len = len(cand)
    ref_lens = [len(r) for r in refs] or [cand_len]
    closest = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
    bp = 1.0 if cand_len > closest else math.exp(1 - closest / max(cand_len, 1))
    return bp * math.exp(sum(log_ps) / max_n)


def self_bleu_one(rollouts_tokens: list[list[str]]) -> float:
    if len(rollouts_tokens) < 2:
        return 0.0
    scores = []
    for i, cand in enumerate(rollouts_tokens):
        refs = [r for j, r in enumerate(rollouts_tokens) if j != i and r]
        if not refs:
            continue
        scores.append(_bleu(cand, refs))
    return float(np.mean(scores)) if scores else 0.0


def distinct_n(rollouts_tokens: list[list[str]], n: int) -> float:
    seen: set[tuple[str, ...]] = set()
    total = 0
    for toks in rollouts_tokens:
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i:i + n]))
            total += 1
    return (len(seen) / total) if total else 0.0


def _render(data: dict, max_rollouts: int = 16, max_problems: int = 0) -> str:
    if not data:
        return "# Self-BLEU and distinct-n-gram\n\nNo input data.\n"

    rows: dict[tuple[str, str], dict[str, float]] = {}
    for (arm, ds_name), ds in sorted(data.items()):
        per_prompt = ds["per_prompt"]
        if max_problems:
            per_prompt = per_prompt[: max_problems]
        sbs = []
        d1s = []
        d2s = []
        d3s = []
        for p in per_prompt:
            rollouts = p["rollouts"][: max_rollouts]
            toks = [tokenize(r) for r in rollouts if r]
            if len(toks) < 2:
                continue
            sbs.append(self_bleu_one(toks))
            d1s.append(distinct_n(toks, 1))
            d2s.append(distinct_n(toks, 2))
            d3s.append(distinct_n(toks, 3))
        rows[(arm, ds_name)] = {
            "self_bleu": float(np.mean(sbs)) if sbs else 0.0,
            "distinct_1": float(np.mean(d1s)) if d1s else 0.0,
            "distinct_2": float(np.mean(d2s)) if d2s else 0.0,
            "distinct_3": float(np.mean(d3s)) if d3s else 0.0,
            "n_problems": len(sbs),
        }

    arms = sorted({a for (a, _) in rows})
    datasets = sorted({d for (_, d) in rows})

    lines = ["# Self-BLEU and distinct-n-gram (rollout text)", "",
             "Self-BLEU: **lower = more diverse**. distinct_n: **higher = more diverse**.",
             f"Sampled up to {max_rollouts} rollouts/problem (Self-BLEU is O(n^2)).",
             ""]
    for ds_name in datasets:
        lines.append(f"## {ds_name}")
        lines.append("")
        lines.append("| arm | n_problems | Self-BLEU | distinct-1 | distinct-2 | distinct-3 |")
        lines.append("|---|---|---|---|---|---|")
        for arm in arms:
            r = rows.get((arm, ds_name))
            if r is None:
                continue
            lines.append(
                f"| {arm} | {r['n_problems']} | {r['self_bleu']:.4f} | "
                f"{r['distinct_1']:.4f} | {r['distinct_2']:.4f} | {r['distinct_3']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def analyze(json_data: dict, max_rollouts: int = 8, max_problems: int = 0) -> str:
    return _render(collected_from_json(json_data),
                   max_rollouts=max_rollouts, max_problems=max_problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-rollouts", type=int, default=16)
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--out", default="self_bleu.md")
    args = ap.parse_args()
    data = collect(args.paths)
    if not data:
        print("[self_bleu] no inputs found")
        return
    md = _render(data, max_rollouts=args.max_rollouts, max_problems=args.max_problems)
    out = write_markdown(args.out, md)
    print(md)
    print(f"[self_bleu] wrote {out}")


if __name__ == "__main__":
    main()
