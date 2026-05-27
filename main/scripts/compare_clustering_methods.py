#!/usr/bin/env python3
"""Label smoke/train rollouts under each answer-clustering method (offline ablation).

Reads train_rollouts.jsonl (one row per rollout) grouped by (step, problem_id),
recomputes cluster ids under four methods from docs/build_spec/answer_clustering.md,
and writes:
  - detail jsonl: per-rollout canon strings + cluster_id per method
  - summary json: agreement / flip rates between methods

Usage (from repo root):
  python main/scripts/compare_clustering_methods.py \\
    --rollouts /path/to/train_rollouts.jsonl \\
    --out-dir main/data/probes/05-26/minority_answer_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from train.clustering import (  # noqa: E402
    answer_hash_clusters,
    canonicalize_answer,
    canonicalize_answer_old,
    sympy_equiv_allowlist,
)

METHODS = (
    "old_canon",
    "hardened_canon",
    "hardened_sympy_blocklist",
    "hardened_sympy_allowlist",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _clusters_for_prompt(
    parsed: list[str | None],
    ok: list[bool],
    method: str,
) -> list[int]:
    if method == "old_canon":
        return answer_hash_clusters(
            parsed, ok, use_sympy=False, canonicalize_fn=canonicalize_answer_old
        )
    if method == "hardened_canon":
        return answer_hash_clusters(parsed, ok, use_sympy=False)
    if method == "hardened_sympy_blocklist":
        return answer_hash_clusters(parsed, ok, use_sympy=True)
    if method == "hardened_sympy_allowlist":
        return answer_hash_clusters(
            parsed,
            ok,
            use_sympy=True,
            sympy_equiv_fn=sympy_equiv_allowlist,
        )
    raise ValueError(f"unknown method: {method}")


def _partition_signature(cluster_ids: list[int]) -> tuple[tuple[int, ...], ...]:
    """Normalize cluster ids to a partition signature for cross-method comparison."""
    idx_by_cid: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        idx_by_cid[cid].append(i)
    parts = [tuple(sorted(idxs)) for idxs in idx_by_cid.values()]
    return tuple(sorted(parts))


def run_compare(
    rollouts_path: Path,
    out_dir: Path,
    *,
    min_step: int | None = None,
    max_step: int | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(rollouts_path)
    if not rows:
        raise SystemExit(f"No rows in {rollouts_path}")

    by_prompt: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        step = int(r["step"])
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        pid = int(r["problem_id"])
        by_prompt[(step, pid)].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "clustering_compare_detail.jsonl"
    summary_path = out_dir / "clustering_compare_summary.json"

    n_prompts = 0
    partition_agree: dict[tuple[str, str], int] = defaultdict(int)
    steps_seen: set[int] = set()

    with detail_path.open("w") as detail_f:
        for (step, pid), group in sorted(by_prompt.items()):
            group.sort(key=lambda r: int(r["rollout_idx"]))
            parsed = [r.get("parsed_answer") for r in group]
            ok = [bool(r.get("parse_ok")) for r in group]
            n_prompts += 1
            steps_seen.add(step)

            labels: dict[str, list[int]] = {}
            for method in METHODS:
                labels[method] = _clusters_for_prompt(parsed, ok, method)

            sigs = {m: _partition_signature(labels[m]) for m in METHODS}
            for i, a in enumerate(METHODS):
                for b in METHODS[i + 1 :]:
                    if sigs[a] == sigs[b]:
                        partition_agree[(a, b)] += 1

            for r_idx, r in enumerate(group):
                rec = {
                    "step": step,
                    "problem_id": pid,
                    "rollout_idx": int(r["rollout_idx"]),
                    "parse_ok": ok[r_idx],
                    "parsed_answer": parsed[r_idx],
                    "canon_old": canonicalize_answer_old(parsed[r_idx])
                    if ok[r_idx]
                    else None,
                    "canon_hardened": canonicalize_answer(parsed[r_idx])
                    if ok[r_idx]
                    else None,
                }
                for method in METHODS:
                    rec[f"cluster_{method}"] = labels[method][r_idx]
                detail_f.write(json.dumps(rec) + "\n")

    pair_stats = []
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1 :]:
            agree = partition_agree[(a, b)]
            pair_stats.append(
                {
                    "a": a,
                    "b": b,
                    "prompts_same_partition": agree,
                    "prompts_differ": n_prompts - agree,
                    "frac_same": agree / n_prompts if n_prompts else 0.0,
                }
            )

    summary = {
        "rollouts_path": str(rollouts_path),
        "n_rollout_rows": sum(len(g) for g in by_prompt.values()),
        "n_prompt_groups": n_prompts,
        "steps": sorted(steps_seen),
        "methods": list(METHODS),
        "partition_agreement": pair_stats,
        "detail_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rollouts",
        type=Path,
        default=_MAIN_ROOT / "data/probes/05-26/minority_answer_smoke/train_rollouts.jsonl",
        help="train_rollouts.jsonl from smoke (local path or after modal volume get)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_MAIN_ROOT / "data/probes/05-26/minority_answer_smoke",
    )
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument(
        "--expect-steps",
        type=int,
        default=None,
        help="Exit 1 unless exactly this many distinct steps are present",
    )
    args = p.parse_args()

    if not args.rollouts.is_file():
        raise SystemExit(f"Rollouts not found: {args.rollouts}")

    summary = run_compare(
        args.rollouts,
        args.out_dir,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    n_steps = len(summary["steps"])
    print(json.dumps(summary, indent=2))
    if args.expect_steps is not None and n_steps != args.expect_steps:
        raise SystemExit(
            f"Expected {args.expect_steps} steps, found {n_steps}: {summary['steps']}"
        )


if __name__ == "__main__":
    main()
