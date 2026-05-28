#!/usr/bin/env python3
"""Filter a Polaris manifest by base-model rollout pass rate.

Reads a manifest jsonl + a rollouts jsonl (typically base-model rollouts at N=8),
groups rewards by problem_id, computes per-prompt pass_rate, and writes a
filtered manifest keeping only prompts with `min_pass < pass_rate < max_pass`.

Default cutoffs (0 < pass_rate < 1) drop the 0/N "never solved" and N/N
"always solved" prompts — the signal-starvation regime that vanilla GRPO
gets ~0 gradient on. This matches Polaris's own 53K→30K refilter recipe
(https://hkunlp.github.io/blog/2025/Polaris/) at the same difficulty thresholds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_pass_rates(rollouts: list[dict]) -> dict[int, tuple[int, int]]:
    """Returns {problem_id: (n_correct, n_rollouts)}."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        pid = int(r["problem_id"])
        grouped[pid].append(int(r.get("reward", 0)))
    return {pid: (sum(v), len(v)) for pid, v in grouped.items()}


def filter_manifest(
    manifest: list[dict],
    pass_rates: dict[int, tuple[int, int]],
    *,
    min_pass: float,
    max_pass: float,
    require_min_rollouts: int,
) -> tuple[list[dict], list[dict], dict]:
    kept: list[dict] = []
    dropped: list[dict] = []
    drop_reasons: Counter[str] = Counter()
    no_rollouts = 0

    for row in manifest:
        pid = int(row["problem_id"])
        if pid not in pass_rates:
            no_rollouts += 1
            dropped.append({**row, "drop_reason": "no_rollouts"})
            drop_reasons["no_rollouts"] += 1
            continue
        n_correct, n = pass_rates[pid]
        if n < require_min_rollouts:
            dropped.append({**row, "drop_reason": f"too_few_rollouts({n})", "pass_rate": n_correct / n, "n_rollouts": n})
            drop_reasons["too_few_rollouts"] += 1
            continue
        pr = n_correct / n
        if pr <= min_pass:
            dropped.append({**row, "drop_reason": f"pass_rate_le_min({pr:.3f})", "pass_rate": pr, "n_rollouts": n})
            drop_reasons["always_wrong"] += 1
            continue
        if pr >= max_pass:
            dropped.append({**row, "drop_reason": f"pass_rate_ge_max({pr:.3f})", "pass_rate": pr, "n_rollouts": n})
            drop_reasons["always_right"] += 1
            continue
        kept.append({**row, "base_pass_rate": pr, "base_n_rollouts": n})

    band_kept = Counter(str(r.get("difficulty_band", "?")) for r in kept)
    band_dropped = Counter(str(r.get("difficulty_band", "?")) for r in dropped)

    stats = {
        "n_manifest": len(manifest),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_reasons": dict(drop_reasons),
        "per_band_kept": dict(band_kept),
        "per_band_dropped": dict(band_dropped),
        "cutoffs": {"min_pass": min_pass, "max_pass": max_pass, "require_min_rollouts": require_min_rollouts},
    }
    return kept, dropped, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="input manifest jsonl (problem_id, problem, gold, ...)")
    p.add_argument("--rollouts", type=Path, required=True, help="rollouts jsonl with {problem_id, reward, ...}")
    p.add_argument("--out", type=Path, required=True, help="output filtered manifest jsonl")
    p.add_argument("--dropped-audit", type=Path, default=None, help="optional jsonl of dropped rows with reasons")
    p.add_argument("--meta", type=Path, default=None, help="optional stats sidecar json")
    p.add_argument("--min-pass", type=float, default=0.0, help="drop prompts with pass_rate <= this (default 0.0 = drop 0/N)")
    p.add_argument("--max-pass", type=float, default=1.0, help="drop prompts with pass_rate >= this (default 1.0 = drop N/N)")
    p.add_argument("--require-min-rollouts", type=int, default=4, help="skip prompts with fewer than this many rollouts")
    p.add_argument("--dry-run", action="store_true", help="compute stats only; do not write outputs")
    args = p.parse_args()

    manifest = _read_jsonl(args.manifest)
    rollouts = _read_jsonl(args.rollouts)
    print(f"manifest: {len(manifest):,} prompts; rollouts: {len(rollouts):,} rows")

    pass_rates = compute_pass_rates(rollouts)
    print(f"prompts with at least one rollout: {len(pass_rates):,}")
    n_rollout_dist = Counter(n for _, n in pass_rates.values())
    print(f"  rollouts-per-prompt distribution: {dict(sorted(n_rollout_dist.items()))}")

    kept, dropped, stats = filter_manifest(
        manifest,
        pass_rates,
        min_pass=args.min_pass,
        max_pass=args.max_pass,
        require_min_rollouts=args.require_min_rollouts,
    )

    print(f"\nKept:   {stats['n_kept']:,} ({100 * stats['n_kept'] / stats['n_manifest']:.2f}%)")
    print(f"Dropped:{stats['n_dropped']:,} ({100 * stats['n_dropped'] / stats['n_manifest']:.2f}%)")
    for reason, count in sorted(stats["drop_reasons"].items()):
        print(f"  {reason}: {count:,}")
    print(f"\nPer-band kept: {stats['per_band_kept']}")
    print(f"Per-band dropped: {stats['per_band_dropped']}")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    _write_jsonl(args.out, kept)
    print(f"\nWrote {args.out} ({len(kept):,} rows)")
    if args.dropped_audit:
        _write_jsonl(args.dropped_audit, dropped)
        print(f"Wrote dropped audit {args.dropped_audit} ({len(dropped):,} rows)")
    if args.meta:
        meta = {
            "source_manifest": str(args.manifest),
            "source_rollouts": str(args.rollouts),
            "output_manifest": str(args.out),
            "dropped_audit": str(args.dropped_audit) if args.dropped_audit else None,
            "stats": stats,
            "materialized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.meta.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Wrote {args.meta}")


if __name__ == "__main__":
    main()
