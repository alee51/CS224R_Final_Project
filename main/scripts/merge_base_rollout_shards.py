#!/usr/bin/env python3
"""Merge sharded base-rollout jsonl files into one rollouts.jsonl for filter_by_rollout_pass_rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stamp",
        required=True,
        help="CS224R_RUN_STAMP directory under probes/base_rollout_pass_polaris_51k/",
    )
    parser.add_argument(
        "--volume-root",
        type=Path,
        default=None,
        help="Local path if shards were pulled from Modal (default: print volume paths only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output merged jsonl (default: <volume-root>/rollouts.jsonl)",
    )
    args = parser.parse_args()

    base = Path("probes/base_rollout_pass_polaris_51k") / args.stamp
    if args.volume_root is not None:
        base = args.volume_root

    shard_dirs = sorted(base.glob("shard_*_of_*"))
    if not shard_dirs:
        raise SystemExit(f"No shard dirs under {base}")

    merged: list[dict] = []
    for shard_dir in shard_dirs:
        rollouts = shard_dir / "rollouts.jsonl"
        if not rollouts.is_file():
            raise SystemExit(f"Missing {rollouts}")
        merged.extend(_read_jsonl(rollouts))
        print(f"{shard_dir.name}: {rollouts.stat().st_size // 1024} KiB")

    merged.sort(key=lambda r: (int(r["problem_id"]), int(r["rollout_idx"])))
    out = args.out or (base / "rollouts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(merged)} rollouts -> {out}")


if __name__ == "__main__":
    main()
