#!/usr/bin/env python3
"""Offline Rank-2 rescore for prompt A/B/C probe rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from train.reward import compute_reward, extract_rank2


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _empty_band_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "parse_ok_minerva": 0,
        "parse_ok_boxed": 0,
        "parse_ok_rank2": 0,
        "has_boxed": 0,
        "has_answer_line": 0,
        "reward": 0,
        "prompt_ids": set(),
        "per_prompt_rewards": defaultdict(list),
    }


def _finalize_band_stats(stats: dict[str, Any]) -> dict[str, Any]:
    n = stats["n"]
    prompt_ids = stats.pop("prompt_ids")
    per_prompt_rewards: dict[int, list[int]] = stats.pop("per_prompt_rewards")
    mixed = sum(
        1
        for pid in prompt_ids
        if per_prompt_rewards[pid]
        and 0 < sum(per_prompt_rewards[pid]) < len(per_prompt_rewards[pid])
    )
    out = {
        "n_rollouts": n,
        "parse_ok_minerva_rate": stats["parse_ok_minerva"] / n if n else 0.0,
        "parse_ok_boxed_rate": stats["parse_ok_boxed"] / n if n else 0.0,
        "parse_ok_rank2_rate": stats["parse_ok_rank2"] / n if n else 0.0,
        "has_boxed_rate": stats["has_boxed"] / n if n else 0.0,
        "has_answer_line_rate": stats["has_answer_line"] / n if n else 0.0,
        "pass_rate": stats["reward"] / n if n else 0.0,
        "mixed_reward_prompt_fraction": mixed / len(prompt_ids) if prompt_ids else 0.0,
    }
    return out


def _accumulate(
    stats: dict[str, Any],
    row: dict[str, Any],
    rank2: dict[str, Any],
    live: dict[str, Any],
) -> None:
    stats["n"] += 1
    stats["parse_ok_minerva"] += int(rank2["parse_ok_minerva"])
    stats["parse_ok_boxed"] += int(rank2["parse_ok_boxed"])
    stats["parse_ok_rank2"] += int(rank2["parse_ok_rank2"])
    stats["has_boxed"] += int(live["has_boxed"])
    stats["has_answer_line"] += int(live["has_answer_line"])
    stats["reward"] += int(rank2["reward"])
    pid = int(row["problem_id"])
    stats["prompt_ids"].add(pid)
    stats["per_prompt_rewards"][pid].append(int(rank2["reward"]))


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"rollouts: {summary['n_rollouts']}")
    print(f"parse_ok_minerva: {summary['parse_ok_minerva_rate']:.1%}")
    print(f"parse_ok_boxed: {summary['parse_ok_boxed_rate']:.1%}")
    print(f"parse_ok_rank2: {summary['parse_ok_rank2_rate']:.1%}")
    print(f"has_boxed: {summary['has_boxed_rate']:.1%}")
    print(f"has_answer_line: {summary['has_answer_line_rate']:.1%}")
    print(f"pass rate (rank2 reward): {summary['pass_rate']:.1%}")
    print(f"mixed_reward prompt fraction: {summary['mixed_reward_prompt_fraction']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Rank-2 rescore for probe rollouts")
    parser.add_argument("--rollouts", required=True, help="Path to phase1_rollouts.jsonl")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument(
        "--prompt-variant",
        default="dapo_answer_v1",
        choices=["dapo_answer_v1", "verl_math_boxed", "hybrid_answer_boxed"],
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for rescored jsonl with rank2 fields",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rollouts_path = Path(args.rollouts)
    manifest = _read_jsonl(manifest_path)
    rollouts = _read_jsonl(rollouts_path)

    gold_by_pid = {int(m["problem_id"]): m["gold"] for m in manifest}
    band_by_pid = {
        int(m["problem_id"]): m.get("difficulty_band") for m in manifest
    }

    overall = _empty_band_stats()
    by_band: dict[str, dict[str, Any]] = defaultdict(_empty_band_stats)
    rescored: list[dict[str, Any]] = []

    for row in rollouts:
        pid = int(row["problem_id"])
        gold = gold_by_pid[pid]
        rank2 = extract_rank2(
            row["completion"], gold, prompt_variant=args.prompt_variant
        )
        live = compute_reward(
            row["completion"], gold, prompt_variant=args.prompt_variant
        )
        out_row = {
            **row,
            **rank2,
            "prompt_variant": args.prompt_variant,
            "has_boxed": live["has_boxed"],
            "has_answer_line": live["has_answer_line"],
        }
        rescored.append(out_row)

        _accumulate(overall, row, rank2, live)
        band = band_by_pid.get(pid)
        if band is not None:
            _accumulate(by_band[band], row, rank2, live)

    overall_summary = _finalize_band_stats(overall)
    _print_summary(f"aggregate ({args.prompt_variant})", overall_summary)

    if by_band:
        print("\n--- per-band ---")
        for band in sorted(by_band):
            band_summary = _finalize_band_stats(by_band[band])
            _print_summary(str(band), band_summary)

    if args.output:
        out_path = Path(args.output)
        _write_jsonl(out_path, rescored)
        print(f"\nWrote rescored rollouts to {out_path}")


if __name__ == "__main__":
    main()
