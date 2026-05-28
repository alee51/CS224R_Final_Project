#!/usr/bin/env python3
"""Compare BeyondAIME rollout jsonl (base vs trained) for regression patterns."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _pass_at_k(n_correct: int, n: int, k: int) -> float:
    if n - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n - n_correct, k) / math.comb(n, k)


def load_rollouts(path: Path) -> dict[int, list[dict]]:
    by_pid: dict[int, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_pid[int(row["problem_id"])].append(row)
    return by_pid


def prompt_passk(rows: list[dict], k: int) -> float:
    n = len(rows)
    c = sum(1 for r in rows if r.get("reward", 0) > 0)
    return _pass_at_k(c, n, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir", type=Path, required=True)
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--trained-label", default="grpo_b200_s359")
    ap.add_argument("--tail-chars", type=int, default=800)
    ap.add_argument("--max-examples", type=int, default=8)
    args = ap.parse_args()

    base_path = args.rollouts_dir / f"{args.base_label}.jsonl"
    trained_path = args.rollouts_dir / f"{args.trained_label}.jsonl"
    base = load_rollouts(base_path)
    trained = load_rollouts(trained_path)

    categories: dict[str, list[int]] = defaultdict(list)
    for pid in sorted(set(base) & set(trained)):
        b_rows, t_rows = base[pid], trained[pid]
        b16 = prompt_passk(b_rows, 16)
        t16 = prompt_passk(t_rows, 16)
        b_ok = any(r.get("reward", 0) > 0 for r in b_rows)
        t_ok = any(r.get("reward", 0) > 0 for r in t_rows)
        if b16 > 0 and t16 == 0:
            categories["base_pass16_trained_fail"].append(pid)
        elif b_ok and not t_ok:
            categories["base_any_correct_trained_none"].append(pid)
        elif not b_ok and t_ok:
            categories["trained_only"].append(pid)

    print(f"=== BeyondAIME rollout compare: {args.base_label} vs {args.trained_label} ===")
    print(f"rollouts_dir: {args.rollouts_dir}\n")
    for name, pids in sorted(categories.items()):
        print(f"  {name}: {len(pids)} prompts")

    def summarize(label: str, rows_by_pid: dict[int, list[dict]]) -> None:
        all_rows = [r for rs in rows_by_pid.values() for r in rs]
        n = len(all_rows)
        print(f"\n--- {label} (n_rollout_rows={n}) ---")
        print(f"  parse_ok rate: {sum(r.get('parse_ok') for r in all_rows) / n:.3f}")
        print(f"  extract_path: {dict(Counter(r.get('extract_path') for r in all_rows))}")
        print(f"  finish_reason: {dict(Counter(r.get('finish_reason') for r in all_rows))}")
        lens = [len(r.get("completion_text", "")) for r in all_rows]
        print(f"  completion len: median={sorted(lens)[len(lens)//2]} max={max(lens)}")

    summarize(args.base_label, base)
    summarize(args.trained_label, trained)

    show = categories["base_pass16_trained_fail"] or categories["base_any_correct_trained_none"]
    print(f"\n=== Examples: base solves, trained does not (up to {args.max_examples}) ===")
    for pid in show[: args.max_examples]:
        gold = base[pid][0]["gold"]
        print(f"\n--- problem_id={pid} gold={gold} ---")
        print(f"  base pass@16={prompt_passk(base[pid], 16):.3f}  trained pass@16={prompt_passk(trained[pid], 16):.3f}")
        for tag, rows in [(args.base_label, base[pid]), (args.trained_label, trained[pid])]:
            best = max(rows, key=lambda r: r.get("reward", 0))
            tail = (best.get("completion_text") or "")[-args.tail_chars :]
            print(f"  [{tag}] reward={best.get('reward')} parse_ok={best.get('parse_ok')} "
                  f"path={best.get('extract_path')} parsed={best.get('parsed_answer')!r}")
            print(f"  ... tail:\n{tail}\n")


if __name__ == "__main__":
    main()
