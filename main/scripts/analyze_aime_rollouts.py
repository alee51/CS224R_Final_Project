#!/usr/bin/env python3
"""Per-problem AIME rollout compare: parsing failures vs wrong answers."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from train.reward import compute_reward, extract_rank2


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


def enrich_row(row: dict) -> dict:
    text = row.get("completion_text") or ""
    gold = str(row.get("gold", ""))
    pv = str(row.get("prompt_variant", "hybrid_answer_boxed"))
    meta = compute_reward(text, gold, prompt_variant=pv)
    r2 = extract_rank2(text, gold, prompt_variant=pv)
    out = {**row, **meta, **r2}
    return out


def classify_prompt(
    base_rows: list[dict],
    other_rows: list[dict],
) -> str:
    b16 = prompt_passk(base_rows, 16)
    o16 = prompt_passk(other_rows, 16)
    b_any = any(r.get("reward", 0) > 0 for r in base_rows)
    o_any = any(r.get("reward", 0) > 0 for r in other_rows)
    if b16 > 0 and o16 == 0:
        return "base_pass16_other_fail"
    if b_any and not o_any:
        return "base_any_other_none"
    if not b_any and o_any:
        return "other_only"
    if b16 == 0 and o16 == 0:
        return "both_fail"
    return "both_some"


def rollout_bucket(row: dict) -> str:
    if not row.get("parse_ok_rank2") and not row.get("parse_ok"):
        return "parse_fail"
    if row.get("reward", 0) > 0:
        return "parsed_correct"
    return "parsed_wrong"


def summarize_label(rows_by_pid: dict[int, list[dict]]) -> dict:
    all_rows = [enrich_row(r) for rs in rows_by_pid.values() for r in rs]
    n = len(all_rows)
    buckets = Counter(rollout_bucket(r) for r in all_rows)
    return {
        "n_rollout_rows": n,
        "parse_ok_rate": sum(r.get("parse_ok") for r in all_rows) / n,
        "parse_ok_rank2_rate": sum(r.get("parse_ok_rank2") for r in all_rows) / n,
        "reward_rate": sum(r.get("reward", 0) > 0 for r in all_rows) / n,
        "extract_path": dict(Counter(r.get("extract_path") for r in all_rows)),
        "rollout_bucket": dict(buckets),
        "has_boxed_rate": sum(r.get("has_boxed") for r in all_rows) / n,
        "has_answer_line_rate": sum(r.get("has_answer_line") for r in all_rows) / n,
        "strict_parse_ok_rate": sum(r.get("strict_parse_ok") for r in all_rows) / n,
        "finish_reason": dict(Counter(r.get("finish_reason") for r in all_rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir", type=Path, required=True)
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--other-labels", nargs="+", default=["grpo_lr3e6_s59", "minority_lr3e6_s54"])
    ap.add_argument("--tail-chars", type=int, default=600)
    ap.add_argument("--max-examples", type=int, default=6)
    args = ap.parse_args()

    base = load_rollouts(args.rollouts_dir / f"{args.base_label}.jsonl")
    others = {
        lab: load_rollouts(args.rollouts_dir / f"{lab}.jsonl") for lab in args.other_labels
    }

    print(f"rollouts_dir: {args.rollouts_dir}\n")
    print("=== Aggregate rollout stats ===")
    print(f"\n[{args.base_label}]")
    for k, v in summarize_label(base).items():
        print(f"  {k}: {v}")
    for lab, by_pid in others.items():
        print(f"\n[{lab}]")
        for k, v in summarize_label(by_pid).items():
            print(f"  {k}: {v}")

    for lab, other in others.items():
        print(f"\n=== Per-prompt: {args.base_label} vs {lab} ===")
        cats: dict[str, list[int]] = defaultdict(list)
        for pid in sorted(set(base) & set(other)):
            cats[classify_prompt(base[pid], other[pid])].append(pid)
        for name, pids in sorted(cats.items()):
            print(f"  {name}: {len(pids)} prompts")

        regress = cats["base_pass16_other_fail"] + cats["base_any_other_none"]
        print(f"\n--- Regressions ({args.base_label} > {lab}), up to {args.max_examples} ---")
        for pid in regress[: args.max_examples]:
            gold = base[pid][0]["gold"]
            b16 = prompt_passk(base[pid], 16)
            o16 = prompt_passk(other[pid], 16)
            print(f"\nproblem_id={pid} gold={gold!r}  base pass@16={b16:.3f}  {lab} pass@16={o16:.3f}")
            for tag, rows in [(args.base_label, base[pid]), (lab, other[pid])]:
                enriched = [enrich_row(r) for r in rows]
                best = max(enriched, key=lambda r: r.get("reward", 0))
                bkt = rollout_bucket(best)
                tail = (best.get("completion_text") or "")[-args.tail_chars :]
                print(
                    f"  [{tag}] best_rollout: bucket={bkt} reward={best.get('reward')} "
                    f"parse_ok={best.get('parse_ok')} path={best.get('extract_path')} "
                    f"parsed={best.get('parsed_answer')!r}"
                )
                print(f"  ... tail:\n{tail}\n")

        # On regressions: is other failing due to parse or wrong answer?
        parse_only = 0
        wrong_only = 0
        both = 0
        for pid in regress:
            o_rows = [enrich_row(r) for r in other[pid]]
            if not any(r.get("parse_ok") for r in o_rows):
                parse_only += 1
            elif not any(r.get("reward", 0) > 0 for r in o_rows):
                wrong_only += 1
            else:
                both += 1
        print(
            f"Regression breakdown for {lab} ({len(regress)} prompts): "
            f"never_parsed={parse_only} parsed_but_wrong={wrong_only} mixed={both}"
        )


if __name__ == "__main__":
    main()
