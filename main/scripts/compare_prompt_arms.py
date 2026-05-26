#!/usr/bin/env python3
"""Side-by-side A/B/C comparison for the prompt probe.

Reads each arm's phase1_rollouts.jsonl + a shared manifest, runs the
offline Rank-2 rescore, and emits a paired comparison table plus the
prompt_probe.md §5 decision verdict.

Paired comparison uses the same problem_ids across arms (same manifest),
so per-prompt deltas are meaningful.
"""

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

from train.reward import extract_rank2  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def score_arm(
    rollouts_path: Path, manifest_path: Path, prompt_variant: str
) -> dict[str, Any]:
    manifest = _read_jsonl(manifest_path)
    rollouts = _read_jsonl(rollouts_path)
    gold_by_pid = {int(m["problem_id"]): m["gold"] for m in manifest}

    n = len(rollouts)
    counts = {
        "parse_ok_minerva": 0,
        "parse_ok_boxed": 0,
        "parse_ok_rank2": 0,
        "has_boxed": 0,
        "has_answer_line": 0,
        "reward": 0,
    }
    paths = defaultdict(int)
    per_prompt: dict[int, list[int]] = defaultdict(list)

    for r in rollouts:
        pid = int(r["problem_id"])
        gold = gold_by_pid[pid]
        rk = extract_rank2(r["completion"], gold, prompt_variant=prompt_variant)
        counts["parse_ok_minerva"] += int(rk["parse_ok_minerva"])
        counts["parse_ok_boxed"] += int(rk["parse_ok_boxed"])
        counts["parse_ok_rank2"] += int(rk["parse_ok_rank2"])
        counts["has_boxed"] += int(r.get("has_boxed", False))
        counts["has_answer_line"] += int(r.get("has_answer_line", False))
        counts["reward"] += int(rk["reward"])
        paths[rk["extract_path"]] += 1
        per_prompt[pid].append(int(rk["reward"]))

    mixed = sum(
        1 for pid, vals in per_prompt.items()
        if vals and 0 < sum(vals) < len(vals)
    )
    all_correct = sum(
        1 for vals in per_prompt.values() if vals and sum(vals) == len(vals)
    )
    all_wrong = sum(
        1 for vals in per_prompt.values() if vals and sum(vals) == 0
    )

    return {
        "n_rollouts": n,
        "n_prompts": len(per_prompt),
        "parse_ok_minerva": counts["parse_ok_minerva"] / n,
        "parse_ok_boxed": counts["parse_ok_boxed"] / n,
        "parse_ok_rank2": counts["parse_ok_rank2"] / n,
        "has_boxed": counts["has_boxed"] / n,
        "has_answer_line": counts["has_answer_line"] / n,
        "pass_rate": counts["reward"] / n,
        "mixed_fraction": mixed / len(per_prompt),
        "all_correct_fraction": all_correct / len(per_prompt),
        "all_wrong_fraction": all_wrong / len(per_prompt),
        "extract_path_counts": dict(paths),
        "per_prompt_rewards": per_prompt,  # for paired comparison
    }


def _fmt(x: float) -> str:
    return f"{x:.1%}"


def render_table(arms: dict[str, dict[str, Any]]) -> str:
    names = list(arms)
    rows = [
        ("rollouts", "n_rollouts", "{:,}"),
        ("prompts", "n_prompts", "{:,}"),
        ("has_answer_line", "has_answer_line", "pct"),
        ("has_boxed", "has_boxed", "pct"),
        ("parse_ok_minerva", "parse_ok_minerva", "pct"),
        ("parse_ok_boxed", "parse_ok_boxed", "pct"),
        ("parse_ok_rank2 ⭐", "parse_ok_rank2", "pct"),
        ("pass_rate (rank2 reward)", "pass_rate", "pct"),
        ("mixed_reward fraction ⭐", "mixed_fraction", "pct"),
        ("all_correct fraction", "all_correct_fraction", "pct"),
        ("all_wrong fraction", "all_wrong_fraction", "pct"),
    ]
    width = 28
    header = "metric".ljust(width) + " | " + " | ".join(n.ljust(14) for n in names)
    lines = [header, "-" * len(header)]
    for label, key, fmt in rows:
        vals = []
        for n in names:
            v = arms[n][key]
            if fmt == "pct":
                vals.append(_fmt(v).rjust(14))
            else:
                vals.append(fmt.format(v).rjust(14))
        lines.append(label.ljust(width) + " | " + " | ".join(vals))
    return "\n".join(lines)


def render_paired_deltas(arms: dict[str, dict[str, Any]], base: str = "A") -> str:
    if base not in arms:
        return ""
    base_per_prompt = arms[base]["per_prompt_rewards"]
    lines = [f"\nPaired prompt-level Rank-2 reward count deltas vs {base}:"]
    for name, summary in arms.items():
        if name == base:
            continue
        their = summary["per_prompt_rewards"]
        common = set(base_per_prompt) & set(their)
        # Per-prompt: did this arm get more correct rollouts than base?
        wins = losses = ties = 0
        net_delta = 0
        for pid in common:
            b = sum(base_per_prompt[pid])
            t = sum(their[pid])
            net_delta += (t - b)
            if t > b:
                wins += 1
            elif t < b:
                losses += 1
            else:
                ties += 1
        avg_delta = net_delta / len(common) if common else 0
        lines.append(
            f"  {name} vs {base}: wins={wins}, losses={losses}, ties={ties}  "
            f"(net +{net_delta} rollouts correct, avg +{avg_delta:.2f}/prompt)"
        )
    return "\n".join(lines)


def render_decision(arms: dict[str, dict[str, Any]]) -> str:
    """Apply prompt_probe.md §5 decision rule."""
    if "A" not in arms:
        return "\n(no Arm A baseline; cannot apply decision rule)"
    a = arms["A"]
    lines = ["\n=== Decision (per prompt_probe.md §5) ==="]
    for name in ("B", "C"):
        if name not in arms:
            continue
        x = arms[name]
        d_parse = (x["parse_ok_rank2"] - a["parse_ok_rank2"]) * 100
        d_mixed = (x["mixed_fraction"] - a["mixed_fraction"]) * 100
        verdict = ""
        if d_parse > 5 and d_mixed > 2:
            verdict = f"✅ {name} BEATS A (parse +{d_parse:.1f}pp, mixed +{d_mixed:.1f}pp)"
        elif abs(d_parse) <= 2 and abs(d_mixed) <= 2:
            verdict = (
                f"= {name} TIES A (parse Δ{d_parse:+.1f}pp, mixed Δ{d_mixed:+.1f}pp) "
                f"— A wins by default"
            )
        else:
            verdict = (
                f"❌ {name} loses to A (parse Δ{d_parse:+.1f}pp, mixed Δ{d_mixed:+.1f}pp)"
            )
        lines.append(f"  {verdict}")
    if "B" in arms and "C" in arms:
        b, c = arms["B"], arms["C"]
        d_parse_cb = (c["parse_ok_rank2"] - b["parse_ok_rank2"]) * 100
        d_mixed_cb = (c["mixed_fraction"] - b["mixed_fraction"]) * 100
        if d_parse_cb > 5 and d_mixed_cb > 2:
            lines.append(
                f"  ✅ C also beats B (parse +{d_parse_cb:.1f}pp, mixed +{d_mixed_cb:.1f}pp) "
                f"— C wins only if it also beat A above"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Shared manifest jsonl")
    p.add_argument("--arm-a", help="Arm A phase1_rollouts.jsonl path (dapo_answer_v1)")
    p.add_argument("--arm-b", help="Arm B phase1_rollouts.jsonl path (verl_math_boxed)")
    p.add_argument("--arm-c", help="Arm C phase1_rollouts.jsonl path (hybrid_answer_boxed)")
    p.add_argument("--out-json", help="Optional path to write the full comparison as JSON")
    args = p.parse_args()

    manifest = Path(args.manifest)
    arms: dict[str, dict[str, Any]] = {}
    if args.arm_a:
        print(f"Scoring Arm A ({args.arm_a})...")
        arms["A"] = score_arm(Path(args.arm_a), manifest, "dapo_answer_v1")
    if args.arm_b:
        print(f"Scoring Arm B ({args.arm_b})...")
        arms["B"] = score_arm(Path(args.arm_b), manifest, "verl_math_boxed")
    if args.arm_c:
        print(f"Scoring Arm C ({args.arm_c})...")
        arms["C"] = score_arm(Path(args.arm_c), manifest, "hybrid_answer_boxed")

    if not arms:
        print("No arms supplied.")
        return

    print("\n" + render_table(arms))
    print(render_paired_deltas(arms))
    print(render_decision(arms))

    if args.out_json:
        # Strip non-serializable per_prompt_rewards for the json dump
        serializable = {
            name: {k: v for k, v in s.items() if k != "per_prompt_rewards"}
            for name, s in arms.items()
        }
        Path(args.out_json).write_text(json.dumps(serializable, indent=2))
        print(f"\nWrote comparison JSON to {args.out_json}")


if __name__ == "__main__":
    main()
