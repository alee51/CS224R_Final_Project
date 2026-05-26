#!/usr/bin/env python3
"""Grade random full-gold Polaris probe rollouts (train reward: Rank-2 + grade_parsed_answer)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from data.gold_utils import is_integer_gold  # noqa: E402
from train.reward import extract_rank2, grade_parsed_answer  # noqa: E402

DEFAULT_MANIFEST = _MAIN_ROOT / "data/probes/05-27/random_fullgold_n800/manifest.jsonl"
DEFAULT_ROLLOUTS = _MAIN_ROOT / "data/probes/05-27/random_fullgold_n800/phase1_rollouts.jsonl"
DEFAULT_OUT_MD = _MAIN_ROOT / "docs/probes/random_fullgold_n800_results.md"
DEFAULT_OUT_JSON = _MAIN_ROOT / "data/probes/05-27/random_fullgold_n800/grading_summary.json"

PROMPT_VARIANT = "hybrid_answer_boxed"
N_ROLLOUTS = 8
K_PASS = 8
BANDS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pass_at_k_unbiased(n_correct: int, n: int = N_ROLLOUTS, k: int = K_PASS) -> float:
    if n - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n - n_correct, k) / math.comb(n, k)


def _grade_rollout(completion: str, gold: str) -> dict[str, Any]:
    r2 = extract_rank2(completion, gold, prompt_variant=PROMPT_VARIANT)
    parsed = r2.get("parsed_answer")
    parse_ok = bool(r2["parse_ok_rank2"])
    correct = bool(parse_ok and parsed is not None and grade_parsed_answer(parsed, gold))
    return {
        "parse_ok_rank2": parse_ok,
        "correct": correct,
        "parsed_answer": parsed,
    }


def _empty_bucket() -> dict[str, Any]:
    return {
        "n_rollouts": 0,
        "n_correct": 0,
        "parse_ok": 0,
        "per_prompt": defaultdict(list),
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    n_rollouts = bucket["n_rollouts"]
    per_prompt: dict[int, list[bool]] = bucket.pop("per_prompt")
    n_prompts = len(per_prompt)
    pass8_vals = [_pass_at_k_unbiased(sum(1 for c in rs if c)) for rs in per_prompt.values()]
    any_correct = sum(1 for v in pass8_vals if v > 0)

    return {
        "n_rollouts": n_rollouts,
        "n_prompts": n_prompts,
        "pass_at_1": bucket["n_correct"] / n_rollouts if n_rollouts else 0.0,
        "pass_at_8_mean": sum(pass8_vals) / n_prompts if n_prompts else 0.0,
        "pass_at_8_any": any_correct / n_prompts if n_prompts else 0.0,
        "parse_ok_rank2_rate": bucket["parse_ok"] / n_rollouts if n_rollouts else 0.0,
    }


def _accumulate(bucket: dict[str, Any], pid: int, graded: dict[str, Any]) -> None:
    bucket["n_rollouts"] += 1
    if graded["parse_ok_rank2"]:
        bucket["parse_ok"] += 1
    if graded["correct"]:
        bucket["n_correct"] += 1
    bucket["per_prompt"][pid].append(graded["correct"])


def analyze(
    manifest_path: Path,
    rollouts_path: Path,
    expected_rollouts: int | None,
) -> dict[str, Any]:
    manifest = _read_jsonl(manifest_path)
    rollouts = _read_jsonl(rollouts_path)

    gold_by_pid = {int(m["problem_id"]): str(m["gold"]) for m in manifest}
    band_by_pid = {int(m["problem_id"]): m["difficulty_band"] for m in manifest}
    integer_by_pid = {
        int(m["problem_id"]): is_integer_gold(m["gold"]) for m in manifest
    }

    overall = _empty_bucket()
    by_band: dict[str, dict[str, Any]] = {b: _empty_bucket() for b in BANDS}
    by_integer = {"integer_gold": _empty_bucket(), "non_integer_gold": _empty_bucket()}

    for row in rollouts:
        pid = int(row["problem_id"])
        gold = gold_by_pid[pid]
        graded = _grade_rollout(row["completion"], gold)
        _accumulate(overall, pid, graded)
        band = band_by_pid[pid]
        if band in by_band:
            _accumulate(by_band[band], pid, graded)
        ig_key = "integer_gold" if integer_by_pid[pid] else "non_integer_gold"
        _accumulate(by_integer[ig_key], pid, graded)

    n_manifest = len(manifest)
    n_rollouts = len(rollouts)
    complete = expected_rollouts is None or n_rollouts >= expected_rollouts

    return {
        "manifest_path": str(manifest_path),
        "rollouts_path": str(rollouts_path),
        "prompt_variant": PROMPT_VARIANT,
        "n_manifest_prompts": n_manifest,
        "n_rollouts": n_rollouts,
        "expected_rollouts": expected_rollouts,
        "rollouts_complete": complete,
        "integer_gold_prompts": sum(1 for v in integer_by_pid.values() if v),
        "non_integer_gold_prompts": sum(1 for v in integer_by_pid.values() if not v),
        "overall": _finalize_bucket(overall),
        "by_band": {b: _finalize_bucket(by_band[b]) for b in BANDS if by_band[b]["n_rollouts"]},
        "by_integer_gold": {k: _finalize_bucket(by_integer[k]) for k in by_integer},
    }


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _band_sort_key(band: str) -> tuple[int, int]:
    num, den = band.split("/")
    return (int(num), int(den))


def render_markdown(results: dict[str, Any]) -> str:
    o = results["overall"]
    lines = [
        "# Random full-gold Polaris probe (n800)",
        "",
        "Uniform random sample from Polaris-53K with **relaxed cleaning** "
        "(non-empty problem string + non-empty gold; **no** integer-gold filter).",
        "",
        f"- **Prompt arm:** `{results['prompt_variant']}`",
        f"- **Grading:** `extract_rank2` + `grade_parsed_answer` (mathd OR sympy)",
        f"- **Manifest:** `{results['manifest_path']}`",
        f"- **Rollouts:** `{results['rollouts_path']}`",
        "",
        "## Run status",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Manifest prompts | {results['n_manifest_prompts']} |",
        f"| Rollouts graded | {results['n_rollouts']} |",
        f"| Expected rollouts | {results['expected_rollouts'] or '—'} |",
        f"| Complete | {'yes' if results['rollouts_complete'] else '**no (partial)**'} |",
        f"| Integer-gold prompts (manifest) | {results['integer_gold_prompts']} |",
        f"| Non-integer-gold prompts (manifest) | {results['non_integer_gold_prompts']} |",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Rollout pass@1 | {_pct(o['pass_at_1'])} ({o['n_rollouts']} rollouts) |",
        f"| Prompt pass@8 (Chen mean) | {_pct(o['pass_at_8_mean'])} |",
        f"| Prompt pass@8 (any correct) | {_pct(o['pass_at_8_any'])} |",
        f"| parse_ok_rank2 (rollout) | {_pct(o['parse_ok_rank2_rate'])} |",
        "",
        "## By difficulty band",
        "",
        "| Band | Rollouts | pass@1 | pass@8 (mean) | pass@8 (any) | parse_ok |",
        "|------|----------|--------|---------------|--------------|----------|",
    ]
    for band in sorted(results["by_band"], key=_band_sort_key):
        b = results["by_band"][band]
        lines.append(
            f"| {band} | {b['n_rollouts']} | {_pct(b['pass_at_1'])} | "
            f"{_pct(b['pass_at_8_mean'])} | {_pct(b['pass_at_8_any'])} | "
            f"{_pct(b['parse_ok_rank2_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## By gold type (diagnostic)",
            "",
            "| Gold type | Prompts | Rollouts | pass@1 | pass@8 (any) | parse_ok |",
            "|-----------|---------|----------|--------|--------------|----------|",
        ]
    )
    for label, key in [
        ("Integer gold", "integer_gold"),
        ("Non-integer gold", "non_integer_gold"),
    ]:
        b = results["by_integer_gold"][key]
        lines.append(
            f"| {label} | {b['n_prompts']} | {b['n_rollouts']} | {_pct(b['pass_at_1'])} | "
            f"{_pct(b['pass_at_8_any'])} | {_pct(b['parse_ok_rank2_rate'])} |"
        )

    if not results["rollouts_complete"]:
        lines.extend(
            [
                "",
                "> **Note:** Rollouts were incomplete when this report was generated. "
                "Re-run after `modal volume get` sync for final numbers.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--expected-rollouts",
        type=int,
        default=6400,
        help="800 prompts × 8 rollouts; set 0 to skip completeness check",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Missing manifest: {args.manifest}")
    if not args.rollouts.is_file():
        raise SystemExit(f"Missing rollouts: {args.rollouts}")

    expected = args.expected_rollouts if args.expected_rollouts > 0 else None
    results = analyze(args.manifest, args.rollouts, expected)
    md = render_markdown(results)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    args.out_json.write_text(json.dumps(results, indent=2) + "\n")

    o = results["overall"]
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    print(f"Rollouts: {results['n_rollouts']} complete={results['rollouts_complete']}")
    print(f"pass@1={o['pass_at_1']:.4f} pass@8_any={o['pass_at_8_any']:.4f} parse_ok={o['parse_ok_rank2_rate']:.4f}")


if __name__ == "__main__":
    main()
