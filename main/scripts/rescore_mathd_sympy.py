#!/usr/bin/env python3
"""Rescore Polaris rollouts: strict Rank-2 vs DeepScaleR mathd OR sympy."""

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

from train.math_grade_deepscaler import (  # noqa: E402
    extract_answer,
    grade_answer_mathd,
    grade_answer_mathd_or_sympy,
    grade_answer_sympy,
)
from train.reward import grade_parsed_answer  # noqa: E402
from train.reward import extract_rank2, normalize_final_answer  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _empty_bucket() -> dict[str, int]:
    return {
        "n_rollouts": 0,
        "parse_ok_rank2": 0,
        "parse_ok_boxed": 0,
        "old_strict_pass": 0,
        "mathd_only_pass": 0,
        "sympy_only_pass": 0,
        "mathd_or_sympy_pass": 0,
        "rescued": 0,
        "regressed": 0,
        "old_strict_pass_boxed": 0,
        "mathd_or_sympy_pass_boxed": 0,
        "rescued_boxed": 0,
        "regressed_boxed": 0,
    }


def _grade_pair(parsed: str, gold: str) -> dict[str, int]:
    old = int(
        normalize_final_answer(parsed) == normalize_final_answer(gold)
    )
    mathd = int(grade_answer_mathd(parsed, gold))
    sympy = int(grade_answer_sympy(parsed, gold))
    mos = int(grade_answer_mathd_or_sympy(parsed, gold))
    train = int(grade_parsed_answer(parsed, gold))
    assert train == mos
    return {
        "old_strict": old,
        "mathd_only": mathd,
        "sympy_only": sympy,
        "mathd_or_sympy": mos,
    }


def _accumulate(bucket: dict[str, int], labels: dict[str, int]) -> None:
    bucket["old_strict_pass"] += labels["old_strict"]
    bucket["mathd_only_pass"] += labels["mathd_only"]
    bucket["sympy_only_pass"] += labels["sympy_only"]
    bucket["mathd_or_sympy_pass"] += labels["mathd_or_sympy"]

    if labels["old_strict"] == 0 and labels["mathd_or_sympy"] == 1:
        bucket["rescued"] += 1
    if labels["old_strict"] == 1 and labels["mathd_or_sympy"] == 0:
        bucket["regressed"] += 1


def _accumulate_boxed(
    bucket: dict[str, int], labels: dict[str, int], parse_ok: bool
) -> None:
    if not parse_ok:
        return
    bucket["old_strict_pass_boxed"] += labels["old_strict"]
    bucket["mathd_or_sympy_pass_boxed"] += labels["mathd_or_sympy"]
    if labels["old_strict"] == 0 and labels["mathd_or_sympy"] == 1:
        bucket["rescued_boxed"] += 1
    if labels["old_strict"] == 1 and labels["mathd_or_sympy"] == 0:
        bucket["regressed_boxed"] += 1


def _finalize(bucket: dict[str, int]) -> dict[str, Any]:
    n = bucket["n_rollouts"]
    p2 = bucket["parse_ok_rank2"]
    pb = bucket["parse_ok_boxed"]
    strict = bucket["old_strict_pass"]
    mos = bucket["mathd_or_sympy_pass"]

    def rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    return {
        "n_rollouts": n,
        "parse_ok_rank2": p2,
        "parse_ok_rank2_rate": rate(p2, n),
        "parse_ok_boxed": pb,
        "parse_ok_boxed_rate": rate(pb, n),
        "pass_old_strict": rate(strict, n),
        "pass_mathd_only": rate(bucket["mathd_only_pass"], n),
        "pass_sympy_only": rate(bucket["sympy_only_pass"], n),
        "pass_mathd_or_sympy": rate(mos, n),
        "rescued": bucket["rescued"],
        "regressed": bucket["regressed"],
        "lift_pp_mathd_or_sympy": rate(mos - strict, n),
        "pass_old_strict_given_parse": rate(strict, p2),
        "pass_mathd_or_sympy_given_parse": rate(mos, p2),
        "pass_old_strict_boxed": rate(bucket["old_strict_pass_boxed"], n),
        "pass_mathd_or_sympy_boxed": rate(bucket["mathd_or_sympy_pass_boxed"], n),
        "rescued_boxed": bucket["rescued_boxed"],
        "regressed_boxed": bucket["regressed_boxed"],
        "lift_pp_mathd_or_sympy_boxed": rate(
            bucket["mathd_or_sympy_pass_boxed"] - bucket["old_strict_pass_boxed"], n
        ),
    }


def analyze(
    manifest_path: Path,
    rollouts_path: Path,
    prompt_variant: str,
) -> dict[str, Any]:
    manifest = _read_jsonl(manifest_path)
    rollouts = _read_jsonl(rollouts_path)
    gold_by_pid = {int(m["problem_id"]): m["gold"] for m in manifest}
    band_by_pid = {int(m["problem_id"]): m["difficulty_band"] for m in manifest}

    overall = _empty_bucket()
    by_band: dict[str, dict[str, int]] = defaultdict(_empty_bucket)

    print(f"  rescoring {len(rollouts)} rollouts ({prompt_variant})...", flush=True)
    for row in rollouts:
        pid = int(row["problem_id"])
        gold = gold_by_pid[pid]
        band = band_by_pid[pid]
        completion = row["completion"]

        rank2 = extract_rank2(completion, gold, prompt_variant=prompt_variant)
        parsed_r2 = rank2.get("parsed_answer")
        parse_ok_r2 = bool(rank2["parse_ok_rank2"])

        boxed_raw = extract_answer(completion)
        parse_ok_boxed = boxed_raw is not None and str(boxed_raw).strip() != ""

        for bucket in (overall, by_band[band]):
            bucket["n_rollouts"] += 1
            if parse_ok_r2:
                bucket["parse_ok_rank2"] += 1
            if parse_ok_boxed:
                bucket["parse_ok_boxed"] += 1

        if parse_ok_r2 and parsed_r2 is not None:
            labels_a = _grade_pair(parsed_r2, gold)
            _accumulate(overall, labels_a)
            _accumulate(by_band[band], labels_a)

        if parse_ok_boxed and boxed_raw is not None:
            labels_b = _grade_pair(str(boxed_raw), gold)
            _accumulate_boxed(overall, labels_b, True)
            _accumulate_boxed(by_band[band], labels_b, True)

    return {
        "manifest": str(manifest_path),
        "rollouts": str(rollouts_path),
        "prompt_variant": prompt_variant,
        "n_manifest": len(manifest),
        "n_rollouts": len(rollouts),
        "overall": _finalize(overall),
        "by_band": {
            b: _finalize(by_band[b]) for b in sorted(by_band, key=_band_sort_key)
        },
    }


def _band_sort_key(band: str) -> tuple[int, int]:
    num, den = band.split("/")
    return (int(num), int(den))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# DeepScaleR mathd OR sympy rescore (Group A n800)",
        "",
        "**Dataset:** `main/data/probes/05-25/group_a_n800/` — 800 prompts × 8 rollouts = 6400.",
        "**Prompt arm:** `dapo_answer_v1` (arm A). Manifest gold is integer-only (Polaris probe).",
        "",
        "## Matchers",
        "",
        "| ID | Extraction | Grading |",
        "|----|------------|---------|",
        "| `old_strict` | Rank-2 `parsed_answer` when `parse_ok_rank2` | "
        "`normalize_final_answer` string equality (legacy strict) |",
        "| `mathd_only` | same | `grade_answer_mathd` (Hendrycks mathd normalize) |",
        "| `sympy_only` | same | `grade_answer_sympy` (integer gold → strict int match) |",
        "| `mathd_or_sympy` | same | mathd OR sympy (DeepScaleR train rule) |",
        "",
        "**Variant B (boxed):** last `\\boxed{}` via `extract_answer` on full completion; "
        "same graders. Parse failures count as fail for pass rates.",
        "",
        "**Rescued** = `parse_ok_rank2`, old_strict fail, mathd_or_sympy pass. "
        "**Regressed** = old_strict pass, mathd_or_sympy fail.",
        "",
        f"- Manifest: `{results['manifest']}`",
        f"- Rollouts: `{results['rollouts']}`",
        "",
        "## Overall (variant A: Rank-2 extraction)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    ov = results["overall"]
    lines.extend(
        [
            f"| Rollouts | {ov['n_rollouts']} |",
            f"| parse_ok_rank2 | {_pct(ov['parse_ok_rank2_rate'])} ({ov['parse_ok_rank2']}) |",
            f"| parse_ok_boxed (full completion) | {_pct(ov['parse_ok_boxed_rate'])} ({ov['parse_ok_boxed']}) |",
            f"| pass old_strict | {_pct(ov['pass_old_strict'])} |",
            f"| pass mathd_only | {_pct(ov['pass_mathd_only'])} |",
            f"| pass sympy_only | {_pct(ov['pass_sympy_only'])} |",
            f"| pass mathd_or_sympy | {_pct(ov['pass_mathd_or_sympy'])} |",
            f"| lift (mathd_or_sympy − strict) | {_pp(ov['lift_pp_mathd_or_sympy'])} pp |",
            f"| rescued (old fail → new pass) | {ov['rescued']} |",
            f"| regressed (old pass → new fail) | {ov['regressed']} |",
            "",
            "## Overall (variant B: boxed extract, mathd_or_sympy)",
            "",
            f"| pass old_strict (boxed) | {_pct(ov['pass_old_strict_boxed'])} |",
            f"| pass mathd_or_sympy (boxed) | {_pct(ov['pass_mathd_or_sympy_boxed'])} |",
            f"| lift | {_pp(ov['lift_pp_mathd_or_sympy_boxed'])} pp |",
            f"| rescued | {ov['rescued_boxed']} |",
            f"| regressed | {ov['regressed_boxed']} |",
            "",
            "## Per difficulty band (100 prompts × 8 rollouts = 800/band)",
            "",
            "| Band | n | parse_ok R2 | parse_ok boxed | strict | mathd∨sympy | rescued | regressed | lift pp |",
            "|------|---|-------------|----------------|--------|-------------|---------|-----------|---------|",
        ]
    )
    for band, b in results["by_band"].items():
        lines.append(
            f"| {band} | {b['n_rollouts']} | {_pct(b['parse_ok_rank2_rate'])} | "
            f"{_pct(b['parse_ok_boxed_rate'])} | {_pct(b['pass_old_strict'])} | "
            f"{_pct(b['pass_mathd_or_sympy'])} | {b['rescued']} | {b['regressed']} | "
            f"{_pp(b['lift_pp_mathd_or_sympy'])} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Graders in `main/train/math_grade_deepscaler.py` (rLLM / DeepScaleR).",
            "Script: `main/scripts/rescore_mathd_sympy.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/probes/05-25/group_a_n800/manifest.jsonl",
    )
    parser.add_argument(
        "--rollouts",
        default="data/probes/05-25/group_a_n800/phase1_rollouts.jsonl",
    )
    parser.add_argument(
        "--prompt-variant",
        default="dapo_answer_v1",
        dest="prompt_variant",
    )
    parser.add_argument(
        "--out-json",
        default="data/probes/05-25/group_a_n800/mathd_sympy_rescore.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/probes/mathd_sympy_rescore_n800.md",
    )
    args = parser.parse_args()

    manifest_path = (_MAIN_ROOT / args.manifest).resolve()
    rollouts_path = (_MAIN_ROOT / args.rollouts).resolve()

    results = analyze(manifest_path, rollouts_path, args.prompt_variant)

    out_json = (_MAIN_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    md = render_markdown(results)
    out_md = (_MAIN_ROOT / args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    print(md)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
