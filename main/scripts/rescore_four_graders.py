#!/usr/bin/env python3
"""Four-grader rescore on a saved rollouts jsonl.

Compares (on identical completions):
  G1 legacy_strict   = Rank-2 extract + normalize_final_answer string ==
  G2 mathd_or_sympy  = Rank-2 extract + DeepScaleR mathd OR sympy
  G3 math_verify     = Rank-2 extract -> math_verify.parse/verify vs gold
  G4 math_verify_box = last \boxed{} (no Rank-2 fallback) -> math_verify

Reports per-grader pass@1, pass@8, non-zero pass rate, and pairwise lifts.
Also dumps disagreements (G2/G3 says correct, G1 says wrong) for hand-check.
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

from train.math_grade_deepscaler import (  # noqa: E402
    extract_answer,
    grade_answer_mathd_or_sympy,
)
from train.reward import extract_rank2, normalize_final_answer  # noqa: E402

from math_verify import parse as mv_parse  # noqa: E402
from math_verify import verify as mv_verify  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mv_grade(pred: str | None, gold: str) -> int:
    if pred is None or str(pred).strip() == "":
        return 0
    try:
        p = mv_parse(str(pred))
        g = mv_parse(str(gold))
        return int(mv_verify(g, p))
    except Exception:
        return 0


def _grade_all(
    completion: str, gold: str, prompt_variant: str
) -> dict[str, int]:
    rank2 = extract_rank2(completion, gold, prompt_variant=prompt_variant)
    parsed_r2 = rank2.get("parsed_answer")
    parse_ok_r2 = bool(rank2["parse_ok_rank2"]) and parsed_r2 is not None

    boxed_raw = extract_answer(completion)

    if parse_ok_r2:
        g1 = int(
            normalize_final_answer(str(parsed_r2))
            == normalize_final_answer(gold)
        )
        g2 = int(grade_answer_mathd_or_sympy(str(parsed_r2), gold))
        g3 = _mv_grade(str(parsed_r2), gold)
    else:
        g1 = g2 = g3 = 0

    # G4: math_verify on last \boxed{} (no Rank-2 fallback)
    g4 = _mv_grade(boxed_raw, gold) if boxed_raw else 0

    return {
        "legacy_strict": g1,
        "mathd_or_sympy": g2,
        "math_verify": g3,
        "math_verify_box": g4,
    }


def analyze(
    manifest_path: Path,
    rollouts_path: Path,
    prompt_variant: str,
    n_rollouts: int = 8,
) -> dict[str, Any]:
    manifest = _read_jsonl(manifest_path)
    rollouts = _read_jsonl(rollouts_path)
    gold_by_pid = {int(m["problem_id"]): m["gold"] for m in manifest}

    graders = ["legacy_strict", "mathd_or_sympy", "math_verify", "math_verify_box"]

    # per-rollout counters
    pass_count = {g: 0 for g in graders}

    # per-prompt aggregation: max reward across rollouts
    # initialize with every pid in the manifest so denominators are correct
    by_prompt: dict[int, dict[str, int]] = {
        int(pid): {g: 0 for g in graders} for pid in gold_by_pid
    }
    seen_pids: set[int] = set()

    # disagreement samples for hand-check
    disagree_samples: dict[str, list[dict[str, Any]]] = {
        "mathd_or_sympy_rescues_strict": [],
        "math_verify_rescues_strict": [],
        "math_verify_box_rescues_strict": [],
        "strict_passes_math_verify_fails": [],  # regressions for math_verify
        "strict_passes_mathd_or_sympy_fails": [],  # regressions for G2
    }
    max_samples = 50

    n_used = 0
    for row in rollouts:
        pid = int(row["problem_id"])
        if pid not in gold_by_pid:
            continue
        gold = gold_by_pid[pid]
        completion = row["completion"]

        labels = _grade_all(completion, gold, prompt_variant)
        n_used += 1
        seen_pids.add(pid)
        for g in graders:
            pass_count[g] += labels[g]
            if labels[g]:
                by_prompt[pid][g] = 1  # max over rollouts

        # collect disagreements
        if labels["legacy_strict"] == 0:
            for g_better, key in [
                ("mathd_or_sympy", "mathd_or_sympy_rescues_strict"),
                ("math_verify", "math_verify_rescues_strict"),
                ("math_verify_box", "math_verify_box_rescues_strict"),
            ]:
                if labels[g_better] == 1 and len(disagree_samples[key]) < max_samples:
                    disagree_samples[key].append({
                        "problem_id": pid,
                        "rollout_idx": row.get("rollout_idx"),
                        "gold": gold,
                        "completion_tail": completion[-400:],
                    })
        # regressions: strict pass, lenient fail
        if labels["legacy_strict"] == 1:
            for g_worse, key in [
                ("math_verify", "strict_passes_math_verify_fails"),
                ("mathd_or_sympy", "strict_passes_mathd_or_sympy_fails"),
            ]:
                if labels[g_worse] == 0 and len(disagree_samples[key]) < max_samples:
                    disagree_samples[key].append({
                        "problem_id": pid,
                        "rollout_idx": row.get("rollout_idx"),
                        "gold": gold,
                        "completion_tail": completion[-400:],
                    })

    n_prompts = len(seen_pids)

    def rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    pass_at_1 = {g: rate(pass_count[g], n_used) for g in graders}
    pass_at_k = {
        g: rate(sum(1 for pid in seen_pids if by_prompt[pid][g]), n_prompts)
        for g in graders
    }
    non_zero_rate = pass_at_k  # identical metric

    return {
        "manifest": str(manifest_path),
        "rollouts": str(rollouts_path),
        "prompt_variant": prompt_variant,
        "n_rollouts_used": n_used,
        "n_prompts": n_prompts,
        "n_rollouts_per_prompt": n_rollouts,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "non_zero_pass_rate": non_zero_rate,
        "lift_pp_pass_at_1": {
            g: pass_at_1[g] - pass_at_1["legacy_strict"] for g in graders
        },
        "lift_pp_non_zero": {
            g: non_zero_rate[g] - non_zero_rate["legacy_strict"] for g in graders
        },
        "disagree_samples": disagree_samples,
    }


def render_markdown(r: dict[str, Any]) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.2f}%"

    def pp(x: float) -> str:
        return f"{100 * x:+.2f} pp"

    g_order = ["legacy_strict", "mathd_or_sympy", "math_verify", "math_verify_box"]
    g_labels = {
        "legacy_strict": "G1 legacy strict (Rank-2 + normalize ==)",
        "mathd_or_sympy": "G2 mathd OR sympy (Rank-2 extract)",
        "math_verify": "G3 math_verify (Rank-2 extract)",
        "math_verify_box": "G4 math_verify on raw `\\boxed{}` (no fallback)",
    }
    lines = [
        "# Four-grader rescore — does lenient grading lift step-0 non-zero rate?",
        "",
        f"- Rollouts: `{r['rollouts']}`",
        f"- Manifest: `{r['manifest']}`",
        f"- Prompt variant: `{r['prompt_variant']}`",
        f"- n_rollouts used: {r['n_rollouts_used']}",
        f"- n_prompts: {r['n_prompts']}  (×{r['n_rollouts_per_prompt']} rollouts)",
        "",
        "## Headline numbers",
        "",
        "| grader | pass@1 | pass@8 (≡ non-zero rate) | lift vs strict (non-zero) |",
        "|---|---|---|---|",
    ]
    for g in g_order:
        lines.append(
            f"| {g_labels[g]} | {pct(r['pass_at_1'][g])} | "
            f"{pct(r['non_zero_pass_rate'][g])} | "
            f"{pp(r['lift_pp_non_zero'][g])} |"
        )

    lines += [
        "",
        "## Disagreement counts (samples capped at 50 each)",
        "",
    ]
    for key, samples in r["disagree_samples"].items():
        lines.append(f"- **{key}**: {len(samples)} samples collected.")
    lines += [
        "",
        "Hand-check the JSON `disagree_samples.*` arrays to decide whether",
        "the lenient grader is genuinely correcting or sneaking false positives.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/probes/05-25/group_a_n800/manifest.jsonl",
    )
    parser.add_argument(
        "--rollouts",
        default="data/probes/05-25/prompt_c/phase1_rollouts.jsonl",
    )
    parser.add_argument(
        "--prompt-variant",
        default="hybrid_answer_boxed",
        dest="prompt_variant",
    )
    parser.add_argument(
        "--out-json",
        default="data/probes/05-25/prompt_c/four_grader_rescore.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/probes/four_grader_rescore.md",
    )
    args = parser.parse_args()

    manifest_path = (_MAIN_ROOT / args.manifest).resolve()
    rollouts_path = (_MAIN_ROOT / args.rollouts).resolve()

    print(f"manifest:  {manifest_path}", flush=True)
    print(f"rollouts:  {rollouts_path}", flush=True)
    print(f"variant:   {args.prompt_variant}", flush=True)
    print("rescoring...", flush=True)

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
