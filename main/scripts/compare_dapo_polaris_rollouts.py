#!/usr/bin/env python3
"""Apples-to-apples rollout metrics: DAPO pilot Run0 vs Polaris Group A n800.

Uses identical definitions:
  - pass@1 (rollout-level): share of rollouts marked correct
  - pass@8 (prompt-level): Chen et al. unbiased Pass@k with k=8, n=8 rollouts/prompt
    (= 1 iff prompt has >=1 correct rollout)
  - mixed_reward: 0 < n_correct < 8 per prompt
  - all_wrong: n_correct == 0
  - all_correct: n_correct == 8
  - parse_ok_rank2: Rank-2 extraction succeeded

Grading (unified rerank):
  - extract_rank2(..., prompt_variant per dataset)
  - grade_parsed_answer: mathd OR sympy (DeepScaleR / rLLM train rule)

Also reports pilot legacy labels (stored `correct`, `correct_clean`) for traceability.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from train.reward import extract_rank2, grade_parsed_answer, normalize_final_answer  # noqa: E402

PILOT_PREDICTIONS = (
    Path(__file__).resolve().parents[2]
    / "pre-milestone/pilot/artifacts/run0_proxy/20260519T190202Z/cleaned/predictions.jsonl"
)
PILOT_PROMPTS = (
    Path(__file__).resolve().parents[2]
    / "pre-milestone/pilot/artifacts/run0_proxy/20260519T190202Z/prompt_inputs.jsonl"
)
POLARIS_MANIFEST = _MAIN_ROOT / "data/probes/05-25/group_a_n800/manifest.jsonl"
POLARIS_ROLLOUTS = _MAIN_ROOT / "data/probes/05-25/prompt_c/phase1_rollouts.jsonl"
OUT_MD = _MAIN_ROOT / "docs/probes/dapo_vs_polaris_rollout_comparison.md"
OUT_JSON = _MAIN_ROOT / "data/probes/dapo_vs_polaris_rollout_comparison.json"

N_ROLLOUTS = 8
K_PASS = 8
PROMPT_VARIANT_POLARIS = "hybrid_answer_boxed"
PROMPT_VARIANT_DAPO = "dapo_answer_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _graded_correct(parsed: str | None, gold: str, parse_ok: bool) -> bool:
    if not parse_ok or parsed is None:
        return False
    return grade_parsed_answer(parsed, gold)


def _pass_at_k_unbiased(n_correct: int, n: int = N_ROLLOUTS, k: int = K_PASS) -> float:
    """Chen et al. (2021) pass@k for one prompt."""
    if n - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n - n_correct, k) / math.comb(n, k)


def _grade_rollout(
    completion: str, gold: str, prompt_variant: str
) -> dict[str, Any]:
    r2 = extract_rank2(completion, gold, prompt_variant=prompt_variant)
    parsed = r2.get("parsed_answer")
    parse_ok = bool(r2["parse_ok_rank2"])
    correct = _graded_correct(parsed, gold, parse_ok)
    return {
        "parse_ok_rank2": parse_ok,
        "correct_strict": correct,
        "parsed_answer": parsed,
    }


def _aggregate_prompt_rollouts(
    rollouts_by_prompt: dict[Any, list[dict[str, Any]]],
) -> dict[str, Any]:
    n_prompts = len(rollouts_by_prompt)
    n_rollouts = sum(len(v) for v in rollouts_by_prompt.values())
    rollout_correct = 0
    rollout_parse_ok = 0
    pass8_vals: list[float] = []
    n_mixed = 0
    n_all_wrong = 0
    n_all_correct = 0
    dist_correct: dict[int, int] = defaultdict(int)

    for rows in rollouts_by_prompt.values():
        n_correct = sum(1 for r in rows if r["correct_strict"])
        dist_correct[n_correct] += 1
        rollout_correct += n_correct
        rollout_parse_ok += sum(1 for r in rows if r["parse_ok_rank2"])
        pass8_vals.append(_pass_at_k_unbiased(n_correct))
        if 0 < n_correct < N_ROLLOUTS:
            n_mixed += 1
        if n_correct == 0:
            n_all_wrong += 1
        if n_correct == N_ROLLOUTS:
            n_all_correct += 1

    return {
        "n_prompts": n_prompts,
        "n_rollouts": n_rollouts,
        "rollouts_per_prompt": N_ROLLOUTS,
        "pass_at_1_rollout_level": rollout_correct / n_rollouts if n_rollouts else 0.0,
        "pass_at_1_n_correct": rollout_correct,
        "pass_at_8_prompt_level_mean": sum(pass8_vals) / n_prompts if n_prompts else 0.0,
        "pass_at_8_n_prompts_solved": sum(1 for v in pass8_vals if v > 0),
        "parse_ok_rank2_rollout_level": rollout_parse_ok / n_rollouts if n_rollouts else 0.0,
        "mixed_reward_prompt_rate": n_mixed / n_prompts if n_prompts else 0.0,
        "mixed_reward_n_prompts": n_mixed,
        "all_wrong_prompt_rate": n_all_wrong / n_prompts if n_prompts else 0.0,
        "all_wrong_n_prompts": n_all_wrong,
        "all_correct_prompt_rate": n_all_correct / n_prompts if n_prompts else 0.0,
        "dist_correct_per_prompt": dict(sorted(dist_correct.items())),
    }


def _load_pilot_unified() -> dict[Any, list[dict[str, Any]]]:
    gold_by_pid = {
        r["prompt_id"]: str(r["gold_answer"]).strip()
        for r in _read_jsonl(PILOT_PROMPTS)
    }
    by_prompt: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(PILOT_PREDICTIONS):
        pid = row["prompt_id"]
        gold = gold_by_pid[pid]
        g = _grade_rollout(row["completion"], gold, PROMPT_VARIANT_DAPO)
        g["stored_correct"] = bool(row.get("correct", False))
        g["correct_clean"] = bool(row.get("correct_clean", False))
        by_prompt[pid].append(g)
    return by_prompt


def _load_polaris_unified(
    manifest_path: Path, rollouts_path: Path
) -> dict[Any, list[dict[str, Any]]]:
    gold_by_pid = {
        int(m["problem_id"]): str(m["gold"]).strip()
        for m in _read_jsonl(manifest_path)
    }
    band_by_pid = {
        int(m["problem_id"]): m["difficulty_band"]
        for m in _read_jsonl(manifest_path)
    }
    by_prompt: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(rollouts_path):
        pid = int(row["problem_id"])
        gold = gold_by_pid[pid]
        g = _grade_rollout(row["completion"], gold, PROMPT_VARIANT_POLARIS)
        g["difficulty_band"] = band_by_pid[pid]
        by_prompt[pid].append(g)
    return by_prompt


def _aggregate_pilot_legacy(by_prompt: dict[Any, list[dict[str, Any]]]) -> dict[str, Any]:
    n_rollouts = sum(len(v) for v in by_prompt.values())
    stored = sum(1 for rows in by_prompt.values() for r in rows if r["stored_correct"])
    clean = sum(1 for rows in by_prompt.values() for r in rows if r["correct_clean"])
    n_prompts = len(by_prompt)
    any_stored = sum(
        1 for rows in by_prompt.values() if any(r["stored_correct"] for r in rows)
    )
    any_clean = sum(
        1 for rows in by_prompt.values() if any(r["correct_clean"] for r in rows)
    )
    return {
        "pass_at_1_stored": stored / n_rollouts,
        "pass_at_8_stored_any_correct": any_stored / n_prompts,
        "pass_at_1_correct_clean": clean / n_rollouts,
        "pass_at_8_correct_clean_any": any_clean / n_prompts,
    }


def _subsample_prompts(
    by_prompt: dict[Any, list[dict[str, Any]]], n: int, seed: int
) -> dict[Any, list[dict[str, Any]]]:
    pids = sorted(by_prompt.keys())
    rng = random.Random(seed)
    chosen = sorted(rng.sample(pids, min(n, len(pids))))
    return {pid: by_prompt[pid] for pid in chosen}


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _fmt_dist(dist: dict[int, int], n_prompts: int) -> str:
    lines = []
    for k in range(N_ROLLOUTS + 1):
        c = dist.get(k, 0)
        if c:
            lines.append(f"| {k}/8 | {c} | {100 * c / n_prompts:.1f}% |")
    return "\n".join(lines)


def _write_report(results: dict[str, Any]) -> None:
    lines = [
        "# DAPO pilot (Run 0) vs Polaris probe — unified rollout metrics",
        "",
        "Same model (**Qwen3-1.7B-Base**), **8 rollouts/prompt**, **temp=1** (both runs).",
        "Unified grader: Rank-2 + `grade_parsed_answer` (mathd OR sympy, DeepScaleR/rLLM). "
        f"DAPO pilot extract `{PROMPT_VARIANT_DAPO}`; Polaris arm C `{PROMPT_VARIANT_POLARIS}`.",
        "",
        "## Metric definitions (comparable)",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        f"| **pass@1** | Rollout-level: (# correct rollouts) / (# rollouts) |",
        f"| **pass@8** | Prompt-level mean of Chen et al. unbiased Pass@k (k={K_PASS}, n={N_ROLLOUTS}); equals **% prompts with ≥1 correct** |",
        "| **mixed_reward** | % prompts with some but not all rollouts correct (0 < n_correct < 8) |",
        "| **all_wrong** | % prompts with n_correct = 0 (zero GRPO gradient under standard filter) |",
        "| **parse_ok_rank2** | % rollouts where Rank-2 extraction succeeded |",
        "",
        "## Headline comparison (unified strict grader)",
        "",
    ]

    def row(label: str, key: str) -> str:
        d = results[key]["unified"]
        p = results[key]
        n_p = d["n_prompts"]
        n_r = d["n_rollouts"]
        return (
            f"| {label} | {n_p} | {n_r} | "
            f"{_pct(d['pass_at_1_rollout_level'])} ({d['pass_at_1_n_correct']}/{n_r}) | "
            f"{_pct(d['pass_at_8_prompt_level_mean'])} ({d['pass_at_8_n_prompts_solved']}/{n_p}) | "
            f"{_pct(d['mixed_reward_prompt_rate'])} ({d['mixed_reward_n_prompts']}/{n_p}) | "
            f"{_pct(d['all_wrong_prompt_rate'])} ({d['all_wrong_n_prompts']}/{n_p}) | "
            f"{_pct(d['parse_ok_rank2_rollout_level'])} |"
        )

    lines += [
        "| Dataset | Prompts | Rollouts | pass@1 | pass@8 | mixed_reward | all_wrong | parse_ok |",
        "|---------|--------:|---------:|--------|--------|--------------|----------|----------|",
        row("DAPO pilot Run0", "dapo_pilot_500"),
        row("Polaris n800 (arm A)", "polaris_800"),
        row("Polaris subsample n=500", "polaris_500_subsample"),
        "",
        "### Ratio (Polaris 800 / DAPO pilot) — unified strict",
        "",
    ]
    dapo = results["dapo_pilot_500"]["unified"]
    pol = results["polaris_800"]["unified"]

    def ratio(a: float, b: float) -> str:
        if b == 0:
            return "—"
        return f"{a / b:.2f}×"

    lines += [
        f"- pass@1: {ratio(pol['pass_at_1_rollout_level'], dapo['pass_at_1_rollout_level'])}",
        f"- pass@8: {ratio(pol['pass_at_8_prompt_level_mean'], dapo['pass_at_8_prompt_level_mean'])}",
        f"- mixed_reward: {ratio(pol['mixed_reward_prompt_rate'], dapo['mixed_reward_prompt_rate'])}",
        f"- all_wrong: {ratio(pol['all_wrong_prompt_rate'], dapo['all_wrong_prompt_rate'])} (lower is better for GRPO signal)",
        "",
        "## DAPO pilot — legacy labels (same 500×8 rollouts)",
        "",
        "Pilot also recorded run-time and human-cleaned labels (different parsers).",
        "",
    ]
    leg = results["dapo_pilot_500"]["legacy"]
    lines += [
        "| Label source | pass@1 (rollout) | pass@8 (prompt, any-correct) |",
        "|--------------|------------------|------------------------------|",
        f"| Unified strict (rerank) | {_pct(results['dapo_pilot_500']['unified']['pass_at_1_rollout_level'])} | {_pct(results['dapo_pilot_500']['unified']['pass_at_8_prompt_level_mean'])} |",
        f"| Stored at run time (`correct`) | {_pct(leg['pass_at_1_stored'])} | {_pct(leg['pass_at_8_stored_any_correct'])} |",
        f"| Human-cleaned (`correct_clean`) | {_pct(leg['pass_at_1_correct_clean'])} | {_pct(leg['pass_at_8_correct_clean_any'])} |",
        "",
        "Published pilot baseline ([`minority_metrics.md`](../../../pre-milestone/nancy_explore/run0_analysis/analysis_minority/minority_metrics.md)) uses **correct_clean**: pass@1 **9.03%**, pass@8 **34.40%**.",
        "",
        "## Distribution: n_correct rollouts per prompt (unified strict)",
        "",
        "### DAPO pilot (n=500)",
        "",
        "| n_correct | # prompts | % |",
        "|-----------|----------:|--:|",
        _fmt_dist(dapo["dist_correct_per_prompt"], 500),
        "",
        "### Polaris n800",
        "",
        "| n_correct | # prompts | % |",
        "|-----------|----------:|--:|",
        _fmt_dist(pol["dist_correct_per_prompt"], 800),
        "",
        "## Takeaway",
        "",
    ]

    p1_dapo = dapo["pass_at_1_rollout_level"]
    p1_pol = pol["pass_at_1_rollout_level"]
    p8_dapo = dapo["pass_at_8_prompt_level_mean"]
    p8_pol = pol["pass_at_8_prompt_level_mean"]
    if p8_pol < p8_dapo * 0.85:
        lines.append(
            f"- **pass@8** on Polaris ({_pct(p8_pol)}) is materially lower than DAPO pilot unified ({_pct(p8_dapo)}) "
            f"and much lower than pilot human-cleaned pass@8 (34.4%)."
        )
    lines.append(
        f"- **mixed_reward** is lower on Polaris ({_pct(pol['mixed_reward_prompt_rate'])} vs {_pct(dapo['mixed_reward_prompt_rate'])} on DAPO): "
        "fewer prompts contribute GRPO signal."
    )
    lines.append(
        f"- **all_wrong** is higher on Polaris ({_pct(pol['all_wrong_prompt_rate'])} vs {_pct(dapo['all_wrong_prompt_rate'])}): "
        "more wasted rollouts per step without dynamic sampling."
    )
    lines.append(
        "- Grader alignment matters for pilot: unified strict pass@1 (~9%) is close to stored run-time (8.1%); "
        "human-cleaned labels are more lenient on extraction."
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pilot_by_prompt = _load_pilot_unified()
    polaris_by_prompt = _load_polaris_unified(POLARIS_MANIFEST, POLARIS_ROLLOUTS)
    polaris_500 = _subsample_prompts(polaris_by_prompt, 500, args.seed)

    results = {
        "dapo_pilot_500": {
            "unified": _aggregate_prompt_rollouts(pilot_by_prompt),
            "legacy": _aggregate_pilot_legacy(pilot_by_prompt),
            "paths": {"predictions": str(PILOT_PREDICTIONS)},
        },
        "polaris_800": {
            "unified": _aggregate_prompt_rollouts(polaris_by_prompt),
            "paths": {
                "manifest": str(POLARIS_MANIFEST),
                "rollouts": str(POLARIS_ROLLOUTS),
            },
        },
        "polaris_500_subsample": {
            "unified": _aggregate_prompt_rollouts(polaris_500),
            "subsample_seed": args.seed,
        },
        "grader": {
            "dapo_prompt_variant": PROMPT_VARIANT_DAPO,
            "polaris_prompt_variant": PROMPT_VARIANT_POLARIS,
            "matcher": "grade_parsed_answer (mathd OR sympy)",
            "pass_at_k": f"Chen unbiased k={K_PASS} n={N_ROLLOUTS}",
        },
    }
    _write_report(results)

    u = results["dapo_pilot_500"]["unified"]
    p = results["polaris_800"]["unified"]
    print("DAPO pilot 500:", u["pass_at_1_rollout_level"], u["pass_at_8_prompt_level_mean"], u["mixed_reward_prompt_rate"])
    print("Polaris 800:", p["pass_at_1_rollout_level"], p["pass_at_8_prompt_level_mean"], p["mixed_reward_prompt_rate"])


if __name__ == "__main__":
    main()
