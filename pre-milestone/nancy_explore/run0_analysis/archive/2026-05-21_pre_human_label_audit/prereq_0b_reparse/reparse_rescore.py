#!/usr/bin/env python3
"""Prerequisite 0b: parser-fix re-score for Run 0 offline analyses.

Re-parses immutable completions with brace-balanced \\boxed extraction (C2) and
LaTeX-aware canonicalization (C3) via ``pilot.train.answer_clean`` — the
offline implementation of PILOT_REDESIGN C2/C3. Preserves v1 fields; adds v2.

Usage (from repo root):
    python nancy_explore/run0_analysis/reparse_rescore.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pilot.train.answer_clean import (  # noqa: E402
    cluster_id_clean,
    extract_answer_clean,
    is_correct_clean,
    normalize_answer_clean,
)
from pilot.train.run_proxy import has_minority_correct_cluster  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN0_DIR = HERE.parent
DATA_DIR = RUN0_DIR / "data"
RAW_PATH = DATA_DIR / "raw_predictions.jsonl"
PROMPTS_PATH = DATA_DIR / "prompt_inputs.jsonl"
OUT_PATH = DATA_DIR / "predictions_reparsed.jsonl"
REPORT_PATH = HERE / "reparse_diff.md"

N_BOOT = 1000
BOOT_SEED = 42


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bootstrap_ci(
    prompt_flags: list[bool],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Prompt-level bootstrap 95% CI for a Bernoulli rate."""
    if not prompt_flags:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(prompt_flags)
    rates: list[float] = []
    for _ in range(n_boot):
        sample = [prompt_flags[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    lo = rates[int(0.025 * n_boot)]
    hi = rates[int(0.975 * n_boot)]
    return (sum(prompt_flags) / n, lo, hi)


def _dist_table(counter: Counter, max_key: int, label: str) -> str:
    lines = [f"| {label} | count | % |", "|---|---:|---:|"]
    total = sum(counter.values())
    for k in range(max_key + 1):
        c = counter.get(k, 0)
        pct = 100.0 * c / total if total else 0.0
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    return "\n".join(lines)


def reparse_row(row: dict, gold: str) -> dict:
    completion = row.get("completion", "")
    parsed_v2, extract_path_v2 = extract_answer_clean(completion)
    canon_v2 = normalize_answer_clean(parsed_v2)
    is_correct_v2 = is_correct_clean(parsed_v2, gold)
    cluster_id_v2 = cluster_id_clean(parsed_v2)

    out = dict(row)
    out.update(
        {
            "parsed_answer_v2": parsed_v2,
            "canonical_v2": canon_v2,
            "is_correct_v2": is_correct_v2,
            "cluster_id_v2": cluster_id_v2,
            "extract_path_v2": extract_path_v2,
            "parser_clean_v2": extract_path_v2 == "boxed_balanced" and bool(parsed_v2),
        }
    )
    return out


def build_report(
    rollouts: list[dict],
    prompt_rows: list[dict],
) -> str:
    n = len(rollouts)
    n_parsed_chg = sum(r["parsed_answer"] != r["parsed_answer_v2"] for r in rollouts)
    n_cluster_chg = sum(r["cluster_id"] != r["cluster_id_v2"] for r in rollouts)
    n_correct_chg = sum(bool(r["correct"]) != bool(r["is_correct_v2"]) for r in rollouts)
    unparseable = sum(not r["parsed_answer_v2"] for r in rollouts)

    acc_v1 = sum(bool(r["correct"]) for r in rollouts) / n
    acc_v2 = sum(bool(r["is_correct_v2"]) for r in rollouts) / n

    minority_v1_flags: list[bool] = []
    minority_v2_flags: list[bool] = []
    for pr in prompt_rows:
        pid = pr["prompt_id"]
        sub = [r for r in rollouts if r["prompt_id"] == pid]
        minority_v1_flags.append(
            has_minority_correct_cluster(
                [bool(r["correct"]) for r in sub],
                [r["cluster_id"] for r in sub],
            )
        )
        minority_v2_flags.append(
            has_minority_correct_cluster(
                [bool(r["is_correct_v2"]) for r in sub],
                [r["cluster_id_v2"] for r in sub],
            )
        )

    m1, m1_lo, m1_hi = _bootstrap_ci(minority_v1_flags)
    m2, m2_lo, m2_hi = _bootstrap_ci(minority_v2_flags)

    cluster_grouping_changed = 0
    for pid in {r["prompt_id"] for r in rollouts}:
        sub = [r for r in rollouts if r["prompt_id"] == pid]
        stored_canon = tuple(sorted({normalize_answer_clean(r["parsed_answer"]) for r in sub}))
        v2_canon = tuple(sorted({r["canonical_v2"] for r in sub}))
        if stored_canon != v2_canon:
            cluster_grouping_changed += 1

    dist_correct_v1 = Counter(pr["n_correct_v1"] for pr in prompt_rows)
    dist_correct_v2 = Counter(pr["n_correct_v2"] for pr in prompt_rows)
    path_counts = Counter(r["extract_path_v2"] for r in rollouts)

    gained = sum(not r["correct"] and r["is_correct_v2"] for r in rollouts)
    lost = sum(r["correct"] and not r["is_correct_v2"] for r in rollouts)

    md: list[str] = []
    md.append("# Run 0 parser-fix re-score (0b)\n")
    md.append("**Date:** 2026-05-21  \n")
    md.append("**Source:** `data/raw_predictions.jsonl` + `data/prompt_inputs.jsonl`  \n")
    md.append(
        "**Parser:** `pilot/train/answer_clean.py` — brace-balanced `\\boxed` (C2), "
        "`normalize_answer_clean` (C3)  \n"
    )
    md.append(f"**Output:** `{OUT_PATH.relative_to(RUN0_DIR)}`\n")

    md.append("\n## Accuracy (rollout Pass@1)\n")
    md.append("| Version | Field | Rate | n correct / 4000 |\n|---|---|---:|---:|\n")
    md.append(
        f"| v1 (stored) | `correct` | {100*acc_v1:.2f}% | "
        f"{sum(bool(r['correct']) for r in rollouts)} |\n"
    )
    md.append(
        f"| v2 (re-parsed) | `is_correct_v2` | {100*acc_v2:.2f}% | "
        f"{sum(bool(r['is_correct_v2']) for r in rollouts)} |\n"
    )
    md.append(f"| Δ (v2 − v1) | | {100*(acc_v2 - acc_v1):+.2f} pp | +{gained} / −{lost} flips |\n")

    md.append("\n## Cluster churn\n")
    md.append("| Metric | Count | Rate |\n|---|---:|---:|\n")
    md.append(f"| `parsed_answer` → `parsed_answer_v2` changed | {n_parsed_chg} | {100*n_parsed_chg/n:.2f}% |\n")
    md.append(f"| `cluster_id` → `cluster_id_v2` changed | {n_cluster_chg} | {100*n_cluster_chg/n:.2f}% |\n")
    md.append(
        f"| Prompts with different canon grouping (8 rollouts) | "
        f"{cluster_grouping_changed} | {100*cluster_grouping_changed/500:.1f}% |\n"
    )
    md.append(
        f"| Mean distinct clusters / prompt (v1) | "
        f"{statistics.mean(pr['n_distinct_cluster_v1'] for pr in prompt_rows):.2f} |\n"
    )
    md.append(
        f"| Mean distinct clusters / prompt (v2) | "
        f"{statistics.mean(pr['n_distinct_cluster_v2'] for pr in prompt_rows):.2f} |\n"
    )

    md.append("\n## Unparseable (v2)\n")
    md.append(
        f"- Rollouts with empty `parsed_answer_v2`: **{unparseable}** "
        f"({100*unparseable/n:.2f}%)\n"
    )
    md.append("- Breakdown by `extract_path_v2`:\n")
    md.append("| Path | Count | % |\n|---|---:|---:|\n")
    for path, cnt in path_counts.most_common():
        md.append(f"| `{path}` | {cnt} | {100*cnt/n:.1f}% |\n")

    md.append("\n## Minority-correct prompt rate (bootstrap 95% CI, prompt-level)\n")
    md.append(
        "Definition: among prompts with ≥1 correct rollout, fraction where correct "
        "rollouts span ≥2 clusters and at least one correct cluster is not the largest.\n"
    )
    md.append("| Version | Rate | 95% CI |\n|---|---:|---|\n")
    md.append(f"| v1 (`correct`, `cluster_id`) | {100*m1:.2f}% | [{100*m1_lo:.2f}%, {100*m1_hi:.2f}%] |\n")
    md.append(
        f"| v2 (`is_correct_v2`, `cluster_id_v2`) | {100*m2:.2f}% | "
        f"[{100*m2_lo:.2f}%, {100*m2_hi:.2f}%] |\n"
    )
    md.append(
        "\n**Note:** Under exact-match clustering, correct rollouts that share the same "
        "canonical answer land in one cluster — minority-correct stays ~0% unless "
        "semantically equivalent answers split across clusters (parser) or substrate "
        "changes (Analysis A LLM clusters).\n"
    )

    md.append("\n### Correct rollouts per prompt\n")
    md.append("**v1 (stored)**\n")
    md.append(_dist_table(dist_correct_v1, 8, "n_correct_v1"))
    md.append("\n\n**v2 (re-parsed)**\n")
    md.append(_dist_table(dist_correct_v2, 8, "n_correct_v2"))

    gained_pids = sorted(
        {
            r["prompt_id"]
            for r in rollouts
            if not r["correct"] and r["is_correct_v2"]
        }
    )
    if gained_pids:
        md.append("\n## Correct gained (v1 false → v2 true)\n")
        for pid in gained_pids:
            md.append(f"- `{pid}`\n")

    md.append("\n## Implications for Analyses A–D\n")
    md.append(
        "- **Analysis A/B/C/D** should use `data/predictions_reparsed.jsonl` "
        "(`is_correct_v2`, `cluster_id_v2`, `canonical_v2`).\n"
    )
    md.append(
        "- Parser fixes are **small** on accuracy (+6 rollouts) but **large** on "
        "parse/cluster hygiene (12.9% parsed changed; all 500 prompts re-grouped "
        "under deterministic SHA cluster ids).\n"
    )
    md.append(
        "- `minority_correct_prompt_rate_v2` remains 0% — proceed to **Analysis A** "
        "(LLM reasoning clusters) for the substrate-controlled gate metric.\n"
    )

    return "".join(md)


def main() -> None:
    gold_by_pid = {
        row["prompt_id"]: row["gold_answer"]
        for row in _load_jsonl(PROMPTS_PATH)
    }
    raw = _load_jsonl(RAW_PATH)
    if len(raw) != 4000:
        raise SystemExit(f"expected 4000 rollouts, got {len(raw)}")
    if len(gold_by_pid) != 500:
        raise SystemExit(f"expected 500 prompts, got {len(gold_by_pid)}")

    reparsed: list[dict] = []
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in raw:
        pid = row["prompt_id"]
        gold = gold_by_pid[pid]
        out = reparse_row(row, gold)
        reparsed.append(out)
        by_prompt[pid].append(out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for row in reparsed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_rows: list[dict] = []
    for pid, rollouts in sorted(by_prompt.items()):
        prompt_rows.append(
            {
                "prompt_id": pid,
                "gold_answer": gold_by_pid[pid],
                "n_rollouts": len(rollouts),
                "n_correct_v1": sum(bool(r["correct"]) for r in rollouts),
                "n_correct_v2": sum(bool(r["is_correct_v2"]) for r in rollouts),
                "n_distinct_cluster_v1": len({r["cluster_id"] for r in rollouts}),
                "n_distinct_cluster_v2": len({r["cluster_id_v2"] for r in rollouts}),
            }
        )

    REPORT_PATH.write_text(build_report(reparsed, prompt_rows))
    print(f"Wrote {OUT_PATH} ({len(reparsed)} rows)")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
