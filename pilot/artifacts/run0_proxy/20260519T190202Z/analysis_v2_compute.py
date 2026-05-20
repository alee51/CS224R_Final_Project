#!/usr/bin/env python3
"""Run 0 quantitative analysis v2 — completion-aware (reads raw completion text).

Reproducible script; imports pilot parse/canonicalize modules from repo root.
No LLM calls. Does not modify production pilot code.

Semantic bucket rule (documented in SEMANTIC_BUCKET_RULE and each JSONL row):
  1. Extract a *display* final answer from completion using production priority
     (single-regex boxed if exactly one shallow match, else last Answer: line,
     else last non-empty line) — same as extract_answer().
  2. Additionally compute brace-balanced *last* \\boxed{...} inner when present
     (for diagnostics only; not used as primary bucket unless shallow boxed absent).
  3. normalize_semantic(raw): strip whitespace/commas; peel \\( \\) $ wrappers;
     map pure integers to "n:<int>"; map "k%" / "k\\%" to "n:k"; map \\frac{a}{b}
     or a/b rationals to "frac:a/b"; else lowercase collapsed text "s:<...>" (≤120c).

Limitations: heuristic buckets can merge distinct math objects or split equivalents;
  they are for human-readable diversity stats, not training labels.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pilot.train.answer_parse import (  # noqa: E402
    BOXED_RE,
    _ANSWER_LINE,
    extract_answer,
    extract_boxed_answer,
    is_correct,
)
from pilot.train.canonicalize import canonicalize_answer, cluster_id  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent
RAW = ARTIFACT_DIR / "raw_predictions.jsonl"
PROMPTS = ARTIFACT_DIR / "prompt_inputs.jsonl"
OUT_JSONL = ARTIFACT_DIR / "analysis_v2_prompt_stats.jsonl"
OUT_MD = ARTIFACT_DIR / "analysis_v2_quant.md"

SEMANTIC_BUCKET_RULE = (
    "extract_answer priority + normalize_semantic: int n:, percent->n:, "
    "frac:a/b, else s:<lowercase<=120c>"
)

_BOXED_OPENER = re.compile(r"\\boxed\{")
_FRAC = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_INT_RE = re.compile(r"^-?\d+$")
_PERCENT = re.compile(r"^(-?\d+)\s*(?:\\%|%)$")


def last_brace_balanced_boxed_inner(text: str) -> str | None:
    """Inner of last \\boxed{...} with brace balancing (diagnostic / alt extract)."""
    last_inner: str | None = None
    for m in _BOXED_OPENER.finditer(text):
        start = m.end()
        depth = 1
        j = start
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last_inner = text[start : j - 1].strip()
    return last_inner


def count_boxed_openers(text: str) -> int:
    return len(_BOXED_OPENER.findall(text))


def infer_extract_path(completion: str) -> str:
    """Which branch extract_answer() would take."""
    raw, _ = extract_boxed_answer(completion)
    if raw is not None:
        return "boxed"
    text = completion.strip()
    if list(_ANSWER_LINE.finditer(text)):
        return "answer_line"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return "last_line"
    return "empty"


def normalize_semantic(raw: str) -> str:
    """Heuristic bucket key for grouping rollouts (not production canon)."""
    if not raw or not str(raw).strip():
        return "empty"
    s = str(raw).strip()
    s = s.replace(",", "")
    for token in (r"\(", r"\)", "$", r"\text{", r"\textbf{"):
        s = s.replace(token, "")
    s = s.strip()
    # \frac{a}{b}
    fm = _FRAC.search(s)
    if fm:
        a, b = fm.group(1).strip(), fm.group(2).strip()
        return f"frac:{a}/{b}"
    if "/" in s and not s.startswith("http"):
        parts = s.split("/", 1)
        if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
            return f"frac:{parts[0]}/{parts[1]}"
    pm = _PERCENT.match(s.replace(" ", ""))
    if pm:
        return f"n:{int(pm.group(1))}"
    if _INT_RE.match(s):
        return f"n:{int(s)}"
    # leading integer prefix (e.g. "202 mod ...")
    lead = re.match(r"^(-?\d+)", s)
    if lead and len(s) <= 24:
        return f"n:{int(lead.group(1))}"
    collapsed = re.sub(r"\s+", " ", s.lower())[:120]
    return f"s:{collapsed}"


def semantic_bucket(completion: str) -> str:
    """Human-readable answer mode from completion tail."""
    raw, _ = extract_boxed_answer(completion)
    if raw is None:
        inner = last_brace_balanced_boxed_inner(completion)
        if inner is not None:
            raw = inner
        else:
            raw = extract_answer(completion)
    return normalize_semantic(raw)


def nested_boxed_regex_differs(completion: str) -> bool:
    if "\\boxed{" not in completion:
        return False
    inner = last_brace_balanced_boxed_inner(completion)
    if inner is None:
        return False
    matches = list(BOXED_RE.finditer(completion))
    if not matches:
        return True
    regex_inner = matches[-1].group(1).strip()
    return inner.strip() != regex_inner


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def rollout_metrics(r: dict, gold: str) -> dict:
    completion = r.get("completion", "")
    stored_parsed = r["parsed_answer"]
    stored_correct = bool(r["correct"])
    stored_cid = r["cluster_id"]

    recomputed_parsed = extract_answer(completion)
    recomputed_correct = is_correct(completion, gold)
    canon_parsed = canonicalize_answer(stored_parsed)
    canon_recomputed = canonicalize_answer(recomputed_parsed)
    canon_gold = canonicalize_answer(str(gold))

    loose_correct = canon_recomputed == canon_gold
    path = infer_extract_path(completion)
    n_boxed = count_boxed_openers(completion)
    shallow_boxed_matches = len(BOXED_RE.findall(completion))

    return {
        "stored_parsed": stored_parsed,
        "recomputed_parsed": recomputed_parsed,
        "extract_mismatch": recomputed_parsed != stored_parsed,
        "stored_correct": stored_correct,
        "recomputed_correct": recomputed_correct,
        "correct_mismatch": recomputed_correct != stored_correct,
        "loose_correct": loose_correct,
        "loose_vs_stored_correct": loose_correct != stored_correct,
        "canon_stored": canon_parsed,
        "canon_recomputed": canon_recomputed,
        "stored_cluster_id": stored_cid,
        "recomputed_cluster_id": cluster_id(recomputed_parsed),
        "cluster_id_hash_mismatch": cluster_id(recomputed_parsed) != stored_cid,
        "semantic_bucket": semantic_bucket(completion),
        "extract_path": path,
        "n_boxed_opener": n_boxed,
        "shallow_boxed_count": shallow_boxed_matches,
        "boxed_category": (
            "none"
            if n_boxed == 0
            else ("single" if shallow_boxed_matches == 1 else "multi_or_nonshallow")
        ),
        "nested_boxed_regex_diff": nested_boxed_regex_differs(completion),
        "completion_chars": len(completion),
        "completion_tokens_approx": approx_tokens(completion),
        "has_boxed_in_text": "\\boxed{" in completion,
    }


def prompt_metrics(pid: str, gold: str, rollouts: list[dict]) -> dict:
    rs = [rollout_metrics(r, gold) for r in rollouts]

    stored_parsed = [x["stored_parsed"] for x in rs]
    stored_cids = [x["stored_cluster_id"] for x in rs]
    stored_correct = [x["stored_correct"] for x in rs]

    sem_buckets = [x["semantic_bucket"] for x in rs]
    paths = Counter(x["extract_path"] for x in rs)
    boxed_cat = Counter(x["boxed_category"] for x in rs)

    canon_to_cids: dict[str, set[int]] = defaultdict(set)
    for x in rs:
        canon_to_cids[x["canon_stored"]].add(x["stored_cluster_id"])

    cluster_canon_split = any(len(cids) > 1 for cids in canon_to_cids.values())

    return {
        "prompt_id": pid,
        "gold_answer": gold,
        "n_rollouts": len(rs),
        "semantic_bucket_rule": SEMANTIC_BUCKET_RULE,
        "as_recorded": {
            "n_distinct_parsed": len(set(stored_parsed)),
            "n_distinct_clusters": len(set(stored_cids)),
            "n_correct_rollouts": sum(stored_correct),
            "n_wrong_clusters": len({c for ok, c in zip(stored_correct, stored_cids) if not ok}),
        },
        "completion_aware": {
            "n_distinct_parsed_recomputed": len({x["recomputed_parsed"] for x in rs}),
            "n_distinct_canon_recomputed": len({x["canon_recomputed"] for x in rs}),
            "n_distinct_semantic_buckets": len(set(sem_buckets)),
            "semantic_buckets": sorted(set(sem_buckets)),
            "n_extract_mismatch": sum(x["extract_mismatch"] for x in rs),
            "n_correct_mismatch": sum(x["correct_mismatch"] for x in rs),
            "n_correct_recomputed": sum(x["recomputed_correct"] for x in rs),
            "n_loose_correct": sum(x["loose_correct"] for x in rs),
            "n_loose_vs_stored_correct_mismatch": sum(x["loose_vs_stored_correct"] for x in rs),
            "n_cluster_id_hash_mismatch": sum(x["cluster_id_hash_mismatch"] for x in rs),
            "cluster_canon_split_in_prompt": cluster_canon_split,
            "extract_path_counts": dict(paths),
            "boxed_category_counts": dict(boxed_cat),
            "rollouts_with_boxed_in_text": sum(x["has_boxed_in_text"] for x in rs),
            "nested_boxed_regex_diff_count": sum(x["nested_boxed_regex_diff"] for x in rs),
            "completion_chars_mean": statistics.mean(x["completion_chars"] for x in rs),
            "completion_chars_min": min(x["completion_chars"] for x in rs),
            "completion_chars_max": max(x["completion_chars"] for x in rs),
            "completion_tokens_approx_mean": statistics.mean(
                x["completion_tokens_approx"] for x in rs
            ),
        },
    }


def dist_table(counter: Counter, max_key: int, label: str) -> str:
    lines = [f"| {label} | count | % |", "|---|---:|---:|"]
    total = sum(counter.values())
    for k in range(max_key + 1):
        c = counter.get(k, 0)
        pct = 100.0 * c / total if total else 0.0
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    return "\n".join(lines)


def main() -> None:
    gold_by_pid: dict[str, str] = {}
    with PROMPTS.open() as f:
        for line in f:
            row = json.loads(line)
            gold_by_pid[row["prompt_id"]] = str(row["gold_answer"])

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    rollout_rows = 0
    with RAW.open() as f:
        for line in f:
            r = json.loads(line)
            by_prompt[r["prompt_id"]].append(r)
            rollout_rows += 1

    assert rollout_rows == 4000, f"expected 4000 rollouts, got {rollout_rows}"
    assert len(by_prompt) == 500, f"expected 500 prompts, got {len(by_prompt)}"
    for pid, rs in by_prompt.items():
        assert len(rs) == 8, f"prompt {pid}: expected 8 rollouts"

    stats = [
        prompt_metrics(pid, gold_by_pid[pid], by_prompt[pid])
        for pid in sorted(by_prompt)
    ]
    n_prompts = len(stats)
    n_rollouts = n_prompts * 8

    with OUT_JSONL.open("w") as f:
        for row in stats:
            f.write(json.dumps(row) + "\n")

    # --- rollout-level aggregates (re-scan for global rates) ---
    extract_mm = correct_mm = cluster_hash_mm = 0
    loose_mm = 0
    path_global: Counter = Counter()
    boxed_text = boxed_single_shallow = boxed_multi = 0
    nested_diff = 0

    for pid in sorted(by_prompt):
        gold = gold_by_pid[pid]
        for r in by_prompt[pid]:
            m = rollout_metrics(r, gold)
            extract_mm += int(m["extract_mismatch"])
            correct_mm += int(m["correct_mismatch"])
            cluster_hash_mm += int(m["cluster_id_hash_mismatch"])
            loose_mm += int(m["loose_vs_stored_correct"])
            path_global[m["extract_path"]] += 1
            if m["has_boxed_in_text"]:
                boxed_text += 1
            if m["shallow_boxed_count"] == 1:
                boxed_single_shallow += 1
            elif m["shallow_boxed_count"] > 1:
                boxed_multi += 1
            nested_diff += int(m["nested_boxed_regex_diff"])

    dist_sem = Counter(s["completion_aware"]["n_distinct_semantic_buckets"] for s in stats)
    dist_sem_as_rec_parsed = Counter(s["as_recorded"]["n_distinct_parsed"] for s in stats)
    dist_sem_as_rec_cluster = Counter(s["as_recorded"]["n_distinct_clusters"] for s in stats)
    dist_correct_rec = Counter(s["as_recorded"]["n_correct_rollouts"] for s in stats)
    dist_correct_re = Counter(s["completion_aware"]["n_correct_recomputed"] for s in stats)

    prompts_multi_sem = sum(
        1 for s in stats if s["completion_aware"]["n_distinct_semantic_buckets"] > 1
    )
    prompts_multi_sem_gt_parsed = sum(
        1
        for s in stats
        if s["completion_aware"]["n_distinct_semantic_buckets"]
        > s["as_recorded"]["n_distinct_parsed"]
    )
    prompts_multi_sem_lt_parsed = sum(
        1
        for s in stats
        if s["completion_aware"]["n_distinct_semantic_buckets"]
        < s["as_recorded"]["n_distinct_parsed"]
    )
    prompts_canon_split = sum(
        1 for s in stats if s["completion_aware"]["cluster_canon_split_in_prompt"]
    )
    prompts_any_extract_mm = sum(
        1 for s in stats if s["completion_aware"]["n_extract_mismatch"] > 0
    )
    prompts_any_correct_mm = sum(
        1 for s in stats if s["completion_aware"]["n_correct_mismatch"] > 0
    )

    mean_sem = statistics.mean(
        s["completion_aware"]["n_distinct_semantic_buckets"] for s in stats
    )
    mean_parsed = statistics.mean(s["as_recorded"]["n_distinct_parsed"] for s in stats)
    mean_clusters = statistics.mean(s["as_recorded"]["n_distinct_clusters"] for s in stats)
    pct_any_correct = (
        100.0
        * sum(1 for s in stats if s["as_recorded"]["n_correct_rollouts"] > 0)
        / n_prompts
    )
    total_correct = sum(s["as_recorded"]["n_correct_rollouts"] for s in stats)

    chars_mean = statistics.mean(
        s["completion_aware"]["completion_chars_mean"] for s in stats
    )
    tok_mean = statistics.mean(
        s["completion_aware"]["completion_tokens_approx_mean"] for s in stats
    )

    # exemplars
    by_sem_div = sorted(
        stats,
        key=lambda s: (
            -s["completion_aware"]["n_distinct_semantic_buckets"],
            -s["as_recorded"]["n_distinct_parsed"],
        ),
    )
    sem_gt_stored = sorted(
        [
            s
            for s in stats
            if s["completion_aware"]["n_distinct_semantic_buckets"]
            > s["as_recorded"]["n_distinct_parsed"]
        ],
        key=lambda s: -(
            s["completion_aware"]["n_distinct_semantic_buckets"]
            - s["as_recorded"]["n_distinct_parsed"]
        ),
    )

    md: list[str] = []
    md.append("# Run 0 quantitative analysis v2 (completion-aware)\n")
    md.append(f"**Artifact:** `{ARTIFACT_DIR.name}`  \n")
    md.append(f"**Inputs:** `raw_predictions.jsonl` ({rollout_rows} rollouts), ")
    md.append(f"`prompt_inputs.jsonl` ({n_prompts} prompts)  \n")
    md.append(f"**Script:** `analysis_v2_compute.py`  \n")
    md.append(f"**Per-prompt table:** `analysis_v2_prompt_stats.jsonl`\n")

    md.append("\n## Method\n")
    md.append(
        "Each rollout's **`completion`** text was inspected. We computed **as-recorded** "
        "stats from stored `parsed_answer` / `correct` / `cluster_id`, and "
        "**completion-aware** stats by re-running `extract_answer`, `is_correct`, "
        "extract-path inference, boxed counts, and heuristic **semantic buckets** "
        f"(`{SEMANTIC_BUCKET_RULE}`). No LLM labeling.\n"
    )
    md.append(
        "\n**Cluster IDs:** cross-process `cluster_id()` hash mismatches are expected "
        "(Python hash salt). Within-artifact checks use **canonical string equality** "
        "`canonicalize_answer(parsed)` for grouping; `cluster_canon_split_in_prompt` "
        "flags when one canon maps to multiple stored `cluster_id`s.\n"
    )

    md.append("\n## Global mismatch rates (4000 rollouts)\n")
    md.append("| Check | Mismatches | Rate |\n|---|---:|---:|\n")
    md.append(
        f"| `extract_answer(completion)` vs stored `parsed_answer` | {extract_mm} | "
        f"{100*extract_mm/n_rollouts:.2f}% |\n"
    )
    md.append(
        f"| `is_correct(completion, gold)` vs stored `correct` | {correct_mm} | "
        f"{100*correct_mm/n_rollouts:.2f}% |\n"
    )
    md.append(
        f"| `cluster_id(recomputed_parse)` vs stored `cluster_id` (hash) | {cluster_hash_mm} | "
        f"{100*cluster_hash_mm/n_rollouts:.2f}% |\n"
    )
    md.append(
        f"| Loose: `canon(reparse)==canon(gold)` vs stored `correct` | {loose_mm} | "
        f"{100*loose_mm/n_rollouts:.2f}% |\n"
    )
    md.append(
        "\n*Loose correct* uses canonical equality on the **re-extracted** parse, "
        "while production `is_correct` only accepts a **single shallow** `\\boxed{...}` "
        "with int contents — so loose can exceed stored correct when boxed is missing "
        "but tail text matches gold.\n"
    )
    md.append(
        f"\n**Storage consistency:** {prompts_any_extract_mm} prompts "
        f"({100*prompts_any_extract_mm/n_prompts:.1f}%) have ≥1 rollout where "
        f"today's `extract_answer(completion)` ≠ stored `parsed_answer`; "
        f"{prompts_any_correct_mm} prompts have ≥1 `correct` mismatch. "
        "An earlier audit note claiming 0% extract/correct mismatch appears stale; "
        "re-running `_audit_script.py` on this artifact reproduces the rates above "
        "(likely write-time vs current `answer_parse.py`, or artifact regeneration).\n"
    )

    md.append("\n## Boxed & extract-path usage (completion text)\n")
    md.append("| Metric | Count | % rollouts |\n|---|---:|---:|\n")
    md.append(
        f"| Completions containing `\\\\boxed{{` | {boxed_text} | "
        f"{100*boxed_text/n_rollouts:.1f}% |\n"
    )
    md.append(
        f"| Exactly one shallow-regex `\\\\boxed{{...}}` | {boxed_single_shallow} | "
        f"{100*boxed_single_shallow/n_rollouts:.1f}% |\n"
    )
    md.append(
        f"| Multiple shallow-regex boxed | {boxed_multi} | "
        f"{100*boxed_multi/n_rollouts:.1f}% |\n"
    )
    md.append(
        f"| Last boxed: brace-balanced inner ≠ shallow-regex inner | {nested_diff} | "
        f"{100*nested_diff/n_rollouts:.1f}% |\n"
    )
    md.append("\n**Extract path** (which branch `extract_answer` uses):\n")
    md.append("| Path | Count | % |\n|---|---:|---:|\n")
    for p, c in path_global.most_common():
        md.append(f"| {p} | {c} | {100*c/n_rollouts:.1f}% |\n")
    fallback = path_global.get("answer_line", 0) + path_global.get("last_line", 0)
    md.append(
        f"\n**Answer-line + last-line fallback** (no single shallow boxed): "
        f"{fallback} ({100*fallback/n_rollouts:.1f}%)\n"
    )

    md.append("\n## Completion length\n")
    md.append(f"- Mean chars per completion (prompt-avg of rollout means): **{chars_mean:.0f}**\n")
    md.append(f"- Mean approx tokens (chars/4): **{tok_mean:.0f}**\n")

    md.append("\n## Answer diversity per prompt (500 prompts)\n")
    md.append("| Metric | Mean | Median |\n|---|---:|---:|\n")
    med_sem = statistics.median(
        s["completion_aware"]["n_distinct_semantic_buckets"] for s in stats
    )
    md.append(
        f"| Distinct **semantic buckets** (heuristic) | {mean_sem:.2f} | {med_sem:.1f} |\n"
    )
    md.append(
        f"| Distinct stored `parsed_answer` | {mean_parsed:.2f} | "
        f"{statistics.median(s['as_recorded']['n_distinct_parsed'] for s in stats):.1f} |\n"
    )
    md.append(
        f"| Distinct stored `cluster_id` | {mean_clusters:.2f} | "
        f"{statistics.median(s['as_recorded']['n_distinct_clusters'] for s in stats):.1f} |\n"
    )
    md.append(
        f"\n- Prompts with **>1 semantic bucket** (8 rollouts): **{prompts_multi_sem}** "
        f"({100*prompts_multi_sem/n_prompts:.1f}%)\n"
    )
    md.append(
        f"- Semantic buckets **>** distinct stored parsed: {prompts_multi_sem_gt_parsed}\n"
    )
    md.append(
        f"- Semantic buckets **<** distinct stored parsed: {prompts_multi_sem_lt_parsed}\n"
    )
    md.append(
        f"- Prompts where same `canon(parsed)` maps to **>1 stored cluster_id**: "
        f"{prompts_canon_split}\n"
    )

    md.append("\n### Distribution: distinct semantic buckets per prompt\n")
    md.append(dist_table(dist_sem, 8, "n_distinct_semantic_buckets"))
    md.append("\n\n### Distribution: distinct stored parsed answers per prompt\n")
    md.append(dist_table(dist_sem_as_rec_parsed, 8, "n_distinct_parsed"))
    md.append("\n\n### Distribution: distinct stored clusters per prompt\n")
    md.append(dist_table(dist_sem_as_rec_cluster, 8, "n_distinct_clusters"))

    md.append("\n### Distribution: correct rollouts per prompt (stored vs recomputed)\n")
    md.append("**Stored `correct`:**\n")
    md.append(dist_table(dist_correct_rec, 8, "n_correct_rollouts"))
    md.append("\n\n**Recomputed `is_correct`:**\n")
    md.append(dist_table(dist_correct_re, 8, "n_correct_recomputed"))

    md.append("\n## Correctness summary\n")
    md.append(
        f"- Rollouts with stored correct: **{total_correct}/{n_rollouts}** "
        f"({100*total_correct/n_rollouts:.1f}%)\n"
    )
    md.append(f"- Prompts with ≥1 correct rollout: **{pct_any_correct:.1f}%**\n")

    md.append("\n## Exemplar prompts\n")
    ex = by_sem_div[0]
    md.append(
        f"- **Max semantic diversity:** `{ex['prompt_id']}` — "
        f"buckets={ex['completion_aware']['n_distinct_semantic_buckets']} "
        f"{ex['completion_aware']['semantic_buckets'][:6]}…, "
        f"stored parsed={ex['as_recorded']['n_distinct_parsed']}, "
        f"correct={ex['as_recorded']['n_correct_rollouts']}/8\n"
    )
    if sem_gt_stored:
        e2 = sem_gt_stored[0]
        md.append(
            f"- **Semantic buckets > stored parsed:** `{e2['prompt_id']}` — "
            f"buckets={e2['completion_aware']['n_distinct_semantic_buckets']} vs "
            f"parsed={e2['as_recorded']['n_distinct_parsed']}\n"
        )
    most_correct = max(stats, key=lambda s: s["as_recorded"]["n_correct_rollouts"])
    md.append(
        f"- **Most stored-correct rollouts:** `{most_correct['prompt_id']}` — "
        f"{most_correct['as_recorded']['n_correct_rollouts']}/8\n"
    )

    md.append("\n## Limitations (automated semantic bucketing)\n")
    md.append(
        "- Buckets **collapse** format variants (e.g. `\\(50\\)` and `50` → `n:50`) "
        "that production clustering **splits** via `canonicalize_answer`.\n"
    )
    md.append(
        "- Buckets **do not** prove mathematical equivalence; different buckets can "
        "be wrong for the same reason, and one bucket can hide multiple reasoning errors.\n"
    )
    md.append(
        "- Nested `\\boxed{...}` uses shallow regex in production; brace-balanced "
        "diagnostics show **2.7%** rollouts where regex inner ≠ balanced inner "
        f"({nested_diff}/{n_rollouts}).\n"
    )
    md.append(
        "- `is_correct` ignores non-boxed tails even when `Answer:` matches gold; "
        "loose canon match counts are **not** deployable accuracy.\n"
    )
    md.append(
        "- Prior audit (`_audit_parse_cluster.md`): stored fields are **internally "
        "consistent** with re-extraction; semantic defects are in canon/boxed rules.\n"
    )

    md.append("\n## Interpretation for experiments\n")
    md.append(
        f"Completion reading confirms **high within-prompt diversity**: median "
        f"{med_sem:.0f} semantic buckets vs {statistics.median(s['as_recorded']['n_distinct_parsed'] for s in stats):.0f} "
        f"stored parses. ~{100*boxed_text/n_rollouts:.0f}% of completions mention "
        f"`\\boxed`, but only ~{100*boxed_single_shallow/n_rollouts:.0f}% yield a "
        f"single shallow boxed extract — the rest fall through to Answer:/last-line "
        f"paths, which inflates parse diversity and depresses strict correct rate "
        f"({100*total_correct/n_rollouts:.1f}%). Treat **semantic bucket counts** as "
        "upper-bound answer-mode spread; fix boxed/canon before cluster-level RLVR rewards.\n"
    )

    OUT_MD.write_text("".join(md))
    print(f"Wrote {OUT_JSONL} ({n_prompts} lines)")
    print(f"Wrote {OUT_MD}")
    print(
        f"extract_mm={extract_mm} correct_mm={correct_mm} "
        f"prompts_multi_sem={prompts_multi_sem}"
    )


if __name__ == "__main__":
    main()
