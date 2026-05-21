#!/usr/bin/env python3
"""Build llm_clusters_handcheck.md for §A.7 manual audit."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RUN0 = Path(__file__).resolve().parent
DATA = RUN0 / "data"
CACHE_DIR = REPO / "pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters"
OUT = RUN0 / "llm_clusters_handcheck.md"

PROMPTS_PATH = DATA / "prompt_inputs.jsonl"
REParsed_PATH = DATA / "predictions_reparsed.jsonl"
SEED = 42
MAX_COMPLETION_CHARS = 2400


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _escape_md(s: str) -> str:
    return s.replace("\r\n", "\n")


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 20].rstrip() + "\n\n… [truncated]"


def _judge_labels(cache: dict) -> dict[int, dict]:
    """rollout index 0-7 -> {cluster_id, macro_micro}."""
    out: dict[int, dict] = {}
    raw = cache.get("raw_response")
    parsed: dict = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            pass
    assign = cache.get("cluster_assignment") or {}
    for i in range(8):
        key = str(i)
        cid = int(assign.get(key, assign.get(i, -999)))
        entry = parsed.get(str(i + 1), {}) if isinstance(parsed, dict) else {}
        cot = entry.get("chain_of_thought", "") if isinstance(entry, dict) else ""
        out[i] = {"cluster_id": cid, "chain_of_thought": cot}
    return out


def _stratum_label(n_correct: int) -> str:
    if n_correct >= 7:
        return "high correctness (7–8/8 correct)"
    if n_correct >= 1:
        return f"mixed ({n_correct}/8 correct)"
    return "none correct (0/8)"


def _pick_prompts(by_prompt: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """Return [(prompt_id, bucket_name), ...] × 10."""
    by_nc: dict[int, list[str]] = defaultdict(list)
    for pid, rows in by_prompt.items():
        nc = sum(1 for r in rows if r.get("is_correct_v2"))
        by_nc[nc].append(pid)

    rng = random.Random(SEED)
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()

    def take(pool: list[str], tag: str, k: int) -> None:
        for pid in rng.sample(pool, min(k, len(pool))):
            if pid not in used:
                chosen.append((pid, tag))
                used.add(pid)

    # High: 7/8, then 6/8, then 5/8 (no 8/8 on Run 0)
    high_pool = by_nc[7] + by_nc[6] + by_nc[5]
    take(high_pool, "high", 3)

    # Mixed: spread across 1–4 correct
    mixed_pool: list[str] = []
    for nc in (2, 3, 4, 1):
        mixed_pool.extend(by_nc[nc])
    rng.shuffle(mixed_pool)
    for pid in mixed_pool:
        if len([c for c in chosen if c[1] == "mixed"]) >= 3:
            break
        if pid not in used:
            chosen.append((pid, "mixed"))
            used.add(pid)

    take(by_nc[0], "none", 4)
    return chosen[:10]


def _cluster_summary(assign: dict[int, dict], correct: list[bool]) -> str:
    cids = [assign[i]["cluster_id"] for i in range(8)]
    uniq = len(set(cids))
    deg = sum(1 for c in cids if c == -1)
    nc = sum(correct)
    return f"{uniq} distinct clusters, {deg} degenerate (-1), {nc}/8 correct"


def _minority_flag(correct: list[bool], cids: list[int]) -> bool:
    from pilot.train.run_proxy import has_minority_correct_cluster

    return has_minority_correct_cluster(correct, cids)


def build() -> str:
    prompts = {r["prompt_id"]: r for r in _load_jsonl(PROMPTS_PATH)}
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in _load_jsonl(REParsed_PATH):
        by_prompt[row["prompt_id"]].append(row)
    for pid in by_prompt:
        by_prompt[pid].sort(key=lambda r: r.get("rollout_idx", 0))

    caches: dict[str, dict] = {}
    for path in CACHE_DIR.glob("*.json"):
        caches[path.stem] = json.loads(path.read_text())

    picked = _pick_prompts(by_prompt)
    lines: list[str] = [
        "# Analysis A — LLM cluster hand-check (10 prompts)\n",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  \n",
        "**Instructions:** For each prompt, read the 8 rollouts and judge whether rollouts "
        "with the *same reasoning approach* share a cluster. Ignore final-answer agreement; "
        "focus on macro/micro strategy. Record disagreements in the **Your notes** section "
        "at the bottom of each prompt block.\n",
        "\n**Strata (design §A.7):** 3 high-correctness, 3 mixed, 4 none-correct. "
        "Run 0 has **no** prompts with 8/8 correct; high stratum uses 5–7/8 correct.\n",
        "\n**Cluster key:** `-1` = degenerate (paper cluster 100). "
        "Same integer cluster ⇒ judge says same strategy.\n",
        "---\n",
    ]

    for n, (pid, bucket) in enumerate(picked, 1):
        rows = by_prompt[pid]
        pr = prompts[pid]
        cache = caches.get(pid, {})
        judge = _judge_labels(cache)
        correct = [bool(r.get("is_correct_v2")) for r in rows]
        nc = sum(correct)
        cids = [judge[i]["cluster_id"] for i in range(8)]
        minority = _minority_flag(correct, cids) if nc else False

        lines.append(f"\n## {n}. `{pid}` — {bucket}\n")
        lines.append(f"**Stratum:** {bucket} — {_stratum_label(nc)}  \n")
        lines.append(f"**Clusters:** {_cluster_summary(judge, correct)}  \n")
        if nc:
            lines.append(
                f"**Minority-correct prompt?** {'yes' if minority else 'no'} "
                "(correct rollouts in ≥2 clusters, one not the majority among correct)  \n"
            )
        lines.append(f"**Gold answer:** `{pr.get('gold_answer', '')}`\n")
        lines.append("\n### Problem\n\n")
        lines.append(_escape_md(pr.get("problem", "")) + "\n")
        lines.append("\n### Rollout summary\n\n")
        lines.append(
            "| # | Correct (v2) | Parsed answer | LLM cluster | Judge macro/micro |\n"
            "|---:|:---:|---|---:|---|\n"
        )
        for i, r in enumerate(rows):
            pa = (r.get("parsed_answer_v2") or "")[:80].replace("|", "\\|").replace("\n", " ")
            cot = (judge[i].get("chain_of_thought") or "—").replace("|", "\\|").replace("\n", " ")
            if len(cot) > 120:
                cot = cot[:117] + "…"
            ok = "✓" if correct[i] else "✗"
            cid = judge[i]["cluster_id"]
            cid_s = "**deg**" if cid == -1 else str(cid)
            lines.append(f"| {i + 1} | {ok} | `{pa}` | {cid_s} | {cot} |\n")

        lines.append("\n### Full completions (expand to read)\n\n")
        for i, r in enumerate(rows):
            comp = _truncate(r.get("completion") or "", MAX_COMPLETION_CHARS)
            cid = judge[i]["cluster_id"]
            lines.append(f"<details>\n<summary>Rollout {i + 1} — cluster {cid}</summary>\n\n")
            lines.append("```\n" + comp + "\n```\n\n</details>\n\n")

        lines.append("### Your notes\n\n")
        lines.append(
            "- [ ] Clustering looks reasonable\n"
            "- [ ] Disagreements (which rollouts should merge/split?):\n"
            "- [ ] Other:\n"
        )

    lines.append("\n---\n\n## Overall sign-off\n\n")
    lines.append(
        "- [ ] Reviewed all 10 prompts\n"
        "- [ ] Comfortable using `llm_clusters_summary.parquet` for Analysis B\n"
        "- [ ] Blockers / follow-ups:\n"
    )
    return "".join(lines)


def main() -> None:
    if not REParsed_PATH.is_file() or not CACHE_DIR.is_dir():
        raise SystemExit("missing reparsed predictions or llm_clusters cache")
    OUT.write_text(build())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
