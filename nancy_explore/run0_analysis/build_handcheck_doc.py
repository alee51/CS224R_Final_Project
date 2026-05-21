#!/usr/bin/env python3
"""Build llm_clusters_handcheck.md for §A.7 manual audit."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n")


def _table_cell(s: str) -> str:
    """Escape characters that break GFM tables."""
    return _normalize_newlines(s).replace("|", "\\|").replace("\n", " ").strip()


def _indent_completion(text: str) -> str:
    """Indent full completion so inner ``` fences do not break MD preview."""
    lines = _normalize_newlines(text or "").split("\n")
    return "\n".join(("    " + line) if line else "    " for line in lines) + "\n"


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

    high_pool = by_nc[7] + by_nc[6] + by_nc[5]
    take(high_pool, "high", 3)

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
    parts: list[str] = [
        "# Analysis A — LLM cluster hand-check (10 prompts)\n\n",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  \n\n",
        "For each prompt: read the summary table, then each rollout below (full text, untruncated). "
        "Judge whether same *reasoning approach* ⇒ same cluster. Record notes at the end of each prompt.\n\n",
        "**Strata:** 3 high, 3 mixed, 4 none-correct (Run 0 has no 8/8-correct prompts; high = 5–7/8).  \n",
        "**Clusters:** `-1` = degenerate (paper id 100).\n\n",
        "---\n\n",
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

        parts.append(f"## {n}. `{pid}` — {bucket}\n\n")
        parts.append(f"- **Stratum:** {bucket} — {_stratum_label(nc)}\n")
        parts.append(f"- **Clusters:** {_cluster_summary(judge, correct)}\n")
        if nc:
            parts.append(
                f"- **Minority-correct prompt?** "
                f"{'yes' if minority else 'no'}\n"
            )
        parts.append(f"- **Gold answer:** `{pr.get('gold_answer', '')}`\n\n")

        parts.append("### Problem\n\n")
        parts.append(_normalize_newlines(pr.get("problem", "")) + "\n\n")

        parts.append("### Rollout summary\n\n")
        parts.append("| # | OK | Parsed answer | Cluster | Judge macro/micro |\n")
        parts.append("|---:|:--:|---|:-:|---|\n")
        for i, r in enumerate(rows):
            pa = _table_cell((r.get("parsed_answer_v2") or "")[:100])
            cot = _table_cell(judge[i].get("chain_of_thought") or "—")
            ok = "yes" if correct[i] else "no"
            cid = judge[i]["cluster_id"]
            cid_s = "deg" if cid == -1 else str(cid)
            parts.append(f"| {i + 1} | {ok} | `{pa}` | {cid_s} | {cot} |\n")
        parts.append("\n")

        parts.append("### Rollouts (full text)\n\n")
        for i, r in enumerate(rows):
            cid = judge[i]["cluster_id"]
            ok = "correct" if correct[i] else "incorrect"
            pa = _table_cell(r.get("parsed_answer_v2") or "")
            cot = judge[i].get("chain_of_thought") or "—"
            parts.append(f"#### Rollout {i + 1} — cluster {cid} ({ok})\n\n")
            parts.append(f"**Parsed answer:** `{pa}`  \n\n")
            parts.append(f"**Judge macro/micro:** {cot}\n\n")
            parts.append(_indent_completion(r.get("completion") or ""))
            parts.append("\n")

        parts.append("### Your notes\n\n")
        parts.append("- [ ] Clustering looks reasonable\n")
        parts.append("- [ ] Disagreements (which rollouts should merge/split?):\n")
        parts.append("- [ ] Other:\n\n")
        parts.append("---\n\n")

    parts.append("## Overall sign-off\n\n")
    parts.append("- [ ] Reviewed all 10 prompts\n")
    parts.append("- [ ] Comfortable using `llm_clusters_summary.parquet` for Analysis B\n")
    parts.append("- [ ] Blockers / follow-ups:\n")

    return "".join(parts)


def main() -> None:
    if not REParsed_PATH.is_file() or not CACHE_DIR.is_dir():
        raise SystemExit("missing reparsed predictions or llm_clusters cache")
    OUT.write_text(build())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
