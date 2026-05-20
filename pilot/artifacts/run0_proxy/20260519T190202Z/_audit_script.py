#!/usr/bin/env python3
"""One-off audit: recompute extract/correct/cluster vs artifacts."""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pilot.train.answer_parse import extract_answer, is_correct
from pilot.train.canonicalize import canonicalize_answer, cluster_id

ART = Path(__file__).parent
PRED = ART / "raw_predictions.jsonl"
PROMPT = ART / "prompt_inputs.jsonl"


def load_gold() -> dict[str, str]:
    gold = {}
    with PROMPT.open() as f:
        for line in f:
            row = json.loads(line)
            gold[row["prompt_id"]] = str(row["gold_answer"])
    return gold


def load_rollouts():
    rows = []
    with PRED.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    gold = load_gold()
    rows = load_rollouts()
    n = len(rows)
    print(f"rollouts={n} prompts_gold={len(gold)}")

    extract_mismatch = []
    correct_mismatch = []
    cluster_mismatch = []
    empty_parsed = []
    boxed_nested_issues = []

    # cluster merge analysis: same canonical form, different cluster_id
    canon_to_cids: dict[str, set[int]] = defaultdict(set)
    cid_to_canon: dict[int, set[str]] = defaultdict(set)

    for i, r in enumerate(rows):
        pid = r["prompt_id"]
        completion = r.get("completion", "")
        stored_parsed = r["parsed_answer"]
        stored_correct = r["correct"]
        stored_cid = r["cluster_id"]
        g = gold.get(pid, "")

        recomputed_parsed = extract_answer(completion)
        recomputed_correct = is_correct(completion, g)
        recomputed_cid = cluster_id(stored_parsed)

        if recomputed_parsed != stored_parsed:
            extract_mismatch.append((i, r))
        if recomputed_correct != stored_correct:
            correct_mismatch.append((i, r, recomputed_correct))
        if recomputed_cid != stored_cid:
            cluster_mismatch.append((i, r, recomputed_cid))

        if not str(stored_parsed).strip():
            empty_parsed.append(r)

        canon = canonicalize_answer(stored_parsed)
        canon_to_cids[canon].add(stored_cid)
        cid_to_canon[stored_cid].add(canon)

        # detect \boxed with nested braces (extract might truncate)
        if "\\boxed{" in completion:
            for m in re.finditer(r"\\boxed\{", completion):
                start = m.end()
                depth = 1
                j = start
                while j < len(completion) and depth:
                    if completion[j] == "{":
                        depth += 1
                    elif completion[j] == "}":
                        depth -= 1
                    j += 1
                inner = completion[start : j - 1] if depth == 0 else None
                regex_inner = None
                matches = list(re.finditer(r"\\boxed\{([^}]*)\}", completion, re.DOTALL))
                if matches:
                    regex_inner = matches[-1].group(1)
                if inner is not None and regex_inner is not None and inner.strip() != regex_inner.strip():
                    boxed_nested_issues.append((pid, inner[:80], regex_inner[:80]))

    print("\n=== FULL DATASET MISMATCH RATES ===")
    print(f"extract:  {len(extract_mismatch)}/{n} = {len(extract_mismatch)/n:.6f}")
    print(f"correct:  {len(correct_mismatch)}/{n} = {len(correct_mismatch)/n:.6f}")
    print(f"cluster:  {len(cluster_mismatch)}/{n} = {len(cluster_mismatch)/n:.6f}")
    print(f"empty parsed_answer: {len(empty_parsed)}")

    # canonicalize splits clusters incorrectly?
    split_canons = [(c, cids) for c, cids in canon_to_cids.items() if len(cids) > 1]
    merge_canons = [(cid, cs) for cid, cs in cid_to_canon.items() if len(cs) > 1]
    print(f"\ncanon forms mapping to >1 cluster_id: {len(split_canons)}")
    print(f"cluster_ids mapping to >1 canon form: {len(merge_canons)}")

    # LaTeX delimiter diversity: same numeric answer different clusters within prompt
    by_prompt: dict[str, list] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)

    latex_split_examples = []
    for pid, rs in by_prompt.items():
        g = gold.get(pid, "")
        gold_canon = canonicalize_answer(g)
        # group by cluster among correct rollouts
        correct_by_cid = defaultdict(list)
        for r in rs:
            if r["correct"]:
                correct_by_cid[r["cluster_id"]].append(r["parsed_answer"])
        if len(correct_by_cid) > 1:
            latex_split_examples.append((pid, g, correct_by_cid))

    print(f"prompts with multiple correct clusters: {len(latex_split_examples)}")

    # correct using parsed vs completion path
    parsed_vs_completion_correct = 0
    for r in rows:
        g = gold.get(r["prompt_id"], "")
        from_canon_parsed = canonicalize_answer(r["parsed_answer"]) == canonicalize_answer(g)
        if from_canon_parsed != r["correct"]:
            parsed_vs_completion_correct += 1
    print(f"stored correct != canon(parsed)==canon(gold): {parsed_vs_completion_correct}/{n}")

    # Stratified sample for manual review
    rng = random.Random(42)
    strata = {
        "extract_mismatch": extract_mismatch[:20],
        "correct_mismatch": correct_mismatch[:20],
        "cluster_mismatch": cluster_mismatch[:20],
        "empty": empty_parsed[:10],
        "multi_correct_cluster": [],
        "nested_boxed": boxed_nested_issues[:10],
    }
    if latex_split_examples:
        for item in rng.sample(latex_split_examples, min(15, len(latex_split_examples))):
            pid, g, cids = item
            strata["multi_correct_cluster"].append(
                (pid, g, {k: v[:3] for k, v in cids.items()})
            )

    # random diverse sample
    diverse = []
    indices = set()
    for bucket, items in [
        ("wrong", [i for i, r in enumerate(rows) if not r["correct"]]),
        ("right", [i for i, r in enumerate(rows) if r["correct"]]),
        ("boxed", [i for i, r in enumerate(rows) if "\\boxed" in r.get("completion", "")]),
        ("no_boxed", [i for i, r in enumerate(rows) if "\\boxed" not in r.get("completion", "")]),
    ]:
        if items:
            for idx in rng.sample(items, min(15, len(items))):
                indices.add(idx)
    diverse_rows = [(i, rows[i]) for i in sorted(indices)[:60]]

    # Write markdown report
    md = []
    md.append("# Run0 parse/cluster audit\n")
    md.append(f"- Rollouts: {n}\n")
    md.append("## Mismatch rates (recompute vs stored)\n")
    md.append(f"| Check | Mismatches | Rate |\n|---|---|---|\n")
    md.append(f"| extract_answer(completion) vs parsed_answer | {len(extract_mismatch)} | {len(extract_mismatch)/n:.4%} |\n")
    md.append(f"| is_correct(completion,gold) vs correct | {len(correct_mismatch)} | {len(correct_mismatch)/n:.4%} |\n")
    md.append(f"| cluster_id(parsed) vs cluster_id | {len(cluster_mismatch)} | {len(cluster_mismatch)/n:.4%} |\n")
    md.append(f"| canon(parsed)==canon(gold) vs stored correct | {parsed_vs_completion_correct} | {parsed_vs_completion_correct/n:.4%} |\n")
    md.append(f"\n- Empty parsed_answer: {len(empty_parsed)}\n")
    md.append(f"- Canon→multiple cluster_ids: {len(split_canons)}\n")
    md.append(f"- Cluster_id→multiple canons: {len(merge_canons)}\n")
    md.append(f"- Prompts w/ >1 correct cluster: {len(latex_split_examples)}\n")
    md.append(f"- Nested \\boxed regex truncation candidates: {len(boxed_nested_issues)}\n")

    def ex_block(title, examples, limit=8):
        md.append(f"\n## {title}\n")
        if not examples:
            md.append("_none_\n")
            return
        for item in examples[:limit]:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], dict):
                i, r = item[0], item[1]
            elif isinstance(item, dict):
                i, r = -1, item
            else:
                md.append(f"- {item}\n")
                continue
            comp = r.get("completion", "")[:300].replace("\n", " ↵ ")
            md.append(f"\n### {r['prompt_id']} (line ~{i+1})\n")
            md.append(f"- gold: `{gold.get(r['prompt_id'], '')[:120]}`\n")
            md.append(f"- stored parsed: `{r['parsed_answer'][:120]}`\n")
            md.append(f"- re-extract: `{extract_answer(r.get('completion',''))[:120]}`\n")
            md.append(f"- correct stored/recomp: {r['correct']} / {is_correct(r.get('completion',''), gold.get(r['prompt_id'],''))}\n")
            md.append(f"- cluster stored/recomp: {r['cluster_id']} / {cluster_id(r['parsed_answer'])}\n")
            md.append(f"- canon(parsed): `{canonicalize_answer(r['parsed_answer'])[:80]}`\n")
            md.append(f"- canon(gold): `{canonicalize_answer(gold.get(r['prompt_id'],''))[:80]}`\n")
            md.append(f"- completion snippet: {comp}...\n")

    ex_block("Extract mismatches", extract_mismatch)
    ex_block("Correct mismatches", [(i, r) for i, r, _ in correct_mismatch])
    ex_block("Cluster mismatches", [(i, {**r, "_recomp_cid": rc}) for i, r, rc in cluster_mismatch])

    md.append("\n## Multi-cluster correct answers (format splits)\n")
    for pid, g, cids in (latex_split_examples[:10] if latex_split_examples else []):
        md.append(f"\n### {pid}\n- gold: `{g[:100]}`\n")
        for cid, answers in cids.items():
            md.append(f"- cluster {cid}: `{answers[0][:80]}` (+{len(answers)-1} more)\n")
            for a in answers[1:3]:
                md.append(f"  - also: `{a[:80]}`\n")

    md.append("\n## Nested boxed truncation samples\n")
    for pid, inner, regex_inner in boxed_nested_issues[:8]:
        md.append(f"- **{pid}**: full-depth=`{inner}` vs regex=`{regex_inner}`\n")

    # Find cluster splits from canonicalize stripping (same semantic, different canon)
    strip_risk = []
    for r in rows:
        p = r["parsed_answer"]
        if "}" in p or "{" in p or "\\" in p:
            c = canonicalize_answer(p)
            if "}" in c or "{" in c:  # still has braces after strip? 
                pass
            # answers where stripping removed structural chars
            if p.count("}") != c.count("}") and p.count("}"):
                strip_risk.append(r)
    md.append(f"\n## Brace-heavy parsed answers: {sum(1 for r in rows if '{' in r['parsed_answer'] or '}' in r['parsed_answer'])}\n")

    # Within-prompt: same correct answer text different clusters?
    within_prompt_cluster_splits = []
    for pid, rs in by_prompt.items():
        correct_parsed = [r for r in rs if r["correct"]]
        by_canon = defaultdict(set)
        for r in correct_parsed:
            by_canon[canonicalize_answer(r["parsed_answer"])].add(r["cluster_id"])
        for canon, cids in by_canon.items():
            if len(cids) > 1:
                within_prompt_cluster_splits.append((pid, canon, cids))
    md.append(f"Correct rollouts: same canon, multiple cluster_ids: {len(within_prompt_cluster_splits)}\n")

    out = ART / "_audit_parse_cluster.md"
    out.write_text("".join(md))
    print(f"\nWrote {out}")

    # Print top split canons
    if split_canons:
        print("\nTop canon→multi-cid (should be 0 if cluster_id deterministic):")
        for c, cids in split_canons[:5]:
            print(f"  {c!r} -> {cids}")
    if merge_canons:
        print("\nTop cid→multi-canon (BUG if cluster_id is hash of canon):")
        for cid, cs in sorted(merge_canons, key=lambda x: -len(x[1]))[:5]:
            print(f"  cid={cid} -> {list(cs)[:3]}...")


if __name__ == "__main__":
    main()
