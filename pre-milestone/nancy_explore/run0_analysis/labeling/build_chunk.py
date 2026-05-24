#!/usr/bin/env python3
"""Build chunk_KKK_in.tsv (id, problem, tail) and chunk_KKK_keys.tsv (id, rollout_key, gold)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from label_paths import ANALYSIS_ROOT, LABELING_ROOT, write_manifest

RAW = ANALYSIS_ROOT / "data" / "raw_predictions.jsonl"
PROMPTS = ANALYSIS_ROOT / "data" / "prompt_inputs.jsonl"
CHUNKS = LABELING_ROOT / "chunks"
TAIL_LEN = 120
PROMPTS_PER_CHUNK = 50


def is_priority_rollout(completion: str, parsed: str) -> bool:
    comp = completion or ""
    pa = (parsed or "").strip()
    if "\\boxed" not in comp:
        return True
    if not pa or len(pa) > 60:
        return True
    if "answer:" in comp[-200:].lower() and len(pa) > 25:
        return True
    return False


def load_data() -> tuple[dict[str, str], dict[str, str], dict[str, list[dict]]]:
    problems: dict[str, str] = {}
    golds: dict[str, str] = {}
    with PROMPTS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = row["prompt_id"]
            problems[pid] = row.get("problem") or ""
            golds[pid] = str(row.get("gold_answer", ""))

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_prompt[r["prompt_id"]].append(r)

    for pid in by_prompt:
        if len(by_prompt[pid]) != 8:
            raise SystemExit(f"prompt {pid}: expected 8 rollouts, got {len(by_prompt[pid])}")

    return problems, golds, by_prompt


def priority_prompt_ids(by_prompt: dict[str, list[dict]]) -> list[str]:
    out: list[str] = []
    for pid in sorted(by_prompt):
        rollouts = by_prompt[pid]
        if any(
            is_priority_rollout(r.get("completion") or "", r.get("parsed_answer") or "")
            for r in rollouts
        ):
            out.append(pid)
    return out


def non_priority_prompt_ids(by_prompt: dict[str, list[dict]]) -> list[str]:
    pri = set(priority_prompt_ids(by_prompt))
    return [pid for pid in sorted(by_prompt) if pid not in pri]


REMAINDER_CHUNK_BASE = 10
REMAINDER_PROMPTS_PER_CHUNK = 4


def _tsv_cell(s: str) -> str:
    return (s or "").replace("\t", " ").replace("\r\n", "\n").replace("\r", "\n")


def build_chunk(
    chunk_k: int,
    prompt_ids: list[str],
    problems: dict[str, str],
    golds: dict[str, str],
    by_prompt: dict[str, list[dict]],
) -> int:
    batch = prompt_ids
    if not batch:
        print(f"chunk {chunk_k:03d}: no prompts")
        return 0

    CHUNKS.mkdir(parents=True, exist_ok=True)
    tag = f"{chunk_k:03d}"
    in_path = CHUNKS / f"chunk_{tag}_in.tsv"
    keys_path = CHUNKS / f"chunk_{tag}_keys.tsv"

    rid = 0
    in_rows: list[list[str]] = []
    key_rows: list[list[str]] = [["id", "rollout_key", "gold"]]
    for pid in batch:
        problem = _tsv_cell(problems[pid])
        gold = _tsv_cell(golds[pid])
        for idx, r in enumerate(by_prompt[pid]):
            rid += 1
            completion = r.get("completion") or ""
            tail = _tsv_cell(completion[-TAIL_LEN:] if completion else "")
            in_rows.append([str(rid), problem, tail])
            key_rows.append([str(rid), f"{pid}#{idx}", gold])

    with in_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "problem", "tail"])
        w.writerows(in_rows)

    with keys_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerows(key_rows)

    manifest = write_manifest(chunk_k, rid)

    print(f"chunk_{tag}: {len(batch)} prompts, {rid} rollouts")
    print(f"  {in_path}")
    print(f"  {keys_path}")
    print(f"  blind manifest: {manifest['output_a']} / {manifest['output_b']}")
    return rid


def remainder_batch(chunk_k: int, remainder_ids: list[str]) -> list[str]:
    if chunk_k < REMAINDER_CHUNK_BASE:
        raise SystemExit(f"--remainder requires chunk >= {REMAINDER_CHUNK_BASE}")
    idx = chunk_k - REMAINDER_CHUNK_BASE
    start = idx * REMAINDER_PROMPTS_PER_CHUNK
    end = start + REMAINDER_PROMPTS_PER_CHUNK
    return remainder_ids[start:end]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, required=True, help="0-based chunk index")
    p.add_argument(
        "--remainder",
        action="store_true",
        help=f"Label non-priority prompts (chunks {REMAINDER_CHUNK_BASE}+, 4 prompts / 32 rollouts each)",
    )
    args = p.parse_args()

    problems, golds, by_prompt = load_data()
    if args.remainder:
        rem = non_priority_prompt_ids(by_prompt)
        n_chunks = (len(rem) + REMAINDER_PROMPTS_PER_CHUNK - 1) // REMAINDER_PROMPTS_PER_CHUNK
        print(f"non-priority prompts: {len(rem)} -> {n_chunks} remainder chunks of {REMAINDER_PROMPTS_PER_CHUNK}")
        batch = remainder_batch(args.chunk, rem)
        if not batch:
            raise SystemExit(f"chunk {args.chunk:03d}: no remainder prompts for this index")
        build_chunk(args.chunk, batch, problems, golds, by_prompt)
        return

    pri = priority_prompt_ids(by_prompt)
    n_chunks = (len(pri) + PROMPTS_PER_CHUNK - 1) // PROMPTS_PER_CHUNK
    print(f"priority prompts: {len(pri)} -> {n_chunks} full chunks of {PROMPTS_PER_CHUNK}")
    start = args.chunk * PROMPTS_PER_CHUNK
    build_chunk(args.chunk, pri[start : start + PROMPTS_PER_CHUNK], problems, golds, by_prompt)


if __name__ == "__main__":
    main()
