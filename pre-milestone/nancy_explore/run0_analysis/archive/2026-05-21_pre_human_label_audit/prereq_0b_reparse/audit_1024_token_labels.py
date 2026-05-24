#!/usr/bin/env python3
"""Audit rollouts at the 1024-token generation cap vs human labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent
RAW = WORKDIR / "data" / "raw_predictions.jsonl"
ROLLOUT_LABELS = WORKDIR / "labels" / "rollout_labels.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
LITERAL = frozenset({"runon", "no_answer", "needs_review"})


@lru_cache(maxsize=1)
def _tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def completion_token_count(text: str, model_id: str) -> int:
    if not text:
        return 0
    return len(_tokenizer(model_id).encode(text, add_special_tokens=False))


def load_rollouts() -> dict[str, dict]:
    by_prompt: dict[str, list] = defaultdict(list)
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_prompt[r["prompt_id"]].append(r)
    out: dict[str, dict] = {}
    for pid, rollouts in by_prompt.items():
        for idx, r in enumerate(rollouts):
            key = f"{pid}#{idx}"
            out[key] = {
                "completion": r.get("completion") or "",
                "parsed_answer": (r.get("parsed_answer") or "").strip(),
            }
    return out


def label_kind(result: str | None) -> str:
    if not result:
        return "(missing)"
    r = result.strip()
    if r in LITERAL:
        return r
    return "extracted_answer"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--cap", type=int, default=1024)
    p.add_argument("--show", type=int, default=25, help="Max examples per bucket")
    args = p.parse_args()

    rollouts = load_rollouts()
    labels: dict[str, dict] = {}
    with ROLLOUT_LABELS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                labels[row["rollout_key"]] = row

    over_cap: list[tuple[str, int]] = []
    at_cap: list[tuple[str, int, str, str, str]] = []

    for key, ro in rollouts.items():
        comp = ro["completion"]
        n = completion_token_count(comp, args.model_id)
        if n > args.cap:
            over_cap.append((key, n))
        elif n == args.cap:
            row = labels.get(key, {})
            human = (row.get("human_result") or row.get("result") or "").strip()
            parsed = ro["parsed_answer"]
            kind = label_kind(human)
            at_cap.append((key, n, kind, human, parsed))

    print(f"Qwen tokenizer: {args.model_id}")
    print(f"completions with tokens > {args.cap}: {len(over_cap)}")
    if over_cap:
        for k, n in over_cap[:10]:
            print(f"  OVER {k} tokens={n}")

    by_kind = Counter(k for _, _, k, _, _ in at_cap)
    print(f"\ncompletions with exactly {args.cap} tokens: {len(at_cap)}")
    print("human label breakdown (result / human_result):")
    for kind, cnt in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {cnt}")

    print("\n--- extracted_answer at 1024 cap (review these) ---")
    shown = 0
    for key, n, kind, human, _ in at_cap:
        if kind != "extracted_answer":
            continue
        tail = rollouts[key]["completion"][-120:].replace("\n", " ")
        parsed = rollouts[key]["parsed_answer"]
        print(f"  {key}  label={human!r}  v1_parse={parsed!r}  tail=...{tail!r}")
        shown += 1
        if shown >= args.show:
            break

    print("\n--- no_answer at 1024 cap ---")
    shown = 0
    for key, n, kind, human, _ in at_cap:
        if kind != "no_answer":
            continue
        print(f"  {key}  label={human!r}")
        shown += 1
        if shown >= args.show:
            break


if __name__ == "__main__":
    main()
