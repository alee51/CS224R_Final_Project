#!/usr/bin/env python3
"""Prepare one blind labeling slot (A or B) and print the spawn prompt.

Run A first, then B. B's slot blocks A's output in .cursorignore so agents cannot peek.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from label_paths import (
    LABELING_ROOT,
    SPAWN,
    agent_key,
    count_data_rows,
    init_output_tsv,
    load_manifest,
    output_key,
    rel,
    update_isolation_cursorignore,
)

PROMPT_TEMPLATE = Path(__file__).resolve().parent / "AGENT_LABEL_PROMPT.md"


def _read_spawn_section() -> str:
    text = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    marker = "## Spawn message (template"
    start = text.find("```\n", text.find(marker))
    if start < 0:
        raise SystemExit("AGENT_LABEL_PROMPT.md: missing spawn code block")
    start += 4
    end = text.find("\n```", start)
    if end < 0:
        raise SystemExit("AGENT_LABEL_PROMPT.md: unclosed spawn code block")
    return text[start:end]


def build_spawn_message(manifest: dict, agent: str) -> str:
    tag = manifest["chunk"]
    n = manifest["n_rows"]
    a = agent_key(agent)
    body = _read_spawn_section()
    repl = {
        "{{CHUNK}}": tag,
        "{{AGENT}}": a,
        "{{N_ROWS}}": str(n),
        "{{INPUT_PATH}}": rel(manifest["input"]),
        "{{OUTPUT_PATH}}": rel(manifest[output_key(agent)]),
    }
    for key, val in repl.items():
        body = body.replace(key, val)
    return body


def prepare(chunk_k: int, agent: str, *, require_a_done: bool = True) -> int:
    manifest = load_manifest(chunk_k)
    a = agent_key(agent)
    tag = manifest["chunk"]
    n = int(manifest["n_rows"])
    in_path = LABELING_ROOT / manifest["input"]
    out_path = LABELING_ROOT / manifest[output_key(agent)]
    out_a = LABELING_ROOT / manifest["output_a"]

    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")

    if a == "B" and require_a_done:
        if not out_a.exists():
            raise SystemExit(
                f"agent A output missing: {out_a}\n"
                "Run prepare_label_slot.py --chunk {chunk_k} --agent A and complete labeling first."
            )
        n_a = count_data_rows(out_a)
        if n_a != n:
            raise SystemExit(f"agent A has {n_a} rows, expected {n}; finish A before starting B")

    init_output_tsv(out_path)

    block: list[str] = []
    if a == "B":
        block.append(manifest["output_a"])
    update_isolation_cursorignore(block)

    msg = build_spawn_message(manifest, agent)
    SPAWN.mkdir(parents=True, exist_ok=True)
    spawn_path = SPAWN / f"chunk_{tag}_{a}.txt"
    spawn_path.write_text(msg, encoding="utf-8")

    print(msg)
    print(f"\n--- spawn saved: {spawn_path} ---", file=sys.stderr)
    if a == "B":
        print(f"--- cursorignore blocks: {manifest['output_a']} ---", file=sys.stderr)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare blind label slot and print spawn prompt")
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--agent", choices=["A", "B", "a", "b"], required=True)
    args = p.parse_args()
    prepare(args.chunk, args.agent)


if __name__ == "__main__":
    main()
