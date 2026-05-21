#!/usr/bin/env python3
"""Paths and manifest for blind A/B labeling (agents never see sibling output)."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

LABELING_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = LABELING_ROOT.parent
CHUNKS = LABELING_ROOT / "chunks"
BLIND = LABELING_ROOT / "blind"
SPAWN = LABELING_ROOT / "spawn"
CURSORIGNORE = LABELING_ROOT / ".cursorignore"
ROLLOUT_LABELS = ANALYSIS_ROOT / "labels" / "rollout_labels.jsonl"
ISOLATION_BEGIN = "# BEGIN run0_label_isolation (auto — do not edit)"
ISOLATION_END = "# END run0_label_isolation"


def chunk_tag(chunk_k: int) -> str:
    return f"{chunk_k:03d}"


def manifest_path(chunk_k: int) -> Path:
    return BLIND / chunk_tag(chunk_k) / "manifest.json"


def load_manifest(chunk_k: int) -> dict:
    path = manifest_path(chunk_k)
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path} (run build_chunk.py first)")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_manifest(chunk_k: int, n_rows: int) -> dict:
    tag = chunk_tag(chunk_k)
    blind_dir = BLIND / tag
    blind_dir.mkdir(parents=True, exist_ok=True)

    token_a = secrets.token_hex(4)
    token_b = secrets.token_hex(4)
    manifest = {
        "chunk": tag,
        "chunk_index": chunk_k,
        "n_rows": n_rows,
        "input": f"chunks/chunk_{tag}_in.tsv",
        "keys": f"chunks/chunk_{tag}_keys.tsv",
        "output_a": f"blind/{tag}/{token_a}.tsv",
        "output_b": f"blind/{tag}/{token_b}.tsv",
    }
    path = manifest_path(chunk_k)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def rel(path: str) -> str:
    return f"nancy_explore/run0_analysis/labeling/{path}"


def agent_key(agent: str) -> str:
    a = agent.strip().upper()
    if a not in ("A", "B"):
        raise ValueError("agent must be A or B")
    return a


def output_key(agent: str) -> str:
    return f"output_{agent_key(agent).lower()}"


def init_output_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id\tresult\n", encoding="utf-8")


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    return max(0, len(lines) - 1)


def update_isolation_cursorignore(block_paths: list[str]) -> None:
    """Hide other agents' blind outputs from Cursor Read/Grep (B cannot see A's file)."""
    lines: list[str] = []
    if CURSORIGNORE.exists():
        lines = CURSORIGNORE.read_text(encoding="utf-8").splitlines()

    kept: list[str] = []
    in_block = False
    for line in lines:
        if line.strip() == ISOLATION_BEGIN:
            in_block = True
            continue
        if line.strip() == ISOLATION_END:
            in_block = False
            continue
        if not in_block:
            kept.append(line)

    while kept and kept[-1] == "":
        kept.pop()

    block = [ISOLATION_BEGIN]
    for p in sorted(set(block_paths)):
        block.append(p)
    block.append(ISOLATION_END)

    out = kept + ([""] if kept else []) + block + [""]
    CURSORIGNORE.write_text("\n".join(out), encoding="utf-8")
