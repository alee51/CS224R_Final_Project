#!/usr/bin/env python3
"""Rebuild labels/rollout_labels.jsonl from all merged chunks (sequential by chunk, id)."""

from __future__ import annotations

import json
from pathlib import Path

from label_paths import LABELING_ROOT, ROLLOUT_LABELS, load_manifest, manifest_path
from merge_chunk_pair import merge_chunk


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--through", type=int, default=11, help="Highest chunk index inclusive")
    args = p.parse_args()

    all_rows: list[dict] = []
    for k in range(args.through + 1):
        tag = f"{k:03d}"
        if not manifest_path(k).exists():
            print(f"skip chunk {tag}: no blind manifest")
            continue
        m = load_manifest(k)
        if not (LABELING_ROOT / m["output_a"]).exists() or not (LABELING_ROOT / m["output_b"]).exists():
            print(f"skip chunk {tag}: missing agent outputs")
            continue
        rows, n_dis, n_ag = merge_chunk(k)
        all_rows.extend(rows)
        print(f"chunk_{tag}: {len(rows)} rows ({n_ag} agreed, {n_dis} dispute)")

    ROLLOUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with ROLLOUT_LABELS.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(all_rows)} rows -> {ROLLOUT_LABELS}")


if __name__ == "__main__":
    main()
