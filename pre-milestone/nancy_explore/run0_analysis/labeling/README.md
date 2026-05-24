# Labeling pipeline (archived)

Frozen artifacts and scripts used to produce **`../labels/rollout_labels.jsonl`** (4000 rollouts, blind A/B + human dispute resolution). Labeling is **complete** — nothing here is required for Analyses A–D.

## Layout

| Path | Contents |
|------|----------|
| `chunks/` | Per-chunk `*_in.tsv` (problem + tail), `*_keys.tsv` (gold), `*_dispute.tsv` |
| `blind/` | Opaque A/B agent outputs + `manifest.json` per chunk |
| `spawn/` | Copy-paste spawn prompts for Cursor agents |
| `labels_archive/` | Human review CSVs (`human_review_queue.csv`, `labeled.csv`, etc.) |
| `audit_1024_extracted/` | One-off 1024-token cap audit snapshot |

## Re-run (only if needed)

From this directory:

```bash
python build_chunk.py --chunk K
python prepare_label_slot.py --chunk K --agent A   # then B
python merge_chunk_pair.py K
python rebuild_rollout_labels.py --through 11    # full rebuild
```

Data symlinks live in **`../data/`**. Canonical labels: **`../labels/rollout_labels.jsonl`**.

See **`FRESH_PLAN.md`** and **`AGENT_LABEL_PROMPT.md`** for the original workflow.
