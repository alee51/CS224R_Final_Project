# Polaris train data layout

| Path | Rows | Purpose |
|------|-----:|---------|
| **`polaris_train.jsonl`** | 51,139 | **Canonical train manifest** (prompt-filtered). Upload to Modal `/vol/data/polaris_train.jsonl`. |
| `polaris_train.meta.json` | — | Provenance for train freeze (filter rule, band counts). |
| `polaris_train_dropped.jsonl` | 2,152 | Audit of rows removed by the train prompt filter. |
| `source/polaris_train_full.jsonl` | 53,291 | Unfiltered clean HF pool (local; not in git). For re-filter / labeling only. |
| `source/polaris_train_full.meta.json` | — | Provenance for full-pool materialization. |
| `polaris_train_labeled.jsonl` | 53,291 | Full-pool heuristic labels (analysis only). |
| `polaris_train_heuristic_summary.json` | — | Label counts for full pool. |

**Pipeline:** `preprocess_polaris.py` → `source/polaris_train_full.jsonl` → `filter_polaris_train.py` → `polaris_train.jsonl`.

Do not train on `source/polaris_train_full.jsonl`.
