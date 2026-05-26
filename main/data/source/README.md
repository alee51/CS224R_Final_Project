# Polaris full pool (unfiltered)

`polaris_train_full.jsonl` is the **53,291-row** clean HF freeze (no prompt filter). It is **not** the training manifest.

- Materialize: `PYTHONPATH=main python3 main/data/preprocess_polaris.py --n 53291`
- Filter to train: `PYTHONPATH=main python3 main/scripts/filter_polaris_train.py`

Large jsonl files here are typically **gitignored** (local only). The canonical committed train file is `../polaris_train.jsonl`.
