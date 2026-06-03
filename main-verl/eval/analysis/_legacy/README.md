# Legacy / superseded analysis scripts

Scripts here are kept for git provenance only. Do not run.

- **compare.py** — early cross-arm comparison driver. Superseded by
  `../posthoc/auc_at_k.py` (single-number AUC@k table) plus
  `../posthoc/diff_at_k_split.py` (solved/unsolved partition). Has multiple
  drift problems vs the locked 2026-06-02 eval spec:
  - Reads `/tmp/<arm>_*.json` (the local-tmp naming used pre-locking; current
    canonical layout is `main-verl/eval/probes/eval_4b/<arm>_step<N>_<split>_<ds>.json`)
  - `K_VALUES = [1, 4, 8, 16]` instead of the locked ladder `{1, 2, 4, 8, 16, 32, 64}`
  - `DATASETS: list[str] = []` is an empty placeholder
  - Writes to `main/data/probes/eval_4b/cross_arm_summary.json`; the `main/`
    tree is dead — canonical writeup folder is `main-verl/writeup/`
