# Polaris train freeze — preprocessing plan

**Status:** frozen (2026-05-26) — full 53,291 rows in `source/polaris_train_full.jsonl`; train manifest is `polaris_train.jsonl` (filtered).  
**Scope:** one-shot materialization of **`main/data/source/polaris_train_full.jsonl`** + meta per PLAN §2. See [`data/README.md`](../data/README.md).

---

## 1. Objective

Produce a **frozen, reproducible** full Polaris pool for downstream filtering. **Training** uses `polaris_train.jsonl` (from `filter_polaris_train.py`), not this file. Prompt wrapping happens at train time in `format_problem`, not in jsonl.

## 2. Source data

| Field | Value |
|-------|--------|
| HF dataset | `POLARIS-Project/Polaris-Dataset-53K` |
| Split | `train` only (53,291 rows as of 2026-05-25) |
| Columns used | `problem`, `answer`, `difficulty` |
| Difficulty semantics | `k/8` = k successes out of 8 on DeepSeek-R1-distill-Qwen-7B reference rollouts; **`7/8` = easiest**, **`0/8` = hardest**. No `8/8` band in released set. |

Record in meta: dataset ID, `datasets` revision / commit SHA if available, download timestamp.

## 3. Locked sampling policy (this freeze)

| Decision | Choice | Rationale (document only) |
|----------|--------|---------------------------|
| Target size | **16,000** rows | PLAN §2 target ≈16k (DAPO-17k parity) |
| Band selection | **All bands** `0/8` … `7/8` | No train/OOD split within Polaris |
| Sampling method | **Stratified proportional (Hamilton)** | See §3.1 — deterministic quotas + within-band shuffle |
| Random seed | **42** | `random.Random(42)` for all within-band shuffles |
| 1.7B 8/8 drop | **No** (v1) | Deferred; HF set already excludes 7B `8/8` |

Alternative explicitly **out of scope** for this freeze: easy-band-only train, hard-band holdout, uniform random without stratification, re-labeling difficulty with 1.7B.

### 3.1 Stratified sampling algorithm (normative)

Bands in fixed order: `BANDS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8"]`.

1. Partition cleaned rows by `difficulty`. **Drop** rows whose `difficulty` ∉ `BANDS` (log count). **Fail** if any unexpected band remains after partition (should not happen on current HF).
2. Let `N_clean = |pool|`, `n = 16000`. For each band `b` with clean count `c_b`:
   - `exact = n * c_b / N_clean`
   - `quota_b = floor(exact)`
   - `remainder_b = exact - quota_b`
3. **Feasibility:** if any `quota_b > c_b` before step 4, raise `ValueError` (band too small). After step 4, if any band draw would exceed `c_b`, raise (should not happen if Hamilton is correct).
4. **Hamilton apportionment:** let `R = n - sum(quota_b)`. While `R > 0`, assign +1 to the band with largest `remainder_b`; tie-break **lower band index in `BANDS`** (i.e. prefer `0/8` over `1/8`). Subtract 1 from `R` and set that band’s remainder to 0 for subsequent ties.
5. **Within-band draw:** `rng = random.Random(seed)`. For each `b` in `BANDS` order, `pool_b = rows with difficulty b`, `rng.shuffle(pool_b)`, take first `quota_b` rows.
6. **Output order:** concatenate bands in `BANDS` order (not global shuffle). Assign `problem_id = 0..n-1` in that concatenation order.
7. **Reproducibility check:** two runs with same HF revision, seed, and `n` must produce identical ordered list of `hf_index` values.

## 4. Cleaning filters (apply before sampling)

Drop a row if any of:

1. `problem` missing, whitespace-only, or not a string.
2. `answer` empty after `str(answer).strip()` (no integer-only filter).

Keep `difficulty` as-is from HF (string `k/8`). Log counts: rows in, rows dropped per reason, rows per band after clean.

**Gold policy:** `gold = str(answer).strip()` — verbatim HF content (LaTeX, fractions, strings, etc.). Correctness at train time via `grade_parsed_answer` (mathd OR sympy), not manifest string shape.

**Probe note:** Group A manifests still use integer gold (`group_a_rollout_judge._clean_polaris_rows`); do not conflate with this freeze.

## 5. Output schema (`source/polaris_train_full.jsonl`)

One JSON object per line, UTF-8, no trailing garbage. Fields:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `problem_id` | int | yes | `0 .. len-1` in output order (stable within file) |
| `problem` | str | yes | verbatim from HF after clean |
| `gold` | str | yes | HF answer verbatim (strip only) |
| `difficulty_band` | str | yes | copy of HF `difficulty` |
| `hf_index` | int | yes | original row index in HF `train` split after load (for traceability) |

Do **not** embed prompt templates or chat messages in jsonl.

## 6. Provenance (`source/polaris_train_full.meta.json`)

JSON object with at least:

```json
{
  "dataset_id": "POLARIS-Project/Polaris-Dataset-53K",
  "dataset_revision": "<sha or unknown>",
  "split": "train",
  "sampling": {
    "method": "stratified_proportional",
    "target_n": 16000,
    "seed": 42,
    "bands": ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8"]
  },
  "cleaning": {
    "drop_empty_gold": true,
    "drop_empty_problem": true,
    "gold_policy": "verbatim_hf_strip_only"
  },
  "counts": {
    "hf_rows": 53291,
    "after_clean": "<n>",
    "written": 16000,
    "dropped_cleaning": "<n>",
    "per_band_after_clean": {},
    "per_band_in_output": {}
  },
  "materialized_at": "<ISO8601 UTC>",
  "output_files": {
    "jsonl": "main/data/source/polaris_train_full.jsonl",
    "meta": "main/data/source/polaris_train_full.meta.json"
  }
}
```

## 7. Implementation location

| Artifact | Path |
|----------|------|
| Script | `main/data/preprocess_polaris.py` |
| CLI | `python -m data.preprocess_polaris` from `main/` **or** `python main/data/preprocess_polaris.py` with repo-root `PYTHONPATH` |
| Args | `--out-dir` (default `main/data`), `--n` (default 16000), `--seed` (default 42), `--dataset` (default HF id), `--dry-run` (stats only, no write; **run before first materialization**) |

Structure:

- `normalize_train_gold` / `is_nonempty_gold` from `main/data/gold_utils.py`.
- `load_and_clean() -> list[dict]`
- `stratified_sample(rows, n, seed) -> list[dict]`
- `write_jsonl(path, rows)`, `write_meta(path, meta)`
- `main()` orchestration + logging

## 8. Validation (must pass before declaring freeze)

1. **Row count:** exactly 16,000 lines in jsonl (unless clean pool &lt; 16k — then fail loud, do not pad).
2. **Uniqueness:** all `hf_index` unique in output.
3. **Schema:** every row has required keys; `gold` non-empty after strip.
4. **Band totals:** sum of `per_band_in_output` = 16000; each band present if present after clean.
5. **Proportion sanity:** each band share within **±1.0 pp** of `band_count_after_clean / N_clean` (loose check for stratified rounding).
6. **Loader smoke:** `JsonlPromptDataset(str(path), seed=0)` loads and `next_batch(8)` returns 8 non-empty problems and golds.
7. **Unit tests:** add `main/tests/test_preprocess_polaris.py` with synthetic tiny HF-like rows (no network): clean filters, stratified counts, determinism same seed.

## 9. Post-run (human / ops, not script)

- Upload **train** jsonl to Modal `main-artifacts` at `/vol/data/polaris_train.jsonl` (not `source/polaris_train_full.jsonl`).
- Full pool stays local under `source/` (gitignored jsonl); **do** commit `polaris_train.meta.json` and `polaris_train.jsonl` for train freeze.
- Update PLAN §2 “TBD” → decided (fix difficulty semantics line 31); add one line to `main/docs/timeline.md` with band histogram summary.
- If re-materializing later: dated note in `main/docs/context.md` per PLAN §2 freeze policy.

## 10. Non-goals

- Parquet / verl `gsm8k.py` schema
- Eval split materialization
- Dynamic mid-training difficulty drops
- Re-scoring or rollout passes on 1.7B

## 11. Acceptance checklist

- [ ] `preprocess_polaris.py` implements plan §3–§7
- [ ] Tests pass: `pytest main/tests/test_preprocess_polaris.py`
- [x] `source/polaris_train_full.jsonl` + meta written locally; `polaris_train.jsonl` via filter script
- [ ] Meta band histogram attached to timeline note
