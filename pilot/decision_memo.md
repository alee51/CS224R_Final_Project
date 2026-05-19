# Pilot Decision Memo

**Status:** BLOCKED — awaiting HF data materialization + GPU runs  
**Frozen scope:** v1.1.0 (`pilot/preflight_lock.json`, approved 2026-05-18)

## Tier summary

| Tier | Eval sets | Used for |
|------|-----------|----------|
| **Pilot (tier 1)** | `aime25_eval_30` + `hmmt_nov25_eval_30` | Gate / tail metrics |
| **Sanity** | `math500_sanity_100` | Pass@1/8 pipeline check only |
| **Paper (tier 2)** | `beyond_aime_eval_100`, `hmmt_feb25_eval_30`, `math500_eval_500` | Post-`ESCALATE` only |

**Model:** `Qwen/Qwen3-1.7B-Base`  
**Train:** `dapo_slice_3k.jsonl` (rows 0–499 for Run0)

## Decision

**Token:** `PENDING`

## Next steps

1. `python pilot/scripts/materialize_data_slices.py`  
2. `python pilot/scripts/preflight_check.py`  
3. Modal + HF setup: `pilot/infra/MODEL_DATA_SETUP.md`  
4. Run0 → Run1/1b → Run2 → (optional Run3) → `python pilot/eval/gate.py`
