# Small-Compute RL Pilot — Source of Truth

**Orchestrator-owned.** Subagents may not change frozen fields without orchestrator approval.

## Objective

Discrimination study: GRPO (majority) vs `inverse_freq` (minority) vs F-GRPO (nearest neighbor), at fixed compute. One decision token after pilot: `ESCALATE` | `PIVOT_WORST_SUBSET` | `PIVOT_SUBSTRATE_OR_ARCH` | `STOP_NO_SIGNAL`.

## Frozen scope (approved 2026-05-18)

| Field | Value |
|-------|--------|
| Model | `Qwen/Qwen3-1.7B-Base` |
| Train data | `pilot/data/dapo_slice_3k.jsonl` — 3000 prompts; stable sort `prompt_id` → shuffle `seed=42` → first 3k from `open-r1/DAPO-Math-17k-Processed` **config `en`** |
| Run0 slice | Rows **0–499** of the **same** train file (no separate draw) |
| **Pilot primary OOD** | `pilot/data/aime25_eval_30.jsonl` (30) — `MathArena/aime_2025` |
| **Pilot secondary OOD** | `pilot/data/hmmt_nov25_eval_30.jsonl` (30) — `MathArena/hmmt_nov_2025` |
| **Pilot sanity** | `pilot/data/math500_sanity_100.jsonl` (100) — MATH-500: proportional `level` × `subject`, `seed=42` |
| **Paper primary OOD** (post-`ESCALATE` only) | `pilot/data/beyond_aime_eval_100.jsonl` (100) — `ByteDance-Seed/BeyondAIME` |
| **Paper secondary OOD** | `pilot/data/hmmt_feb25_eval_30.jsonl` (30) — `MathArena/hmmt_feb_2025` |
| **Paper MATH-500 full** | `pilot/data/math500_eval_500.jsonl` (500) |
| Rollouts / prompt (N) | 8 |
| Pilot steps | 100 |
| Optimizer / KL / clip | See `configs/shared_train.yaml` |
| Clustering | Exact-answer canonicalization (`pilot/train/canonicalize.py`) |

### Two-tier eval

| Tier | When | Sets used |
|------|------|-----------|
| **Tier 1 — pilot discrimination** | Run0–Run3 gate | `aime25_eval_30` + `hmmt_nov25_eval_30` only |
| **Tier 2 — paper headline** | After `ESCALATE` | `beyond_aime_eval_100`, `hmmt_feb25_eval_30`, `math500_eval_500` |

**Sanity (`math500_sanity_100`):** report Pass@1 / Pass@8; **not** in gate logic.

### Metrics & gates

| Rule | Value |
|------|--------|
| Pass@k | k = 8 |
| Cover@τ | τ = 0.15 |
| Worst-subset | bottom 25% prompts |
| Run0 minority gate | ≥ 15% prompts with correct minority cluster |
| Tail gain (Run2 vs Run1) | ≥ 3 pp on tier-1 pooled tail metrics + bootstrap CI excludes 0 |
| Pass@1 regression cap | ≤ 2 pp |
| Run1 vs Run1b noise stop | max tail \|Δ\| > **6 pp** → `STOP_NO_SIGNAL` |
| F-GRPO equivalence | \|Δ\| ≤ 1.5 pp on tail metrics → `PIVOT_SUBSTRATE_OR_ARCH` |

**Do not** interpret pilot numbers on Beyond-AIME; tier-2 eval runs only after `ESCALATE`.

## Run matrix

| Run | ID | Objective | Seed | Budget cap |
|-----|-----|-----------|------|------------|
| 0 | `run0_proxy` | None (rollouts only) | 42 | $24 |
| 1 | `run1_grpo` | `grpo` | 42 | $36 |
| 1b | `run1b_grpo` | `grpo` | 43 | $36 |
| 2 | `run2_inverse_freq` | `inverse_freq` (γ=1, w_max=8) | 42 | $36 |
| 3 | `run3_f_grpo` | `f_grpo` | 42 | $36 |

**Pilot total ceiling:** $405 (~29% of $1400).

## Execution sequence

1. **Materialize data** — `python pilot/scripts/materialize_data_slices.py` (HF pulls + SHA update).
2. **Preflight** — `python pilot/scripts/preflight_check.py` exit 0.
3. **HF + Modal setup** — see `pilot/infra/MODEL_DATA_SETUP.md`.
4. **Run0 → Run1/1b → Run2 → (conditional Run3) → gate.**

## Artifact layout

```
pilot/artifacts/<run_id>/
  config.snapshot.yaml
  git_sha.txt
  raw_predictions.jsonl   # fields: prompt_id, eval_split, correct, cluster_id, ...
  metrics.json            # per-split + pooled pilot_eval
  metrics_ci.json
  train.log
  cost.json
```

## Data / model setup (required before GPU)

See **`pilot/infra/MODEL_DATA_SETUP.md`**: HuggingFace dataset pulls, `Qwen/Qwen3-1.7B-Base` weights on Modal volume, and `HF_TOKEN` for gated assets.
