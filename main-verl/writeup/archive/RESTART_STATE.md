# Restart state — 2026-06-04 03:00 PDT (overnight session complete)

Resume context for fresh post-compact session. Read this first.

## Session-end summary

**Phase 1 + Phase 3 + Tier 1 analysis essentially DONE for the 4-arm × 6-dataset
locked eval.** Canonical pointer: [`eval_complete.md`](eval_complete.md).
Single-line summary of every analysis file: [`INDEX.md`](INDEX.md).

One cell is missing: **polyepo × math500 GEN crashed mid-JSON-write**
(`ap-h8zHYGx8IuvDhiPOfYtITd`). 23/24 Phase 1 cells, 17/18 Phase 3 KL cells
complete. v1 poster ships with this. Re-fire deferred —
see [`eval_pipeline_bugs.md`](eval_pipeline_bugs.md) Bug 5.

## What's on disk

### Phase 1 GEN (23/24 cells on abao `/vol/probes/eval_4b/`)

| arm | aime25 | aime26 | hmmt_feb25 | hmmt_nov25 | beyondaime | math500 |
|---|---|---|---|---|---|---|
| base | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GRPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Minority | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Poly-EPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

### Phase 3 KL (17/18 cells on abao `/vol/probes/kl/`)

| arm | aime25 | aime26 | hmmt_feb25 | hmmt_nov25 | beyondaime | math500 |
|---|---|---|---|---|---|---|
| GRPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Minority | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Poly-EPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (input GEN missing) |

(Base auto-skipped: KL(π_base ‖ π_base) = 0.)

### Tier 1 analysis markdowns at `main-verl/writeup/results/`

- `auc_at_k.md` — pass@k ladder + AUC@k (4 arms × 6 datasets, polyepo/math500 missing)
- `comparison.md` — headline pass@k tables
- `coverage.md`, `diff_at_k_split.md`, `potential_at_k.md`, `reflective_actions.md`, `self_bleu.md` — 5 smallood diversity/reflection analyses
- `kl_summary.md` — per-token KL from base (15 smallood + 2 math500)
- `grader_sanity_all.md` — 3-way grader verification (eval.md §8)
- `eval_pipeline_bugs.md` — 5 bugs hit + fixed
- `eval_complete.md` — single canonical completion record
- `INDEX.md` — file index

### Memory updates
- `project_eval_findings_2026_06_04.md` — main eval-time findings
- (existing) `feedback_eval_verification.md` — protocol that flagged the grader risk

## Headline finding (one paragraph for the poster intro)

At step 400, the three trained arms (GRPO, Minority-CoT, Poly-EPO-CoT)
underperform Qwen3-4B-Base on every OOD dataset at every k≤16. The
exception is hmmt_nov25, where base saturates at pass@32=0.132 while
all three trained arms reach 0.167 — a depth-vs-breadth crossover.
Polyepo specifically suffers a complete repetition-collapse failure
mode on aime26 (0/30 prompts solved across 64 rollouts each, verified
real not grader-artifact). Per-token KL(π_arm ‖ π_base) is heavy-tailed
(mean ~2.5 bits/token, median ~0.27) — RL training shifts a small
fraction of high-leverage tokens hard, leaves the rest near-base.
Minority is **not** the most-divergent trained arm in either eval-time
KL or eval-time distinct-answers diversity, contradicting the naïve
"minority = high entropy" prediction.

## Grader is verified

3 independent verifications, all passing — pass@k is genuine policy
behavior, not grader artifact:

1. **gt-in-preds match**: 452/452 cases where policy produced exact
   ground-truth string were rewarded. Zero strict-equality misses.
2. **Local rescore**: 200 sampled rollouts across 20 cells, 100% match
   between stored reward and recomputed reward via `math.compute_score`.
3. **math_dapo tripwire**: smallood agreement 97.6–100% across all 4
   arms. math500 agreement is lower (58.3–71.0%) due to latex
   normalization bias in `is_equiv` — math grader is consistently
   LOOSER, never stricter, so cross-arm comparisons within our eval
   are valid.
4. **Pass@k recompute**: 140/140 saved values reproduce exactly from
   independent recompute on `n_correct` using
   `1 − C(n−c, k) / C(n, k)`. Max delta: 3e-18 (floating-point noise).

Full detail: `grader_sanity_all.md`.

## Bugs hit + fixed this session

See `eval_pipeline_bugs.md`. Summary:

1. `kl_from_base.py` stale `parents[2]` after posthoc/ reorg → `parents[3]`+guard
2. `kl_from_base.py` vLLM OOM at default `max_num_seqs=256` for teacher-forcing → `max_num_seqs=16, gpu_mem_util=0.70`
3. `kl_from_base.py` `max_model_len=5120` truncated polyepo rollouts → raised to 8192
4. `run_eval.py` `max_num_seqs=4096` caused KV preemption thrash on long-rollout arms → lowered to 128, GPU util steady at 35% (bandwidth-bound)
5. `run_eval.py` json.dump hung on polyepo math500 post-generation → only 2/500 prompts written, no fix applied (deferred)

Also patched `kl_from_base.py` to skip files that fail to parse (Bug 5
mitigation — future re-runs won't crash on truncated GEN JSONs).

## Apps stopped at session end

| app | reason |
|---|---|
| `ap-h8zHYGx8IuvDhiPOfYtITd` | polyepo math500 GEN, killed because json.dump hung |
| `ap-F226GQblGB5rxcFgFReI91` | smallood KL re-run for consistent n; completed cleanly |

## Apps in flight at session end

| app | what | ETA |
|---|---|---|
| `ap-nr3cBVsR1NcW0Ip0op570D` | math500 KL — grpo done, minority in progress (~8% at session start), polyepo will fail | ~30 min for minority then crash on polyepo (corrupted JSON) |

The crash on polyepo math500 KL is expected and benign — kl_from_base.py
was patched mid-session to handle this gracefully, but the patch went in
after this app was already launched.

## Resume checklist

1. Read this doc + `eval_complete.md` + `INDEX.md`
2. `MODAL_PROFILE=abao modal app list` — confirm `ap-nr3c` is stopped or completed
3. Pull final 2 math500 KL cells if not already in `kl_summary.md`:
   `MODAL_PROFILE=abao modal volume get main-artifacts probes/kl/{grpo,minority}_math500.json /tmp/`
4. If re-firing polyepo math500 GEN: use the updated `run_eval.py` config
   (`max_num_seqs=128`, `gpu_mem_util=0.98`) and watch for the json.dump
   hang. Consider patching `json.dump(..., indent=None)` first.
5. v1 poster work — pull selected tables from `comparison.md`, `auc_at_k.md`,
   `kl_summary.md`. Verification sentence: "all numbers verified per
   `grader_sanity_all.md`".

## What to NOT redo

- All Tier 1 analyses (auc_at_k, coverage, etc.) — committed, no need to re-run
- Grader sanity — fully verified, 100% match rates documented
- KL on smallood cells — re-run done at consistent n on 2026-06-04 by `ap-F226`
- Phase 4 training-time analyses — done in earlier session
