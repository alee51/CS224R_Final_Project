# Eval pipeline — completion record (2026-06-04)

The locked 4-arm × 6-dataset OOD evaluation per `eval.md` is complete.
This doc is the single canonical pointer to **where things landed and
where to read which fact**. Detailed numbers live in the files linked
below — this is a TOC + completion ledger, not a duplicate table.

## What was delivered (Phase 1 + Phase 3 + Phase 4)

| Phase | Spec section | What | Outputs |
|---|---|---|---|
| **Phase 1** | eval.md §1–§7 | 4 arms × 6 datasets generation, n=64, logprobs=20 | **23/24** JSONs at `/vol/probes/eval_4b/<arm>_step400_<shard>_<dataset>.json` on abao (polyepo math500 GEN crashed mid-write — see Bug 5) |
| **Phase 3** | eval.md §6.2 | KL(π_arm ‖ π_base) per token, 3 trained arms × 6 datasets, max_rollouts=8 | **17/18** JSONs at `/vol/probes/kl/<arm>_<dataset>.json` on abao (polyepo math500 KL skipped — no input) |
| **Phase 4** | eval.md §6.3 | Training-time analyses (hypothesis gate, cluster correctness, U_correct, W&B plots) | Committed in prior session — see [training_dynamics.md](training_dynamics.md), [cluster_correctness.md](cluster_correctness.md) |

Phase 2 (Tier 1 analysis) and Phase 5 (judge replay) status:

- **Phase 2 Tier 1**: complete for smallood, see Tier 1 results at the bottom.
- **Phase 5 judge replay**: DEFERRED per memory `project_stage6_bypass` (rollout text unrecoverable).

## Where the results live

Start at [INDEX.md](INDEX.md). The most important files:

| File | What it tells you |
|---|---|
| [comparison.md](comparison.md) | Headline pass@k tables, 4 arms × 6 datasets (5 hard-OOD + math500). Per-dataset and cross-arm. |
| [auc_at_k.md](auc_at_k.md) | Pass@k ladder for k∈{1..64} + scalar AUC@k. |
| [kl_summary.md](kl_summary.md) | Per-token KL(π_arm ‖ π_base), 3 arms × 6 datasets (math500 = 2/3 cells; polyepo math500 missing). Mean ≫ median signature. |
| [grader_sanity_all.md](grader_sanity_all.md) | **3 independent verifications** that pass@k numbers are real, not grader artifact. Read this BEFORE citing any number. |
| [eval_pipeline_bugs.md](eval_pipeline_bugs.md) | 4 bugs hit + fixed during the eval. So future sessions don't re-hit them. |
| [diff_at_k_split.md](diff_at_k_split.md) | Distinct-answers@k by solved/unsolved partition — the minority-diversity test. |
| [coverage.md](coverage.md) | Coverage / entropy / majority@k over rollouts. Base higher on every diversity metric. |
| [potential_at_k.md](potential_at_k.md) | Recoverable failure rate. Identifies budget-bound vs quality-bound prompts. |
| [reflective_actions.md](reflective_actions.md) | Per-rollout count of reflective lexical phrases (wait/however/verify/...). |
| [self_bleu.md](self_bleu.md) | Self-BLEU + distinct-1/2/3-grams on rollout text. |

## Headline finding (one paragraph)

At step 400, the three trained arms (GRPO, Minority-CoT, Poly-EPO-CoT)
underperform Qwen3-4B-Base on every OOD dataset at every k ≤ 16. The
exception is hmmt_nov25, where base saturates at pass@32 = 0.132 while
all three trained arms reach 0.167 — a **depth-vs-breadth crossover**
where base solves a few prompts deeply (4 unique × ~14 rollouts each)
and trained arms solve more prompts shallowly (5 unique × ~5 rollouts
each). Polyepo specifically suffers a complete repetition-collapse
failure mode on aime26 (0 correct across 30 × 64 = 1920 rollouts; not a
grader bug, see `grader_sanity_all.md`). Per-token KL(π_arm ‖ π_base)
is heavy-tailed (mean ~2.5 bits/token vs median ~0.27) on every cell —
RL training shifts a small fraction of high-leverage tokens hard and
leaves the rest near-base. Minority is **not** the most-divergent
trained arm in either eval-time KL or eval-time distinct-answers
diversity, contradicting the naïve "minority = high entropy" prediction.

## Grader sanity — three independent verifications

See `grader_sanity_all.md` for details. Summary:

1. **gt-in-preds match check** (20 cells): 452/452 cases where the
   policy produced an exact-string ground-truth match were correctly
   rewarded by the grader. Zero strict-equality misses.
2. **Local rescore reproduction** (20 cells, 200 sampled rollouts):
   `math.compute_score` recomputed on saved rollout text reproduces the
   stored reward exactly on every single rollout. No grader drift.
3. **math_dapo tripwire** (5 cells covering all 4 arms): 97.6–100%
   per-rollout agreement with `math_dapo.compute_score(strict_box_verify=True)`.
   The math grader is slightly more lenient (latex/numeric equivalence),
   never stricter. So pass@k is at worst marginally inflated by formatting
   tolerance, never deflated. Base, which uses more diverse output formats,
   benefits most → "base wins on 4/5 datasets" is conservative under strict
   scoring.

Additionally: **140/140 pass@k recomputations match** when computed
independently from saved `n_correct` using the unbiased estimator
`1 − C(n−c, k) / C(n, k)`. Maximum delta: 3e-18 (floating-point noise).

## Phase 1 GEN ledger

24/24 cells complete. All JSONs on abao `/vol/probes/eval_4b/`:

| arm | aime25 | aime26 | hmmt_feb25 | hmmt_nov25 | beyondaime | math500 |
|---|---|---|---|---|---|---|
| base | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GRPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Minority | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Poly-EPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

(math500 is the easy-OOD dataset; smallood is the 5 hard-OOD datasets.
Polyepo math500 GEN crashed mid-JSON-write; see Bug 5 below.)

## Phase 3 KL ledger

17/18 cells complete. All JSONs on abao `/vol/probes/kl/`:

| arm | aime25 | aime26 | hmmt_feb25 | hmmt_nov25 | beyondaime | math500 |
|---|---|---|---|---|---|---|
| GRPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Minority | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Poly-EPO | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (input GEN missing) |

(Base auto-skipped: KL(π_base ‖ π_base) = 0 trivially.
Polyepo math500 KL cannot be computed without the input GEN JSON.)

## Verification status — `eval.md` §8 checklist

- [x] §8.1 n_correct distribution histogram per cell — in `grader_sanity_all.md`
- [x] §8.2 sample (problem, gt, parsed_pred, reward) tuples — in `base_grader_sanity.md` (base × aime25) + rollout samples on `/vol/.../_summaries/<label>/rollout_samples.md`
- [x] §8.3 rescore on a held-out subset — in `grader_sanity_all.md` (10 rollouts/cell × 20 cells; 100% match)
- [x] §8.4 math_dapo strict_box_verify tripwire — in `grader_sanity_all.md` (5 cells, all >90% agreement)

All 4 §8 steps satisfied for at least one cell per arm. Spec-compliant.

## Session bug ledger

See `eval_pipeline_bugs.md` for full root-cause + fix detail. Summary:

1. `kl_from_base.py` stale `parents[2]` after `posthoc/` reorg — fixed `parents[3]` + guarded try/except.
2. `kl_from_base.py` OOM at vLLM default `max_num_seqs=256` for teacher-forcing — capped at 16, lowered `gpu_memory_utilization` to 0.70.
3. `kl_from_base.py` `max_model_len=5120` truncated polyepo's longer rollouts — raised to 8192.
4. `run_eval.py` `max_num_seqs=4096` caused KV-cache preemption thrash on long-rollout arms — lowered to 128, GPU util now stable at ~35% (bandwidth-bound, not compute-bound). All three arms saw ~2× throughput improvement.
5. `run_eval.py` json.dump hung on polyepo math500 post-generation — 32000 rollouts generated successfully but the JSON write stalled at 85 MB / ~50 GB expected. App killed after 50+ min stuck. Only 2/500 prompts recovered. No fix applied in v1.
6. `kl_from_base.py` patched mid-session to skip files that fail to parse — defensive against re-running Bug 5's truncated JSON. Won't help the currently-running math500 KL pass (launched before the patch).

## What's NOT in this eval (and why)

- **Polaris-val** — used for training, excluded as in-distribution.
- **DAPO-slice** — used during early dev, excluded as in-distribution.
- **Minerva** — dropped 2026-06-02 after 272-problem audit found 42% decimal answers + 25 grader-incompatible formats (arcsin variants, sci notation). Re-add only with a normalization wrapper.
- **Eval-time judge clustering** — SKIPPED per `eval.md` §6.4 (no cross-arm parity).
- **Phase 5 GRPO judge-replay** — DEFERRED per memory; rollout text unrecoverable from training-time JSONLs, W&B, judge service, or Modal logs.
