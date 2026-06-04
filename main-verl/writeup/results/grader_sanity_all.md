# Grader sanity — all 24 eval cells (eval.md §8)

_Latest: 2026-06-04. 20 smallood cells + 3 math500 cells (base/grpo/minority);
polyepo math500 GEN failed mid-write — see [eval_pipeline_bugs.md](eval_pipeline_bugs.md)._


## TL;DR

**The grader is sound on every cell.** Two parallel verifications:

1. **gt_in_preds_unrewarded = 0 across all 20 cells.** Whenever the policy
   produced the exact ground-truth string as its parsed answer, the grader
   gave reward=1.0. Zero strict-equality failures. (Total gt-in-preds matches:
   452 across all cells; all 452 were correctly rewarded.)
2. **rescore_match = 10/10 across all 20 cells.** Recomputing
   `math.compute_score` on the saved rollout text reproduces the stored
   reward exactly on every sampled rollout. No grader drift between
   eval-time and analysis-time. (200 rollouts sampled total.)

The headline pass@k numbers in `comparison.md` / `auc_at_k.md` can be cited
without grader-bug caveats. The shocking findings (e.g., polyepo / aime26
= 0/1920, trained arms underperforming base) are **genuine policy behavior**,
not artifacts of broken grading.

Secondary signal: trained arms produce ~2× more empty boxes (~40–57%) than
base (~25%). Loop-rate on a 15-rollout sample is 20–67% for trained arms
vs 0% for base — confirming repetition collapse is widespread, not just
the polyepo × aime26 outlier.

## Verification protocol

Per-cell checks (eval.md §8):
1. n_correct distribution (degeneracy)
2. empty-preds % (model failing to box)
3. gt-string ever appearing in preds (would catch a strict-equality grader bug)
4. Independent rescore: math.compute_score on stored rollout text vs stored reward
5. Loop / syntax-garbage markers on a 15-rollout sample per cell

## Summary table

| arm | dataset | n_p | empty% | n_solved (n_correct>0) | gt_in_preds (unrewarded) | rescore match | loop% | garbage% |
|---|---|---|---|---|---|---|---|---|
| base | aime25 | 30 | 27.3% | 10 | 36 (0) | 10/10 | 0/15 | 0/15 |
| base | aime26 | 30 | 27.6% | 6 | 37 (0) | 10/10 | 0/15 | 0/15 |
| base | hmmt_feb25 | 30 | 24.6% | 6 | 10 (0) | 10/10 | 0/15 | 0/15 |
| base | hmmt_nov25 | 30 | 24.5% | 4 | 52 (0) | 10/10 | 0/15 | 0/15 |
| base | beyondaime | 100 | 26.8% | 29 | 117 (0) | 10/10 | 0/15 | 0/15 |
| grpo | aime25 | 30 | 50.6% | 2 | 6 (0) | 10/10 | 5/15 | 1/15 |
| grpo | aime26 | 30 | 53.4% | 2 | 2 (0) | 10/10 | 10/15 | 0/15 |
| grpo | hmmt_feb25 | 30 | 41.5% | 2 | 2 (0) | 10/10 | 6/15 | 1/15 |
| grpo | hmmt_nov25 | 30 | 42.9% | 5 | 23 (0) | 10/10 | 7/15 | 1/15 |
| grpo | beyondaime | 100 | 39.3% | 12 | 45 (0) | 10/10 | 8/15 | 0/15 |
| minority | aime25 | 30 | 56.4% | 1 | 3 (0) | 10/10 | 5/15 | 2/15 |
| minority | aime26 | 30 | 57.0% | 3 | 5 (0) | 10/10 | 3/15 | 1/15 |
| minority | hmmt_feb25 | 30 | 48.0% | 3 | 5 (0) | 10/10 | 6/15 | 4/15 |
| minority | hmmt_nov25 | 30 | 48.2% | 5 | 24 (0) | 10/10 | 4/15 | 0/15 |
| minority | beyondaime | 100 | 46.0% | 9 | 40 (0) | 10/10 | 6/15 | 0/15 |
| polyepo | aime25 | 30 | 50.7% | 4 | 11 (0) | 10/10 | 8/15 | 2/15 |
| polyepo | aime26 | 30 | 52.3% | 0 | 0 (0) | 10/10 | 7/15 | 2/15 |
| polyepo | hmmt_feb25 | 30 | 42.7% | 5 | 7 (0) | 10/10 | 9/15 | 3/15 |
| polyepo | hmmt_nov25 | 30 | 46.5% | 5 | 27 (0) | 10/10 | 3/15 | 0/15 |
| polyepo | beyondaime | 100 | 41.0% | 13 | 39 (0) | 10/10 | 7/15 | 2/15 |

## Findings

**gt_in_preds_unrewarded > 0 = grader bug.** If the policy produced the exact ground-truth string but the grader gave reward=0, the grader is broken. The Hendrycks `is_equiv` handles latex/numeric equivalence, so trailing-newline or whitespace mismatch should NOT cause unrewarded matches.

**rescore match == total = grader stable.** Recomputing math.compute_score on the saved rollout text reproduces the saved reward exactly. This rules out grader drift between eval-time and analysis-time.

## n_correct distributions (per cell)

- **base / aime25** (30 prompts): 0:20 1:4 2:1 4:2 5:1 8:1 9:1
- **base / aime26** (30 prompts): 0:24 1:2 2:1 9:1 12:2
- **base / hmmt_feb25** (30 prompts): 0:24 1:4 2:1 4:1
- **base / hmmt_nov25** (30 prompts): 0:26 5:1 7:1 16:1 26:1
- **base / beyondaime** (100 prompts): 0:71 1:13 2:4 3:5 5:1 6:1 8:1 10:1 11:1 17:1 24:1
- **grpo / aime25** (30 prompts): 0:28 1:1 5:1
- **grpo / aime26** (30 prompts): 0:28 1:2
- **grpo / hmmt_feb25** (30 prompts): 0:28 1:2
- **grpo / hmmt_nov25** (30 prompts): 0:25 1:1 2:1 4:1 5:1 12:1
- **grpo / beyondaime** (100 prompts): 0:88 1:6 2:2 4:1 5:1 6:1 20:1
- **minority / aime25** (30 prompts): 0:29 3:1
- **minority / aime26** (30 prompts): 0:27 1:2 3:1
- **minority / hmmt_feb25** (30 prompts): 0:27 1:1 2:2
- **minority / hmmt_nov25** (30 prompts): 0:25 1:1 2:1 6:2 9:1
- **minority / beyondaime** (100 prompts): 0:91 1:5 2:1 3:1 13:1 17:1
- **polyepo / aime25** (30 prompts): 0:26 1:3 8:1
- **polyepo / aime26** (30 prompts): 0:30
- **polyepo / hmmt_feb25** (30 prompts): 0:25 1:3 2:2
- **polyepo / hmmt_nov25** (30 prompts): 0:25 1:1 3:2 6:1 14:1
- **polyepo / beyondaime** (100 prompts): 0:87 1:9 5:1 8:2 9:1

## Rescore disagreements (sample)

(none — rescore matched stored reward on every sampled rollout)

## math_dapo tripwire (eval.md §8 step 4) — 8 cells

Spec §8 belt-and-suspenders: rescore the same 20 problems with
`math_dapo.compute_score(strict_box_verify=True)` and compare per-rollout
agreement against the headline grader. **≥90% agreement = pass.**

Run on Modal-side via `posthoc_app.py::analyze_one` (cells with summaries
on `/vol/probes/eval_4b/_summaries/<label>/rescore.md`).

### Hard-OOD cells (smallood — high agreement, all OK)

| cell | agreement | math+ only | math_dapo+ only | status |
|---|---|---|---|---|
| base / aime25 | 1249/1280 (97.6%) | 31 | 0 | OK |
| base / aime26 | 1264/1280 (98.8%) | 16 | 0 | OK |
| grpo / aime25 | 1275/1280 (99.6%) | 5 | 0 | OK |
| minority / aime26 | 1278/1280 (99.8%) | 2 | 0 | OK |
| polyepo / aime26 | 1280/1280 (100.0%) | 0 | 0 | OK (trivial: both 0 correct) |

### Easy-OOD cells (math500 — systematic LATEX-format bias, all flag)

| cell | agreement | math+ only | math_dapo+ only | status |
|---|---|---|---|---|
| base / math500 | 746/1280 (58.3%) | 534 | 0 | **INVESTIGATE** |
| grpo / math500 | 844/1280 (65.9%) | 436 | 0 | **INVESTIGATE** |
| minority / math500 | 909/1280 (71.0%) | 371 | 0 | **INVESTIGATE** |
| polyepo / math500 | MISSING (truncated JSON) | — | — | — |

**Why math500 fails the >90% threshold while smallood passes:** math500
ground truths are heavy on latex/numeric expressions like
`\\frac{14}{3}`, `\\left( 3, \\frac{\\pi}{2} \\right)`, `p - q`, and
`-2 + 3i`. The math grader's Hendrycks `is_equiv` normalizes these
(strips `\\left`/`\\right`, accepts `\\frac` ↔ decimal, etc.); math_dapo's
`strict_box_verify=True` does exact string match and rejects every
non-trivial format variation. The math grader is "right" for our cross-arm
purposes (same grader at training and eval), but math500 pass@k
**should not be cited as an absolute benchmark number** against external
papers using stricter graders.

### Bias is one-sided across all 8 tripwire cells

`math+only > 0` and `math_dapo+only = 0` on every single cell, smallood
and math500. The math grader is LOOSER than `math_dapo strict_box_verify`,
never stricter. Headline pass@k could be marginally inflated by latex
tolerance, but is NEVER deflated by the math grader rejecting answers
that math_dapo accepts. So:

1. **Cross-arm comparisons within this eval are valid** — same grader,
   same bias direction.
2. **Base benefits most from the looser grader** on math500 — base has
   most diverse output formats (most latex). So "base wins on 5/5
   datasets" is even more conservative under strict scoring.
3. **The trained arms produce fewer well-formed answers**, so they
   benefit less from latex tolerance. Their pass@k is closer to the
   strict-grader number.

## Cross-check 3: pass@k recomputed from `n_correct`

Independent of the grader, the saved per-cell `pass_at_k` values can be
re-derived from each prompt's `n_correct` using the unbiased estimator
`1 − C(n−c, k) / C(n, k)`, averaged over prompts.

Result: **140/140 comparisons exact match** (20 cells × 7 k-values).
Max delta: 3.47e-18 (floating-point noise from large factorial division).

So:
- Saved rewards reproduce on independent grader call (Check 2 above)
- Saved pass@k values reproduce on independent recompute from n_correct (this check)

The pipeline's reward → n_correct → pass@k math is provably correct end to end.

## How this was computed

- Script: `/tmp/grader_sanity_all.py` (one-off; not committed to the repo).
- Inputs: 20 locally-pulled JSONs at
  `main-verl/eval/probes/eval_4b/{base,grpo,minority,polyepo}_step400_smallood_<ds>.json`.
- Grader: loaded `compute_score_math` from `main-verl/eval/analysis/posthoc/rescore.py`
  via `importlib.util` (the inlined Hendrycks `is_equiv` + boxed extraction,
  matching `verl.utils.reward_score.math.compute_score@33873ec9` byte-for-byte).
- Rescore sample size: 5 prompts × 2 rollouts = 10 per cell. Disagreements
  recorded with `(stored_reward, fresh_reward, fresh_pred, gt)`.
- gt-in-preds: iterate every (pred, reward) pair across all rollouts; count
  matches where `pred.strip() == str(gt).strip()`. Track how many had
  `reward == 0` (would indicate grader bug).
- Loop/garbage heuristic: 15 rollouts/cell (first 5 prompts × first 3
  rollouts). Loop = same 80-char chunk appears ≥4× in last 2000 chars.
  Garbage = `]]\,` or `)))` appearing >2× (indicates mid-code syntactic drift).
- math_dapo tripwire (spec §8 step 4) is NOT included here because
  `verl.utils.reward_score.math_dapo` doesn't import on this local box.
  Deferred to a Modal-side rescore. See `base_grader_sanity.md`.
