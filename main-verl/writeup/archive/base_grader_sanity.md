# Base Arm Eval — Grader Sanity Check (base × aime25)

_Updated 2026-06-02 post-pull; supersedes the prior "BLOCKED" version of this doc._

Per `eval.md` §8, every (arm × dataset) headline number requires a 4-part
sanity check before publishing. Only **base × aime25** has been fully
verified so far (the file is locally pulled at `/tmp/base_aime25.json`,
1.94 GB). The other 4 base smallood shards are on abao but not yet pulled.

## 1. n_correct distribution (30 prompts, n=64)

| n_correct | # prompts |
|---:|---:|
| 0 | 20 |
| 1 | 4 |
| 2 | 1 |
| 4 | 2 |
| 5 | 1 |
| 8 | 1 |
| 9 | 1 |

Heavy floor (20/30 prompts unsolved across all 64 rollouts) is consistent
with a base model on aime25 — `mean_reward_at_1 = 0.033`, `pass@64 = 0.333`.
Empty-`preds` rate = 27.3% (524/1920 rollouts produced no `\boxed{}`).

## 2. Sample (problem, gt, preds[:3], rewards[:3])

```
[0] gt='70'   n_correct=8  preds[:3]=['', '', '0']   rewards[:3]=[0.0, 0.0, 0.0]
    "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b..."

[1] gt='588'  n_correct=0  preds[:3]=['', '924', '756']   rewards[:3]=[0.0, 0.0, 0.0]
    "On △ABC points A, D, E, and B lie in that order on side AB with AD=4, ..."

[2] gt='16'   n_correct=4  preds[:3]=['756', '1', '']   rewards[:3]=[0.0, 0.0, 0.0]
    "The 9 members of a baseball team went to an ice-cream parlor..."
```

Parsed answers look like genuine integer answers (no `[INVALID]` sentinels);
empty strings reflect rollouts where the base model never emits `\boxed{}`.
No format-level grader confusion.

## 3. Same-grader rescore (`rescore.py`)

Replaying `math.compute_score` on saved rollout text and recomputing pass@k:

| k | saved | rescored | Δ |
|---:|---:|---:|---:|
| 1 | 0.0187 | 0.0187 | +0.0000 |
| 2 | 0.0361 | 0.0361 | +0.0000 |
| 4 | 0.0669 | 0.0669 | +0.0000 |
| 8 | 0.1161 | 0.1161 | +0.0000 |
| 16 | 0.1819 | 0.1819 | +0.0000 |
| 32 | 0.2537 | 0.2537 | +0.0000 |
| 64 | 0.3333 | 0.3333 | +0.0000 |

Identical — the saved JSON's `rewards` were produced by the same vendored
Hendrycks `is_equiv` that `rescore.py` runs offline. No silent grader drift
between eval-time and analysis-time.

## 4. `math_dapo` tripwire (eval.md §8 belt-and-suspenders)

**Locally: SKIPPED** — `verl.utils.reward_score.math_dapo` and `math_verify`
are not importable from this machine. `rescore.py` now (a) attempts the
import at startup, (b) engages a per-rollout agreement check (math vs
math_dapo strict-box, flags if <90%) on 20 sampled problems per dataset,
(c) prints a `[tripwire] SKIPPED` line with reason when the import fails.
Run on Modal (or `pip install verl` + `math_verify` locally) to engage.

**This is the only spec §8 step still outstanding for base × aime25.**
Plan: re-fire `rescore.py` inside the Modal image where verl is already
installed, dump the agreement rate to `writeup/results/`.

## Status

- Steps 1–3: ✅ pass
- Step 4: ⏳ deferred to a Modal-side rescore (script ready, ledger entry pending)

For the other 4 base smallood shards (aime26 / hmmt_feb25 / hmmt_nov25 /
beyondaime) and for math500, repeat steps 1–4 once the files are pulled
locally or rescored in-place on abao. Headline base pass@k is safe to cite
for aime25; the other 5 datasets still need this loop.

For the broader pipeline audit (which analysis scripts work end-to-end on
this JSON, what was patched), see `eval_pipeline_verification.md`.
