# Restart state — 2026-06-04 ~01:25 PDT

Resume context after overnight session. Read this first.

## Session-end summary

**Tier 1 analysis is COMPLETE for 4 arms × 5 OOD datasets** (20 eval JSONs).
All 6 analysis scripts produced annotated markdown results + a central
INDEX.md. KL Phase 3 produced 15 (arm, dataset) JSONs and a summary.

Start here: `main-verl/writeup/results/INDEX.md`. Headline tables also in
`main-verl/writeup/results/comparison.md`.

## Committed this session (chronological)

| sha | what |
|---|---|
| bbe2a34 | Tier 1 5 analyses (auc_at_k, diff_at_k_split, potential_at_k, reflective_actions, self_bleu) for 4 arms × 5 datasets. Also: 3 codepath fixes in analysis_io.py (strict=False, drop_heavy, write_markdown path); run_eval.py timeout 3h→6h. |
| 241e44e | Annotate Tier 1 results (TL;DR + How-this-was-computed) + INDEX.md with audit findings. 8 items flagged, 1 contradicting "base dominates" framing. |
| 31c3f3d | Verify polyepo/aime26 = 0 is real (repetition collapse, NOT grader bug) via rollout spot-check. |
| a3cd30c | coverage.md (annotated) + hmmt_nov25 depth-vs-breadth analysis. |
| afc3954 | Populate comparison.md with actual pass@k tables for all 4 arms × 5 datasets. |
| 70ec4f5 | kl_summary.md aggregated from 15 KL JSONs; INDEX updated. |
| bbe393b | Fix INDEX layout (kl_summary row was misplaced). |
| d9ce269 | Eval-time epilogue to minority_diagnostic.md: training-time diversity finding does NOT carry into eval-time. |

## Headline numbers (pass@k)

See `main-verl/writeup/results/comparison.md` for the full per-dataset
tables. Cross-arm summary:

- **Base wins on every (arm, dataset, k) for k≤16** across all 5 OOD datasets.
- **Only crossover is hmmt_nov25 at k≥32**: base saturates at 0.133; all 3
  trained arms reach 0.167. Explanation: depth vs breadth — base solves 4
  prompts deeply (54 correct rollouts), trained arms solve 5 prompts
  shallowly (24-27 correct).
- **polyepo / aime26 = 0/1920 verified real** (repetition collapse, not
  grader bug).
- **Minority is NOT the most-diverse trained arm at eval-time** on
  beyondaime unsolved: grpo 20.50 > polyepo 19.26 > minority 18.37
  (`diff_at_k_split.md`).
- **All trained arms collapse lexical diversity ~2×** vs base
  (`self_bleu.md`, `coverage.md`).

## Per-token KL from base

15 JSONs at `/vol/probes/kl/<arm>_<dataset>.json` (also pulled to `/tmp/`).
Per-arm mean (averaged over 5 datasets): grpo 2.71 > minority 2.57 >
polyepo 2.33 bits/token. **Mean ≫ median** for every cell — divergence
concentrated in a small fraction of high-leverage tokens. See
`main-verl/writeup/results/kl_summary.md`.

n_prompts inconsistency: my later overnight kl_pass run overwrote
grpo/{aime25, aime26, beyondaime} with `max_prompts=20` cap; the original
n=30/100 data was preserved for the other 12 cells. Killed the run at
~01:21 to prevent overwriting minority + polyepo data.

## What's deferred / still pending

1. **token_entropy_split.md** — requires `drop_heavy=False` in analysis_io
   (needs logprobs). 20 JSONs × multi-GB-of-logprobs = OOM risk if loaded
   all at once. Either run per-file with subprocess + merge, or stream
   parse. Not load-bearing for the headline poster numbers — kl_summary
   covers most of what token entropy would tell us.
2. **GRPO KL re-run at n=30/100** — my overnight run overwrote 3 cells
   with smaller n. Re-firing kl_from_base.py with `CS224R_KL_MAX_PROMPTS=0`
   would restore parity.
3. **Final consolidation commit + push** — locally everything's committed
   but not pushed to remote. `git push origin main` when ready.

## How to re-run any Tier 1 analysis

```
cd /Users/nancybao/Desktop/dev/cs224r_finalproject
PATHS=$(ls main-verl/eval/probes/eval_4b/*_step400_smallood_*.json | grep -v aime25-aime26)
python3 main-verl/eval/analysis/posthoc/<script>.py $PATHS --out <script>.md
# (output goes to main-verl/writeup/results/<script>.md via the patched
# write_markdown path; ~10-40 min per analysis depending on what it reads.)
```

## Memory updates this session

- `project_eval_findings_2026_06_04.md` (NEW): the eval-time findings,
  superseding the older training-time-only memory.
- `MEMORY.md` index updated.

## State of in-flight / background jobs

None. All sessions ended cleanly. KL Phase 3 Modal app `ap-3RW1A9wICGJ70fWGnvRITB`
stopped by `modal app stop -y` at 01:21 to preserve high-n data.
