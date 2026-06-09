# Eval audit caveats + interpretive notes

For numbers + the file index, see [`INDEX.md`](INDEX.md).
For the at-a-glance overview, see [`../eval_summary.md`](../eval_summary.md).

## The hmmt_nov25 crossover (2026-06-04)

Why does base lose to all 3 trained arms on hmmt_nov25 at k≥32? **Depth vs breadth.**

*Source: per-prompt `n_correct` distributions in `grader_sanity_all.md:75,80,85,90`.*

| arm | n=0 prompts | n=1 | n=2-4 | n=5-9 | n=10-31 | total_correct (of 1920) |
|---|---|---|---|---|---|---|
| **base** | 26 | 0 | 0 | 2 | **2** | **54** |
| grpo | 25 | 1 | 2 | 1 | 1 | 24 |
| minority | 25 | 1 | 1 | 3 | 0 | 24 |
| polyepo | 25 | 1 | 2 | 1 | 1 | 27 |

Base solves **4 unique prompts** but **deeply** (concentrated: 5, 16, 26, 7
correct rollouts on those 4 — 54 total). Trained arms solve **5 unique prompts**
but **shallowly** (mostly 1-12 correct out of 64).

- Low k (1-16): base wins because each *attempt* is more likely correct
- pass@64: trained arms win because they cover one more unique prompt
  (5/30 = 0.167 vs 4/30 = 0.133)

Trained arms trade depth for breadth — more answer diversity occasionally
stumbles onto a correct answer on more prompts via stochastic exploration.
The "diversity hypothesis" works **only at saturated sample budgets**.

This is hmmt_nov25-specific. Other datasets don't show the crossover —
trained arms don't even cover the same number of unique prompts as base.

Bottom line: at step-400 the trained arms underperform base on the OOD math
panel except on hmmt_nov25, where base saturates early and all 3 trained
arms catch and slightly pass it by k=32. The story is qualitatively
consistent across all five result files in `results/`.

## Polyepo × math500 reconciliation (2026-06-08)

`eval_complete.md` Bug 5 reports the original locked-config GEN crashed
mid-write (2/500 prompts saved on 2026-06-02). A re-run JSON now exists on
`abao:/vol/probes/eval_4b/polyepo_step400_math500_math500.json`, written
2026-06-08 00:31 PDT — Anastasia ran it before her CoT-diversity pass at
01:10 PDT.

**Size anomaly:** 152 MiB vs. 21–44 GiB for the sibling math500 files
(~140× smaller); likely a logprobs-stripped re-run that's still rich
enough for CoT clustering.

**Status:**
- CoT diversity@k for polyepo×math500 is **safe to cite** (cluster
  assignments only need rollout text).
- Pass@k for polyepo×math500 **NOT yet reconciled into `comparison.md`** —
  schema parity (n=64, prompt count) needs verification first. See
  [`../paper.md`](../paper.md) "What awaits eval" for the open question.

## Audit caveats

### 1. Trained arms beat base on hmmt_nov25 (not just AUC: every k≥32)
- `auc_at_k.md` lines 25–30: hmmt_nov25 AUC base=7.685 < grpo=7.850 <
  minority=7.988 < polyepo=7.998.
- The crossover is real in the pass@k ladder, not a trapezoid artifact:
  at k=32 base=0.132 vs grpo=0.139, minority=0.141, polyepo=0.142
  (`auc_at_k.md` lines 43, 48, 53, 58).
- At k=64 the gap widens: base=0.133 (essentially flat from k=32) while
  trained arms all reach 0.167.
- **Interpretation hypothesis** (NOT verified — flag for investigation):
  base's hmmt_nov25 pass@k saturates at ~0.13 because its early rollouts
  cover a small set of solvable prompts and the rest are quality-bound;
  trained arms have a flatter curve but reach more total prompts by k=64.
  Compare to `potential_at_k.md` lines 68–71 — base has pot@8 = 0.000 on
  hmmt_nov25 (failures are NOT budget-bound), while trained arms still
  have 0.038–0.138 potential at k=16, consistent with this hypothesis.

### 2. polyepo / aime26 = 0 across the entire pass@k ladder — VERIFIED REAL (2026-06-04)
- `auc_at_k.md` line 55: `pass@{1,2,4,8,16,32,64}` all 0.000 for
  polyepo / aime26.
- **30 prompts × 64 rollouts = 1920 rollouts with zero correct answers**,
  while polyepo solves prompts on aime25 (pass@64=0.133), beyondaime
  (0.130), hmmt_feb25 (0.167), hmmt_nov25 (0.167).
- **Spot-check 2026-06-04 confirms this is a genuine polyepo failure
  mode, not a grader bug.** Sampled rollouts show:
  - Prompt 7 rollout 0: infinite loop "### Step 47 ... Step 48 ... Step 49"
    never reaches `\boxed{}` → pred = `""`.
  - Prompt 15 rollout 0: sentence loop "Therefore, the common difference d
    must be ... Therefore, the common difference d must be ...".
  - Prompt 0 rollout 0: syntax noise `)))`, `]]`, `}}}` mid-code.
  - Some preds ARE non-empty (e.g., `"39"`, `"278"`, `"220"`) but none
    match ground truth (277, 244, 178). Grader correctly rejects wrong
    answers — not a parser bug.
- Polyepo's training has induced a **repetition-collapse failure mode**
  triggered specifically on aime26 problems. Matches the audit's
  polyepo/hmmt_nov25 wait=1.296 outlier (8.6x base) — same repetition
  signature.
- `diff_at_k_split.md` cross-confirms: polyepo/aime26 solved partition
  has n_partition=0 (line 47); unsolved has n_partition=30 (line 94) →
  polyepo never lands a correct answer on any aime26 prompt.
- `potential_at_k.md` line 44: polyepo/aime26 pot@k = 0.000 for all
  k, consistent (no prompts ever solved → no recoverable failures).

### 3. Self-BLEU vs distinct-n direction disagreement
- `self_bleu.md` aime26 block: grpo has the **lowest** Self-BLEU
  (0.3317, "most diverse") AND the **lowest** distinct-1 (0.0435,
  "least diverse") simultaneously. Same direction-flip on hmmt_feb25
  (grpo SB=0.3520 tied lowest distinct-1=0.0520) and beyondaime
  (polyepo SB=0.4291 highest = "least diverse" but distinct-1=0.0489
  middle of trained arms).
- Likely cause: Self-BLEU is dominated by repeated *high-order* n-grams
  (BLEU-4 uses geometric mean of 1..4-gram precisions), while distinct-1
  is unigram only. An arm that repeats long boilerplate phrases looks
  *more* diverse to Self-BLEU (lower n-gram overlap?) — actually the
  reverse: repeating long phrases inflates 4-gram overlap → higher
  Self-BLEU. The direction flip might mean grpo has high unigram
  *concentration* (small vocabulary) but those unigrams arrange into
  *different* 4-grams across rollouts. Worth a one-paragraph note when
  citing self_bleu, not a deal-breaker.

### 4. Reflective-actions outlier: polyepo/hmmt_nov25 wait = 1.296
- `reflective_actions.md` line 71: polyepo on hmmt_nov25 has wait=1.296
  per rollout vs base=0.150 (8.6×), driving total/roll=2.083 (next
  closest cell ~1.5).
- The whole-word `\bwait\b` regex doesn't dedupe repeats within a
  rollout, so a single "wait wait wait …" loop in even a fraction of
  rollouts would inflate this average a lot.
- minority/aime26 alternatively=0.235 (`reflective_actions.md` line 43)
  also looks like a single-prompt repetition spike vs base=0.052 (4.5×).
- **Recommended sanity check**: pull the distribution (not just mean)
  of `wait` counts in polyepo/hmmt_nov25 rollouts; if the upper tail is
  heavy, flag in the writeup as "lexical repetition, not reflection".

### 5. Small-n on solved partition in diff_at_k_split.md
- For trained arms on aime25/26/hmmt_feb25/nov25 the solved partition
  often has n_partition = 1–5 prompts (lines 36, 37, 46, 63, 64, 72, 73
  in `diff_at_k_split.md`). Solved-side numbers there are extremely
  noisy and shouldn't be cited without an n_partition footnote.
- minority/aime25 solved row has n_partition=1 (`diff_at_k_split.md`
  line 37) — that is literally one prompt. diff@1=0.000 for that row
  means the first rollout's answer was empty/[INVALID]; not a real
  comparison.

### 6. Monotonicity check on pass@k (sanity, no failures found)
- All 23 pass@k rows in `auc_at_k.md` lines 39–62 are monotonically
  non-decreasing in k. No violations. (pass@k is non-decreasing by
  construction; finding a violation would have meant a corrupted JSON.)

### 7. diff@k unsolved > solved on some cells (mild)
- `diff_at_k_split.md` aime25 base: solved diff@64=36.100 (line 35) vs
  unsolved=38.150 (line 82). The unsolved side is slightly more diverse,
  consistent with "model wanders more when it can't solve". Not a bug,
  but the gap is mild — so calling it "diversity goes to wrong answers"
  is technically true but the magnitude is small for base.
- For minority on beyondaime: solved=18.889 (line 55) vs unsolved=18.374
  (line 102) — solved is marginally *higher*, opposite direction. Not a
  clear "diversity → wrong answers" pattern for minority on this dataset.
