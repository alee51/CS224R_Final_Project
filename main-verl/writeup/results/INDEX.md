# Eval results index — 4 arms × 5 OOD datasets

Eval setup: 4 arms × 5 smallood datasets × n=64 rollouts × temp=1.0, max_tokens=4096
Grader: `verl.utils.reward_score.math.compute_score` (Hendrycks `is_equiv`, mathd ∨ sympy fallback)
Generated: 2026-06-04

Arms:
- **base** — Qwen3-4B-Base (no RL)
- **grpo** — GRPO RL, step 400, FSDP→HF merged
- **minority** — minority-CoT RL, step 400, FSDP→HF merged
- **polyepo** — Poly-EPO CoT RL, step 400, FSDP→HF merged

Datasets ("smallood" panel):
- aime25 (30 prompts), aime26 (30), hmmt_feb25 (30), hmmt_nov25 (30), beyondaime (100)

Each table below: each result file has a `## TL;DR` block at the top and a
`## How this was computed` block at the bottom; the body in between is the
raw numeric tables.

| Artifact | TL;DR | Full content |
|---|---|---|
| [auc_at_k.md](auc_at_k.md) | Scalar AUC of pass@k over k∈{1..64}; base wins on 4 of 5 datasets but loses on hmmt_nov25 to all 3 trained arms. polyepo collapses to 0 on aime26. | AUC@k table + per-(arm,dataset) pass@k ladder |
| [diff_at_k_split.md](diff_at_k_split.md) | distinct_answers@k partitioned by solved vs unsolved; tests whether minority's extra diversity goes to wrong answers. Base most diverse everywhere; minority is NOT the most diverse trained arm on beyondaime/unsolved. | 2 partitions × 5 datasets × 4 arms × k∈{1,4,8,16,32,64} |
| [potential_at_k.md](potential_at_k.md) | Fraction of failed-at-k prompts that are eventually solvable in n=64; budget-bound vs quality-bound failure. Base has most recoverable failures except on hmmt_nov25 where the pattern flips. | 5 datasets × 4 arms × k∈{1,4,8,16,32} |
| [reflective_actions.md](reflective_actions.md) | Per-rollout count of 7 "reflective" lexical phrases. GRPO bumps `however` ~1.5–2× over base; polyepo/hmmt_nov25 has wait=1.296 (likely repetition artifact). | 5 datasets × 4 arms × 7 phrases + total/roll + total/1k_tok |
| [self_bleu.md](self_bleu.md) | Self-BLEU (lower = more diverse) and distinct-1/2/3-grams on rollout text. Base ~2-3× more diverse than trained arms at distinct-1 level; the two metrics disagree on direction on aime26 / hmmt_feb25 / beyondaime. | 5 datasets × 4 arms; subsampled to 8 rollouts × 16 prompts |
| [coverage.md](coverage.md) | coverage / distinct_answers / entropy / majority @k over rollouts. Base higher entropy + distinct_answers than any trained arm; base majority-vote rate non-zero at k≥4 while trained arms ≈ 0. | 5 datasets × 4 arms × k∈{1,2,4,8,16,32} |

## Headline numbers

From `auc_at_k.md`:

| arm \ dataset | aime25 | aime26 | beyondaime | hmmt_feb25 | hmmt_nov25 |
|---|---|---|---|---|---|
| base | 14.566 | 9.360 | 12.180 | 7.322 | 7.685 |
| grpo | 2.826 | 2.133 | 4.932 | 2.133 | 7.850 |
| minority | 1.562 | 3.695 | 3.681 | 3.818 | 7.988 |
| polyepo | 5.088 | 0.000 | 5.115 | 5.951 | 7.998 |

pass@64 spot-checks (from the ladder in `auc_at_k.md`):

| arm \ dataset | aime25 | aime26 | beyondaime | hmmt_feb25 | hmmt_nov25 |
|---|---|---|---|---|---|
| base | 0.333 | 0.200 | 0.290 | 0.200 | 0.133 |
| grpo | 0.067 | 0.067 | 0.120 | 0.067 | 0.167 |
| minority | 0.033 | 0.100 | 0.090 | 0.100 | 0.167 |
| polyepo | 0.133 | 0.000 | 0.130 | 0.167 | 0.167 |

## The hmmt_nov25 crossover explained (2026-06-04)

Why does base lose to all 3 trained arms on hmmt_nov25 at k≥32? **Depth vs breadth.**

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
consistent across all five result files in this directory.

## Important caveats and audit notes

### 1. Trained arms beat base on hmmt_nov25 (not just AUC: every k≥32)
- `auc_at_k.md` lines 23–26: hmmt_nov25 AUC base=7.685 < grpo=7.850 <
  minority=7.988 < polyepo=7.998.
- The crossover is real in the pass@k ladder, not a trapezoid artifact:
  at k=32 base=0.132 vs grpo=0.139, minority=0.141, polyepo=0.142
  (`auc_at_k.md` lines 34, 39, 44, 49).
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
- `auc_at_k.md` line 46: `pass@{1,2,4,8,16,32,64}` all 0.000 for
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
- All 20 pass@k rows in `auc_at_k.md` lines 30–49 are monotonically
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
