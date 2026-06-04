# Self-BLEU and distinct-n-gram (rollout text)

## TL;DR

**What it measures.** Two complementary diversity metrics on the *raw
rollout text* (not the parsed answer), to catch cases where parsed-answer
diversity hides near-identical reasoning chains or vice-versa.
- **Self-BLEU**: for each rollout, BLEU-4 against the other rollouts as
  references; lower = more diverse (less mutual overlap).
- **distinct-n**: unique n-grams / total n-grams across the rollout set;
  higher = more diverse. Reported for n ∈ {1, 2, 3}.

**How to read.** Compare arms within a dataset. The two metrics *should*
agree on direction (low Self-BLEU ↔ high distinct-n). When they disagree,
the diversity signal is fragile.

**Headline.** **Base is consistently the most diverse arm at the n-gram
level** — distinct-1 ≈ 0.10–0.13 across datasets vs 0.04–0.06 for all three
trained arms (a ~2-3× gap). Self-BLEU is much closer (all arms within
~0.10 in a 0.33–0.45 range) and the per-dataset Self-BLEU ranking is
*not* always base-first: on aime26 grpo has the lowest SB (0.3317), and
on hmmt_feb25 grpo (0.3520) ties base (0.4183) very loosely. **Direction
disagreement on aime26**: grpo has the *lowest* Self-BLEU (most diverse)
but ALSO the *lowest* distinct-1 (least diverse) simultaneously. Mild
similar disagreement on hmmt_feb25 and beyondaime. Among trained arms,
minority and polyepo are not consistently more diverse than grpo on
either metric.

Self-BLEU: **lower = more diverse**. distinct_n: **higher = more diverse**.
Sampled up to 8 rollouts/problem (Self-BLEU is O(n^2)).

## aime25

| arm | n_problems | Self-BLEU | distinct-1 | distinct-2 | distinct-3 |
|---|---|---|---|---|---|
| base | 16 | 0.3259 | 0.1159 | 0.3577 | 0.5213 |
| grpo | 16 | 0.3648 | 0.0520 | 0.1566 | 0.2328 |
| minority | 16 | 0.3663 | 0.0515 | 0.1607 | 0.2409 |
| polyepo | 16 | 0.3925 | 0.0443 | 0.1374 | 0.2083 |

## aime26

| arm | n_problems | Self-BLEU | distinct-1 | distinct-2 | distinct-3 |
|---|---|---|---|---|---|
| base | 16 | 0.3751 | 0.1133 | 0.3477 | 0.5093 |
| grpo | 16 | 0.3317 | 0.0435 | 0.1369 | 0.2055 |
| minority | 16 | 0.3695 | 0.0565 | 0.1669 | 0.2482 |
| polyepo | 16 | 0.3843 | 0.0455 | 0.1419 | 0.2165 |

## beyondaime

| arm | n_problems | Self-BLEU | distinct-1 | distinct-2 | distinct-3 |
|---|---|---|---|---|---|
| base | 16 | 0.3656 | 0.1348 | 0.3978 | 0.5540 |
| grpo | 16 | 0.3690 | 0.0437 | 0.1270 | 0.1846 |
| minority | 16 | 0.3705 | 0.0596 | 0.1578 | 0.2192 |
| polyepo | 16 | 0.4291 | 0.0489 | 0.1422 | 0.2083 |

## hmmt_feb25

| arm | n_problems | Self-BLEU | distinct-1 | distinct-2 | distinct-3 |
|---|---|---|---|---|---|
| base | 16 | 0.4183 | 0.0983 | 0.3216 | 0.4734 |
| grpo | 16 | 0.3520 | 0.0520 | 0.1477 | 0.2137 |
| minority | 16 | 0.3652 | 0.0500 | 0.1461 | 0.2146 |
| polyepo | 16 | 0.4453 | 0.0500 | 0.1562 | 0.2361 |

## hmmt_nov25

| arm | n_problems | Self-BLEU | distinct-1 | distinct-2 | distinct-3 |
|---|---|---|---|---|---|
| base | 16 | 0.3646 | 0.1038 | 0.3188 | 0.4705 |
| grpo | 16 | 0.3496 | 0.0412 | 0.1256 | 0.1875 |
| minority | 16 | 0.3793 | 0.0634 | 0.1778 | 0.2552 |
| polyepo | 16 | 0.4397 | 0.0586 | 0.1779 | 0.2648 |

## How this was computed

- **Script**: `main-verl/eval/analysis/posthoc/self_bleu.py`. Self-BLEU
  uses an in-process BLEU-4 with `+1e-9` smoothing on the n-gram match
  count and a brevity penalty against the closest-length reference. If
  `sacrebleu` is importable the script will delegate to it; this run
  used the in-process fallback.
- **Inputs**: same 20 probe JSONs; reads `per_prompt[i].rollouts[j]` text.
- **Subsampling** (Self-BLEU is O(n²) in rollouts × tokens):
  - `max_rollouts = 8` per problem (out of the available 64)
  - `max_problems = 16` per dataset (so beyondaime's 100-prompt panel is
    truncated to the first 16 prompts, matching the small-dataset panels)
  - `n_problems` column in each table is the count of prompts that had
    ≥2 non-empty rollouts (so it equals 16 across the board here, which
    confirms no problem was dropped for emptiness).
- **Eval probe sampling**: as in `auc_at_k.md` (n=64 generated; only the
  first 8 are scored here, T=1.0, top_p=1.0).
- **Tokenization**: `\w+|[^\w\s]` regex, case-folded — i.e. whitespace +
  punctuation tokenization, NOT BPE. Numeric strings are kept whole. This
  is consistent with classical Self-BLEU literature but means n-gram
  counts are not directly comparable to model-token counts.
- **Limitations / caveats**:
  - Subsampling to 8 rollouts × 16 problems means each Self-BLEU value is
    averaged over **16 numbers**; small differences between arms
    (e.g. 0.36 vs 0.37) are within noise.
  - For beyondaime, only the first 16 prompts contribute — order is
    dataset-defined and identical across arms, so cross-arm comparison is
    fair, but absolute beyondaime values are not representative of the
    full 100-prompt panel.
  - Self-BLEU and distinct-n can disagree (see TL;DR) — Self-BLEU
    weights n-gram overlap at higher orders (uses geometric mean of
    1..4), distinct-1 looks only at unigrams.

