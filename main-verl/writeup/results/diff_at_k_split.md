# distinct_answers@k split by solved vs unsolved

## TL;DR

**What it measures.** Counts the number of distinct parsed answers a policy
emits in its first k rollouts, averaged over prompts, then **partitioned**
into two groups: `solved` (n_correct > 0 across all 64 rollouts) and
`unsolved` (n_correct == 0). If a method's diversity is genuine reasoning
breadth, it should hold up on solved prompts; if the "extra" distinct answers
only show up on unsolved prompts, diversity is "going to wrong answers".

**How to read.** Compare arms within a (partition, dataset, k) cell. Look
especially at the `unsolved` partition at large k: that's where a diversity-
seeking objective should pay off. `n_partition` is the prompt count behind the
row (a row with `n_partition=1` is statistically meaningless).

**Headline.** Base has the highest diff@k on essentially every (partition,
dataset) cell — its rollouts spray across many distinct answers. Among
trained arms, the picture is mixed: minority is **not** the consistently most
diverse arm. On `beyondaime/unsolved` minority is actually the LEAST diverse
trained arm (diff@64 = 18.374 vs grpo 20.500, polyepo 19.264). Solved-side
ns are tiny for trained arms (often 1-5 prompts), so solved-vs-unsolved
comparisons should be read with that small-n caveat in mind.

Load-bearing for the minority-CoT diversity story:
if minority's distinct-answers advantage is concentrated in the
unsolved partition, diversity is going to wrong answers.

## Partition: solved (n_correct > 0)

### aime25

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 10 | 0.800 | 2.600 | 5.100 | 10.100 | 19.800 | 36.100 |
| grpo | 2 | 1.000 | 2.000 | 3.500 | 8.000 | 13.000 | 22.500 |
| minority | 1 | 0.000 | 1.000 | 3.000 | 6.000 | 11.000 | 24.000 |
| polyepo | 4 | 0.750 | 3.000 | 5.500 | 10.000 | 18.750 | 28.750 |

### aime26

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 6 | 0.500 | 3.000 | 5.333 | 9.333 | 17.167 | 33.000 |
| grpo | 2 | 0.500 | 2.000 | 4.000 | 6.500 | 12.000 | 27.500 |
| minority | 3 | 0.667 | 1.333 | 3.333 | 6.000 | 12.000 | 20.333 |
| polyepo | 0 | — | — | — | — | — | — |

### beyondaime

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 29 | 0.793 | 2.793 | 4.793 | 8.586 | 15.414 | 27.103 |
| grpo | 12 | 0.833 | 2.583 | 3.917 | 6.333 | 11.917 | 21.000 |
| minority | 9 | 0.667 | 2.444 | 3.556 | 6.111 | 11.000 | 18.889 |
| polyepo | 13 | 0.615 | 2.154 | 4.000 | 6.769 | 11.538 | 19.385 |

### hmmt_feb25

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 6 | 0.833 | 3.000 | 6.000 | 10.833 | 21.000 | 37.833 |
| grpo | 2 | 0.000 | 1.500 | 3.000 | 5.500 | 11.000 | 21.500 |
| minority | 3 | 0.667 | 2.000 | 3.667 | 7.333 | 13.667 | 25.333 |
| polyepo | 5 | 1.000 | 2.600 | 4.000 | 6.200 | 12.200 | 18.600 |

### hmmt_nov25

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 4 | 0.750 | 3.000 | 4.750 | 6.750 | 11.250 | 20.750 |
| grpo | 5 | 0.400 | 2.000 | 3.000 | 5.400 | 9.600 | 17.800 |
| minority | 5 | 0.600 | 2.600 | 4.000 | 6.000 | 9.000 | 13.600 |
| polyepo | 5 | 0.400 | 2.200 | 4.200 | 6.400 | 10.000 | 14.600 |

## Partition: unsolved (n_correct == 0)

### aime25

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 20 | 0.650 | 2.550 | 5.550 | 10.750 | 20.200 | 38.150 |
| grpo | 28 | 0.321 | 1.929 | 3.536 | 6.786 | 13.071 | 24.571 |
| minority | 29 | 0.379 | 1.759 | 3.379 | 6.448 | 11.724 | 21.448 |
| polyepo | 26 | 0.462 | 1.885 | 3.423 | 7.192 | 13.077 | 23.231 |

### aime26

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 24 | 0.792 | 2.833 | 5.125 | 9.917 | 18.417 | 34.542 |
| grpo | 28 | 0.536 | 1.821 | 3.036 | 5.893 | 11.429 | 19.714 |
| minority | 27 | 0.370 | 1.519 | 2.889 | 6.148 | 10.630 | 18.667 |
| polyepo | 30 | 0.500 | 1.567 | 3.267 | 6.367 | 11.900 | 19.567 |

### beyondaime

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 71 | 0.761 | 2.690 | 4.873 | 8.887 | 16.437 | 28.845 |
| grpo | 88 | 0.705 | 2.091 | 3.886 | 6.852 | 11.818 | 20.500 |
| minority | 91 | 0.549 | 1.967 | 3.451 | 6.198 | 10.648 | 18.374 |
| polyepo | 87 | 0.517 | 2.046 | 3.793 | 6.586 | 11.195 | 19.264 |

### hmmt_feb25

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 24 | 0.750 | 2.917 | 5.417 | 9.625 | 17.750 | 29.792 |
| grpo | 28 | 0.786 | 2.321 | 4.393 | 7.571 | 12.571 | 21.143 |
| minority | 27 | 0.370 | 2.000 | 3.333 | 6.222 | 9.815 | 17.259 |
| polyepo | 25 | 0.560 | 2.040 | 3.800 | 6.720 | 12.720 | 20.760 |

### hmmt_nov25

| arm | n_partition | diff@1 | diff@4 | diff@8 | diff@16 | diff@32 | diff@64 |
|---|---|---|---|---|---|---|---|
| base | 26 | 0.885 | 2.846 | 4.923 | 9.038 | 16.962 | 31.077 |
| grpo | 25 | 0.560 | 1.840 | 3.520 | 6.800 | 11.600 | 20.320 |
| minority | 25 | 0.640 | 1.880 | 3.040 | 5.840 | 9.920 | 17.960 |
| polyepo | 25 | 0.560 | 2.360 | 3.960 | 6.440 | 10.640 | 17.760 |

## How this was computed

- **Script**: `main-verl/eval/analysis/posthoc/diff_at_k_split.py`. Per prompt,
  takes the first k parsed answers (excluding empty or `[INVALID]`), counts
  unique values; partitions prompts by `n_correct > 0` vs `== 0`; averages
  the count within partition.
- **Inputs**: same 20 probe JSONs (`*_step400_smallood_*.json`), all 64
  rollouts per prompt. No subsampling.
- **Eval probe sampling**: as in `auc_at_k.md` (n=64, T=1.0, top_p=1.0,
  max_tokens=4096, vLLM B200:1 enforce_eager).
- **Limitations / caveats**:
  - Distinct-answer count depends on the answer-extractor (verl math
    grader's parsing). Two semantically-identical answers in different
    surface forms (e.g. `42` vs `42.0`) may be counted as distinct.
  - `n_partition` shrinks the cross-arm comparison: the solved partition
    has only 1-5 prompts for trained arms on aime25/26/hmmt_feb25/nov25,
    so headline solved-side differences within those datasets are noisy.
  - polyepo/aime26 has `n_partition=0` in the solved table (consistent
    with `pass@k = 0` in `auc_at_k.md`).
