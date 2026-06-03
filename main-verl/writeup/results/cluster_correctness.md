# Cluster-correctness by rank (training-time)

Sampled training steps 100-400 every 10, n=8 rollouts/prompt. Per-prompt clusters are non-degenerate judge clusters (CoT-based; a cluster can contain rollouts with the same reasoning chain but different final answers). The "correct cluster" for a prompt is the cluster holding the most correct rollouts.

GRPO has no clusters during training (no judge); shown as n/a — Phase 5 will retroactively add cluster IDs via judge pass for cross-arm parity.

## Headline

**Cluster rarity is uncorrelated with correctness.** In the only regime where the question is well-posed — a *unique* (untied) smallest cluster on a prompt where geometry permits the correct cluster to be smallest (n_correct ≤ 2) — P(rarest cluster = correct cluster) sits at chance for ~3 clusters (0.31–0.40 vs chance ≈ 0.33).

The minority objective's reward signal at the rarest cluster is approximately random with respect to correctness.

## Why naive aggregate numbers are misleading

Cluster sizes are heavily skewed toward singletons (size=1 occurrences: 2469 minority, 3064 polyepo). Many prompts have multiple clusters tied at the smallest size, so the script's "rarest cluster" is actually a *set* of tied clusters that trivially contains the correct one.

Two stratifications matter:

**1. Unique vs tied rarest.** Restrict to prompts where exactly one cluster has the strictly smallest size:

|  | rarest = correct (all prompts) | rarest = correct (unique rarest only) |
|---|---|---|
| minority | 398/900 = 0.442 | 58/354 = **0.164** |
| polyepo | 473/1015 = 0.466 | 66/368 = **0.179** |

**2. n_correct constrains geometry.** At n_correct ≥ 5 (5+ of 8 rollouts correct), a unique-rarest cluster *cannot* be the correct cluster by arithmetic: the correct rollouts have too much mass to fit in a strictly smallest cluster while all others are strictly larger. Those rows are zeros-by-construction, not data.

## The well-posed slice

Unique rarest AND n_correct ≤ 2 (geometry permits the correct cluster to be the unique smallest):

| n_correct | minority | polyepo |
|---|---|---|
| 1 | 32/98 = **0.327** | 33/89 = **0.371** |
| 2 | 16/51 = 0.314 | 23/57 = 0.404 |

Chance for ~3 clusters ≈ 0.333. Rarity carries no signal about correctness here.

## CoT clustering carries real information (just not in the rare direction)

Of multi-rollout clusters (size ≥ 2):
- minority: 37.1% are **mixed** (both correct and wrong rollouts in the same CoT cluster)
- polyepo: 34.2% are mixed

So the judge's CoT clustering is genuinely informative — same reasoning chain frequently produces divergent final answers, and the clustering captures this. It just doesn't separate clusters along a rare↔correct axis.

## Raw output (unstratified)

```
Sampling steps 100..400 every 10

=== minority ===
  prompts with >=1 correct (across sampled steps): 1794
  eligible (>=2 non-deg clusters): 970

  P(cluster at rank R is the correct cluster):
    rank | count | total | P(correct)
      1  |   536 |  900 |  0.596
      2  |   186 |  900 |  0.207
      3  |    83 |  575 |  0.144
      4  |    40 |  412 |  0.097
      5  |    30 |  270 |  0.111
      6  |    16 |  176 |  0.091
      7  |     8 |  119 |  0.067
      8  |     1 |   69 |  0.014

  rarest cluster == correct: 0.442
  most-common cluster == correct: 0.772

  Cluster size histogram (size: #occurrences):
    size=1: 2469
    size=2: 342
    size=3: 198
    size=4: 147
    size=5: 120
    size=6: 89
    size=7: 56

=== polyepo ===
  prompts with >=1 correct (across sampled steps): 1877
  eligible (>=2 non-deg clusters): 1097

  P(cluster at rank R is the correct cluster):
    rank | count | total | P(correct)
      1  |   593 | 1015 |  0.584
      2  |   213 | 1015 |  0.210
      3  |    95 |  684 |  0.139
      4  |    43 |  506 |  0.085
      5  |    37 |  355 |  0.104
      6  |    15 |  257 |  0.058
      7  |    13 |  180 |  0.072
      8  |     6 |  103 |  0.058

  rarest cluster == correct: 0.466
  most-common cluster == correct: 0.785

  Cluster size histogram (size: #occurrences):
    size=1: 3064
    size=2: 364
    size=3: 235
    size=4: 138
    size=5: 129
    size=6: 112
    size=7: 73
```

The rank-1 and rank-N rows in the table above are subject to the same tie-inflation as the rarest=correct headline (clusters tied at a frequency get sequential ranks by cluster_id), so read them only as a coarse signal. The well-posed measurement is the unique-rarest stratification above.

## Reproduce

```
python main-verl/eval/analysis/training/cluster_correctness.py --step-min 100 --step-max 400 --sample-every 10
# stratified analysis (unique vs tied × n_correct):
python /tmp/rarest_when_v2.py   # see also: minority_diagnostic.md §2
```
