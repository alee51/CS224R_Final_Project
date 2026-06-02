# Cluster-correctness by rank (training-time)

Sampled training steps 100-400 every 10, n=8 rollouts/prompt.
P(rank-r cluster is correct cluster), where rank 1 = most common.

GRPO has no clusters during training (no judge); shown as n/a — Phase 5 will retroactively add cluster IDs via judge pass for cross-arm parity.

Headline:
- **minority**: rarest-correct **0.442**, most-common-correct **0.772**
- **polyepo**: rarest-correct **0.466**, most-common-correct **0.785**
- chance with ~3 distinct clusters per prompt: ~0.33

Both set arms show the SAME inversion — most-common cluster is far more likely to be the correct cluster than the rarest. Minority objective (reward rarest) trains against this signal ~half the time it engages.

## Raw output

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

=== grpo ===
  prompts with >=1 correct (across sampled steps): 1870
  eligible (>=2 non-deg clusters): 0

  P(cluster at rank R is the correct cluster):
    rank | count | total | P(correct)

  rarest cluster == correct: n/a (no clusters; e.g. GRPO has no judge during training)
  most-common cluster == correct: n/a

  Cluster size histogram (size: #occurrences):

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
