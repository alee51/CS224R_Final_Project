# Self-BLEU and distinct-n-gram (rollout text)

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

