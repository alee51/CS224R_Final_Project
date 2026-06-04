# coverage@k — 4 arms × 5 OOD datasets

## TL;DR

For each (arm, dataset, k), four quantities computed over the first k
rollouts per problem:
- **coverage** = fraction of problems with ≥1 correct rollout
- **distinct_answers** = unique parsed answers across the k rollouts
- **entropy** = Shannon entropy (bits) of the answer distribution
- **majority** = fraction correct using majority-vote over the k rollouts

Higher entropy / distinct_answers = more answer-space exploration. Higher
coverage = more recoverable problems within budget. Majority captures
self-consistency, the standard test-time aggregation.

Headline finding: base has higher entropy / distinct_answers than any
trained arm at every k on every dataset — trained arms collapse the
answer distribution. Majority for base is generally non-zero at k≥4;
trained arms majority is near 0 across the board (they don't repeat the
same correct answer reliably enough for majority-vote to help).

## base_step400_smallood_aime25

# coverage / distinct / entropy / majority @k — base_step400_smallood

- ckpt: ``
- n_rollouts: 64

## aime25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.01875, 'pass@2': 0.03606150793650794, 'pass@4': 0.06686403011759966, 'pass@8': 0.11609901235996176, 'pass@16': 0.1819086550805851, 'pass@32': 0.25369896446218815, 'pass@64': 0.3333333333333333}`
- mean_reward_at_1: 0.0333

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.03 | 0.70 | 0.00 | 0.033 |
| 2 | 0.03 | 1.30 | 0.44 | 0.033 |
| 4 | 0.10 | 2.57 | 1.23 | 0.033 |
| 8 | 0.10 | 5.40 | 2.36 | 0.033 |
| 16 | 0.13 | 10.53 | 3.32 | 0.033 |
| 32 | 0.23 | 20.07 | 4.20 | 0.000 |
| 64 | 0.33 | 37.47 | 5.02 | 0.133 |

## base_step400_smallood_aime26

# coverage / distinct / entropy / majority @k — base_step400_smallood

- ckpt: ``
- n_rollouts: 64

## aime26 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.019270833333333334, 'pass@2': 0.03574735449735449, 'pass@4': 0.061935819630161246, 'pass@8': 0.09569955452388931, 'pass@16': 0.12790513558001812, 'pass@32': 0.1584270766254452, 'pass@64': 0.2}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.73 | 0.00 | 0.000 |
| 2 | 0.00 | 1.40 | 0.56 | 0.000 |
| 4 | 0.00 | 2.87 | 1.45 | 0.000 |
| 8 | 0.03 | 5.17 | 2.28 | 0.000 |
| 16 | 0.13 | 9.80 | 3.18 | 0.067 |
| 32 | 0.17 | 18.17 | 3.97 | 0.100 |
| 64 | 0.20 | 34.23 | 4.78 | 0.100 |

## base_step400_smallood_beyondaime

# coverage / distinct / entropy / majority @k — base_step400_smallood

- ckpt: ``
- n_rollouts: 64

## beyondaime (n=100 prompts)

- saved pass@k: `{'pass@1': 0.01828125, 'pass@2': 0.03366567460317461, 'pass@4': 0.058486423786860066, 'pass@8': 0.09464110501575819, 'pass@16': 0.14385970026790276, 'pass@32': 0.2087916035811618, 'pass@64': 0.29}`
- mean_reward_at_1: 0.0300

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.03 | 0.77 | 0.00 | 0.030 |
| 2 | 0.03 | 1.39 | 0.53 | 0.030 |
| 4 | 0.06 | 2.72 | 1.33 | 0.040 |
| 8 | 0.11 | 4.85 | 2.11 | 0.060 |
| 16 | 0.14 | 8.80 | 2.84 | 0.070 |
| 32 | 0.19 | 16.14 | 3.53 | 0.070 |
| 64 | 0.29 | 28.34 | 4.13 | 0.030 |

## base_step400_smallood_hmmt_feb25

# coverage / distinct / entropy / majority @k — base_step400_smallood

- ckpt: ``
- n_rollouts: 64

## hmmt_feb25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.005208333333333333, 'pass@2': 0.010300925925925927, 'pass@4': 0.020151637245767337, 'pass@8': 0.038601465029280996, 'pass@16': 0.07114107761913155, 'pass@32': 0.12324572816376095, 'pass@64': 0.2}`
- mean_reward_at_1: 0.0333

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.03 | 0.77 | 0.00 | 0.033 |
| 2 | 0.03 | 1.50 | 0.50 | 0.033 |
| 4 | 0.03 | 2.93 | 1.47 | 0.033 |
| 8 | 0.03 | 5.53 | 2.37 | 0.000 |
| 16 | 0.07 | 9.87 | 3.10 | 0.000 |
| 32 | 0.17 | 18.40 | 3.90 | 0.000 |
| 64 | 0.20 | 31.40 | 4.45 | 0.033 |

## base_step400_smallood_hmmt_nov25

# coverage / distinct / entropy / majority @k — base_step400_smallood

- ckpt: ``
- n_rollouts: 64

## hmmt_nov25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.028125, 'pass@2': 0.04837962962962963, 'pass@4': 0.07465254169709484, 'pass@8': 0.10097925252213487, 'pass@16': 0.12174119880999917, 'pass@32': 0.13227229801104615, 'pass@64': 0.13333333333333333}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.87 | 0.00 | 0.000 |
| 2 | 0.07 | 1.57 | 0.57 | 0.000 |
| 4 | 0.10 | 2.87 | 1.41 | 0.033 |
| 8 | 0.13 | 4.90 | 2.15 | 0.067 |
| 16 | 0.13 | 8.73 | 2.87 | 0.100 |
| 32 | 0.17 | 16.20 | 3.65 | 0.100 |
| 64 | 0.20 | 29.70 | 4.34 | 0.067 |

## grpo_step400_smallood_aime25

# coverage / distinct / entropy / majority @k — grpo_step400_smallood

- ckpt: `/vol/merged_hf/grpo_step400`
- n_rollouts: 64

## aime25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.003125, 'pass@2': 0.006084656084656085, 'pass@4': 0.01153967624419766, 'pass@8': 0.020800282037722547, 'pass@16': 0.03418070559794515, 'pass@32': 0.04911961141469338, 'pass@64': 0.06666666666666667}`
- mean_reward_at_1: 0.0333

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.03 | 0.37 | 0.00 | 0.033 |
| 2 | 0.03 | 1.00 | 0.36 | 0.033 |
| 4 | 0.03 | 1.93 | 0.98 | 0.033 |
| 8 | 0.03 | 3.53 | 1.70 | 0.000 |
| 16 | 0.03 | 6.87 | 2.63 | 0.033 |
| 32 | 0.03 | 13.07 | 3.54 | 0.000 |
| 64 | 0.07 | 24.43 | 4.36 | 0.000 |

## grpo_step400_smallood_aime26

# coverage / distinct / entropy / majority @k — grpo_step400_smallood

- ckpt: `/vol/merged_hf/grpo_step400`
- n_rollouts: 64

## aime26 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0010416666666666667, 'pass@2': 0.0020833333333333333, 'pass@4': 0.004166666666666667, 'pass@8': 0.008333333333333333, 'pass@16': 0.016666666666666666, 'pass@32': 0.03333333333333333, 'pass@64': 0.06666666666666667}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.53 | 0.00 | 0.000 |
| 2 | 0.03 | 1.03 | 0.29 | 0.033 |
| 4 | 0.03 | 1.83 | 0.78 | 0.033 |
| 8 | 0.03 | 3.10 | 1.46 | 0.033 |
| 16 | 0.03 | 5.93 | 2.36 | 0.000 |
| 32 | 0.03 | 11.47 | 3.28 | 0.000 |
| 64 | 0.07 | 20.23 | 3.97 | 0.000 |

## grpo_step400_smallood_beyondaime

# coverage / distinct / entropy / majority @k — grpo_step400_smallood

- ckpt: `/vol/merged_hf/grpo_step400`
- n_rollouts: 64

## beyondaime (n=100 prompts)

- saved pass@k: `{'pass@1': 0.00703125, 'pass@2': 0.012956349206349208, 'pass@4': 0.022538622799728035, 'pass@8': 0.0367006101476505, 'pass@16': 0.05685600586325685, 'pass@32': 0.08412841739065995, 'pass@64': 0.12}`
- mean_reward_at_1: 0.0100

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.01 | 0.72 | 0.00 | 0.010 |
| 2 | 0.02 | 1.33 | 0.46 | 0.010 |
| 4 | 0.02 | 2.15 | 1.05 | 0.010 |
| 8 | 0.03 | 3.89 | 1.76 | 0.010 |
| 16 | 0.03 | 6.79 | 2.48 | 0.030 |
| 32 | 0.07 | 11.83 | 3.10 | 0.020 |
| 64 | 0.12 | 20.56 | 3.64 | 0.020 |

## grpo_step400_smallood_hmmt_feb25

# coverage / distinct / entropy / majority @k — grpo_step400_smallood

- ckpt: `/vol/merged_hf/grpo_step400`
- n_rollouts: 64

## hmmt_feb25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0010416666666666667, 'pass@2': 0.0020833333333333333, 'pass@4': 0.004166666666666667, 'pass@8': 0.008333333333333333, 'pass@16': 0.016666666666666666, 'pass@32': 0.03333333333333333, 'pass@64': 0.06666666666666667}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.73 | 0.00 | 0.000 |
| 2 | 0.00 | 1.30 | 0.44 | 0.000 |
| 4 | 0.00 | 2.27 | 1.10 | 0.000 |
| 8 | 0.00 | 4.30 | 1.99 | 0.000 |
| 16 | 0.00 | 7.43 | 2.70 | 0.000 |
| 32 | 0.03 | 12.47 | 3.30 | 0.000 |
| 64 | 0.07 | 21.17 | 3.86 | 0.000 |

## grpo_step400_smallood_hmmt_nov25

# coverage / distinct / entropy / majority @k — grpo_step400_smallood

- ckpt: `/vol/merged_hf/grpo_step400`
- n_rollouts: 64

## hmmt_nov25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0125, 'pass@2': 0.023627645502645504, 'pass@4': 0.04248843729277362, 'pass@8': 0.07040106899744435, 'pass@16': 0.10461467826437501, 'pass@32': 0.13902971454720217, 'pass@64': 0.16666666666666666}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.53 | 0.00 | 0.000 |
| 2 | 0.00 | 1.00 | 0.43 | 0.000 |
| 4 | 0.00 | 1.87 | 0.88 | 0.000 |
| 8 | 0.03 | 3.43 | 1.58 | 0.000 |
| 16 | 0.03 | 6.57 | 2.46 | 0.000 |
| 32 | 0.10 | 11.27 | 3.06 | 0.000 |
| 64 | 0.20 | 19.90 | 3.65 | 0.000 |

## minority_step400_smallood_aime25

# coverage / distinct / entropy / majority @k — minority_step400_smallood

- ckpt: `/vol/merged_hf/minority_step400`
- n_rollouts: 64

## aime25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0015625, 'pass@2': 0.0030753968253968257, 'pass@4': 0.0059555811571940604, 'pass@8': 0.011155913978494626, 'pass@16': 0.019495647721454172, 'pass@32': 0.029365079365079365, 'pass@64': 0.03333333333333333}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.37 | 0.00 | 0.000 |
| 2 | 0.00 | 0.80 | 0.26 | 0.000 |
| 4 | 0.00 | 1.73 | 0.70 | 0.000 |
| 8 | 0.00 | 3.37 | 1.56 | 0.000 |
| 16 | 0.00 | 6.43 | 2.54 | 0.000 |
| 32 | 0.03 | 11.70 | 3.35 | 0.000 |
| 64 | 0.03 | 21.53 | 4.18 | 0.000 |

## minority_step400_smallood_aime26

# coverage / distinct / entropy / majority @k — minority_step400_smallood

- ckpt: `/vol/merged_hf/minority_step400`
- n_rollouts: 64

## aime26 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0026041666666666665, 'pass@2': 0.005158730158730159, 'pass@4': 0.010122247823860727, 'pass@8': 0.019489247311827957, 'pass@16': 0.036162314388120835, 'pass@32': 0.0626984126984127, 'pass@64': 0.1}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.40 | 0.00 | 0.000 |
| 2 | 0.03 | 0.80 | 0.20 | 0.033 |
| 4 | 0.03 | 1.50 | 0.78 | 0.033 |
| 8 | 0.03 | 2.93 | 1.42 | 0.033 |
| 16 | 0.07 | 6.13 | 2.42 | 0.033 |
| 32 | 0.07 | 10.77 | 3.19 | 0.033 |
| 64 | 0.10 | 18.83 | 3.90 | 0.000 |

## minority_step400_smallood_beyondaime

# coverage / distinct / entropy / majority @k — minority_step400_smallood

- ckpt: `/vol/merged_hf/minority_step400`
- n_rollouts: 64

## beyondaime (n=100 prompts)

- saved pass@k: `{'pass@1': 0.00625, 'pass@2': 0.011418650793650795, 'pass@4': 0.019391572863941983, 'pass@8': 0.029808800338523636, 'pass@16': 0.04257582524077221, 'pass@32': 0.06134893782121159, 'pass@64': 0.09}`
- mean_reward_at_1: 0.0100

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.01 | 0.56 | 0.00 | 0.010 |
| 2 | 0.01 | 1.05 | 0.38 | 0.010 |
| 4 | 0.02 | 2.01 | 0.98 | 0.010 |
| 8 | 0.02 | 3.46 | 1.64 | 0.010 |
| 16 | 0.04 | 6.19 | 2.38 | 0.010 |
| 32 | 0.06 | 10.68 | 3.01 | 0.020 |
| 64 | 0.09 | 18.42 | 3.54 | 0.020 |

## minority_step400_smallood_hmmt_feb25

# coverage / distinct / entropy / majority @k — minority_step400_smallood

- ckpt: `/vol/merged_hf/minority_step400`
- n_rollouts: 64

## hmmt_feb25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0026041666666666665, 'pass@2': 0.005175264550264551, 'pass@4': 0.010218253968253968, 'pass@8': 0.019907407407407412, 'pass@16': 0.037698412698412696, 'pass@32': 0.06693121693121692, 'pass@64': 0.1}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.40 | 0.00 | 0.000 |
| 2 | 0.00 | 1.03 | 0.41 | 0.000 |
| 4 | 0.00 | 2.00 | 0.91 | 0.000 |
| 8 | 0.00 | 3.37 | 1.62 | 0.000 |
| 16 | 0.00 | 6.33 | 2.46 | 0.000 |
| 32 | 0.03 | 10.20 | 3.01 | 0.000 |
| 64 | 0.10 | 18.07 | 3.60 | 0.000 |

## minority_step400_smallood_hmmt_nov25

# coverage / distinct / entropy / majority @k — minority_step400_smallood

- ckpt: `/vol/merged_hf/minority_step400`
- n_rollouts: 64

## hmmt_nov25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0125, 'pass@2': 0.023892195767195767, 'pass@4': 0.04374181586965828, 'pass@8': 0.0739968784796354, 'pass@16': 0.11007427228920601, 'pass@32': 0.14095921397145422, 'pass@64': 0.16666666666666666}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.63 | 0.00 | 0.000 |
| 2 | 0.03 | 1.07 | 0.19 | 0.000 |
| 4 | 0.10 | 2.00 | 0.87 | 0.000 |
| 8 | 0.10 | 3.20 | 1.53 | 0.000 |
| 16 | 0.13 | 5.87 | 2.35 | 0.033 |
| 32 | 0.17 | 9.77 | 2.86 | 0.033 |
| 64 | 0.17 | 17.23 | 3.49 | 0.000 |

## polyepo_step400_smallood_aime25

# coverage / distinct / entropy / majority @k — polyepo_step400_smallood

- ckpt: `/vol/merged_hf/polyepo_step400`
- n_rollouts: 64

## aime25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.005729166666666666, 'pass@2': 0.010995370370370372, 'pass@4': 0.020314427992243964, 'pass@8': 0.035135630341410234, 'pass@16': 0.05549152812403461, 'pass@32': 0.0832541202965736, 'pass@64': 0.13333333333333333}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.50 | 0.00 | 0.000 |
| 2 | 0.00 | 0.97 | 0.38 | 0.000 |
| 4 | 0.03 | 2.03 | 0.95 | 0.000 |
| 8 | 0.03 | 3.70 | 1.74 | 0.000 |
| 16 | 0.10 | 7.57 | 2.78 | 0.033 |
| 32 | 0.13 | 13.83 | 3.62 | 0.033 |
| 64 | 0.13 | 23.97 | 4.32 | 0.033 |

## polyepo_step400_smallood_aime26

# coverage / distinct / entropy / majority @k — polyepo_step400_smallood

- ckpt: `/vol/merged_hf/polyepo_step400`
- n_rollouts: 64

## aime26 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0, 'pass@2': 0.0, 'pass@4': 0.0, 'pass@8': 0.0, 'pass@16': 0.0, 'pass@32': 0.0, 'pass@64': 0.0}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.50 | 0.00 | 0.000 |
| 2 | 0.00 | 0.90 | 0.42 | 0.000 |
| 4 | 0.00 | 1.57 | 0.69 | 0.000 |
| 8 | 0.00 | 3.27 | 1.62 | 0.000 |
| 16 | 0.00 | 6.37 | 2.48 | 0.000 |
| 32 | 0.00 | 11.90 | 3.27 | 0.000 |
| 64 | 0.00 | 19.57 | 3.91 | 0.000 |

## polyepo_step400_smallood_beyondaime

# coverage / distinct / entropy / majority @k — polyepo_step400_smallood

- ckpt: `/vol/merged_hf/polyepo_step400`
- n_rollouts: 64

## beyondaime (n=100 prompts)

- saved pass@k: `{'pass@1': 0.00609375, 'pass@2': 0.01168154761904762, 'pass@4': 0.021532793180730776, 'pass@8': 0.03707062490395411, 'pass@16': 0.05794017029466887, 'pass@32': 0.0846781710690545, 'pass@64': 0.13}`
- mean_reward_at_1: 0.0100

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.01 | 0.53 | 0.00 | 0.010 |
| 2 | 0.02 | 1.07 | 0.29 | 0.020 |
| 4 | 0.03 | 2.06 | 1.00 | 0.020 |
| 8 | 0.04 | 3.82 | 1.80 | 0.020 |
| 16 | 0.06 | 6.61 | 2.46 | 0.020 |
| 32 | 0.09 | 11.24 | 3.05 | 0.020 |
| 64 | 0.14 | 19.28 | 3.54 | 0.020 |

## polyepo_step400_smallood_hmmt_feb25

# coverage / distinct / entropy / majority @k — polyepo_step400_smallood

- ckpt: `/vol/merged_hf/polyepo_step400`
- n_rollouts: 64

## hmmt_feb25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0036458333333333334, 'pass@2': 0.0072585978835978835, 'pass@4': 0.014384920634920636, 'pass@8': 0.028240740740740743, 'pass@16': 0.054365079365079366, 'pass@32': 0.10026455026455026, 'pass@64': 0.16666666666666666}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.63 | 0.00 | 0.000 |
| 2 | 0.00 | 1.13 | 0.42 | 0.000 |
| 4 | 0.00 | 2.13 | 0.94 | 0.000 |
| 8 | 0.03 | 3.83 | 1.78 | 0.000 |
| 16 | 0.07 | 6.63 | 2.49 | 0.000 |
| 32 | 0.10 | 12.63 | 3.27 | 0.000 |
| 64 | 0.20 | 20.40 | 3.70 | 0.000 |

## polyepo_step400_smallood_hmmt_nov25

# coverage / distinct / entropy / majority @k — polyepo_step400_smallood

- ckpt: `/vol/merged_hf/polyepo_step400`
- n_rollouts: 64

## hmmt_nov25 (n=30 prompts)

- saved pass@k: `{'pass@1': 0.0140625, 'pass@2': 0.02627314814814815, 'pass@4': 0.04632086617477945, 'pass@8': 0.07466659307158538, 'pass@16': 0.10819947066968588, 'pass@32': 0.1416602739989061, 'pass@64': 0.16666666666666666}`
- mean_reward_at_1: 0.0000

| k | coverage | distinct_answers | entropy(bits) | majority |
|---|---|---|---|---|
| 1 | 0.00 | 0.53 | 0.00 | 0.000 |
| 2 | 0.00 | 1.07 | 0.45 | 0.000 |
| 4 | 0.00 | 2.33 | 1.13 | 0.000 |
| 8 | 0.10 | 4.00 | 1.85 | 0.000 |
| 16 | 0.13 | 6.43 | 2.47 | 0.000 |
| 32 | 0.17 | 10.53 | 2.99 | 0.000 |
| 64 | 0.17 | 17.23 | 3.47 | 0.033 |


## How this was computed

- Script: `main-verl/eval/analysis/posthoc/coverage.py`
- Invocation: per-file loop (one invocation per eval JSON, output
  concatenated by `tier1` wrapper) over the 20 base/grpo/minority/polyepo
  × 5-dataset JSONs at `main-verl/eval/probes/eval_4b/*_step400_smallood_*.json`.
- Inputs:
  - `per_prompt[i].rewards` (binary 0/1 per rollout from
    `verl.utils.reward_score.math.compute_score`)
  - `per_prompt[i].preds` (parsed `\boxed{...}` strings)
- Sampling params reused from the eval probe: n=64 rollouts, temp=1.0,
  top_p=1.0, max_tokens=4096.
- k ladder: {1, 2, 4, 8, 16, 32}; pass@k saved in the eval JSON also
  reported per-block for sanity-cross-check.
