# Training-time diff@k split (hypothesis gate)

Steps sampled: [200, 400], every 10. n=8 rollouts/prompt during training.

Hypothesis: minority adds diversity but on **unsolved** prompts (wrong-answer diversity), not on solved prompts (correct-answer diversity).

If `Δ(minority − grpo)` is larger on the unsolved row than the solved row, the hypothesis holds.

## distinct_answers@k by partition

| arm | partition | n_prompts/step | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|
| grpo | solved | 61.3 | 0.962 | 1.679 | 2.748 | 4.384 |
| grpo | unsolved | 66.7 | 0.961 | 1.730 | 3.002 | 5.063 |
| minority | solved | 59.1 | 0.959 | 1.638 | 2.712 | 4.407 |
| minority | unsolved | 68.9 | 0.950 | 1.709 | 2.998 | 5.181 |
| polyepo | solved | 60.8 | 0.961 | 1.654 | 2.732 | 4.382 |
| polyepo | unsolved | 67.2 | 0.952 | 1.701 | 3.000 | 5.189 |

## Δ(minority − grpo) by partition

| partition | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| solved | -0.003 | -0.040 | -0.036 | +0.022 |
| unsolved | -0.011 | -0.020 | -0.004 | +0.118 |

**Verdict gate:** if unsolved Δ > solved Δ across the k ladder, hypothesis SUPPORTED. If similar, minority's diversity is real on solved prompts too (good for the arm).

## Δ(polyepo − grpo) by partition (context)

| partition | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| solved | -0.001 | -0.025 | -0.016 | -0.002 |
| unsolved | -0.009 | -0.029 | -0.002 | +0.127 |
