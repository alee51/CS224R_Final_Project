# 4B verl held-out eval — current cross-arm pass@k

Step 400 checkpoints, `n_rollouts=16`, `temp=1.0`, `top_p=1.0`, scored with verl
`math.compute_score` (Hendrycks `is_equiv`, identical to training grader — see
`main/docs/STANDARDS.md` §"Reward path"). The eval probe is
`main-verl/eval/run_eval.py`.

Coverage state as of last update: GRPO and Poly-EPO have AIME-25 + MATH-500;
Poly-EPO has the full hard-OOD set. Minority numbers and the missing GRPO/Minority
cells are pending — see `writeup/eval_panel_candidates.md` for the locked panel.

## aime25
n=30

| arm | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| grpo | 0.073 | 0.179 | 0.227 | 0.267 |
| polyepo | 0.062 | 0.159 | 0.206 | 0.233 |

## beyondaime
n=100

| arm | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| polyepo | 0.040 | 0.099 | 0.137 | 0.190 |

## hmmt_feb25
n=30

| arm | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| polyepo | 0.008 | 0.029 | 0.047 | 0.067 |

## hmmt_nov25
n=30

| arm | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| polyepo | 0.042 | 0.092 | 0.125 | 0.167 |

## math500
n=500

| arm | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| grpo | 0.680 | 0.825 | 0.860 | 0.880 |
| polyepo | 0.683 | 0.832 | 0.868 | 0.892 |

## polaris_val
n=1024

| arm | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| polyepo | 0.192 | 0.393 | 0.497 | 0.589 |

