# Reflective-action frequency in rollout text

Counts per rollout, then averaged. `total_per_1k_tokens` is the
rate normalized by rollout length to control for verbosity.

## aime25

| arm | n_roll | wait | however | verify | because | alternatively | let_me_check | let_me_reconsider | total/roll | total/1k_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1920 | 0.089 | 0.342 | 0.107 | 0.234 | 0.045 | 0.001 | 0.000 | 0.818 | 1.259 |
| grpo | 1920 | 0.097 | 0.885 | 0.078 | 0.206 | 0.054 | 0.000 | 0.000 | 1.321 | 1.173 |
| minority | 1920 | 0.090 | 0.461 | 0.047 | 0.156 | 0.031 | 0.000 | 0.000 | 0.785 | 0.919 |
| polyepo | 1920 | 0.071 | 0.706 | 0.057 | 0.173 | 0.006 | 0.000 | 0.000 | 1.014 | 1.104 |

## aime26

| arm | n_roll | wait | however | verify | because | alternatively | let_me_check | let_me_reconsider | total/roll | total/1k_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1920 | 0.164 | 0.357 | 0.114 | 0.249 | 0.052 | 0.000 | 0.000 | 0.935 | 1.244 |
| grpo | 1920 | 0.141 | 0.584 | 0.051 | 0.328 | 0.037 | 0.000 | 0.000 | 1.140 | 0.985 |
| minority | 1920 | 0.086 | 0.551 | 0.049 | 0.217 | 0.235 | 0.000 | 0.000 | 1.138 | 0.981 |
| polyepo | 1920 | 0.451 | 0.521 | 0.032 | 0.160 | 0.006 | 0.000 | 0.000 | 1.170 | 1.177 |

## beyondaime

| arm | n_roll | wait | however | verify | because | alternatively | let_me_check | let_me_reconsider | total/roll | total/1k_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 6400 | 0.034 | 0.433 | 0.109 | 0.334 | 0.014 | 0.000 | 0.000 | 0.923 | 1.678 |
| grpo | 6400 | 0.046 | 0.531 | 0.071 | 0.348 | 0.011 | 0.000 | 0.000 | 1.007 | 1.239 |
| minority | 6400 | 0.092 | 0.521 | 0.060 | 0.274 | 0.025 | 0.000 | 0.000 | 0.972 | 1.180 |
| polyepo | 6400 | 0.039 | 0.566 | 0.051 | 0.236 | 0.011 | 0.000 | 0.000 | 0.902 | 1.153 |

## hmmt_feb25

| arm | n_roll | wait | however | verify | because | alternatively | let_me_check | let_me_reconsider | total/roll | total/1k_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1920 | 0.145 | 0.444 | 0.110 | 0.342 | 0.028 | 0.000 | 0.000 | 1.070 | 1.786 |
| grpo | 1920 | 0.080 | 0.914 | 0.094 | 0.395 | 0.042 | 0.000 | 0.000 | 1.525 | 1.360 |
| minority | 1920 | 0.060 | 0.529 | 0.123 | 0.274 | 0.010 | 0.000 | 0.000 | 0.996 | 1.098 |
| polyepo | 1920 | 0.032 | 0.582 | 0.104 | 0.279 | 0.052 | 0.000 | 0.000 | 1.048 | 1.181 |

## hmmt_nov25

| arm | n_roll | wait | however | verify | because | alternatively | let_me_check | let_me_reconsider | total/roll | total/1k_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1920 | 0.150 | 0.426 | 0.127 | 0.328 | 0.058 | 0.000 | 0.000 | 1.089 | 1.607 |
| grpo | 1920 | 0.066 | 0.644 | 0.066 | 0.322 | 0.027 | 0.000 | 0.000 | 1.126 | 1.221 |
| minority | 1920 | 0.066 | 0.546 | 0.088 | 0.369 | 0.014 | 0.000 | 0.000 | 1.083 | 1.180 |
| polyepo | 1920 | 1.296 | 0.511 | 0.119 | 0.126 | 0.030 | 0.000 | 0.000 | 2.083 | 1.344 |

