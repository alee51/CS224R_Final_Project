# Reflective-action frequency in rollout text

## TL;DR

**What it measures.** Per-rollout count of seven self-monitoring / hedging
phrases (`wait`, `however`, `verify`, `because`, `alternatively`,
`let me check`, `let me reconsider`), matched as case-insensitive whole-word
regex. Averaged within (arm, dataset). `total/roll` is per-rollout count
across all seven patterns; `total/1k_tok` normalizes by rollout length so
arms with longer rollouts don't look artificially more reflective.

**How to read.** Compare arms within a dataset. Higher = more "reflective"
surface text, which is a *very* rough proxy for chain-of-thought
self-correction. The two `let_me_*` patterns are essentially always 0 across
all (arm, dataset) cells — the model rarely uses those exact phrases.

**Headline.** GRPO and polyepo bump the `however` rate ~1.5–2× over base on
several datasets; minority sits between base and GRPO. **Outlier:
polyepo/hmmt_nov25 has wait=1.296** (vs base=0.150) — 8.6× baseline and
~3× the next-highest cell, driving total/roll=2.083. **minority/aime26 has
alternatively=0.235** vs base=0.052 — 4.5× baseline. These two are large
enough to warrant a sanity check that they aren't a single-prompt or single-
rollout repetition artifact.

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

## How this was computed

- **Script**: `main-verl/eval/analysis/posthoc/reflective_actions.py`. Seven
  regex patterns (`\bwait\b`, `\bhowever\b`, `\bverify\b`, `\bbecause\b`,
  `\balternatively\b`, `\blet me check\b`, `\blet me reconsider\b`), all
  case-insensitive. Counts are summed over a rollout, then averaged across
  rollouts within (arm, dataset).
- **Inputs**: same 20 probe JSONs (`*_step400_smallood_*.json`); reads
  `per_prompt[i].rollouts[j]` text.
- **Tokenization**: `total/1k_tok` uses **whitespace tokens**
  (`len(rollout.split())` divided by 1000), NOT model tokens. So the
  per-1k-tok rate is an underestimate relative to the BPE-token count the
  policy actually produced, but the *relative* comparison across arms is
  preserved.
- **n_roll**: number of dataset_size × n=64 rollouts per (arm, dataset).
  30-prompt datasets → 1920 rollouts; beyondaime (100 prompts) → 6400
  rollouts.
- **Limitations / caveats**:
  - "Reflective" is a surface lexical proxy; nothing here measures
    whether the reflection improved the answer.
  - polyepo/hmmt_nov25 `wait=1.296` looks like a token-level repetition
    artifact (the policy may have entered a `wait wait wait …` loop on a
    subset of prompts). Worth a spot-check of high-count rollouts before
    citing this as "more reflective".
  - Empty rollouts are skipped; multi-word phrases use word boundaries on
    the ends only (intermediate spaces are literal).
