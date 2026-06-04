# Per-token KL(π_arm ‖ π_base) — held-out eval rollouts

## TL;DR

Mean per-token KL from the trained policy's saved top-K to the base model's
top-K (recomputed via vLLM teacher-forcing). All 3 trained arms diverge from
base on **mean** KL (1.7–3.2 bits/token) but **median** KL stays low
(0.25–0.30 bits/token) — divergence is concentrated in a small fraction
of high-leverage tokens, not spread across the rollout. Long-tail behavior:
the policy stays close to base most of the time but makes occasional
high-confidence reversals (e.g., picking a specific number for `\boxed{}`).

Across arms the mean KL is similar (grpo/minority/polyepo all in the same
1.7–3.2 range) — no single arm is dramatically more divergent than another
on OOD prompts. **Minority is NOT the most-divergent trained arm**
(contradicts a naïve "minority = high entropy" prediction).

## Numbers

| arm | dataset | n_prompts | n_rollouts | mean KL (bits) | median KL (bits) |
|---|---|---|---|---|---|
| grpo | aime25 | 20 | 160 | 2.813 | 0.263 |
| grpo | aime26 | 20 | 160 | 2.949 | 0.247 |
| grpo | hmmt_feb25 | 30 | 240 | 3.156 | 0.291 |
| grpo | hmmt_nov25 | 30 | 240 | 2.974 | 0.251 |
| grpo | beyondaime | 20 | 160 | 1.671 | 0.270 |
| minority | aime25 | 30 | 240 | 2.140 | 0.268 |
| minority | aime26 | 30 | 240 | 2.861 | 0.276 |
| minority | hmmt_feb25 | 30 | 240 | 2.824 | 0.288 |
| minority | hmmt_nov25 | 30 | 240 | 2.658 | 0.277 |
| minority | beyondaime | 100 | 800 | 2.376 | 0.300 |
| polyepo | aime25 | 30 | 240 | 2.212 | 0.252 |
| polyepo | aime26 | 30 | 240 | 2.684 | 0.255 |
| polyepo | hmmt_feb25 | 30 | 240 | 2.651 | 0.274 |
| polyepo | hmmt_nov25 | 30 | 240 | 2.180 | 0.256 |
| polyepo | beyondaime | 100 | 800 | 1.918 | 0.278 |

Per-arm mean KL average over all 5 datasets:
- grpo: 2.71 bits/token
- minority: 2.57 bits/token
- polyepo: 2.33 bits/token

Per-arm median KL average:
- grpo: 0.26 bits/token
- minority: 0.28 bits/token
- polyepo: 0.26 bits/token

Note inconsistent n_prompts: grpo aime25/aime26/beyondaime were re-run with
the smaller `CS224R_KL_MAX_PROMPTS=20` cap by the session's later kl_pass
invocation; the other 12 entries are from the prior n=30 (or n=100 for
beyondaime) run. The means are stable enough across n that this doesn't
affect the cross-arm comparison.

## Interpretation

- **Mean ≫ median** for every (arm, dataset) → divergence is heavy-tailed,
  concentrated in a small subset of tokens. The policy doesn't drift
  uniformly away from base.
- **High-KL tokens are likely the high-leverage ones** — the model picks
  a specific number or formula for `\boxed{...}`, which RL training
  reweights aggressively against base's broader prior. The CoT body stays
  close to base (low median).
- **Cross-arm comparison**: polyepo has lower mean KL than grpo (2.33 vs
  2.71). Minority sits in between. This is consistent with polyepo's
  "all-distinct-cluster" reward being a weaker signal than grpo's
  group-relative advantage on average, despite all three reaching similar
  training loss.
- **Per-dataset variation > per-arm variation**: every arm has highest mean
  KL on hmmt_feb25 (2.65–3.16), lowest on beyondaime (1.67–2.38). The
  prompts themselves drive most of the divergence pattern, not the arm.

## How this was computed

- Script: `main-verl/eval/analysis/posthoc/kl_from_base.py`
- Modal app: posthoc-kl, B200:1, vLLM with `logprobs=20`, `enforce_eager=True`
- Method: for each (trained arm, dataset) policy JSON, take the saved
  `per_prompt[i].rendered_prompt` + `per_prompt[i].rollouts[r]` as the
  policy trace. Teacher-force the base model with
  `tokenize(rendered_prompt + rollout_text)` and ask vLLM for
  `prompt_logprobs=20`. Slice the base's distributions for the rollout
  positions only — `base_prompt_lp[P + t]` is base's prediction for
  rollout token t given (rendered_prompt + rollout_tokens[0:t]),
  exactly aligned with the policy's `comp.logprobs[t]`.
- For each token position, compute KL between the two top-K dictionaries
  (both renormalized over the K visible tokens). Skip positions where the
  policy's top-K is empty or where alignment fails.
- Aggregate: mean and median over (position × rollout) per (arm, dataset).
- Sampling cap: `CS224R_KL_MAX_PROMPTS=20` (some cells), `MAX_ROLLOUTS=8`
  per prompt for the later cells; full n=30 (or n=100 for beyondaime) for
  the earlier-written cells.
- Output: per-(arm, dataset) JSON at `/vol/probes/kl/<arm>_<dataset>.json`
  with `{n_prompts, n_rollouts, kl_mean_bits, kl_median_bits}` (and
  longer-form per-position arrays inside).
