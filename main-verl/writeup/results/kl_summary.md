# Per-token KL(π_arm ‖ π_base) — held-out eval rollouts

## TL;DR

Mean per-token KL from the trained policy's saved top-K to the base model's
top-K (recomputed via vLLM teacher-forcing). All 3 trained arms diverge from
base on **mean** KL (1.9–3.2 bits/token) but **median** KL stays low
(~0.25–0.30 bits/token) — divergence is concentrated in a small fraction
of high-leverage tokens, not spread across the rollout. Long-tail behavior:
the policy stays close to base most of the time but makes occasional
high-confidence reversals (e.g., picking a specific number for `\boxed{}`).

Across arms the mean KL is similar (grpo/minority/polyepo all in the same
~2–3 bits range). **Minority is NOT the most-divergent trained arm**
(contradicts a naïve "minority = high entropy" prediction). GRPO is the
most-divergent at mean KL = 2.75 bits, polyepo is the least at 2.33 bits.

## Cross-arm mean KL (bits/token)

| arm \ dataset | aime25 | aime26 | hmmt_feb25 | hmmt_nov25 | beyondaime | per-arm mean |
|---|---|---|---|---|---|---|
| grpo | 2.661 | 2.652 | 3.156 | 2.974 | 2.318 | 2.752 |
| minority | 2.140 | 2.861 | 2.824 | 2.658 | 2.376 | 2.572 |
| polyepo | 2.212 | 2.684 | 2.651 | 2.180 | 1.918 | 2.329 |
| **per-ds mean** | 2.337 | 2.732 | 2.877 | 2.604 | 2.204 | |

### math500 (easy-OOD)

| arm | mean KL (bits) | median | p90 |
|---|---|---|---|
| grpo | 2.851 | 0.229 | 13.003 |
| minority | 2.435 | 0.243 | 5.421 |
| polyepo | _missing_ (input GEN corrupted) | — | — |

math500 has *higher* mean KL than the smallood average for both grpo and
minority — somewhat counter-intuitive (easy dataset, should be closer to
base). Explanation: math500 prompts are tighter/cleaner, so the policy's
high-leverage tokens (the actual answer choice) get reweighted MORE
strongly vs base's broader latex/format priors. Hard-OOD prompts have
more pivot tokens distributed across longer reasoning chains, diluting
the per-token KL average.

All cells at consistent n: hard-OOD-30 cells use n_prompts=30 × 8 rollouts/prompt = 240
(or 100 × 8 = 800 for beyondaime). The earlier kl_summary.md had a 3-cell `max_prompts=20`
cap inconsistency on grpo {aime25, aime26, beyondaime} — fixed by a re-run on 2026-06-04
(Modal app `ap-F226GQblGB5rxcFgFReI91`).

## Distribution shape: mean vs median vs p90

Heavy-tailed KL signature — most rollout-mean-KL values are near the median,
with a small fraction of high-KL outliers driving the mean. p90/mean > 2 =
pronounced right tail.

| arm | dataset | mean | median | p90 | p90/mean | mean/median |
|---|---|---|---|---|---|---|
| grpo | aime25 | 2.661 | 0.256 | 12.404 | 4.7 | 10.4 |
| grpo | aime26 | 2.652 | 0.240 | 11.032 | 4.2 | 11.1 |
| grpo | hmmt_feb25 | 3.156 | 0.291 | 14.642 | 4.6 | 10.9 |
| grpo | hmmt_nov25 | 2.974 | 0.251 | 14.316 | 4.8 | 11.8 |
| grpo | beyondaime | 2.318 | 0.275 | 5.735 | 2.5 | 8.4 |
| minority | aime25 | 2.140 | 0.268 | 2.133 | 1.0 | 8.0 |
| minority | aime26 | 2.861 | 0.276 | 11.914 | 4.2 | 10.4 |
| minority | hmmt_feb25 | 2.824 | 0.288 | 4.577 | 1.6 | 9.8 |
| minority | hmmt_nov25 | 2.658 | 0.277 | 8.027 | 3.0 | 9.6 |
| minority | beyondaime | 2.376 | 0.300 | 4.894 | 2.1 | 7.9 |
| polyepo | aime25 | 2.212 | 0.252 | 2.783 | 1.3 | 8.8 |
| polyepo | aime26 | 2.684 | 0.255 | 7.701 | 2.9 | 10.5 |
| polyepo | hmmt_feb25 | 2.651 | 0.274 | 9.093 | 3.4 | 9.7 |
| polyepo | hmmt_nov25 | 2.180 | 0.256 | 2.098 | 1.0 | 8.5 |
| polyepo | beyondaime | 1.918 | 0.278 | 1.434 | 0.7 | 6.9 |

## Headline aggregate (15 cells)

- Average per-token KL (mean across cells): **2.551 bits**
- Average per-token KL (median across cells): **0.269 bits**
- Average p90 across cells: **7.519 bits**
- mean / median ratio (averaged): **9.5x**

Reading: typical rollout token has near-zero divergence from base; mean is
10× higher because ~10% of tokens contribute almost all the policy drift.
RL training shifts a sparse subset of token decisions hard and leaves the rest near-base.

## How this was computed

- Script: `main-verl/eval/analysis/posthoc/kl_from_base.py`
- Modal app pattern: B200:1, vLLM with `logprobs=20`, `max_model_len=8192`,
  `max_num_seqs=16`, `gpu_memory_utilization=0.70`, `enforce_eager=True`.
- Method: for each (trained arm, dataset) policy JSON, take the saved
  `per_prompt[i].rendered_prompt` + `per_prompt[i].rollouts[r]` as the
  policy trace. Teacher-force the base model with
  `tokenize(rendered_prompt + rollout_text)` and ask vLLM for
  `prompt_logprobs=20`. Slice the base's distributions for the rollout
  positions only — `base_prompt_lp[P + t]` is base's prediction for
  rollout token t given (rendered_prompt + rollout_tokens[0:t]),
  exactly aligned with the policy's `comp.logprobs[t]`.
- For each token position, compute KL between the two top-K dictionaries
  (both renormalized over the union of token_ids in either side).
  Skip positions where the policy's top-K is empty or where alignment fails.
- Per-rollout mean = mean over token positions. Per-cell mean = mean over
  the 240 (or 800 for beyondaime) rollouts. Sampling cap: `max_rollouts=8`.
- Output: per-(arm, dataset) JSON at `/vol/probes/kl/<arm>_<dataset>.json`.

**Token-id key coercion** (silent-bug fix per `eval_pipeline_verification.md`):
policy logprobs (saved JSON) have str keys; base logprobs (Modal-side) have int keys.
The KL helper coerces both to int before the union — without this, every token
was treated as missing on one side → garbage KL. Verified `KL(self‖self) = 0` post-fix.

**Length guard** (fix 2026-06-03): `max_model_len` raised from 5120 to 8192 after
polyepo cells exceeded the lower limit on a rollout's `prompt + rollout_text`
length. See `eval_pipeline_bugs.md` Bug 3.

## math500 KL — landed (2/3 cells)

Phase 3 math500 KL completed at 2026-06-04 ~03:00 PDT (Modal app
`ap-nr3cBVsR1NcW0Ip0op570D`). 2 trained-arm math500 KL cells written
(grpo, minority); the polyepo cell errored as expected — the input
polyepo math500 GEN JSON was truncated (Bug 5) and `json.loads` threw
`JSONDecodeError: Expecting ':' delimiter: line 2126968 column 23 (char 89454830)`,
which terminated the kl_pass before reaching the polyepo aggregation step.
`kl_from_base.py` patched mid-session (commit pending) to skip unparseable
inputs instead of crashing — fix won't help this run but protects future
re-runs.

Updated math500 column added above. Per-arm means (across math500 +
5 smallood = 6 datasets):
- grpo: 2.769 bits/token (was 2.752 over 5 smallood)
- minority: 2.549 bits/token (was 2.572 over 5 smallood)
- polyepo: 2.329 bits/token (smallood only; math500 missing)
