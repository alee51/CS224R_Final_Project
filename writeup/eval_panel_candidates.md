# Eval panel candidates — datasets + methods

Decision doc for which held-out datasets and which evaluation metrics to run on
the 3 final 4B step-400 checkpoints (GRPO, Minority-CoT, Poly-EPO-CoT). Locked
items go into `writeup/eval.md` once you sign off here.

## 1. OOD eval datasets — candidate list

All sourced as verl-format parquets at `main-verl/data/<name>.parquet`. Question
format = chat template + `\nPlease reason step by step, and put your final
answer within \boxed{}.` Scorer = `verl.utils.reward_score.math.compute_score`
(Hendrycks `is_equiv`).

### Tier A — paper-matching, already have parquet

| dataset | n | type | HF source | in paper Fig 1? |
|---|---|---|---|---|
| **AIME-25** | 30 | hard OOD | MathArena (BDP+25) | ✓ |
| **HMMT Feb 2025** | 30 | hard OOD | MathArena (BDP+25) | ✓ |
| **HMMT Nov 2025** | 30 | hard OOD | MathArena (BDP+25) | ✓ |
| **BeyondAIME** | 100 | hard OOD | `ByteDance-Seed/BeyondAIME` | ✓ |

### Tier B — paper-matching, parquet TO ADD

| dataset | n | type | HF source | status |
|---|---|---|---|---|
| **AIME 2026** | 30 | hard OOD | MathArena (BDP+25) | not yet pulled — pull from MathArena GitHub like the other AIME sets |
| **Minerva Math** | 272 | mixed (numeric + symbolic, undergrad STEM) | `math-ai/minervamath` (split=`test`) | Tajwar maxrl ships the preprocess script at `examples/maxrl_data_preprocess/minerva.py` — run it, drop parquet at `main-verl/data/minerva.parquet`. ~20 min |

> Note on "Minerva": this is **OCWCourses from the Minerva paper (Lewkowycz 2022)** — 272 undergrad STEM problems pulled from MIT OpenCourseWare. The RL-reasoning community calls this single 272-problem set "Minerva." About 81/272 problems have symbolic answers, which is exactly the regime where the V2 silent-grader bug bit us, so when we ship the headline number we MUST also ship a strict-vs-is_equiv rescore diff per the in-loop-eval verification rule.

### Tier C — currently in our registry but NOT in paper, candidates for drop

| dataset | n | type | drop? | rationale |
|---|---|---|---|---|
| **MATH-500** | 500 | easy OOD | **keep** | Not in Poly-EPO Fig 1, but in basically every other RL-reasoning paper. Low-variance ceiling check (n=500). Cheap. |
| **Polaris-val** | 1024 | in-distribution | **drop** | In-distribution → no clean OOD signal. Expensive (1024 problems × n_rollouts). Polaris was our training corpus; held-out 1024 doesn't tell us about generalization. |
| **DAPO slice 3k** | 3000 | in-distribution-ish | **drop** | 3000 problems is a lot of compute for an in-dist redundant signal. Paper does not use DAPO at eval. |

### Recommended final panel (6 datasets, 762 problems)

`aime25, aime26, hmmt_feb25, hmmt_nov25, beyondaime, minerva, math500`

That's actually 7 — `math500` is the one extra-vs-paper. If you want strict paper-match drop `math500`; if you want the easy-OOD ceiling check keep it. **I lean toward keep**, since MATH-500 is the standard in every other RL-for-math paper and dropping it would force us to handwave why we don't report it.

## 2. Eval methods — candidate list

### Tier A — Paper-matching (Poly-EPO Figs 1-4)

| # | metric | scope | implementation | cost |
|---|---|---|---|---|
| A1 | **pass@k** at k ∈ {1, 4, 8, 16, 32, 64} | held-out eval | already in `run_eval.py`; needs `n_rollouts=64` for direct pass@64 on AIME-style | **GPU** (already scheduled) |
| A2 | **majority@k** at k ∈ {1, 4, 8, 16, 64} | held-out eval | add to `analysis/coverage.py` (stub exists) | **analysis-only** |
| A3 | **majority vote share / k** (winner-share among k votes) | held-out eval | add to `analysis/coverage.py` | **analysis-only** |
| A4 | **\|U_correct\| trajectory** = avg # distinct judge CoT clusters (degenerate excluded) among correct rollouts per prompt, per training step | training | DONE — `analysis/u_correct.py`. Minority + Poly-EPO only (GRPO had no judge at train time → trivially 1.0). For a real cross-arm view, run the judge over GRPO rollouts post-hoc — see C1 below. | **analysis-only**, done for set arms |
| A5 | **non_zero_rate** = fraction of training prompts with ≥1 correct rollout, per step | training | DONE in `u_correct.py` | **analysis-only**, done |
| A6 | **token-position branching** (# active branches vs token index) | held-out eval | new `analysis/branching.py` — requires building a prefix tree across rollouts | **analysis-only**, ~3 hrs of work |

### Tier B — Beyond-paper, strong contenders

Surveyed from 8 other 2024–2026 RL-for-reasoning papers. All Tier B methods work on rollout JSONs we already have (`rewards, parsed_answer, cluster_id, response_length, finish_reason`).

| # | metric | from paper | what it tells us | implementation |
|---|---|---|---|---|
| B1 | **diff@k split by solved vs unsolved** — # distinct answers in first k rollouts, separately for prompts the model got right vs wrong | Song/Kempe/Munos (arxiv 2509.06941) | **Directly explains our Minority negative result.** If on solved prompts diff@k is low (collapse to one right answer) but on unsolved prompts diff@k is HIGH, that's "diversity went to the wrong answers." | trivial — group by `n_correct > 0`, count distinct `preds` |
| B2 | **AUC@k** — area under the pass@k curve as a single scalar per (arm, dataset) | Wu et al. (2601.08763), Tajwar (2602.02710) | **One scalar per cell** = clean poster table; stops k-cherry-picking arguments. | trivial — `trapz(pass_at_k_vector, ks)` |
| B3 | **difficulty-stratified pass@k** — bucket problems by base-model solve rate (0%/25%/50%/75%/100%), report pass@k per bucket | Tajwar (2602.02710), Chen et al. (2508.10751) | Shows whether Poly-EPO's parity with GRPO actually hides a long-tail win or loss. Likely shows Minority's loss is concentrated on hardest prompts. | needs a base-model eval pass to compute solve-rate buckets (~$20 one-time on Qwen3-4B-Base, then permanent) |
| B4 | **per-rollout entropy split by correct vs incorrect** | Song et al. (2509.06941), Cheng et al. (2506.14758) | Is the model uncertain for the "right" reason? If incorrect rollouts have HIGHER token entropy than correct, model knows when it's guessing. | needs token-level logprobs from rollouts — we DON'T currently save these. Would need to re-roll a small subset with logprob capture. **Expensive.** Skip. |
| B5 | **Self-BLEU + distinct-n-gram on rollout text** | Yao et al. (2505.23433) | Catches the "different answers but identical reasoning" failure mode. Text-level diversity sanity check. | 30 min of pandas |
| B6 | **Potential@k** — fraction of currently-failed problems that become solvable within k extra trials | Yao et al. (2505.23433) | Forward-looking ceiling — at what point does additional sampling stop helping? Direct analog of test-time-compute scaling. | trivial from `per_prompt.n_correct` |
| B7 | **cluster-conditional correctness** P(correct \| cluster rank) | our own, from `analysis/cluster_correctness.py` | DONE for training data — produced our headline 35-45% finding. Can also extend to eval. | already have for training; eval would need judge-on-eval ($5 AIME-25) |
| B8 | **pivotal-word / reflective-action frequency** ("wait", "however", "verify", "because") | Cheng et al. (2506.14758) | Behavioral proxy: did training increase reflective reasoning? If so, GRPO might have done it as much as Poly-EPO, blunting the "exploration" narrative. | trivial — `len(re.findall(...))` on rollout text |

### Tier C — Cross-arm judge passes (paid, decisive)

| # | metric | what it gives us | implementation | cost |
|---|---|---|---|---|
| C1 | **Post-hoc judge cluster pass over GRPO training rollouts** | Lets us plot `\|U_correct\|` trajectory for GRPO on the same axes as Minority and Poly-EPO. Without this, GRPO is stuck at 1.0 by construction and the training-time diversity comparison is moot. | Run the production judge endpoint over the saved per-rollout text at `main/data/probes/per_rollout_v2/grpo/`. ~40 sampled training steps × 128 prompts × 1 judge call each = ~5k judge calls. | ~$10–15 of judge compute, one-time |
| C2 | **Judge cluster pass over held-out eval rollouts, all 3 arms** | The eval-time analog of the paper's Fig 2 left. Tells us whether the diversity Minority/Poly-EPO maintain in training transfers to held-out problems. Apples-to-apples cross-arm since judge is run identically on all 3. Combined with diff@k split (B1) and majority@k (A2), gives the full diversity story. | New `analysis/u_correct_eval.py` that POSTs saved eval rollouts to the judge endpoint and writes back `cluster_id` per rollout. Then run the existing `u_correct` aggregation on the augmented JSON. | ~$5 per dataset per arm at AIME-25 scale (30 prompts × 16-32 rollouts × 3 arms). Full panel = ~$30–50. |

### Tier D — Considered and rejected

- **OOD non-math transfer** (e.g., HumanEval, MMLU-STEM) — out of scope for a math RL poster. Skip.
- **Temperature sweep at eval** — interesting but expensive (3× the eval bill). Skip unless poster narrative depends on it.
- **KL divergence from base model** — interesting "did we move the policy?" metric, but requires either logit capture or new generation. Skip.
- **Calibration / agreement-vs-correctness curves** — interesting but cross-cutting with majority vote share (A3). A3 covers the bases.

## 3. Recommendation: what to actually do

**Lock in for the poster** (analysis-only on already-saved data, no extra GPU):

1. **A1 pass@k** (k up to 32 with n=32 rollouts per AIME-style dataset, n=16 for MATH-500/Minerva). Optional bump to n=64 for paper-matching AIME pass@64 if budget permits.
2. **A2 majority@k** + **A3 majority vote share / k**
3. **A4 + A5** done already for set arms
4. **B2 AUC@k** as the single poster-table headline scalar
5. **B1 diff@k split by solved/unsolved** — load-bearing for the Minority story
6. **B6 Potential@k** — cheap forward-looking ceiling per benchmark
7. **B5 Self-BLEU + distinct-n-gram** — quick text-level sanity check
8. **B8 reflective-action frequency** — orthogonal behavioral angle

**Pay for** (judge cost, decisive):

9. **C1 Judge pass on GRPO training rollouts** — closes the cross-arm `|U_correct|` story at training time, ~$10–15
10. **C2 Judge pass on all 3 arms' eval rollouts, AIME-25 only first** — closes the eval-time diversity story, ~$5 to start

**Defer / cut**:
- A6 branching (3 hrs, can ship later)
- B3 difficulty-stratified (needs base-model eval pass; ~$20 one-time)
- B4 token entropy (we don't have logprobs)

**One open call on dataset panel**: decide on `math500` keep-or-drop. Default keep.
**One open call on n_rollouts**: 32 or 64 on AIME-style. Default 32 to start, bump to 64 only on AIME-25/26 if the pass@k curve at n=32 looks promising.

## 4. Sources

- Poly-EPO paper (Hamid/Orney et al. 2026, arxiv 2509.25424) — Figs 1-4 in §6
- Tajwar maxrl preprocess scripts (`examples/maxrl_data_preprocess/`) — Minerva dataset identity
- Song/Kempe/Munos, Outcome-based Exploration (arxiv 2509.06941) — diff@k split
- Yao et al., Diversity-Aware Policy Optimization (arxiv 2505.23433) — Potential@k, Div-Equ
- Cheng et al., Reasoning with Exploration (arxiv 2506.14758) — reflective action frequency
- Wu et al., Rewarding the Rare (arxiv 2601.08763) — AUC@k as scalar headline
- Tajwar et al., MaxRL (arxiv 2602.02710) — difficulty stratification
