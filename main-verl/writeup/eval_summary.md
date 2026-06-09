# Eval summary — what we have

**One locked eval run on abao, 2026-06-02 → 06-04.** Single source of truth
for the paper's results. Start here.

## Setup

**4 arms × 6 datasets × 64 rollouts/prompt** · temp=1.0 · max_tokens=4096 · logprobs=20

```
arms     : base | grpo | minority | polyepo                (all step 400)
datasets : aime25 | aime26 | hmmt_feb25 | hmmt_nov25 | beyondaime | math500
prompts  :   30   |   30   |     30     |     30      |    100     |   500
```

Grader: `verl.utils.reward_score.math.compute_score` (Hendrycks `is_equiv`).
Full spec: [`eval.md`](eval.md).

## What we measured per (arm, dataset)

| metric | what it answers | source |
|---|---|---|
| **pass@k** | correctness in k attempts (k=1,2,4,8,16,32,64) | `results/comparison.md`, `results/auc_at_k.md` |
| **coverage / entropy / majority@k** | how spread the answer distribution is | `results/coverage.md` |
| **diff@k split** | does extra diversity go to wrong answers? | `results/diff_at_k_split.md` |
| **potential@k** | budget-bound vs quality-bound failure | `results/potential_at_k.md` |
| **self-BLEU + distinct-n** | rollout-text diversity | `results/self_bleu.md` |
| **reflective_actions** | wait/however/verify lexical counts | `results/reflective_actions.md` |
| **KL(π_arm ‖ π_base)** | per-token divergence from base (3 trained arms) | `results/kl_summary.md` |
| **grader_sanity** | 3-way verification that numbers aren't a grader bug | `results/grader_sanity_all.md` |
| **CoT diversity@k** ¹ | distinct correct reasoning clusters in k draws | `../eval/results/cot_diversity_results.md` |

¹ Anastasia's track: post-hoc judge clustering on the same GEN data, math500 + beyondaime only.

## Coverage

```
GEN (rollouts):  ████████████████████████ 23/24 cells   (polyepo×math500 — re-run exists, schema unverified)
KL  (per-token): ███████████████████░░░░░ 17/18 cells   (same gap)
CoT diversity:   ████████░░░░░░░░░░░░░░░░  7/8 cells   (4 arms × {math500, beyondaime})
```

## Headline: pass@64

| arm | aime25 | aime26 | beyondaime | hmmt_feb25 | hmmt_nov25 | math500 |
|---|---|---|---|---|---|---|
| **base** | **0.333** | **0.200** | **0.290** | **0.200** | 0.133 | **0.928** |
| grpo | 0.067 | 0.067 | 0.120 | 0.067 | **0.167** | 0.816 |
| minority | 0.033 | 0.100 | 0.090 | 0.100 | **0.167** | 0.804 |
| polyepo | 0.133 | 0.000⚠ | 0.130 | 0.167 | **0.167** | _missing_ |

Bold = winner per column. Base wins everywhere except **hmmt_nov25** — see
the depth-vs-breadth crossover in [`results/CAVEATS.md`](results/CAVEATS.md).
Polyepo × aime26 = 0/1920 is a real repetition collapse, not a grader bug.

## Headline: AUC of pass@k over k ∈ {1..64}

| arm | aime25 | aime26 | beyondaime | hmmt_feb25 | hmmt_nov25 | math500 |
|---|---|---|---|---|---|---|
| base | 14.566 | 9.360 | 12.180 | 7.322 | 7.685 | **54.799** |
| grpo | 2.826 | 2.133 | 4.932 | 2.133 | 7.850 | 46.288 |
| minority | 1.562 | 3.695 | 3.681 | 3.818 | 7.988 | 44.900 |
| polyepo | 5.088 | 0.000 | 5.115 | 5.951 | 7.998 | _missing_ |

## Where to go next

- **Writing the paper:** [`paper.md`](paper.md) — section-by-section TODO + open decisions
- **Citing a specific result file:** [`results/INDEX.md`](results/INDEX.md) — flat file table
- **Audit caveats + interpretive notes:** [`results/CAVEATS.md`](results/CAVEATS.md) — 7 caveats + hmmt_nov25 crossover + polyepo×math500 reconciliation
- **Specs:** [`eval.md`](eval.md), [`training.md`](training.md), [`algorithm_descriptions.md`](algorithm_descriptions.md)
- **Ops:** [`MODAL_STATUS.md`](MODAL_STATUS.md) (accounts, ckpts, budgets); historical run docs in [`archive/`](archive/)
