# Results

Final numbers only. Mid-run diagnostics live in [`timeline.md`](../../timeline.md) and W&B. If a number might change, it doesn't belong here yet.

**Last updated:** 2026-05-27 (three-arm canonical eval; arms still training).

---

## Three-arm checkpoint eval (canonical, 2026-05-27)

Mid-training snapshot. GRPO at step 299/799; minority_answer and poly_epo_answer at step 133/799 (BeyondAIME used slightly newer ckpts — see footnote). Harness: [`probes/checkpoint_rollout_eval.py`](../../../probes/checkpoint_rollout_eval.py). Train grader (mathd ∨ sympy, Rank-2 parse).

### Polaris 2k — on-distribution (pass@8)

Same 2000-prompt slice from `polaris_train.jsonl`, seed 42, prompt `hybrid_answer_boxed` (matches train), 8 rollouts/prompt.

| Variant | pass@8 | Δ vs base | frac 0/8 |
|---------|--------|-----------|----------|
| **base** (Qwen3-1.7B-Base) | **0.306** | — | 0.694 |
| GRPO `s299` | 0.317 | +1.1 pp | 0.683 |
| minority_answer `s133` | 0.307 | +0.1 pp | 0.693 |
| poly_epo_answer `s133` | 0.319 | +1.3 pp | 0.682 |

Run `20260527T213611Z` · config `checkpoint_eval_2k_polaris_arms_latest_b200.yaml` · ~49 min on B200.

### DAPO 2k — easier OOD (pass@8)

`dapo_n2000_seed43`, prompt `dapo_answer_v1` (≠ train prompt — **rerun with hybrid prompt queued**; GRPO's +2.4 pp may move under fair prompt).

| Variant | pass@8 | Δ vs base | mean_reward |
|---------|--------|-----------|-------------|
| **base** | **0.248** | — | 0.051 |
| GRPO `s299` | **0.272** | **+2.4 pp** | 0.056 |
| minority_answer `s133` | 0.252 | +0.5 pp | 0.052 |
| poly_epo_answer `s133` | 0.263 | +1.5 pp | 0.055 |

Run `20260527T203133Z` · config `checkpoint_eval_ood_aime_dapo_arms_latest_b200.yaml`.

### BeyondAIME — hard OOD (pass@16)

100 problems, prompt `dapo_answer_v1` (≠ train prompt — **rerun with hybrid prompt queued**). Newer ckpts for set arms.

| Variant | pass@1 | pass@4 | pass@8 | **pass@16** | Δ vs base |
|---------|--------|--------|--------|-------------|-----------|
| **base** | 0.009 | 0.035 | 0.068 | **0.130** | — |
| GRPO `s359` | 0.005 | 0.020 | 0.038 | 0.070 | **−6.0 pp** |
| minority_answer `s159` | 0.005 | 0.020 | 0.038 | 0.070 | **−6.0 pp** |
| poly_epo_answer `s133` | 0.005 | 0.020 | 0.040 | 0.080 | **−5.0 pp** |

Run `20260527T221956Z` · config `checkpoint_eval_beyondaime_pass16_arms_latest_b200.yaml` · ~6 min on 4× B200.

### AIME-25 — exploratory (pass@8, n=30, not decision-grade)

| Variant | pass@8 | Δ vs base |
|---------|--------|-----------|
| base | 0.033 | — |
| GRPO `s299` | 0.033 | 0.0 |
| minority_answer `s133` | 0.067 | +3.3 pp (~1 problem) |
| poly_epo_answer `s133` | 0.000 | −3.3 pp |

---

## Cross-slice summary

| Slice | minority Δ | GRPO Δ | poly Δ | Use in writeup |
|-------|------------|--------|--------|----------------|
| Polaris 2k pass@8 | +0.1 pp | +1.1 pp | +1.3 pp | Primary (on-distribution) |
| DAPO 2k pass@8 | +0.5 pp | **+2.4 pp** | +1.5 pp | Primary (easier OOD) |
| BeyondAIME pass@16 | −6.0 pp | −6.0 pp | −5.0 pp | Primary (hard OOD; note prompt confound) |
| AIME-25 pass@8 | +3.3 pp | 0 | −3.3 pp | Exploratory only (n=30) |

**Project-level reads (from [`timeline.md`](../../timeline.md) §2026-05-27 three-arm verdicts):**

1. Training is not obviously broken — Polaris pass@8 does not regress; W&B optimization stable.
2. **Minority-answer hypothesis not supported** at these ckpts — flat on Polaris and DAPO; no decision-grade win vs GRPO.
3. GRPO has the best OOD signal (DAPO +2.4 pp) but **regresses vs base on BeyondAIME**.
4. Poly-EPO-answer tracks GRPO on Polaris/DAPO; marginally better than GRPO on BeyondAIME but still far below base.

---

## Per-arm writeups

One file per arm, created when training + final eval finishes:

- `grpo.md` *(not yet — arm in progress)*
- `minority_answer.md` *(not yet)*
- `minority_cot.md` *(not in scope unless compute opens up)*
- `poly_epo_answer.md` *(not yet)*

**Per-arm file template:**
```
# <arm name>

## Setup
- Config: <path>
- Wandb run: <id>
- Train wall-clock / cost: <hours / $>
- Final checkpoint: <step>

## Headline numbers
[table row from cross-slice summary above]

## Training curve highlights
- [3 bullets: when reward took off, notable inflections, final stable loss]

## Eval-time observations
- [3 bullets: where it wins/loses vs GRPO; pass@k distribution shape]

## Caveats
- [anything that makes the comparison less than apples-to-apples]
```

---

## Probe findings (locked, cite from paper)

Frozen findings that lock the train stack. Cite from `paper/method.md` § Prompt + parser, § Reward, § Data.

- **Prompt format** → arm C `hybrid_answer_boxed` ([`probes/group_a_results.md`](../../probes/group_a_results.md), [`probes/prompt_probe.md`](../../probes/prompt_probe.md))
- **Parser** → Rank-2 ([`probes/group_a_results.md`](../../probes/group_a_results.md))
- **Reward grader** → mathd ∨ sympy ([`probes/mathd_sympy_rescore_n800.md`](../../probes/mathd_sympy_rescore_n800.md))
- **Train data** → Polaris over DAPO ([`probes/dapo_vs_polaris_rollout_comparison.md`](../../probes/dapo_vs_polaris_rollout_comparison.md), [`probes/integer_vs_random_fullgold_unified_grade.md`](../../probes/integer_vs_random_fullgold_unified_grade.md))

## Active training runs (not results yet — see [`handoff/`](../../handoff/))

For paper, only the *final* numbers from completed runs land here. Currently in flight on B200, 2026-05-27 launch:

| Arm | Workspace | Wandb | Step / 799 |
|-----|-----------|-------|------------|
| GRPO | alee72 | [t11jct0t](https://wandb.ai/224r-project/cs224r-minority-voting/runs/t11jct0t) | ~326 |
| minority_answer | alee72 | [o5ypkzja](https://wandb.ai/224r-project/cs224r-minority-voting/runs/o5ypkzja) | ~148 |
| poly_epo_answer | chicken602 | [fdx95beu](https://wandb.ai/224r-project/cs224r-minority-voting/runs/fdx95beu) | ~146 |
