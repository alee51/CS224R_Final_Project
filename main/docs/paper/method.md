# Method (paper-bound)

Source of truth for the paper's Method section. Cross-link to [`PLAN.md`](../PLAN.md) for strategy and `main/train/` for implementation. Write math + prose here, not code.

---

## 1. Problem setup

- **Task:** math reasoning with verifiable answers on Polaris (51,139 prompts, prompt-filtered from 53k).
- **Backbone:** Qwen3-1.7B-Base.
- **Compute envelope:** 1× B200 (or H200 fallback), collocated vLLM rollout + HF train; bs=64, 8 rollouts/prompt, `max_response_length=4096`, KL=0.
- **Hypothesis (from PLAN §3):** A set-based minority-voting objective improves pass@k generalization on harder held-out reasoning evals compared to vanilla GRPO, by preventing collapse to a single output mode.

## 2. Baseline: GRPO

- Per-trajectory advantage A_i = r_i − mean(r) over the N=8 rollouts of a prompt.
- Clipped importance ratio (asymmetric clip à la DAPO: clip_low=0.20, clip_high=0.28).
- Inner PPO epochs = 1 (REINFORCE-with-clip semantics).
- KL coefficient = 0 (no reference model).
- vLLM rollout logprobs reused as `old_logprobs` (no separate HF forward over rollouts).
- Zero-advantage prompts filtered before backward.
- Reference: standard GRPO formulation; calibrated against Poly-EPO Table 1 (PLAN §5).

## 3. Set-based minority-voting objective (Minority-answer / Minority-CoT)

For each prompt, sample N=8 rollouts. For each of the C(8,4)=70 size-4 subsets G:

- f(G) = r(minority(G)) where minority(G) is the cluster with the lowest frequency in G (ties broken at random).
- Per-rollout marginal advantage A_i = (mean f(G) over the 35 sets containing rollout i) − (mean f(G) over all 70 sets).

**Clustering substrates:**

- **Minority-answer:** exact answer match (Rank-2 parsed answer, hash, allowlist sympy merge for canonicalization). See [`build_spec/answer_clustering.md`](../build_spec/answer_clustering.md).
- **Minority-CoT** *(in scope if compute allows)*: LLM-judged CoT clusters via an in-loop Qwen-3-4B-Instruct judge ([`judge/poly_epo_a1.md`](../../judge/poly_epo_a1.md)).

**Tiebreak for minority(G):** random pick (settled in milestone; r=0.994 with averaging, random preferred for simplicity).

## 4. Poly-EPO-answer (stretch)

- Same answer-hash substrate as Minority-answer; subset score differs:
- f_poly(G) = mean(r) · diversity(G), where diversity(G) = distinct answer-hash clusters in G / 4.
- Same per-rollout marginal-advantage averaging as §3.
- Cite [Poly-EPO (Hasan Orney et al., May 2026)] for original formulation; we use answer-hash diversity rather than LLM-judged CoT clusters for cost parity. Note this deviation in the paper.

## 5. Training procedure

- Optimizer: AdamW (fused on B200).
- LR: 1e-6 (matches Poly-EPO Table 1; LR sweep deferred unless training diverges).
- Schedule: constant after warmup.
- Gradient checkpointing: on.
- Mixed precision: bf16 throughout.
- Attention: FlashAttention-2 (`build_hf` `attn_implementation="flash_attention_2"`).
- `token_budget` (greedy chunk packing for backward): 130k on B200 prod.
- Total steps: 799 (one epoch on 51,139-prompt manifest / batch 64).
- Multi-leg Modal training: 23h legs, self-spawn on budget exhaustion (`train_remote.spawn(leg_number=N+1)`); same W&B run id across legs.
- Canonical config: [`configs/train_real.yaml`](../../configs/train_real.yaml), B200 overlay [`configs/train_real_b200.yaml`](../../configs/train_real_b200.yaml).

## 6. Evaluation harness

**Datasets:**

- **On-distribution:** Polaris 2k held-out slice (seed 42, distinct from train manifest).
- **Easier OOD:** DAPO 2k (`dapo_n2000_seed43`).
- **Hard OOD:** BeyondAIME (100 problems), AIME-25, AIME-26, HMMT *(eval mix still being finalized — see PLAN §4)*.

**Metric:** pass@k for k ∈ {1, 4, 8, 16}. Primary: pass@8 on Polaris (decision-grade) and DAPO; pass@16 on BeyondAIME.

**Harness:** [`probes/checkpoint_rollout_eval.py`](../../probes/checkpoint_rollout_eval.py) — HF checkpoint load → vLLM weight sync → rollouts → train grader (mathd ∨ sympy) on Rank-2 parse.

**Open:** OOD eval should use Math-Verify (format-agnostic) per STANDARDS; current eval uses train grader for consistency with train metrics — note this in writeup and consider rerun if Math-Verify changes verdicts.

**Prompt fairness note:** initial three-arm BeyondAIME used `dapo_answer_v1` (≠ train prompt arm C). Rerun with `hybrid_answer_boxed` is queued; treat current BeyondAIME numbers cautiously.

## 7. Data

- **Source:** [POLARIS-Project/Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) — pre-filtered from DeepScaleR + AReal-boba-Data, 8 difficulty bands.
- **Cleaning:** keep all non-empty `problem` + non-empty `answer`. Do not filter to integer-only (random full-gold n800 matches integer-stratified pass rates under arm C grader).
- **Prompt filter** (decided 2026-05-27, see [`decisions.md`](../decisions.md)): drop if last sentence starts with "Prove" OR (gold appears in prompt AND ("prove" in problem OR contains "show that")). Rejects 4.0% of rows.
- **Train manifest (canonical):** `main/data/polaris_train.jsonl` (51,139 rows) + `polaris_train.meta.json` (frozen).
- **Why Polaris over DAPO:** mentor recommendation + difficulty bands. On the same n800 manifest with arm C, Polaris pass@8 is 33.1% vs DAPO pilot 34.4% — ~1 pp gap, much smaller than the 8 pp scare from the arm-A comparison. See [`probes/dapo_vs_polaris_rollout_comparison.md`](../probes/dapo_vs_polaris_rollout_comparison.md).

## 8. Prompt + parser

- **Train prompt:** `hybrid_answer_boxed` (arm C). Format: `…\n\nAnswer: \boxed{N}`. Locked 2026-05-25 after A/B/C ablation (n=6400 each).
- **Parser:** Rank-2 — hybrid regex (arm C) → last `\boxed{}` → Minerva `Answer:` line. `parse_ok` 55.9% → 87.6% vs Minerva-only on the original Group A n200.
- **Justification:** arm C had highest parse_rate, lowest residual, and 28% higher mixed_reward density vs DAPO arm A — directly improves GRPO signal-per-step. See [`probes/group_a_results.md`](../probes/group_a_results.md), [`probes/prompt_probe.md`](../probes/prompt_probe.md).

## 9. Reward

After Rank-2 extraction, mark a rollout correct iff `grade_answer_mathd(parsed, gold) OR grade_answer_sympy(parsed, gold)` — the rLLM / DeepScaleR rule. SymPy catches all observed mathd-only gaps on n800 probes (0 mathd∧¬sympy); mathd is ~100× cheaper. Reward is 0/1. OOD eval optionally uses Math-Verify (STANDARDS §reward).

---

## Notes / open spikes

- LR sweep — not done; running at Poly-EPO's 1e-6 throughout.
- Diversity metrics in-loop (C4/C4b from PLAN §5) — not yet implemented; needed for "consolation" pass criterion.
- BeyondAIME with hybrid prompt — queued.
- Pass@k bootstrap CI — methodology TBD (PLAN §4 open).
