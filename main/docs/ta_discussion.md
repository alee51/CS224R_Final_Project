# TA office-hours agenda

**Date:** 2026-05-27 (deadline: poster 2026-06-03, experiments target 2026-05-31).
**Goal:** Get advice on the open questions in §3 — not status updates.

---

## 1. Where we are

**Three arms training in parallel on B200, all fresh runs launched 2026-05-27:**

| Arm | Workspace | Wandb | Step / 799 | $/step | min/step |
|-----|-----------|-------|-----------|--------|----------|
| GRPO | alee72 (Anastasia) | [t11jct0t](https://wandb.ai/224r-project/cs224r-minority-voting/runs/t11jct0t) | ~326 | ~$0.22 | ~2.1 |
| minority_answer | alee72 | [o5ypkzja](https://wandb.ai/224r-project/cs224r-minority-voting/runs/o5ypkzja) | ~148 | ~$0.43 | ~4.1 |
| poly_epo_answer | chicken602 (Nancy) | [fdx95beu](https://wandb.ai/224r-project/cs224r-minority-voting/runs/fdx95beu) | ~146 | ~$0.46 | ~4.2 |

**First three-arm checkpoint eval (canonical) ran 2026-05-27 evening on B200**, mid-training (GRPO at step 299, set-arms at step 133):

| Slice | base | GRPO Δ | minority Δ | poly Δ |
|-------|------|--------|------------|--------|
| Polaris 2k pass@8 (training distribution) | 0.306 | +1.1 pp | **+0.1 pp** | +1.3 pp |
| DAPO 2k pass@8 (easier OOD) | 0.248 | +2.4 pp | +0.5 pp | +1.5 pp |
| **BeyondAIME pass@16 (hard OOD)** | **0.130** | **−6.0 pp** | **−6.0 pp** | **−5.0 pp** |

**Budget:** Just received +$1500 stretch credit ($500/person). Pre-stretch: ~$381 (chicken602) + ~$340–420 (alee72) ≈ ~$721–801 team. With stretch: ~$1221–1301 team. One epoch of all three arms costs ~$686; two epochs ~$1574.

**Blockers / risks:**
- BeyondAIME used `dapo_answer_v1` prompt (≠ train prompt arm C) — confound for hard-OOD regression.
- H200 GRPO line on alee72 crashed at step 537 (`pcas3emd`); fresh B200 lines are isolated.
- minority arm is the headline hypothesis and is **flat** at step 133.

## 2. What we've learned (locked decisions, citable from paper)

- **Prompt:** arm C (`hybrid_answer_boxed`, `Answer: \boxed{N}`) — +28% mixed_reward density vs DAPO control. Source: [`probes/group_a_results.md`](./probes/group_a_results.md), [`probes/prompt_probe.md`](./probes/prompt_probe.md).
- **Parser:** Rank-2 (hybrid regex → last `\boxed{}` → Minerva `Answer:` line) — `parse_ok` 55.9% → 87.6%.
- **Reward grader:** mathd ∨ sympy from rLLM/DeepScaleR — see [`decisions.md`](./decisions.md) §2026-05-26.
- **Train data:** Polaris 51,139 prompts (prompt-filter rejects proof endings + gold-leak). 1 pp behind DAPO pilot on n800, but mentor recommendation + difficulty bands. See [`probes/dapo_vs_polaris_rollout_comparison.md`](./probes/dapo_vs_polaris_rollout_comparison.md).
- **Batch size:** bs=64 on single H200/B200 is the wall (bs=128 OOMs in `_completion_logprobs_hf`).
- **Hardware:** B200 is ~2× faster than H200 on GRPO, ~1.5–1.8× on set arms (paired smoke). $/step often wins despite higher $/s. Source: [`timeline.md`](./timeline.md) §2026-05-27 GRPO smoke H200 vs B200.
- **Efficiency dead-end:** `vllm_sleep=1 + gc_off` does not unlock bs=128; allocator OOMs at device cap. [`efficiency/B200_sleep_gc_off_give_up_2026-05-27.md`](./efficiency/B200_sleep_gc_off_give_up_2026-05-27.md).

## 3. Open questions (in priority order)

### Q1. The minority hypothesis looks unsupported. What's the paper story?

At step 133 (~1/6 epoch), minority_answer is +0.1 pp on Polaris and +0.5 pp on DAPO. PLAN's success criterion was "minority > GRPO on pass@4 and pass@16 on at least 2 of the held-out evals." We're not on track for that.

- (a) Is one epoch enough to falsify the hypothesis at this scale (1.7B), or do we need to keep training? The headline arms keep training, but slope is shallow on a flat curve.
- (b) PLAN has a "consolation" pass criterion — *matches GRPO on pass@1 AND improves cluster diversity*. We haven't measured diversity yet (C4/C4b in §5 Training-time reporting is unimplemented). Should we pivot eval effort there instead of more pass@k?
- (c) If we have to publish a negative result, what's the standard practice in this area for framing "well-motivated but not supported at this scale"?

### Q2. BeyondAIME regression: real or eval-prompt artifact?

All three trained arms drop 5–6 pp vs base on pass@16. The eval used `dapo_answer_v1` while training was `hybrid_answer_boxed`. Trained models likely emit boxed answers; the eval parser may be biased toward `Answer:`-style.

- (a) Is re-running BeyondAIME with the matching `hybrid_answer_boxed` prompt the right next step before believing the regression?
- (b) Even with prompt-fair eval, base > trained on hard OOD is publication-worthy if real — is there prior work we should cite for this pattern?

### Q3. With +$1500 stretch credit, what buys the most paper signal?

Roughly we can afford one of:

- (i) **2 epochs of all three arms** (~$813 over current 1-epoch budget): tests whether the minority gap closes with more training.
- (ii) **More evals at current ckpts**: BeyondAIME with hybrid prompt + AIME-26 + HMMT + diversity metrics on saved rollouts.
- (iii) **A 4th arm** (minority_CoT — requires in-loop judge) — was deprioritized; ~$430/epoch + judge cost.

Best use of credit?

### Q4. Scope cut for the poster

Poster is 2026-06-03. Internal target was experiments done 2026-05-31. If we run out of time on (i)–(iii), what's the minimum we can defend?

- (a) Three-arm Polaris + DAPO pass@8 at one ckpt — already in hand.
- (b) Plus BeyondAIME with fair prompt.
- (c) Plus one diversity panel.

Cut order if anything has to go?

## 4. Out of scope for this meeting

- Implementation details (clustering, trainer loop, configs) — not blocked.
- Probe re-runs — frozen, decisions locked.
- Compute/efficiency tuning — at acceptable speed; sleep+gc_off explored and ruled out.

---

## Notes from meeting

- [fill in during meeting]

## Action items

- [ ] [Owner — what — by when]
