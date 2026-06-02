# Audit findings

Discrepancies and corrections discovered while building the `writeup/`
folder. Filed against the docs in `main/docs/`, `main-verl/eval/`, and the
live poster `poster-overleaf/draft_nancy_claude.tex`.

## A. Doc-vs-code disagreements (action: trust code; fix the doc if reused)

1. **Eval `README.md` claims `n_rollouts=16` is hardcoded.**
   `main-verl/eval/README.md:100` says "`n_rollouts=16` per prompt (configurable
   via `CS224R_EVAL_N_ROLLOUTS`)." This matches the default in
   `main-verl/eval/run_eval.py:46`, but the writeup recommends `n=32` for the
   four small AIME-style datasets to reduce per-problem pass@k variance.
   Action: when launching the headline panel, override `CS224R_EVAL_N_ROLLOUTS=32`
   for `aime25, hmmt_feb25, hmmt_nov25, beyondaime` and leave `=16` for
   `math500`. The probe already supports this.

2. **STANDARDS §"Sequence length cap" overstates the Dr.GRPO claim.**
   `main/docs/STANDARDS.md:50` reads "We do NOT use Dr.GRPO `T_max`
   normalization explicitly; `loss_agg_mode: seq-mean-token-sum-norm` on the
   set arms is the closest we get to it." The closer reading of the verl
   source is that `seq-mean-token-sum-norm` divides by a max-length factor,
   so the set arms *are* effectively doing the `T_max` normalization. The
   GRPO arm uses `seq-mean-token-mean` and is the one that does *not* do it.
   The writeup phrases this as "set arms use `seq-mean-token-sum-norm`,
   GRPO uses `seq-mean-token-mean`" without re-litigating the Dr.GRPO label.

3. **Eval `PLAN.md` §2.2 says "Math-Verify on parsed answer, not train
   grader, on the OOD splits."** This contradicts both `STANDARDS.md:154`
   ("Eval grader matches training grader. This is non-negotiable for the
   writeup") and `run_eval.py:206-210`, which uses
   `verl.utils.reward_score.math.compute_score` (the training grader) on all
   datasets including OOD. The writeup follows the code.
   Action: PLAN.md §2.2 is stale; do not cite it.

4. **Eval `README.md` table omits `n_rollouts=32` for AIME-style.** The
   table at `README.md:108` only lists `k ∈ {1,4,8,16}`. The writeup adds
   the explicit n=32 recommendation for the four small datasets.

5. **STANDARDS.md §"Algorithms" advantage-scale row.** `main/docs/STANDARDS.md:34`
   reports per-prompt advantage scales of `±0.875 / ±0.50 / ±0.125` at
   step 200; the same numbers appear in `eval/PLAN.md:11`. These match the
   "Advantage range" column in W&B but are not derivable from code alone.
   The writeup carries them only as context in the diagnostic section and
   marks them with W&B provenance; they are `[unverified from source code]`
   for the paper.

6. **Eval `README.md` Polaris-val and DAPO are in the dataset registry but
   excluded from the writeup panel.** The user has locked the headline panel
   at `aime25, math500, hmmt_feb25, hmmt_nov25, beyondaime` (see context).
   The code's registry (`run_eval.py:151-159`) still includes
   `polaris_val` and `dapo_slice_3k`; these are runnable but not part of the
   writeup table.

## B. "We do NOT X" → positive statements (flip for writeup)

Every entry in `main/docs/STANDARDS.md` §"What we do NOT do" (`:166-176`)
and `main-verl/eval/README.md` §"What we do NOT use" (`:89-91`) is recast in
the writeup as a positive description.

| original ("do NOT")                                                                 | writeup phrasing                                                                                       |
|---|---|
| "No DAPO seq-mean loss aggregation on GRPO."                                        | "GRPO uses `seq-mean-token-mean`; set arms use `seq-mean-token-sum-norm`."                            |
| "No standard-deviation normalization in GRPO advantage."                            | "GRPO advantage is the raw mean-centered reward, matching Poly-EPO paper Appendix §A."                |
| "No KL penalty in the loss."                                                        | "PPO surrogate uses asymmetric clipping (0.20 / 0.28) without a KL penalty term."                     |
| "No entropy coefficient."                                                           | "Loss is the clipped PPO surrogate without an entropy bonus."                                         |
| "No `math_dapo` reward path."                                                       | "Reward is Hendrycks `is_equiv` applied to the last `\\boxed{...}` (`verl.utils.reward_score.math`)."  |
| "No pre-milestone `grade_parsed_answer`."                                           | "Same paragraph as above — single grader for train and eval."                                          |
| "No `hybrid_answer_boxed` prompt."                                                  | "Prompts are rendered through Qwen3-4B-Base's `apply_chat_template` with the boxed-answer suffix baked into the parquet." |
| eval README: "We do NOT use `math_dapo.compute_score(strict_box_verify=True)`."     | (delete entirely; the positive statement in §3 of `eval.md` is sufficient)                            |
| eval README: "We do NOT use any of the pre-milestone prompt templates."             | (delete entirely; the positive statement in §2 of `eval.md` is sufficient)                            |

## C. Numbers that cannot be verified from source alone (flag for user)

These are reported in docs / live results files but are not derivable from
the YAMLs or kernel code. The writeup carries them as live diagnostics with
explicit provenance and does **not** treat them as locked-in claims.

1. **Per-prompt advantage scales** (`±0.875 / ±0.50 / ±0.125`) —
   `STANDARDS.md:34`, `PLAN.md:11`. Source: W&B snapshot, step 200.
2. **`train/fraction_filtered`** values (0.53 / 0.15 / 0.20) —
   `PLAN.md:11`. Source: W&B.
3. **Cluster-correctness table in `minority_diagnostic.md`** (P(rank-r cluster
   correct)) — derived from per-rollout JSONLs by
   `main/scripts/analyze_minority_vs_grpo.py`. The script is committed; the
   numbers should be regenerated with `--sample-every 5` before the final
   report. Currently uses sample-every-10.
4. **Token-level entropy 80–200× gap** — `minority_diagnostic.md:74-86`.
   Source: W&B `actor/entropy`. The 80–200× claim depends on which step
   range is summarized; the writeup repeats it only with explicit
   step-200 anchoring.
5. **`degenerate_rollouts ≈ 250-280 / 1024`** — `PLAN.md:22`. Source: W&B.

## D. Poster contradictions with the audit (flag only — do not edit `.tex`)

Findings against the live poster
`poster-overleaf/draft_nancy_claude.tex` (most recent .tex by mtime,
2026-06-02 12:37):

1. **`draft_nancy_claude.tex:108`** describes the reward as
   "binary (mathd $\\vee$ SymPy on `\\boxed{}`)." This is the pre-milestone
   1.7B path. The 4B production reward is **Hendrycks `is_equiv`** via
   `verl.utils.reward_score.math.compute_score`
   (`run_eval.py:206`, `STANDARDS.md:67-77`). The poster sentence should
   match the writeup's `training.md` §4.

2. **`draft_nancy_claude.tex:107`** lists the judge as "64-way async."
   The actual config is `judge_http_batch_size: 64` with
   `judge_concurrency: 2` (`minority_cot_train_4b_1epoch.yaml:64-66`). "64-way
   async" reads as 64 concurrent connections; production is 2 connections
   carrying batches of 64 each.

3. **`draft_nancy_claude.tex:103`** says "Qwen3-1.7B-Base (preliminary),
   Qwen3-4B-Base (main, VeRL)" — fine — but `:143-144` then says
   "Preliminary 1.7B sanity (Polaris-51K, $n_\\text{rollouts}{=}16$): all
   three arms beat the base model; qualitative ordering GRPO ≥ Min. ≥ Poly-EPO
   held on Polaris-2k, DAPO-2k, MATH-500." That 1.7B ordering is from the
   pre-milestone pipeline (different reward function, different prompt
   template). The writeup omits the 1.7B comparison; if the poster keeps it,
   it needs a caveat that the 1.7B reward path is `mathd∨sympy` rather than
   `is_equiv`.

4. **`draft_nancy_claude.tex:131`** displays the minority subset score as
   `(1/|min-cluster(G)|) Σ_{i ∈ min-cluster(G)} r_i`. This is the
   *mean reward of rollouts in the rarest cluster* — which matches the
   kernel (`objective_minority.py:528`: `return float(rewards4[mask].mean())`).
   The formula is mathematically right but the prefactor `1/|min-cluster(G)|`
   is the implicit division inside `.mean()`. No correction required; this
   is fine as poster shorthand.

5. **`draft_nancy_claude.tex:136`** writes the marginal advantage as
   `(1/35) Σ_{G ∋ i} f(G) − (1/70) Σ_G f(G)`. This matches the kernel
   exactly (`objective_minority.py:502-506`). No correction.

6. **`draft_nancy_claude.tex:153-159`** Minority-CoT cells are all `[TODO]`,
   consistent with the writeup's `pending` status. No correction.

7. The poster's "Approach" block describes only Minority-CoT in detail. The
   Poly-EPO-CoT subset score formula on `:92-94`
   (`f_{poly}(G) = (1/|G|) Σ r_i · d(G)/|G|`) is correct and matches
   `objective_poly_epo.py:43-55`. No correction.

## E. Items the user should confirm before submission

1. Whether to commit the `|U_correct|@k` eval-time variant — requires
   running the judge over saved eval rollouts for all 3 arms (set arms
   need it for parity with GRPO, which had no judge during training).
   Estimated ~$10–15 of judge compute for a full AIME-25 panel across
   3 arms.
2. Whether to re-run the cluster-correctness diagnostic with
   `--sample-every 5` for the final paper figure (currently
   `--sample-every 10`).
3. Whether to leave Poly-EPO's `hmmt_*` / `beyondaime` numbers as-is in the
   poster (currently the only arm with full hard-OOD coverage) or wait for
   GRPO and Minority-CoT to land before publishing the cross-arm table.
