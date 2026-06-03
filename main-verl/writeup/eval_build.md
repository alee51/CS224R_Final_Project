# Eval build plan — runs, accounts, sequencing (LOCKED 2026-06-02)

Concrete implementation plan for the spec in `main-verl/writeup/eval.md`.
Operational state (accounts, budgets, ckpt paths): `main-verl/writeup/MODAL_STATUS.md`.

This doc defines **how we run the evals**. It does not re-define the metrics,
the dataset panel, the scorer, or the sampling config — those live in
`eval.md` and must not be duplicated here.

## Cost estimate

| bundle | runs | est cost |
|---|---|---|
| GEN sweep (Phase 1) | 4 arms × 6 datasets = 24 jobs × ~1 GPU-hr | ~$120 |
| BASE-FWD for KL (Phase 3) | 3 trained arms × 6 datasets = 18 forward jobs × ~0.3 GPU-hr | ~$30 |
| Judge on GRPO training rollouts (Phase 5) | ~5k judge calls, one-time | ~$15 |
| **Total** | | **~$165** |

GPU rate assumed ~$5/hr on B200:1. Judge eval-time is SKIPPED in v1 (no
cross-arm parity).

## Account assignments

**All 4 arms run on `abao`** (workspace `nbao0`, $910 budget). Reasons: anastasia
OOC; stonedpinecones low; one-workspace runs concurrently under abao's 10-GPU
cap; analysis scripts only ever need one set of credentials.

| arm | ckpt origin | ckpt destination |
|---|---|---|
| Base | n/a | loaded directly from HF (`Qwen/Qwen3-4B-Base`) |
| GRPO | anastasia | abao (relocate via `modal volume cp`) |
| Minority | emma | abao (relocate via `modal volume cp`) |
| Poly-EPO | stonedpinecones | abao (relocate via `modal volume cp`) |

## Phase 0 — Pre-flight (no compute, except for one tiny schema probe)

Order:

1. **Patch `main-verl/eval/run_eval.py`:**
   - Add `CS224R_EVAL_LOGPROBS` env var (default 0; production = 20)
   - Pass `logprobs=N` to vLLM `SamplingParams`
   - Save per-token top-N logprobs in the output JSON under `per_prompt[i].logprobs`
   - Add `CS224R_EVAL_BASE=1` mode that skips FSDP merge and loads `Qwen/Qwen3-4B-Base` directly from HF
   - **Extend pass@k ladder to `{1, 2, 4, 8, 16, 32, 64}`** (current code stops at k=32; spec locks k=64). Without this patch the headline table is missing the k=64 column.
2. **Add `main-verl/eval/launchers/base.sh`** mirroring the trained-arm launchers.
3. **Strip `polaris_val` from existing launchers' default `CS224R_EVAL_DATASETS`** — it's still listed but excluded from the locked panel.
4. **Relocate all 3 trained ckpts to abao** via `modal volume cp` — GRPO from anastasia, Minority from emma, Poly-EPO from stonedpinecones. Current local workspace has all source profiles configured. Downloads can take a long time; can run in background.
5. **Update launchers' `CS224R_EVAL_CKPT_PATH`** for relocated arms; pin all to abao paths.
6. **Pull all training-time per-rollout JSONLs locally** (NOT ckpts — those are huge). From each source account into `main/data/probes/per_rollout_v2/<arm>/`. Sizes estimated ~3–16 GB per arm × 3 arms ≈ ~10–50 GB total; should fit on local. Local copy makes Phase 4 analysis scripts (training dynamics, hypothesis gate, cluster-correctness) run without Modal round-trips.
7. **Logprobs schema probe (~$3, 5 min compute).** Before firing the full sweep, run base arm × AIME-25 only at n=8, logprobs=20. Verify `per_prompt[i].logprobs` JSON shape matches what the new analysis scripts (Phase 2, esp. per-rollout token entropy) expect. If shape wrong, fix downstream BEFORE committing to the full 24-run sweep.

## Phase 1 — GEN foundation sweep (24 runs, ~$120)

The single biggest spend. Every other bundle attaches downstream.

**Gating analysis before Phase 1 minority+poly_epo launch:** run the
training-time diff@k split (see Phase 4) as a hypothesis-validation gate. If
the answer-level diversity-on-unsolved-prompts hypothesis already collapses
on saved training data, the minority eval narrative is in trouble — reassess
before spending Phase 1 budget on the set arms.

Per-job command pattern:

```bash
CS224R_EVAL_CKPT_PATH=<ckpt> \
CS224R_EVAL_LABEL=<arm>_step400 \
CS224R_EVAL_DATASETS=aime25,aime26,hmmt_feb25,hmmt_nov25,beyondaime,math500 \
CS224R_EVAL_N_ROLLOUTS=64 \
CS224R_EVAL_LOGPROBS=20 \
bash main-verl/eval/launchers/<arm>.sh
```

### Parallel execution (all 4 arms concurrently on abao)

All 4 arms launch in parallel on abao. Abao's workspace GPU cap is 10
concurrent; default mode uses 4 GPUs (one per arm) and stays well below.

**Default: 1 GPU per arm (4 concurrent jobs).**
Each arm runs one Modal app with `CS224R_EVAL_DATASETS=aime25,aime26,hmmt_feb25,hmmt_nov25,beyondaime,math500`,
processing the 6 datasets sequentially in a single vLLM session.

**Scale-up trigger: if any arm exceeds ~1 hr wall-clock per arm**, split into
2 GPUs per arm (8 concurrent total, still within 10-GPU cap). The bottleneck
is MATH-500 (500 prompts × 64 rollouts = ~32k generations vs ≤6.4k for the
other 5 datasets combined). Cleanest split:

| GPU | datasets |
|---|---|
| A | `math500` only (the long pole) |
| B | `aime25, aime26, hmmt_feb25, hmmt_nov25, beyondaime` (5 small datasets bundled) |

This is **dataset partitioning across separate Modal apps**, not tensor-parallel
(TP=2 within one job). TP communication overhead on B200 makes TP=2 sub-2×;
dataset partitioning is ~2× with zero overhead.

If MATH-500 alone still exceeds ~3 hr per arm, shard MATH-500 itself into
two 250-prompt halves on separate GPUs (3 GPUs per arm × 4 arms = 12 — exceeds
the 10-GPU cap, so run in two waves of 2 arms each).

Outputs land on abao's `main-artifacts` volume at `/vol/probes/eval_4b/<arm>_step400_<dataset>.json`.

**Recommended launch order:** schema probe (Phase 0 step 7) first → if clean,
fire all 4 arms together. No staggering needed; the probe already validated
the pipeline.

## Phase 2 — Tier 1 analysis (free, per-dataset as Phase 1 lands)

Run as soon as all 4 arms' JSONs for a dataset are local. Per `eval.md` §6.1:

```bash
python main-verl/eval/analysis/posthoc/auc_at_k.py main-verl/eval/probes/eval_4b/*.json        # cross-arm AUC@k table (replaces legacy compare.py)
python main-verl/eval/analysis/posthoc/diff_at_k_split.py main-verl/eval/probes/eval_4b/*.json # cross-arm solved/unsolved partition
python main-verl/eval/analysis/posthoc/coverage.py main-verl/eval/probes/eval_4b/*.json        # majority@k, distinct, entropy, coverage
# NEW analysis scripts to write:
#   - AUC@k                               (5 lines, trivial)
#   - Potential@k                         (trivial from n_correct)
#   - Self-BLEU + distinct-n-gram         (~30 min, sacrebleu)
#   - reflective-action frequency         (regex)
#   - diff@k split by solved/unsolved     (group-by)
#   - per-rollout token entropy split     (from saved logprobs)
```

Outputs flow to `main-verl/writeup/results/comparison.md` and (new) sub-files for
diversity / behavioral metrics.

## Phase 3 — KL from base (18 forward jobs, ~$30)

After Phase 1 fully done — UNLESS overlapped with Phase 1 (see below).

### Phase 1 / Phase 3 overlap (saves ~1.5 hr wall-clock)

Phase 3 is **per-(arm × dataset) independent** — each cell only needs THAT
arm's logprobs on THAT dataset, not the rest of Phase 1. So instead of
running Phase 1 → Phase 3 sequentially, fire each Phase 3 cell as soon as
its corresponding Phase 1 cell's JSON lands on abao.

The monitor agent polls every 15 min for new JSONs. When it sees a new
`<arm>_step400_<shard>.json` for a trained arm (not base), it fires
`kl_from_base.py` for that cell immediately. Phase 3 cells run concurrent
with Phase 1's still-in-flight cells, sharing the 10-GPU abao cap:

- Phase 1 at full sweep uses 8 GPUs (4 arms × 2 shards)
- That leaves 2 slots for Phase 3 concurrency
- As Phase 1 cells finish, more slots open up for Phase 3

Wall-clock impact: Phase 3 mostly hides inside Phase 1's tail (math500 is
the long pole). Eval headline + KL diagnostic both land by ~Phase 1 finish
time instead of Phase 1 + 2 hr for Phase 3.

Base arm is excluded (KL(base ‖ base) = 0).

New script: `main-verl/eval/analysis/posthoc/kl_from_base.py`
- Load Qwen3-4B-Base via vLLM with `logprobs=20`
- For each (trained arm × dataset) saved rollout, teacher-force the rollout token sequence through base
- Compute per-token KL using policy's saved top-20 logprobs and base's top-20 logprobs at each step
- Aggregate: per-(arm, dataset) KL distribution + token-position curves

Base arm itself is skipped (KL(base ‖ base) = 0 trivially).

## Phase 4 — Training-time analysis (free, can start anytime)

Already exists for minority; needs parity work for poly_epo + GRPO.

| task | implementation | status |
|---|---|---|
| **Training-time diff@k split by solved/unsolved (HYPOTHESIS GATE for Phase 1)** | new — group per-rollout JSONLs by `n_correct > 0` per prompt per step, count distinct `parsed_answer` in each group | **Validates the load-bearing minority hypothesis ("diversity goes to wrong answers") on data already on disk, BEFORE Phase 1 minority+poly_epo spend.** If the hypothesis collapses here, reassess Phase 1 priority. |
| Refresh `u_correct.py` on final step-400 per-rollout JSONLs | existing | run when minority + poly_epo full JSONLs synced |
| Run `cluster_correctness.py` for poly_epo | existing | adds the parallel-to-minority plot (memory expects ~45–50% rarest-correct vs minority's 35%) |
| Pull W&B aggregate plots for all 4 arms (pass@8, fraction_filtered, actor/entropy, ppo_kl, distinct_clusters_mean, etc.) | new W&B export script or manual | poster figure source |
| GRPO `|U_correct|` | (deferred — see Phase 5) | not blocking; poster v1 plots minority + poly_epo with footnote |

Outputs to (new) `main-verl/writeup/results/training_dynamics.md`.

## Phase 5 — Judge on GRPO training rollouts — **DEFERRED**

Original goal: cluster GRPO's training rollouts retroactively so all 3 arms
share the same `|U_correct|` training-time axes.

**Why deferred (2026-06-02 audit):** rollout text from training is
unrecoverable from any storage. Verified locations: per-rollout JSONLs
intentionally drop text after `_extract_boxed_answer`
(`main-verl/train/objective_minority.py:329`); W&B `log_val_generations`
only has 32 (prompt, completion) pairs per run; the judge service didn't
persist either side; Modal app stdout logs expired (apps no longer in
`modal app list`). Full report: rollout-text hunter agent 2026-06-02 17:35 PDT.

**v1 poster:** plot minority + poly_epo `|U_correct|` trajectory; GRPO with
explicit footnote ("no judge during training → no on-policy cluster IDs").

**Post-deadline options if we want GRPO on the trajectory:**
- (a) Replay step-400 GRPO ckpt on a fixed prompt subset → judge → cluster → single point per arm. ~$5, <30 min. Seed-determinism caveat (vLLM + FSDP not bit-exact).
- (b) Replay ~40 saved ckpts × 3 arms for the full trajectory. ~$70–100 + ~6 GPU-hr.
- (c) Use existing held-out eval rollouts (`/probes/eval_4b/*.json`, full text, 30 AIME-25 × 16 rollouts × 3 arms) and cluster eval-time instead — measures something different (eval distribution) but cleanly cross-arm-matched.

## Build-only nuances (not in eval.md or MODAL_STATUS.md)

Everything that's part of the **spec** (sampling, scorer, metrics, datasets,
grader failure modes, sanity-check protocol) lives in `eval.md`. Everything
about **Modal accounts, ckpts, existing-JSON inventory** lives in
`MODAL_STATUS.md`. This list captures *only* the items that don't fit either:

1. **Base arm is in GEN but excluded from BASE-FWD.** Phase 1 sweeps all 4 arms. Phase 3 KL pass is 3 trained × 6 datasets = 18 jobs only — base scoring its own rollouts is degenerate (KL=0). Don't accidentally loop base into Phase 3.

2. **AIME-25 and AIME-26 problems are fully disjoint** despite shared `problem_id` naming conventions. Don't dedup across them in any analysis script.

3. **Phase 1 hypothesis-validation gate (Phase 4 first row).** Run the training-time diff@k solved/unsolved split BEFORE committing Phase 1 budget for minority + poly_epo. If the answer-level "diversity goes to wrong answers" hypothesis collapses on training data already on disk, reassess Phase 1 priority. Costs ~30 min of analysis, $0.

4. **Per-rollout JSONLs go local for Phase 4** (Phase 0 step 6). Analysis scripts then run without Modal round-trips. Ckpts stay on abao (too big to mirror).

5. **Pre-existing eval JSONs from anastasia/stonedpinecones are discarded for v1** (no logprobs, mixed grader provenance). Phase 1 is full from-scratch; do not splice old numbers in.

6. **The two load-bearing minority diagnostics** are diff@k split solved/unsolved (Tier 1) and per-rollout token-entropy split correct/incorrect (Tier 1, needs logprobs). KL from base (Tier 2) is complementary, not a substitute. See eval.md §6.1–§6.2 for definitions.

19. **Polaris-val and DAPO-slice are excluded** from the headline panel as in-distribution. Strip them from launcher defaults in Phase 0 step 3.

20. **AIME-26 and AIME-25 parquets share a problem_id naming scheme** but the problems are completely disjoint (AIME-26 was Feb 2026). Don't accidentally dedup across them in analysis scripts.
