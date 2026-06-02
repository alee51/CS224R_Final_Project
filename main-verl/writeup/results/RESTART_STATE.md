# Restart state — 2026-06-02 16:22 PDT

Brief autonomous-run log. Next agent / Nancy: read this on return to resume cleanly.

## Done ✓ (committed + pushed)

| commit | content |
|---|---|
| `8b53894` | Eval plan locked: writeup/ moved to main-verl/writeup/, 6-dataset panel, 4 arms, AIME-26 + Minerva parquets, deletions/renames |
| `87d1dd0` | Phase 0a code patches: run_eval.py logprobs+base mode, pass@k ladder to k=64, base.sh launcher, 7 new analysis scripts including kl_from_base.py |
| `06d2610` | Phase 4 hypothesis gate (**SUPPORTED**): at k=8, minority Δ over GRPO is +0.118 on unsolved vs +0.022 on solved; polyepo +0.127 vs −0.002. "Diversity goes to wrong answers." |
| `6354903` | Phase 4 u_correct + cluster_correctness: minority rarest-correct 44.2%, polyepo 46.6%, most-common-correct 77/79% for both. GRPO has trivial |U_correct|=1 (no judge during training). |

**Schema probe (Wave 2) VERIFIED.** Base × AIME-25 × n=8 ran in 84s on Modal,
output JSON has correct `logprobs` field structure: `per_prompt[i].logprobs`
is `list[8 rollouts][~2349 tokens][dict{token_id: logprob}` ×20]. All
run_eval.py patches confirmed working. Modal app `ap-erBRBOQLyxWeunEu0EsOkb`
stopped. Local copy: `/tmp/schema_probe.json`.

## Per-rollout JSONLs local ✓

`main/data/probes/per_rollout_v2/{grpo,minority,polyepo}/` — 257 MB total,
1232 step files, all 3 arms have step_400. Gitignored. Used by Phase 4
analysis (already done above).

## Blocked / NOT done

### Phase 0c ckpt relocations — NEEDS RESTART
- First-attempt `modal volume get` downloads produced 0-byte file stubs after ~9 min stuck.
- All local /tmp/ckpt_* cleaned; all modal procs killed.
- Sources still healthy on respective accounts (anastasia / emma / stonedpinecones); destinations on abao either missing or partial.
- **Restart recipe:** sequential per-arm `modal volume get --force` from source, verify each downloaded file `>1 KB` before `modal volume put --force` to abao, verify abao has `actor/config.json` after.

### Phase 1 GEN sweep — blocked on Phase 0c for 3 trained arms
- Base arm could fire now (no ckpt needed) — but `modal run --detach` with `.remote()` inside may cancel on disconnect per Modal warning. **Do NOT fire during Nancy's disconnect.**
- Once ckpts on abao, fire all 4 launchers in parallel: `base.sh`, `grpo.sh`, `minority.sh`, `polyepo.sh` — all set to abao profile, n=64, logprobs=20, 6-dataset panel.

### Phase 2 Tier 1 analysis — blocked on Phase 1
Scripts written and in `main-verl/eval/analysis/`:
- `compare.py` (existing) — cross-arm pass@k table
- `coverage.py` (existing) — coverage/distinct/entropy
- `auc_at_k.py` (new) — AUC@k scalar
- `potential_at_k.py` (new) — failed-problem ceiling
- `self_bleu.py` (new) — rollout text diversity
- `reflective_actions.py` (new) — regex behavioral metric
- `diff_at_k_split.py` (new) — eval-time solved/unsolved split
- `token_entropy_split.py` (new) — from saved logprobs

### Phase 3 KL — blocked on Phase 1
`kl_from_base.py` written (Modal app, B200:1, enforce_eager=True, vLLM
prompt_logprobs=20). Skips base arm. Outputs to `/vol/probes/kl/`. **Caveat
from Agent A:** "teacher-forcing alignment relies on vLLM `prompt_logprobs`
where slot 0 is None; I align by taking `prompt_logprobs[1:]` and trimming
to `min(len(policy), len(base))`. Math is correct but a few-token offset is
the most likely failure mode — sanity-check on a real run."

### Phase 5 GRPO judge pass — DEFERRED
Per-rollout training JSONLs **do not contain rollout text** (only
`parsed_answer`). Judge service needs full rollout text to cluster. Sourcing
rollouts elsewhere is out of scope. GRPO will be absent from the training-time
`|U_correct|` plot; minority + poly_epo will be plotted on their own axes. The
$15 budget assumed for this is freed.

### Phase 4 remainder
- W&B aggregate plots for all 4 arms (pass@8, fraction_filtered, actor/entropy, ppo_kl, distinct_clusters_mean)
- Output to `main-verl/writeup/results/training_dynamics.md`

## Notes for next agent

- `analysis_io.py` (NOT `_io.py` — stdlib clash) is the shared helper for new analysis scripts.
- `compare.py` output path is `main-verl/writeup/results/comparison.md`.
- `cluster_correctness.py` already gracefully handles GRPO's null cluster_ids.
- `training_diff_at_k_split.py` (new) is the training-time hypothesis-gate analog of `diff_at_k_split.py` (eval-time).
- Avoid touching `main/docs/` (DEAD folder per memory).
- All eval-related docs live in `main-verl/writeup/`.
