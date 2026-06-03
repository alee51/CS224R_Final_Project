# Modal status — single source of truth

Operational state for Modal accounts, budgets, and what artifacts live where.
**Do not duplicate this info anywhere else.** Other docs link here; they do
not restate run IDs, ckpt paths, budgets, or inventory.

_Last verified: 2026-06-02 (base smallood + math500 landed on abao; trained-arm eval still blocked)._

## Accounts

| profile | workspace | role | budget status (as of 6/2, 2:06pm ET)|
|---|---|---|---|
| `chicken602` | chicken602 | default (Nancy) |  |
| `anastasia` | alee72 | trained GRPO | **OUT OF CREDITS** — cannot run more compute on this account |
| `emma` | emmagao | trained Minority-CoT | ok |
| `stonedpinecones` | stonedpinecones | trained Poly-EPO-CoT + judge | $17.37 |
| `abao` | nbao0 |  | $910.16 |

GRPO ckpt lives on Anastasia's account, but evals must be done on a different modal account.

## Step-400 checkpoints

All three arms reached step 400. Verl paths:

| arm | account | ckpt path | training run_id (W&B) |
|---|---|---|---|
| GRPO | anastasia | `/vol/checkpoints/main-verl/grpo_train_4b_1epoch_lr3e6/global_step_400/actor` | `rof8t8kf` |
| Minority-CoT | emma | `/vol/checkpoints/main-verl/minority_cot_train_4b_1epoch_lr3e6/global_step_400/actor` | `yfpxs7wo` |
| Poly-EPO-CoT | stonedpinecones | `/vol/checkpoints/main-verl/poly_epo_cot_train_4b_1epoch_lr3e6/global_step_400/actor` | `m29o33k1` → `4x6ywtp7` (Modal.Retries split; stitch when plotting) |

Launcher `CS224R_EVAL_CKPT_PATH` values in `main-verl/eval/launchers/*.sh`
already point at these paths.

## Training-time per-rollout JSONLs

`/vol/per_rollout/<run_id>/step_<N>.jsonl` on each arm's account. All three
arms have step 50 → 400 complete; minority also has 260, 270 from
crash-recovery cycles.

Naming gotcha: poly_epo's per_rollout dir is `/vol/per_rollout/unknown_run/`,
NOT named by run_id like the others. Don't assume run_id-named paths everywhere.

## Held-out eval JSONs already on Modal

### Quarantined (pre-locked-config, provenance unclear)

Surviving outputs from prior eval probes. Grader version + sampling config
unknown — quarantine until rescored via `main-verl/eval/analysis/posthoc/rescore.py`
with the current `math.compute_score` grader. Do NOT cite numbers from these
files until rescore confirms.

| account | path | files present |
|---|---|---|
| anastasia | `/vol/probes/eval_4b/` | `grpo_step400_aime25.json`, `grpo_step400_v3_aime25.json`, `grpo_step400_panel_math500.json`, `grpo_step400_extras_hmmt_feb25.json` (+ legacy v1/v2 files) |
| stonedpinecones | `/vol/probes/eval_4b/` | `polyepo_step400_v2_aime25.json`, `polyepo_step400_panel_polaris_val-math500.json`, `polyepo_step400_extras_hmmt_feb25.json`, `polyepo_step400_extras_hmmt_nov25.json`, `polyepo_step400_extras_beyondaime.json` |
| emma | _no `probes/eval_4b/` directory_ | **gap — minority has zero eval outputs saved** |

GRPO has partial OOD coverage but is missing HMMT-Nov, BeyondAIME. Poly-EPO is
the only arm with the full hard-OOD set on disk.

### Fresh (locked-config, abao)

Base arm (Qwen3-4B-Base) full panel produced under the locked spec
(`n=64`, `logprobs=20`, `math.compute_score` grader). Citable.

| account | path | files present |
|---|---|---|
| abao | `/vol/probes/eval_4b/` | `base_step400_math500_math500.json`, `base_step400_smallood_aime25.json`, `base_step400_smallood_aime26.json`, `base_step400_smallood_hmmt_feb25.json`, `base_step400_smallood_hmmt_nov25.json`, `base_step400_smallood_beyondaime.json`, `base_step400_smallood_<all-five>.json` (combined, 1.4 GB), `base_schemaprobe_aime25.json` |

Locally pulled so far: `/tmp/base_aime25.json` (1.94 GB, full file with
logprobs). Other 4 smallood shards downloaded on abao but not yet analyzed
into `writeup/results/`; only `base × aime25` has pass@k + AUC@k extracted
(see `writeup/results/auc_at_k.md`).

### Trained-arm fresh eval — BLOCKED

GRPO, Minority-CoT, Poly-EPO-CoT have **no fresh-config eval JSONs** yet.
Phase 1 fired all 3 × 2 shards on abao at 17:13 PDT; all 4 GRPO/Poly-EPO jobs
failed at the FSDP-merge step (corrupted `model_world_size_*_rank_*.pt` after
Modal CLI transfer to abao; Minority jobs status unknown). See
`writeup/results/PHASE1_PROGRESS.md` and memory
`project_modal_cli_unreliable_large_pt`. Workaround in progress: HF-merge
locally and upload merged weights to abao (GRPO already at
`/tmp/merged_grpo_hf/`, ready to push).

## Eval re-run scoping (what this implies)

Locked plan: see `main-verl/writeup/eval.md` (spec) and `main-verl/writeup/eval_build.md` (runs).

- **Base (4th arm, Qwen3-4B-Base)** runs on abao; no ckpt needed.
- **GRPO** needs ckpt relocation anastasia → abao (anastasia OOC).
- **Poly-EPO** existing eval JSONs have no logprobs → re-run from scratch with `logprobs=20`. Stonedpinecones budget ($17) below sweep cost (~$35); decision pending whether to relocate to abao or top up.
- **Minority** has zero saved eval — full from-scratch sweep on emma.

All 4 arms must clear the same 6-dataset panel under the same sampling + logprobs config.
