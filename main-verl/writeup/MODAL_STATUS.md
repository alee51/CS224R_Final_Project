# Modal status — single source of truth

Operational state for Modal accounts, budgets, and what artifacts live where.
**Do not duplicate this info anywhere else.** Other docs link here; they do
not restate run IDs, ckpt paths, budgets, or inventory.

_Last verified: 2026-06-02 (post step-400 inventory sweep)._

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

These are the surviving outputs from prior eval probes. **Provenance unclear**
(grader version, sampling config) — quarantine until rescored via
`main-verl/eval/analysis/rescore.py` with the current `math.compute_score`
grader. Do NOT cite numbers from these files until rescore confirms.

| account | path | files present |
|---|---|---|
| anastasia | `/vol/probes/eval_4b/` | `grpo_step400_aime25.json`, `grpo_step400_v3_aime25.json`, `grpo_step400_panel_math500.json`, `grpo_step400_extras_hmmt_feb25.json` (+ legacy v1/v2 files) |
| stonedpinecones | `/vol/probes/eval_4b/` | `polyepo_step400_v2_aime25.json`, `polyepo_step400_panel_polaris_val-math500.json`, `polyepo_step400_extras_hmmt_feb25.json`, `polyepo_step400_extras_hmmt_nov25.json`, `polyepo_step400_extras_beyondaime.json` |
| emma | _no `probes/eval_4b/` directory_ | **gap — minority has zero eval outputs saved** |

GRPO has partial OOD coverage but is missing HMMT-Nov, BeyondAIME. Poly-EPO is
the only arm with the full hard-OOD set on disk.

## Eval re-run scoping (what this implies)

Locked plan: see `main-verl/writeup/eval.md` (spec) and `main-verl/writeup/eval_build.md` (runs).

- **Base (4th arm, Qwen3-4B-Base)** runs on abao; no ckpt needed.
- **GRPO** needs ckpt relocation anastasia → abao (anastasia OOC).
- **Poly-EPO** existing eval JSONs have no logprobs → re-run from scratch with `logprobs=20`. Stonedpinecones budget ($17) below sweep cost (~$35); decision pending whether to relocate to abao or top up.
- **Minority** has zero saved eval — full from-scratch sweep on emma.

All 4 arms must clear the same 6-dataset panel under the same sampling + logprobs config.
