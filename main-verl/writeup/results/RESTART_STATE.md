# Restart state — 2026-06-02 ~20:00 PDT

Resume context for fresh post-compact session. Read this first.

## Latest pushed commit

`29f5f6e` Phase 2 (base smallood): grader sanity check blocked — Modal download failure
(NOTE: that agent commit was WRONG — modal volume get DOES work on JSON eval outputs.
The agent had a broken environment. The 5 base smallood JSONs ARE pullable from abao.)

Prior commits this session, in order:
- `8b53894` Eval plan locked (writeup moved to main-verl/writeup/, 6 datasets, 4 arms)
- `87d1dd0` Phase 0a code patches: run_eval.py logprobs + base mode + k=64 ladder + 7 new analysis scripts
- `06d2610` Phase 4 hypothesis gate — SUPPORTED ("diversity goes to wrong answers")
- `6354903` u_correct + cluster_correctness for set arms
- `784888c` W&B plots + training_dynamics.md
- `a3a3f7b` .spawn() + KL alignment fix + 2-GPU partitioned launchers
- `c8b9ab4` Phase 5 deferred (training rollout text unrecoverable)
- `f3ebd6e` Phase 3 overlap with Phase 1 doc
- `4289d0c` run_eval: bump vLLM batching for B200 (max_num_seqs=4096, gpu_mem_util=0.95)
- `9195971`, `2f6079b` Poster plots (by Nancy)

## What works

- **GRPO** is locally merged at `/tmp/merged_grpo_hf/` (~2.1 GB, model.safetensors + config). Verified `torch.load`-clean. Ready to upload to abao.
- **5 base smallood eval JSONs landed on abao** at `/vol/probes/eval_4b/base_step400_smallood_{aime25,aime26,hmmt_feb25,hmmt_nov25,beyondaime}.json`. Pullable via `MODAL_PROFILE=abao modal volume get` (verified). Schema confirmed: pass@k for k∈{1,2,4,8,16,32,64}, per_prompt has rewards/preds/rollouts/logprobs/rendered_prompt/n_correct.
- **Per-rollout training JSONLs local** at `main/data/probes/per_rollout_v2/{grpo,minority,polyepo}/` (gitignored, 257 MB).
- **run_eval.py patched** (committed) to skip Modal-side FSDP merge when ckpt_path contains `config.json + model*.safetensors`. So uploading HF-format ckpts to `merged_hf/<arm>_step400/` on abao means run_eval.py just loads them directly.
- **maxrl model_merger.py** local at `/tmp/maxrl/scripts/model_merger.py`. PYTHONPATH=/tmp/maxrl. Two patches applied: DeviceMesh API fallback + Vision2Seq import bypass.

## What's broken

- **Modal CLI silently corrupts large `.pt` files during transfer** (see memory `project_modal_cli_unreliable_large_pt`). Both minority + polyepo `model_world_size_4_rank_0.pt` files in /tmp at correct size but unparseable by torch.load. Re-downloads produce same corruption.
- **Cannot merge minority + polyepo locally** as long as rank_0 is corrupt.
- **All 3 trainee accounts (anastasia/emma/stonedpinecones) OOC** — can't run a Modal app from them to merge in-cloud.
- **Agents stall on watchdog (600s no stdout)** mid-long-pipeline. Do NOT use agents for long sequential bash workflows. Use direct bash with run_in_background.

## What's running NOW

- `ap-QpdzD6kW95b9FRpJFTTzDI` — base × math500 on abao, running ~3 hr with the OLD slow code (max_num_seqs=256). Will finish eventually. Don't restart — sunk compute.
- `a92b9285327b73719` — Phase 1 health monitor agent, just started, polls abao every 15 min. Should commit incremental analysis as JSONs land.

## Path forward options (Nancy decides)

1. **Try Modal Python SDK** with `modal.Volume.from_name` (not `lookup` — API drifted) + `.read_file()` to bypass the CLI corruption bug. ~10 min to try.
2. **HF Hub intermediate** — push minority + polyepo HF-merged from somewhere with compute, pull on abao. Blocked: no source account has compute.
3. **Poster v1 with base + training-time only.** Most-realistic given the day's failures. Drop trained-arm eval-time numbers from v1 headline; ship "trained-arm OOD eval pending" + the strong training-time story (hypothesis gate SUPPORTED + cluster inversion + W&B plots + token entropy gap).
4. **Re-fire modal volume get for the rank_0 files until one comes down non-corrupt** — unreliable but cheap to script. Use torch.load as the integrity check.

## Quick verification commands for resume

```bash
# What's pullable from abao
MODAL_PROFILE=abao modal volume ls main-artifacts probes/eval_4b/
MODAL_PROFILE=abao modal app list | head -10

# Is GRPO local merge still good
ls -la /tmp/merged_grpo_hf/
python3 -c "import torch; torch.load('/tmp/merged_grpo_hf/model.safetensors')"  # safetensors uses safetensors lib not torch

# Verify a fresh download from emma is corrupt
MODAL_PROFILE=emma modal volume get --force main-artifacts checkpoints/main-verl/minority_cot_train_4b_1epoch_lr3e6/global_step_400/actor/model_world_size_4_rank_0.pt /tmp/test_min.pt
python3 -c "import torch; torch.load('/tmp/test_min.pt', map_location='cpu', weights_only=False)"  # expect EOF/zip error
```

## What the poster CAN show right now from existing data (no more compute)

- Training-time hypothesis gate SUPPORTED (committed)
- Cluster-correctness inversion 44.2% rarest / 77.2% most-common for minority + polyepo (committed)
- |U_correct| training trajectory for set arms (committed)
- W&B aggregate plots for all 3 arms (committed)
- Training-time pass@8: GRPO 43%, Poly-EPO 41%, Minority 37% (committed)
- 285x token-entropy gap minority vs GRPO (committed)
- Base smallood pass@k for 5 OOD datasets (data on abao, analysis not yet committed — run base smallood analysis FIRST when resuming)

## Resume checklist

1. Read this doc + MEMORY.md
2. Check `git log` for any new commits since 29f5f6e
3. Pull 5 base smallood JSONs from abao + run base-only Tier 1 analysis + commit
4. Decide on path forward for trained-arm ckpts (option 1, 2, 3, or 4 above)
