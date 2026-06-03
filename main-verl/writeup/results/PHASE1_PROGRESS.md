# Phase 1 Eval Progress Tracker

Long-running monitor started 2026-06-02 17:13 PDT. Polls abao volume every ~15 min for new `<arm>_step400_<shard>.json` files.

Arms: {base, grpo, polyepo, minority} × Shards: {math500, smallood} = 8 cells total.

## Initial state (cycle 0, 17:13 PDT)
- 6 cs224r-eval-* apps running on abao (state=ephemeral detached)
- 0 JSONs landed in `probes/eval_4b/` (only pre-existing `base_schemaprobe_aime25.json`)
- Per instructions, minority shards may land later; not flagging missing minority in first 30 min.

## Landed JSONs

### Cycle 1 (19:48-19:52 PDT)
**Base arm** (2/2 shards):
- base_step400_math500_math500.json ✓ (61M, 500 prompts, pass@1=0.358)
- base_step400_smallood_combined (1.4G, aggregated OOD eval: aime25, aime26, beyondaime, hmmt_feb25, hmmt_nov25) ✓
- base_step400_smallood_aime25.json ✓ (24M)
- base_step400_smallood_aime26.json ✓ (80M)
- base_step400_smallood_beyondaime.json ✓ (88M)
- base_step400_smallood_hmmt_feb25.json (downloading)
- base_step400_smallood_hmmt_nov25.json (downloading)

**Status:** 5/7 base shards downloaded. Trained arms (grpo, minority, polyepo) have not fired yet (0/6 shards).

## Modal app failures / anomalies

### Cycle 1 (19:54 PDT)

**Base eval (COMPLETE):**
- ap-QpdzD6kW95b9FRpJFTTzDI (base math500): stopped at ~19:54 PDT ✓
- ap-EHXNQSlrprrUz9uYzQOatm (base smallood): stopped at ~19:54 PDT ✓

**TRAINED-ARM EVAL (CRITICAL FAILURE):**
All 4 trained-arm jobs launched at 17:13 PDT but **FAILED** at checkpoint merge stage:

| App ID | Arm | Shard | Launched | Status | Error |
|--------|-----|-------|----------|--------|-------|
| ap-MKqv19OCLLd6BmayhEkXZ0 | GRPO | math500 | 17:13 | FAILED | `EOFError: Ran out of input` when merging FSDP checkpoint. Model weights corrupted/incomplete at `/vol/checkpoints/.../grpo_train_4b_1epoch_lr3e6/global_step_400/actor/model_world_size_*_rank_*.pt` |
| ap-SvWOt3mhqdCW3UfXzep4l9 | GRPO | smallood | 17:13 | FAILED | Same checkpoint corruption |
| ap-whs4s4nOpJIyOxfLpMB65J | PolyEPO | math500 | 17:13 | FAILED | `OSError: Invalid JSON in config.json` at `/vol/checkpoints/.../poly_epo_cot_train_4b_1epoch_lr3e6/global_step_400/actor/config.json` |
| (polyepo smallood log) | PolyEPO | smallood | 17:13 | FAILED | Same config corruption |
| (minority jobs) | Minority | * | (not found in `/tmp/` logs) | UNKNOWN | Job logs not found; check emma account or re-fire |

**Root cause:** Checkpoint relocation from source accounts (anastasia/stonedpinecones/emma) to abao volume **did not complete successfully** — files are truncated/corrupted. Phase 0 step "relocate all 3 trained ckpts to abao via `modal volume cp`" may have failed silently or timed out.

## Sanity check (n_correct dist + sample tuples per arm)

### Base arm (n=500 prompts for math500)
Preliminary stats from `base_step400_math500_math500.json`:
- **pass@1:** 0.358 (179/500 correct on first rollout)
- **pass@8:** 0.806 (403/500 correct on any of first 8 rollouts)
- **pass@64:** 0.928 (464/500 correct on any of 64 rollouts)
- **Reward distribution** (first rollout): mean=0.384 (384/1000 expected on random sample)

**Trained arms:** Blocked by checkpoint corruption. Cannot perform grader sanity checks until GRPO/PolyEPO/Minority checkpoints are verified on abao.

## Exit Criteria & Final Status

**Session started:** ~17:13 PDT (2026-06-02), when base eval jobs launched  
**Cycle 1 timestamp:** 19:54-19:58 PDT  
**Session duration:** ~45 min (well within 5-hour window, but Phase 1 blocked by upstream issue)

**JSONs landed:** 2/8 shards (25%)
- Base math500: 1/1 ✓
- Base smallood: ~1/1 partial ✓ (downloads interrupted, but 1.4GB combined file received)
- GRPO × 2: 0/2 (eval failed at checkpoint merge)
- PolyEPO × 2: 0/2 (eval failed at checkpoint merge)
- Minority × 2: 0/2 (unknown — jobs may not have launched)

**Apps failed:** 4 (GRPO math500, GRPO smallood, PolyEPO math500, PolyEPO smallood) due to corrupted checkpoints  
**Commits made:** 2 (cycle 1 status, critical failure flag)

**Monitoring loop status:** `/loop 15m bash /tmp/phase1_monitor.sh` scheduled but Phase 1 cannot progress until checkpoints are fixed.
