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
- Both base eval apps completed and stopped:
  - ap-QpdzD6kW95b9FRpJFTTzDI (base math500): stopped at ~19:54 PDT
  - ap-EHXNQSlrprrUz9uYzQOatm (base smallood): stopped at ~19:54 PDT
  
**Note:** Trained-arm apps (grpo, minority, polyepo) have not yet launched. Awaiting launcher invocation or auto-fire trigger.

## Sanity check (n_correct dist + sample tuples per arm)

(populated when each arm's first JSON lands)
