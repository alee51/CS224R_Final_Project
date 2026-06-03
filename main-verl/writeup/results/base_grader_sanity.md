# Base Arm Eval — Grader Sanity Check (BLOCKED)

**Status: UNABLE TO COMPLETE — Modal Download Failure**

## Summary

Attempted to pull all 5 base-arm smallood eval JSONs from abao Modal account per eval.md §8 sanity-check protocol.

## Findings

### Modal File Inventory (✓ confirmed files exist)
All 5 required JSONs are present on Modal abao account `main-artifacts` volume:
- `probes/eval_4b/base_step400_smallood_aime25.json`
- `probes/eval_4b/base_step400_smallood_aime26.json`
- `probes/eval_4b/base_step400_smallood_hmmt_feb25.json`
- `probes/eval_4b/base_step400_smallood_hmmt_nov25.json`
- `probes/eval_4b/base_step400_smallood_beyondaime.json`

(Verified via `modal volume ls main-artifacts probes/eval_4b/ | grep base_step400_smallood`)

### Download Failure (✗)

Multiple download strategies attempted:

1. **`modal volume get` CLI with loop** — created empty 0-byte files, process hung indefinitely
2. **Sequential `modal volume get` with --force flag** — same result
3. **Python subprocess wrapper calling `modal volume get`** — returned zero status but files remained 0 bytes
4. **Modal Python SDK** — Version 1.4.3 does not support `Volume.lookup()` API for direct file access

**Symptom:** `modal volume get` CLI command creates placeholder files but downloads stall. No error messages; process runs but does not complete. Killing the process (SIGTERM) does not prevent file creation with 0 bytes.

## Impact

Cannot run mandatory pre-headline sanity checks:
1. ~~n_correct distribution histogram~~
2. ~~Sample tuples: (problem_id, gt, pred[0], reward[0])~~
3. ~~Rescore validation: math.compute_score vs math_dapo (threshold: <10% disagreement)~~

**Tier 1 analysis blocked** until JSONs are available locally.

## Recommendation

1. Verify Modal abao account credentials and workspace access
2. Check if file permissions or volume mount issues exist on Modal side
3. Try downloading via:
   - Alternate credentials (if multiple are configured)
   - Direct Modal app that reads the files and streams to stdout
   - Modal Volume API without CLI (if Python SDK supports read operations)
4. Confirm eval runs completed successfully on abao (check Modal app logs for step-400 completion status)

## Next Steps (User Action Required)

**Do not proceed with push/commit until grader sanity check passes.** Per eval.md §8, publishing pass@k without sanity checks exposes risk of silent-grader bugs (user has been burned by 2 in the past).

---

_Generated: 2026-06-02 19:49 PDT_
