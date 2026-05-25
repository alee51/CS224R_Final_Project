# Open issues

Tracked bugs affecting Group A+ runs.

---

## 1. Phase 2 wandb step collision

**Status:** fixed (`b8fcf24`+ follow-up)  
**Severity:** high (readout)

Phase 1 logs batch scalars with `step=rollouts_done` (up to ~1600). Phase 2 resumed the same run but logged per-judge scalars with `step=judged_count` (1..200). Wandb dropped them (`current step 1602`). Judge Table + volume jsonl OK; `$`/call / wall-clock / cluster histogram panels broken.

**Fix (landed):** `PHASE2_STEP_OFFSET = 2000`; Phase 2 per-call logs use `step=2000 + judged_count`.

**Note:** Full run `ap-c7FATv5JQ8K4BhL5UFlNVM` started before this fix — Phase 2 scalars on that wandb run are still lost. Re-run or read judge metrics from `phase2_judge_results` Table / volume only for that run.

---

## 2. Git SHA unknown in Modal containers

**Status:** fixed  
**Severity:** medium (repro / STANDARDS)

`_git_metadata()` returned `unknown` because only `main/` is mounted — no `.git`.

**Fix (landed):** `launch_probe_a.sh` exports `CS224R_GIT_SHA`, `CS224R_GIT_SHA_SHORT`, `CS224R_GIT_DIRTY` at launch; `_git_metadata()` reads env first.

**Note:** Same full run above logged `unknown` unless relaunched via updated launcher.

---

## 3. `judge_vram_gb_used` always 0

**Status:** fixed  
**Severity:** medium (readout)

`torch.cuda.max_memory_allocated()` returned 0 under vLLM v1.

**Fix (landed):** `_reset_vram_peak()` after LLM init each phase; `_vram_gb_used()` falls back to `nvidia-smi` when PyTorch reports 0.

**Note:** Same full run above used old VRAM path.
