# Main-Runs Playbook

**STATUS: POST-MORTEM REFERENCE (First Pilot Matrix, 2026-05-19, FAILED)**

The first pilot matrix launched on 2026-05-19 was killed after structural failures
(cost overrun, no mid-run durability, logging gaps, substrate parser bug). This doc
describes that matrix and lessons learned. **It is NOT the playbook for the next
pilot attempt.**

**For guidance on the Stage 1 redesign and next pilot launch:** see
`./PILOT_REDESIGN.md`.

**For root-cause synthesis:** see `../analysis/0519_perf_consolidated.md`.

**For chronological incident log:** see `../incidents/0519-11` through `0519-25`.

---

Last updated: 2026-05-19

This doc originally existed as a checklist for **planning main runs after the
first pilot**. The first pilot failed structurally, so the prescriptive guidance
has been superseded. The sections below are preserved as **post-mortem
observations and lessons learned** — they document three recurring issues that
nearly broke the pilot and are likely to recur if not addressed up front in
future iterations:

1. **A100 only ~40% utilized** during pilot decode (perf antipattern).
2. **Modal billing on a personal workspace** — at first launch this was treated as
   an ops mistake vs. a planned shared team workspace. **Superseded (2026-05-19):**
   Stage 1 intentionally uses **personal Modal workspaces per operator**; see
   `nancy_explore/narrative/decisions.md` and `./PERSONAL_WORKSPACE_COLLAB.md`.
3. **OOM "lever not wired" antipattern** — overnight runs crashing repeatedly
   while a knob that didn't actually affect peak memory was being tweaked
   (debugging antipattern).

Scope of this doc: **post-mortem observations and lessons** from the first matrix.
This is not prescriptive for the next attempt. Background and diagnoses live in
`../incidents/` and `../decisions/`. Pointers below.

Related docs:
- `./RUNBOOK.md` — frozen pilot scope (do not edit without orchestrator sign-off)
- `../decisions/training_parallelization_plan.md` — current parallelization plan (P0–P3)
- `../incidents/0519-12_grpo-oom-root-cause.md` — full memory math + ordered fixes
- `../incidents/0519-14_main-run-preemption-no-resume.md` — preemption, no mid-run checkpoint, preds wipe on restart
- `../incidents/0519-22_main-matrix-operator-notes.md` — detached launch ops: timing, mid-run pull, Modal UI containers
- `../incidents/0519-23_per-app-gpu-chart-spike.md` — per-app GPU chart 1→2 (preemption overlap)
- `../incidents/0519-24_modal-observability-budget-gaps.md` — billing API, no wandb, YAML budget enforcement gaps
- `../incidents/0519-25_blocking-launch-client-abort.md` — never use blocking `modal run` for long jobs
- `../incidents/0519-11_grpo-smoke-debug-history.md` — chronological "what we tried" ledger
- `../decisions/efficiency_parallelization_note.md` — superseded; historical
- `../decisions/decision_memo.md` — pilot decision token (PENDING)

---

## 1) A100 utilization — post-mortem: what was wired, why it was still only ~40%

### What is actually wired in the code today

Verified against the current source (commits before this doc was written):

- `pilot/train/rollout_engine.py` — `batch_generate_rollouts()` exists and
  chunks prompts.
- `pilot/train/hf_grpo_train.py:393` — `_train_step_microbatch_backward` calls
  `.backward()` **per micro-batch**, freeing each chunk's autograd graph before
  the next forward. This is Tier 1 from the OOM analysis. **It is wired into
  `run_grpo_training()`.**
- `pilot/train/hf_grpo_train.py:846` — `policy.gradient_checkpointing_enable()`.
- `pilot/infra/modal_app.py:57` — image env sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Net effect: the original OOM bug is gone. Pilot runs at
`batch_prompts=32, rollouts=8, max_new_tokens=2048` fit in 80 GB with margin.

### Why utilization is still ~40% on the A100

The dominant cost is **decode** (`model.generate`), and decode is still
effectively serial-per-prompt:

- Per-prompt seed is `seed + i` (different seed per row in a chunk).
- `pilot/train/rollout_engine.py` defaults `allow_seeded_prompt_batching: false`.
- Under that default, `batch_generate_rollouts()` chunks prompts but the inner
  loop falls back to per-prompt `model.generate(...)`. The chunk size doesn't
  actually parallelize decode across prompts.

Secondary contributors (smaller, listed for completeness):
- `pilot/train/hf_grpo_train.py` still does `F.log_softmax(...)`-then-index on
  the full `[B, L, V]` tensor in completion-logprob forwards. T2.a from the OOM
  doc (use `gather` on logits) was never landed. Saves ~1 GB headroom per
  micro-batch in the main config; not throughput-load-bearing but cheap to do.
- No prompt-length bucketing before `generate` (P2-1 in the parallelization
  plan). Helps only once true batching is on.

### Lessons for the next pilot: measured bottlenecks and known opt opportunities

The pilot achieved 40% A100 utilization despite multiple knobs in place. Here's
what was true, what was wired, and what remains low-hanging fruit for the next
attempt:

**What was true in the first pilot:**
- `_train_step_microbatch_backward` (per-micro-batch gradient freeing) was wired and working. Original OOM bug was gone.
- `policy.gradient_checkpointing_enable()` was wired.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set.
- At the first pilot's config (`batch_prompts=32, rollouts=8, max_new_tokens=2048`), memory was safe.

**Why utilization was still ~40%:**
The dominant cost is decode (`model.generate`), and decode remained **effectively serial-per-prompt**:
- Per-prompt seed is `seed + i` (different seed per row).
- `allow_seeded_prompt_batching: false` was the default.
- `batch_generate_rollouts()` chunked prompts but the inner loop fell back to per-prompt `model.generate()`.

**Known optimization opportunities (for the next pilot):**
The `PILOT_REDESIGN.md` §4.B addresses these under the "perf bundle":
1. **Enable seeded prompt batching** — expected rollout time ~26 min → ~10-12 min.
2. **Raise `completion_logprob_micro_batch_size`** (with per-mb backward) — expected train time ~73 min → ~40-48 min.
3. **FlashAttention-2** — expected ~15-25% across both phases.
4. **Fused AdamW** — expected ~10-15% on optimizer step.

All four are folded into Stage 1's redesign spec. See `./PILOT_REDESIGN.md` §4.B for
acceptance criteria, fallback logic (grad checkpointing OOM handling), and implementation order.

---

## 2) Modal workspace — post-mortem: first pilot on personal workspace

> **Supersession (2026-05-19):** The team will **not** migrate to a shared Modal
> team workspace for Stage 1. Each operator uses their **personal** workspace
> (~$400/teammate credits; operator ~$600). Modal credits do not transfer across
> workspaces. Decision record: `nancy_explore/narrative/decisions.md` (2026-05-19). Ops
> cheat sheet: `./PERSONAL_WORKSPACE_COLLAB.md`. Current spec: `./PILOT_REDESIGN.md`
> §2 ("Infra discipline").

### What happened in the first pilot

The first pilot matrix ran entirely on `chicken602` (personal workspace). At the
time, early docs treated that as a mistake vs. a planned shared team workspace;
the matrix failed for structural reasons (cost, durability, logging) before any
team-workspace migration happened.

`modal profile list` confirmed: only `chicken602`. Spawn manifests under
`pilot/artifacts/matrix_logs/` show URLs like
`https://modal.com/apps/chicken602/main/ap-...`.

### Key gotcha (still true): volumes and secrets are workspace-scoped

Modal volumes and secrets are **per workspace**, not global to the GitHub org:

- `pilot/infra/modal_volumes.py` — `ARTIFACTS_VOLUME_NAME = "pilot-artifacts"`,
  `HF_CACHE_VOLUME_NAME = "hf-cache"`. Each operator's profile has its own pair.
  Another teammate's volume does **not** see your artifacts.
- `pilot/infra/modal_app.py` — `modal.Secret.from_name("huggingface")` and
  `wandb-api-key` must exist on **the profile you launch from** or runs fail at
  `from_pretrained` / wandb init.

**Historical note:** An early redesign draft required `MODAL_PROFILE=team` and
`modal profile activate team` before the matrix. That requirement is **withdrawn**.
Do not switch profiles expecting shared pilot state — use artifact pull + agreed
off-Modal sharing (HF Hub, Drive, git LFS) instead.

### Stage 1 collaboration model (current)

- **Launch:** `modal profile current` → your personal profile; detached runs only.
- **During/after run:** `pull_run_artifacts.py` into your local repo clone; optional
  mid-run volume pull per `../incidents/0519-22_main-matrix-operator-notes.md`.
- **Share weights/checkpoints:** HuggingFace repo, shared drive, or git LFS (mind
  size limits) — not cross-workspace Modal volumes.
- **Metrics:** wandb project `cs224r-minority-voting`; include operator in run name.

---

## 3) OOM debugging — post-mortem: the "lever not wired" antipattern

The pilot wasted real hours on this. The lesson generalizes and is critical for
the next iteration.

### The antipattern that almost broke the pilot

Overnight runs OOM'd repeatedly. The debugging loop kept reducing the same YAML
knob (`completion_logprob_micro_batch_size`), going 16 → 8 → 4 → 2, plus
shrinking `batch_prompts` and `max_new_tokens`. Every reduction still OOM'd.

**Root cause:** The differentiable completion-logprob path was accumulating every
micro-batch's autograd graph in a Python list and only calling `.backward()` once
at the very end of the step. Peak memory scaled with the *total* completion count,
not with the *per-iteration* micro-batch size. Reducing `completion_logprob_micro_batch_size`
actually made things worse on the iteration-count axis while the ceiling stayed
pinned at ~80 GB.

**The knob being tweaked was not on the critical memory path.** Full analysis in
`../incidents/0519-12_grpo-oom-root-cause.md` §4 (memory math).

### How to avoid this antipattern in the next pilot

Before you shrink a knob to fix an OOM:

1. **Answer this question:** *Which line of code allocates the tensor that fails to fit?*
   If you can't name the line, the knob is probably not on the critical path.
2. **Use the memory math from `../incidents/0519-12_grpo-oom-root-cause.md` §4.**
   The analysis predicts peak memory for any given config. Compare predicted peak
   to observed peak. If they disagree, there is a *new* bug.
3. **Only after math agrees with observation** should you reduce a knob, and only
   the knob that the math says is on the critical path.

### Smoke gate for the next pilot

`PILOT_REDESIGN.md` §6 prescribes a **32-prompt smoke** end-to-end before the
matrix launches:
- 5 training steps, single A100, forced preemption mid-step-3.
- Pass criteria: step 1 completes in <60 min, peak `nvidia-smi` < 60 GB, mechanism
  checks pass.
- **Do not launch the matrix without passing the smoke.** This is the highest-leverage
  insurance against silent OOM loops.

For detailed smoke spec, see `./PILOT_REDESIGN.md` §6.

---

## 4) Lessons archive — technical details from the first pilot

This section preserves technical observations from the first pilot that do not
roll into Stage 1's redesign but remain valuable as post-mortem reference.

### Observed failure modes (first pilot only)

**Cost blowout.** Measured ~99 min/step × 100 planned steps × 4 runs ≈ ~$1,275,
against an intended ~$210 pilot budget and a $1,400 team total. The pilot was
never affordable as written. Stage 1's redesign budget-caps individual runs to
$50 and the matrix burst to $150 for three GRPO runs (`pilot_total` $200 ceiling; see `PILOT_REDESIGN.md` §2). Run 0 is not in the redesign matrix.

**No mid-run durability.** `artifacts_volume.commit()` ran only in the `finally` block;
no per-step checkpoint; preemption produced zero salvageable weights. `run1_grpo`
entered a death spiral: preempt → restart → bootstrap wipes `raw_predictions.jsonl`
→ replay step 1 → preempt mid-step-2 → repeat. Stage 1 prescribes time-gated
checkpointing (Branch A in §4.A of `PILOT_REDESIGN.md`) and resume-on-boot logic.

**Logging gaps.** Completed-step milestone math broke (`done % 25 == 0` with mb=8);
first log fired 200/500 instead of 25/500. No mid-rollout heartbeat. No wandb.
Modal volume not committed mid-run, so `volume get` returned stale data.
Stage 1 prescribes per-rollout and per-step diagnostics + wandb (Branch C in
§4.C of `PILOT_REDESIGN.md`).

**Substrate parser bug.** `canonicalize_answer` is documented broken
(`nancy_explore/narrative/decisions.md` 2026-05-18: "strips all `}` and breaks LaTeX").
Salvaged step-1 data showed `"12"` and `"\\( 12 \\)"` in different exact-match
clusters — a known bug, not a new finding. Stage 1 includes a rewritten
canonicalization (Branch C, §4.C.3 of `PILOT_REDESIGN.md`).

### What the next pilot (Stage 1) addresses

See `./PILOT_REDESIGN.md`:
- **§2**: Locked constraints (scope, budget, step count, eval, infra).
- **§3**: End-to-end pipeline (3 implementation branches, smoke, matrix launch).
- **§4.A**: Checkpoint/resume + dead-code deletion.
- **§4.B**: Perf bundle (seeded batching, gradient-checkpointing vs. logprob micro-batch trade-off, FlashAttention-2, fused AdamW).
- **§4.C**: Substrate fix + logging + mechanism diagnostics.
- **§5**: Kill rules (mechanism layer, outcome layer).
- **§6**: Smoke spec and pass criteria.
- **§7**: Implementation order and timeline.
