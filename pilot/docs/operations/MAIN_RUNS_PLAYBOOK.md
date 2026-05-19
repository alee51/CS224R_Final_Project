# Main-Runs Playbook

Last updated: 2026-05-19

This is the doc for **planning main runs after the pilot completes**. It exists
because three issues from the pilot are likely to recur if you don't address
them up front:

1. **A100 only ~40% utilized** during pilot decode.
2. **Modal billing on a personal workspace** when the project should be on the
   shared team workspace.
3. **OOM "lever not wired" antipattern** — overnight runs crashing repeatedly
   while a knob that didn't actually affect peak memory was being tweaked.

Scope of this doc: the *what-to-do-differently* checklist. Background and
diagnoses live in `../incidents/` and `../decisions/`. Pointers below.

Related docs:
- `./RUNBOOK.md` — frozen pilot scope (do not edit without orchestrator sign-off)
- `../decisions/training_parallelization_plan.md` — current parallelization plan (P0–P3)
- `../incidents/0519-12_grpo-oom-root-cause.md` — full memory math + ordered fixes
- `../incidents/0519-14_main-run-preemption-no-resume.md` — preemption, no mid-run checkpoint, preds wipe on restart
- `../incidents/0519-11_grpo-smoke-debug-history.md` — chronological "what we tried" ledger
- `../decisions/efficiency_parallelization_note.md` — superseded; historical
- `../decisions/decision_memo.md` — pilot decision token (PENDING)

---

## 1) A100 utilization — what's wired vs. still leaving headroom

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

### What to try for main runs (in order)

Each step should run on a labelled perf-only branch and be gated on
`preflight_lock.json` tolerances before promoting. Do not silently overwrite
`latest` for a baseline run.

1. **Flip seeded prompt batching on for one labelled run.**
   In a per-run config under `pilot/configs/`, add:
   ```yaml
   allow_seeded_prompt_batching: true
   ```
   Then launch only that run and compare wall-clock + `gpu_seconds` against the
   matching baseline run on the same prompt slice. If gate metrics stay within
   tolerances, promote to all main runs.
   *Files touched:* `pilot/configs/run*.yaml`.

2. **If (1) holds, tune `rollout_micro_batch_size`.**
   Default is `ROLLOUT_MICRO_BATCH_SIZE = 8` (hard-coded in
   `pilot/train/rollout_engine.py:22` but overridable via the `run_grpo_training`
   config key). Try 12 and 16 on a fixed debug slice. Stop at the first OOM and
   step back one.

3. **Tune `completion_logprob_micro_batch_size`.**
   Default 16 (`pilot/train/hf_grpo_train.py:48`). With per-mb backward, raising
   this lowers wall-clock without changing peak memory much. Try 24, 32 on the
   same slice.

4. **(Optional, ~30 min) Land T2.a from the OOM doc**: replace
   `F.log_softmax`-then-index with `F.cross_entropy(..., reduction='none')` or a
   `gather` on logits. Saves ~1 GB headroom per micro-batch. Math is identical
   so this is gate-safe.

5. **(Optional, ~1–2 hr) Length-bucket prompts inside `batch_generate_rollouts`**
   (P2-1 in `../decisions/training_parallelization_plan.md`). Only useful once
   step 1 is on.

6. **(Post-pilot only) vLLM rollout backend** (`P3-1`). Material throughput win
   on the same hardware, but requires a parity gate on the 50-prompt slice.

### Acceptance bar for any of these

- `gpu_seconds` in `cost.json` drops at the same `debug_max_prompts`, OR
  measured `nvidia-smi` utilization rises materially during decode.
- Pilot-eval gate metrics stay within `preflight_lock.json` tolerances on the
  baseline slice — no silent regressions.
- Rollback path is a single-line config flip (or a single-file revert).

---

## 2) Modal — switch to the shared team workspace

### Where pilot runs are billing today

`modal profile list` shows only `chicken602` (personal). Spawn manifests under
`pilot/artifacts/matrix_logs/` confirm pilot runs are on `chicken602`:
e.g. `https://modal.com/apps/chicken602/main/ap-...`.

### One-time team-workspace setup

1. **In the Modal web UI**, accept the team workspace invite (or have a
   teammate add you). Note the workspace slug (e.g. `cs224r-rl-pilot`).

2. **Create a profile bound to that workspace** (uses the modal CLI; the exact
   subcommand surface depends on your client version — `modal token --help` to
   confirm):

   ```bash
   # Open a browser auth flow against the team workspace and save under a new profile.
   modal token new --profile team --workspace <team-workspace-slug>
   ```

   If `--workspace` is not supported by your client version, switch the active
   workspace in the Modal web UI before running `modal token new --profile team`,
   and the token will be minted against whichever workspace the web UI has
   selected. Verify with `modal profile list` — both `chicken602` and `team`
   should appear.

3. **Verify the active workspace before launching anything:**
   ```bash
   MODAL_PROFILE=team modal profile current
   # Should print the team workspace name, NOT chicken602.
   ```

### Volumes and secrets do **not** carry over across workspaces

This is the part that bites if you just flip `MODAL_PROFILE` and launch.

- `pilot/infra/modal_volumes.py` references `ARTIFACTS_VOLUME_NAME =
  "pilot-artifacts"` and `HF_CACHE_VOLUME_NAME = "hf-cache"`. Volumes are
  **workspace-scoped**. Under `MODAL_PROFILE=team`, `modal.Volume.from_name(...,
  create_if_missing=True)` will create *new, empty* volumes in the team
  workspace. Pilot artifacts on `chicken602` will not appear.
- `pilot/infra/modal_app.py:80` uses `modal.Secret.from_name("huggingface")`.
  Secrets are also workspace-scoped. You must recreate the secret in the team
  workspace or runs will fail at `from_pretrained`.

### Migration checklist (before the first team-workspace launch)

```bash
# 1) HF token secret in the team workspace
MODAL_PROFILE=team modal secret create huggingface HF_TOKEN=hf_xxxxxxxx

# 2) Confirm fresh (empty) volumes will be created on first use; or pre-create:
MODAL_PROFILE=team modal volume create pilot-artifacts
MODAL_PROFILE=team modal volume create hf-cache

# 3) (Optional) Copy HF weights cache from chicken602 → team to avoid re-download.
#    Cheaper to just let Modal re-download into hf-cache on first run for
#    Qwen3-1.7B (~3.4 GB). Skip this unless you have a reason.

# 4) Dry-run a tiny smoke against the team workspace
MODAL_PROFILE=team modal run pilot/infra/modal_app.py \
  --run-id run0_proxy --debug-max-prompts 2 --wait
```

### Make the workspace explicit in launch scripts

`pilot/scripts/launch_pilot_matrix.sh` does not set or check `MODAL_PROFILE`.
Two options, in order of preference:

- **Per-invocation env var** (simplest, no code change):
  ```bash
  MODAL_PROFILE=team ./pilot/scripts/launch_pilot_matrix.sh
  ```

- **Add a guardrail to `launch_pilot_matrix.sh`** that aborts if
  `MODAL_PROFILE` is unset or equals `chicken602` for main runs. One-liner near
  the top of the script after the venv-activation block:
  ```bash
  if [[ "${MODAL_PROFILE:-}" != "team" ]]; then
    echo "ERROR: MODAL_PROFILE must be 'team' for main runs (got '${MODAL_PROFILE:-unset}')." >&2
    exit 1
  fi
  ```
  Implement this *before* the first main run, not during it.

### Pilot vs. main split

The current pilot matrix is already running on `chicken602`. Don't switch
mid-flight — re-pulling artifacts from a different workspace is a needless
mess. Switch the launcher to `MODAL_PROFILE=team` **after** the pilot decision
token is set and before the first main run.

---

## 3) OOM — what went wrong overnight and how to avoid it on main runs

The pilot wasted real hours on this. The lesson generalizes.

### The antipattern

Overnight runs OOM'd repeatedly. The fix-attempt loop kept reducing the same
YAML knob (`completion_logprob_micro_batch_size`), going 16 → 8 → 4 → 2, plus
shrinking `batch_prompts` and `max_new_tokens`. Every reduction still OOM'd.

The reason — fully written up in `../incidents/0519-12_grpo-oom-root-cause.md` —
is that the differentiable completion-logprob path was accumulating every
micro-batch's autograd graph in a Python list and only calling `.backward()`
once at the very end of the step. The peak scaled with the *total* completion
count, not with the *per-iteration* micro-batch size. So reducing
`completion_logprob_micro_batch_size` actually made it strictly worse on the
iteration-count axis while the ceiling stayed pinned at ~80 GB.

**The knob being tweaked was not on the critical memory path.**

### Pre-launch checklist for main runs

Before kicking off a main-run matrix, walk through this list. If any item
doesn't pass, do not launch.

- [ ] **Run the 64-prompt smoke** end-to-end on the current main-run config:
      `MODAL_PROFILE=team modal run --wait pilot/infra/modal_app.py
      --run-id <main-run-id> --debug-max-prompts 64`.
      Confirm step 1 actually completes (not just "groups ready"). Confirm peak
      `nvidia-smi` < 60 GB. If it OOMs here, stop and diagnose; do not just
      shrink batch sizes.
- [ ] **Verify the per-micro-batch backward is still wired.** Grep for
      `_train_step_microbatch_backward` in `pilot/train/hf_grpo_train.py` and
      confirm it's called inside `run_grpo_training`. The fix is easy to break
      accidentally with a refactor.
- [ ] **Verify `policy.gradient_checkpointing_enable()` is still called** after
      model load.
- [ ] **Verify the Modal image env still sets**
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [ ] **If you're tempted to shrink a knob to make an OOM go away**, first
      answer: *which line of code allocates the tensor that fails to fit?* If
      you can't name the line, the knob is probably not on the critical path —
      stop and read `../incidents/0519-12_grpo-oom-root-cause.md` §4 (memory
      math) instead of guessing.
- [ ] **Don't run an unattended overnight matrix on a config that has not
      passed the 64-prompt smoke first.** This is the single highest-leverage
      rule in this doc.

### If an OOM happens on a main run

1. Pull `train.log` and the last `nvidia-smi` line if logged.
2. Read `../incidents/0519-12_grpo-oom-root-cause.md` §4. The memory math
   predicts peak for any given config; compute the predicted peak and compare
   to observed peak. If they disagree, there is a *new* bug — find it before
   touching knobs.
3. Only after the math agrees with observation should you reduce a knob, and
   only the knob that the math says is on the critical path.

---

## 4) After the pilot finishes — what to do next

This section is intentionally a stub. Detailed planning happens once
`../decisions/decision_memo.md` lands a decision token. Outline:

1. **Read the gate decision** in `pilot/gate_decision.json` (written by
   `pilot/eval/gate.py`). It will be one of:
   `ESCALATE` / `PIVOT_WORST_SUBSET` / `PIVOT_SUBSTRATE_OR_ARCH` /
   `STOP_NO_SIGNAL`.
2. **Branch on the decision** — each branch has different next steps; spelling
   them out now is premature.
   - `ESCALATE`: run the paper tier-2 eval (`beyond_aime_eval_100`,
     `hmmt_feb25_eval_30`, `math500_eval_500`) against the winning objective at
     scaled compute. **At this point: switch to the team workspace per §2 and
     apply throughput fixes per §1 before scaling.**
   - `PIVOT_WORST_SUBSET`: re-scope on the worst-subset signal; design the
     next pilot.
   - `PIVOT_SUBSTRATE_OR_ARCH`: F-GRPO equivalence detected; pick a new
     direction from the synthesis notes (`research/` or wherever the early
     direction docs live).
   - `STOP_NO_SIGNAL`: don't escalate; write up the negative result.
3. **Re-freeze scope** before the first main run. The pilot
   `preflight_lock.json` is for the pilot; main runs need their own lock
   covering the tier-2 splits and scaled budget caps.
4. **Migrate the launcher to the team workspace** per §2 *before* the first
   main run, not during it.

Worry about the specifics in this section when you reach it.
