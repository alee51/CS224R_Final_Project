# Pilot Redesign — Stage 1 Spec

Status: **drafted 2026-05-19, ready for independent audit**. Supersedes the matrix-launch instructions in `MAIN_RUNS_PLAYBOOK.md` for the next pilot attempt. The original `final_decision.md` framing (Tier 1 vs Tier 2 escalation) is preserved; this doc revises the implementation, budget, decision rules, and observability stack only.

---

## 1. Context

The first pilot launch (2026-05-19, runs `run0_proxy`, `run1_grpo`, `run1b_grpo`, `run2_inverse_freq`, `run3_f_grpo`) failed structurally, not by bad luck. Root causes documented in `pilot/docs/incidents/0519-11` through `0519-25` and synthesized in `pilot/docs/analysis/0519_perf_consolidated.md`:

- **Cost mismatch.** Measured ~99 min/step × 100 planned steps × 4 runs ≈ ~$1,275, against an intended ~$210 pilot budget and a $1,400 team total. The pilot was never affordable as written.
- **No mid-run durability.** `artifacts_volume.commit()` ran only in the `finally` block; no per-step checkpoint; preemption produced zero salvageable weights. `run1_grpo` entered a death spiral: preempt → restart → bootstrap wipes `raw_predictions.jsonl` → replay step 1 → preempt mid-step-2 → repeat.
- **Logging gaps.** `completed N/500` milestone math broke after the OOM patch (`done % 25 == 0` with mb=8); first log fired 200/500 instead of 25/500. No mid-rollout heartbeat. No wandb. Modal volume not committed mid-run, so `volume get` returned stale data.
- **Substrate parser bug.** `canonicalize_answer` is documented broken (`nancy_explore/decisions.md` 2026-05-18: "strips all `}` and breaks LaTeX"). Salvaged step-1 data showed `"12"` and `"\\( 12 \\)"` in different exact-match clusters — a known bug, not a new finding.

The current redesign is **Stage 1 of a 2-stage plan**:

- **Stage 1 (this doc):** ~$200 matrix to validate the rig, the substrate, and the mechanism. Decides which variant gets Stage 2.
- **Stage 2 (post-pilot):** mentor-prescribed 400-step / DaPO-17k / 1-epoch headline run on the winning variant, evaluated on AIME-25 + AIME-26 + Beyond-AIME + HMMT + Minerva at 64-sample Pass@k + Cover@τ. Not in scope here.

Research framing: the project is **"kill the LM-judge"** — keep Poly-EPO's set-RL + minority-voting objective structure, replace the expensive Qwen-3-4B-Instruct clustering judge with a cheap substrate (exact-match canonicalization for Stage 1). Reference: `nancy_explore/why_stop_poly_epo.md`, `nancy_explore/context.md`. `inverse_freq` is **one mathematical instantiation** of "minority-weight × reward" — not the only one, but the one Stage 1 tests.

---

## 2. Locked constraints

These are not re-negotiable in the implementing agent's scope.

**Scope.**

- Single-direction Tier 1 framing (per `nancy_explore/agent_outputs/final_decision.md`).
- Four runs: `run0_proxy` (validity check, no training), `run1_grpo` (vanilla GRPO baseline), `run2_inverse_freq` (per-prompt inverse-cluster-frequency advantage weighting), `run3_f_grpo` (F-GRPO novelty separator).
- Single seed: **42** across all matrix runs. No multi-seed.

**Budget.**

- **$50/run hard cap.** Enforced in code, including during the train phase (current `budget_cap_usd` is only checked between GRPO steps; this changes).
- **$200 matrix burst cap.** Total bundle of in-flight runs cannot exceed this.
- Team total: $1,400 (Modal credits). Stage 1 takes ~$200; Stage 2 reserved ~$720; ~$480 slack.

**Step / token budget.**

- **Target ~25 steps/run** for `inverse_freq`, `f_grpo`, and `grpo` baseline. Actual stop is whichever hits first: 25 steps OR $50 cap.
- `run0_proxy`: 500 prompts, no training (validity gate only).
- `**max_new_tokens = 1536`** for both train and eval. Aligned across both. Tile-aligned for A100 bf16. Justification: salvage data shows median completion 636, p90 1531; completions past 1500 in the salvage exhibited repetition pathology (not signal we lose by truncating). Note: `execute.py:63,152` currently clamps to 1024 — Branch B lifts the clamp to 1536 so train/eval/Run0 all share the same horizon.
- `batch_prompts = 32`, `rollouts_per_prompt = 8`. Unchanged.

**Data pinning.**

- DaPO 3k slice (`pilot/data/dapo_slice_3k.jsonl`) is the train set for Stage 1. Record its SHA-256 in `step_diagnostics.jsonl` step-0 row and in the run-final `metrics.json`, so reproducibility survives accidental file edits.

**Eval.**

- **Pilot eval: AIME-25 only, 16-sample Pass@k.** Cover@τ computed on those 16 samples.
- Full mentor-prescribed eval (AIME-26 + Beyond-AIME + HMMT + Minerva at 64-sample) is **Stage 2 only**.
- **Mini-eval every 5 steps** during training (at steps 5, 10, 15, 20). ~5 min/eval, ~$0.85/run total mid-training eval cost. Full end-of-run eval at step 25 (or $50 cap, whichever first); no mini-eval at step 25 to avoid double-eval.
- **Qualitative CoT diversity sample** at end of run: emit 16 generations × 5 random prompts to a separate artifact file for manual inspection (mentor-prescribed in `ifdita_meeting_transcript.md`).

**Infra discipline.**

- **Team workspace** (`MODAL_PROFILE=team`) before matrix launch. Verify with `modal profile current`.
- **Detached launches only** (`modal run --detach` or equivalent). No client-bound runs.
- **Smoke gate mandatory** before matrix launch. Spec in §6.

**Prompt template.**

```python
PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step, "
    "and put your final answer within \\boxed{}.\n\n"
    "{problem}\n\n"
)
```

Replaces the current "Answer: " template in `pilot/train/rollout_engine.py:15-19`. Industry standard; matches DAPO paper convention; subsumes the substrate fix.

**Substrate.**

- Exact-match canonicalization on `\boxed{<integer>}` extraction.
- `canonicalize_answer` rewrite (current implementation broken per `decisions.md`).
- Integer normalization per `decisions.md` 2026-05-18 (strip, optional leading-zero drop, compare as int).
- **No LM-judge clustering** (team has explicitly leaned away — this is the research bet).

**Parameter alignment to Poly-EPO paper (free / cheap):**

- KL coef: `0.001` → `0.0` (paper).
- Clip ratio: symmetric `0.2` → asymmetric `(ε_low=0.2, ε_high=0.28)` (paper). See §4.B5 for the code surface.

---

## 3. End-to-end pipeline

```
   [redeploy gated by smoke]
            │
            ▼
   ┌────────────────┐
   │  Branch A      │  checkpoint/resume + delete dead code
   │  Branch B      │  perf bundle (seeded batching, FA2, fused AdamW, ...)   [parallel]
   │  Branch C      │  substrate fix + logging + mechanism diagnostics
   └────────┬───────┘
            │   merge
            ▼
   ┌────────────────┐
   │  32-prompt     │   1×A100, 3 steps, GRPO only, preempt mid-step-2,
   │  smoke         │   verify resume + GRPO mechanism + parser_clean_rate
   └────────┬───────┘
            │   pass
            ▼
   ┌────────────────┐
   │  Team workspace│   modal profile activate team
   │  switch + cap  │   verify budget_cap_usd enforcement live
   │  audit         │
   └────────┬───────┘
            │
            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Matrix launch (4 parallel runs, detached, single seed 42) │
   │   run0_proxy  +  run1_grpo  +  run2_inverse_freq  +  run3  │
   └────────────────────────────────────────────────────────────┘
```

---

## 4. Code changes — by branch

Each branch is independently implementable. The orchestrator may spawn a Cursor agent per branch (§4.A / §4.B / §4.C) with that section as context — no model pinning. Branches touch mostly disjoint files; conflicts resolved in the merge step before smoke.

### Branch A — Checkpoint/resume + dead code removal

**Files touched:** `pilot/train/hf_grpo_train.py`, `pilot/infra/modal_app.py` (resume logic + `@modal.exit` hook live here — `run_pilot_remote` is at `modal_app.py:91`), `pilot/configs/shared_train.yaml`, and the four per-run yamls (`run0_proxy.yaml`, `run1_grpo.yaml`, `run2_inverse_freq.yaml`, `run3_f_grpo.yaml`) for cap bumps.

**Add: time-gated checkpoint commit.**

Constants come from `shared_train.yaml` keys `checkpoint.min_interval_seconds` and `checkpoint.target_interval_seconds`. `step_is_natural_boundary` means "the optimizer.step() for this training step has returned and we are between steps" — i.e., not mid-rollout, not mid-backward.

```python
# in run_grpo_training loop, after each step's update has returned
elapsed = time.monotonic() - last_commit_time
should_commit = (
    step == 1                                # always after step 1
    or step == total_steps                   # always last step
    or elapsed >= ckpt_target_interval_s     # default 3600s
    or elapsed >= ckpt_min_interval_s        # default 1800s (always a step boundary here)
)
if should_commit:
    save_checkpoint(out_dir, step, policy, optimizer, preds_offset_bytes)
    artifacts_volume.commit()
    last_commit_time = time.monotonic()
```

**Signatures:**

```python
def save_checkpoint(out_dir: Path, step: int, policy, optimizer, preds_offset_bytes: int) -> None:
    ckpt_dir = out_dir / f"checkpoint_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save({"cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()}, ckpt_dir / "rng_state.pt")
    state = {"step": step,
             "rng_state_path": str(ckpt_dir.relative_to(out_dir) / "rng_state.pt"),
             "optimizer_state_path": str(ckpt_dir.relative_to(out_dir) / "optimizer.pt"),
             "preds_offset_bytes": preds_offset_bytes,
             "wall_seconds_elapsed": time.time() - run_t0,
             "usd_spent_estimate": _estimated_usd(...)}
    tmp = out_dir / "training_state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.replace(out_dir / "training_state.json")  # atomic

def load_checkpoint(out_dir: Path, state: dict, policy, optimizer) -> None:
    ckpt_dir = out_dir / f"checkpoint_step{state['step']}"
    # Note: weights are reloaded via from_pretrained at boot; here we restore optimizer + RNG only.
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))
    rng = torch.load(ckpt_dir / "rng_state.pt")
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state_all(rng["cuda"])
```

**Add: in-train 60s budget check.** §2 locks "budget cap enforced every 60s during the train phase." Inside `_train_step_microbatch_backward`, poll `time.monotonic()` before each micro-batch backward; if `_estimated_usd(...) >= budget_cap_usd`, raise a `BudgetCapHit` sentinel that the outer loop catches → save checkpoint → exit cleanly. Without this, a step that costs $10+ can push past the cap before the inter-step check runs.

**Add: `@modal.exit` flush handler.** Modal grants ~30s grace on preemption. Register an exit hook on the training class that calls `artifacts_volume.commit()` so the most-recent partial state survives. Quantified value: `0519-14` lost a full step on `run1_grpo` preempt → ~$4/incident.

**Add: `training_state.json` cursor.** Written atomically alongside each checkpoint. Schema:

```json
{
  "step": 7,
  "rng_state_path": "checkpoint_step7/rng_state.pt",
  "optimizer_state_path": "checkpoint_step7/optimizer.pt",
  "preds_offset_bytes": 524288,
  "wall_seconds_elapsed": 12483.4,
  "usd_spent_estimate": 13.20
}
```

**Add: resume logic on boot.** In `run_pilot_remote` startup, before training loop:

```python
state_path = run_dir / "training_state.json"
if state_path.exists():
    state = json.load(state_path.open())
    resume_step = state["step"] + 1
    load_checkpoint(run_dir, state, policy, optimizer)  # signature per the block above
    # Truncate preds file to the known-good offset, then reopen for append.
    with open(preds_path, "rb+") as f:
        f.truncate(state["preds_offset_bytes"])
    preds_file_handle = open(preds_path, "ab")
    logger.info("resuming from step %s", resume_step)
else:
    resume_step = 1
    preds_path.write_text("")  # ONLY on cold boot, never on resume
```

Policy weights are reloaded via `AutoModelForCausalLM.from_pretrained(checkpoint_step{state['step']}_dir, ...)` at the top of `run_pilot_remote` — not inside `load_checkpoint`, which only restores optimizer + RNG.

**Delete:** the unconditional `pred_path.write_text("")` at `hf_grpo_train.py:~865` (incident 0519-14). This is the wipe bug.

**Delete:** dead code from the OOM patch.

- `HFPolicyModel` class (unreachable from active training loop)
- `_differentiable_loss` function (replaced by current loss path)
- Duplicate `policy.save_pretrained()` at `hf_grpo_train.py:962-966` (kept) and `991-996` (delete — saves 3.4GB twice per run end)

**Config additions to `shared_train.yaml`:**

```yaml
checkpoint:
  min_interval_seconds: 1800
  target_interval_seconds: 3600
  always_save_first_step: true
  always_save_last_step: true
  retention_keep_last: 2     # GC: keep step 1 + the last N step checkpoints; delete others
budget_cap_usd: 50.0          # ENFORCED at every step boundary AND every 60s during train phase
```

**Per-run yaml caps must be bumped too.** `pilot/infra/config_resolver.py` deep-merges with per-run leaf values winning, so the shared value alone will not apply:

- `pilot/configs/run0_proxy.yaml:9` → `budget_cap_usd: 50` (was 24)
- `pilot/configs/run1_grpo.yaml:5` → `budget_cap_usd: 50` (was 36)
- `pilot/configs/run2_inverse_freq.yaml:9` → `budget_cap_usd: 50` (was 36)
- `pilot/configs/run3_f_grpo.yaml:10` → `budget_cap_usd: 50` (was 36)

(Optional: remove `budget_cap_usd` from per-run yamls and rely on shared. Pick one approach and apply uniformly.)

**Checkpoint retention.** Implement GC after each successful new checkpoint: list `checkpoint_step*` dirs, keep step 1 and the two most-recent, delete the rest. A 25-step run otherwise produces ~85 GB of stale checkpoints × 4 runs = ~340 GB, which will exhaust the artifacts volume.

**Acceptance:** in smoke, kill the container mid-step-2 (per §6). Restart. Verify `training_state.json` cursor is read, training resumes at step 2 (not step 1), step 1's completions in `raw_predictions.jsonl` are retained, and final file has 3×256=768 lines after the run finishes step 3.

---

### Branch B — Perf bundle

**Files touched:** `pilot/configs/shared_train.yaml`, `pilot/configs/*.yaml`, `pilot/infra/modal_app.py`, `pilot/train/hf_grpo_train.py`.

Apply as a **single bundle**, validate together in smoke. Source for each: `pilot/docs/analysis/0519_perf_consolidated.md`.

**B1. Enable seeded prompt batching.**

- `shared_train.yaml`: `allow_seeded_prompt_batching: true`
- Also delete the explicit `allow_seeded_prompt_batching: false` lines in `run0_proxy.yaml:12` and `run3_f_grpo.yaml:8` (per-run wins; must remove or override).
- Code path already exists in `rollout_engine.py:107-147` (batched + per-row `Generator`). Validate `mean_reward` parity vs sequential on a fixed 32-prompt slice in smoke: run both paths on the same seed and require `|mean_reward_batched - mean_reward_serial| < 0.01` AND `|advantage_l2_batched - advantage_l2_serial| / advantage_l2_serial < 0.05`. If parity fails, fall back to serial and document.
- Expected: rollout phase ~26 min → ~10-12 min.

**B2. Disable gradient checkpointing + raise logprob micro-batch.**

- `hf_grpo_train.py:846-847`: comment out `policy.gradient_checkpointing_enable()`.
- `shared_train.yaml`: `completion_logprob_micro_batch_size: 32` (was lower).
- **Fallback:** smoke run includes VRAM monitoring. If smoke OOMs with checkpointing off, revert to `gradient_checkpointing_enable()` + drop `completion_logprob_micro_batch_size` to 16. Document which path was taken in smoke output.
- Expected if no OOM: train phase ~73 min → ~40-48 min.

**B3. FlashAttention-2.**

- `modal_app.py:63`: add `flash-attn` to the image pip_install list with version pin matching the torch/CUDA combo. Example: `.pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")`.
- `hf_grpo_train.py:841-845`: `AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True, attn_implementation="flash_attention_2").to(device)`.
- Expected: ~15-25% on both phases.

**B4. Fused AdamW.**

- `hf_grpo_train.py:862`: `optimizer = AdamW(policy.parameters(), lr=lr, fused=True)`.
- Expected: ~10-15% on optimizer step.

**B5. Parameter alignment to paper.**

- `shared_train.yaml`: `kl_coef: 0.0` (was 0.001).
- `shared_train.yaml`: `clip_ratio_high: 0.28`, `clip_ratio_low: 0.2` (asymmetric; paper-aligned). Delete the legacy `clip_eps: 0.2` key.
- **Code surface for asymmetric clip is wider than one line.** Today, `hf_grpo_train.py:855` reads a single `clip_eps`, and `objectives.py`'s `_clip_surrogate_scalar` / `_clip_surrogate_tensor` (called at `hf_grpo_train.py:457,477` and via `_differentiable_loss`) use `torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)`. The implementing agent must:
  1. Add `clip_ratio_low` / `clip_ratio_high` to `GRPOConfig` (drop `clip_eps`).
  2. Update both `_clip_surrogate_*` to `torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)`.
  3. Update `hf_grpo_train.py:855` to read both keys.
  4. Audit all `clip_eps` references in `hf_grpo_train.py` and `objectives.py` (grep first).

**B6. Lift `execute.py` `max_new_tokens` clamp from 1024 → 1536.**

Both `execute.py:63` (`_rollout_engine_config`, used by eval and Run0) and `execute.py:152` (Run0 direct path) currently force `max_new_tokens` to `min(config_value, 1024)`. §2 locks 1536 for train/eval alignment, so:

- `execute.py:63`: `max_new_tokens=min(int(config.get("max_new_tokens", 2048)), 1536)`
- `execute.py:152`: same change.

Without this, eval and Run0 silently run at the old 1024 horizon while training runs at 1536 — the exact behavioral asymmetry called out in `0519_perf_consolidated.md` §B2.

**Skip for now:** vLLM. High effort, parity risk with seeded sampling. Reconsider only if step time > 60 min after B1-B4 land.

**Files touched (revised):** `pilot/configs/shared_train.yaml`, `pilot/configs/run0_proxy.yaml`, `pilot/configs/run3_f_grpo.yaml`, `pilot/infra/modal_app.py`, `pilot/infra/execute.py`, `pilot/train/hf_grpo_train.py`, `pilot/train/objectives.py`.

**Acceptance:** smoke step 1 completes under 60 min with all of B1-B4 enabled (B5/B6 are correctness, not perf). If B2 OOMs, smoke notes the fallback and step time target drops to <80 min. Eval after smoke must use `max_new_tokens=1536` (validates B6).

---

### Branch C — Substrate fix + logging + mechanism diagnostics

**Files touched:** `pilot/train/rollout_engine.py`, `pilot/train/canonicalize.py`, `pilot/train/answer_parse.py`, `pilot/train/hf_grpo_train.py`, `pilot/infra/modal_app.py`.

**C1. Prompt template update.**

`rollout_engine.py:15-19`:

```python
PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step, "
    "and put your final answer within \\boxed{}.\n\n"
    "{problem}\n\n"
)
```

**C2. Answer extraction.**

New extractor in `answer_parse.py`:

```python
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")

def extract_boxed_answer(text: str) -> tuple[str | None, bool]:
    """Returns (raw_extracted, parser_clean). parser_clean is True iff
    exactly one \\boxed{...} is found and the contents parse as an int."""
    matches = BOXED_RE.findall(text)
    if len(matches) != 1:
        return (None, False)
    raw = matches[0].strip()
    try:
        return (str(int(raw.replace(",", ""))), True)
    except ValueError:
        return (raw, False)
```

**C3. Rewrite `canonicalize_answer`.**

The current implementation is documented broken (strips all `}`, per `decisions.md` 2026-05-18).

Two distinct call sites must keep working:
- **Rollout completion side:** input is the completion text; first call `extract_boxed_answer` to get `(raw, parser_clean)`, then canonicalize the `raw`.
- **Gold side:** input is the bare gold integer string (e.g., `"12"`), with no `\boxed{}` wrapper. Must canonicalize directly.

New `canonicalize_answer(text: str) -> str` (single function, both sides):

```python
def canonicalize_answer(text: str) -> str:
    s = (text or "").strip().replace(",", "")
    try:
        return str(int(s))           # integer normalization: drops leading zeros, handles "-"
    except ValueError:
        return s.lower()              # non-integer fallback; cluster_id hash still works
```

There is no separate "flag" — `parser_clean` (from C2) is logged per-rollout in C4 and is the signal that consumers (e.g., `is_correct`, cluster aggregation) can branch on. `cluster_id(answer)` (canonicalize.py:15) keeps its current shape: `hash(canonicalize_answer(answer)) % (2**31)`.

**C4. Per-rollout logging fields.**

In `_build_step_groups`, after reward computation, log per rollout to `step_diagnostics.jsonl`:

```json
{
  "step": 1,
  "prompt_id": "0b4478a7-...",
  "rollout_idx": 3,
  "reward": 1.0,
  "raw_advantage": 0.875,
  "weighted_advantage": 4.375,
  "cluster_id": "12",
  "cluster_size": 1,
  "is_minority_correct": true,
  "completion_tokens": 612,
  "parser_clean": true
}
```

**C5. Per-step aggregate logging.**

Append to `step_diagnostics.jsonl` one row per step with phase markers + aggregates:

```json
{
  "step": 1,
  "phase": "step_complete",
  "wall_seconds": 3247.1,
  "build_seconds": 612.4,
  "train_seconds": 2634.7,
  "mean_reward": 0.172,
  "advantage_var": 1.247,
  "kl": 0.0034,
  "clip_frac": 0.082,
  "advantage_l2": 12.4,
  "grad_norm": 0.84,
  "parser_clean_rate": 0.97,
  "num_minority_correct_prompts": 5,
  "num_clusters_mean": 6.5,
  "usd_spent_estimate": 1.85
}
```

`volume.commit()` after each step's diagnostic write (cheap, ~50KB file).

**C6. Mechanism check.**

For each minority-correct prompt at each step, compute a per-variant mechanism signal:

- **Vanilla GRPO:** `expected_advantage = reward - group_mean_reward`. Mechanism check: `weighted_advantage == expected_advantage` (should be identity).
- **inverse_freq:** `expected_advantage = (reward - group_mean) × normalized_inverse_freq(cluster_size, total_rollouts) × γ`. Mechanism check: correlation between `weighted_advantage` and `expected_advantage` across the 8 rollouts of each minority-correct prompt should be > 0.95.
- **F-GRPO:** spec-dependent on the F-GRPO formulation in `objectives.py`. The implementing agent must derive the expected advantage formula from the code and log the correlation against it.

Per-step log entry: `mechanism_signal_per_variant: float`. If < 0.9 at step 1, kill the run and alert (this is the "implementation bug" tripwire — see §5).

**C7. wandb integration.**

In `run_pilot_remote`, initialize wandb. Default is offline-to-local-then-sync (per §8 risk #3 — Modal egress to `api.wandb.ai` may be blocked). The Modal volume holds the run dir until end-of-run sync.

```python
import wandb
wandb_mode = os.environ.get("WANDB_MODE", "offline")  # "offline" or "online"
wandb.init(
    project="cs224r-minority-voting",
    name=run_name,
    config=asdict(run_config),
    mode=wandb_mode,
    dir=str(out_dir),
)
# At end-of-run (after final volume.commit), if WANDB_API_KEY is set and mode==offline,
# subprocess.run(["wandb", "sync", str(out_dir / "wandb")]) to push.
```

Per-step `wandb.log({...})` with the aggregate fields from C5. Histograms of `weighted_advantage` and `reward` distributions. Sample 4 random completions every 5 steps and log as `wandb.Table`.

**C8. Heartbeat.**

Config keys in `shared_train.yaml`:

```yaml
heartbeat:
  seconds: 60
  completions: 32
```

In `rollout_engine.py`'s generation loop:

```python
hb_s = float(shared.get("heartbeat", {}).get("seconds", 60))
hb_c = int(shared.get("heartbeat", {}).get("completions", 32))
if time.monotonic() - last_heartbeat >= hb_s or completions_done % hb_c == 0:
    logger.info("rollout heartbeat: step=%s completions=%s/%s",
                step, completions_done, total_completions)
    last_heartbeat = time.monotonic()
```

**C9. Modal image: add wandb.**

`modal_app.py:63-71`: add `wandb` to the pip_install list. Attach the `wandb-api-key` Modal secret to the `run_pilot_remote` function decorator (`@app.function(secrets=[modal.Secret.from_name("wandb-api-key")], ...)`). Secret *creation* is operator-owned and listed in §7.

**Acceptance:** smoke produces `step_diagnostics.jsonl` with all required fields, wandb dashboard renders the run, GRPO mechanism check passes (identity correlation ≥0.9 per §6 criterion 5), and `parser_clean_rate > 0.9` on the smoke prompts. inverse_freq and F-GRPO mechanism correlations are validated at matrix step 1, not smoke (see §6 deferral note).

---

## 5. Decision rules v2

Two layers: **mechanism layer** (checked from step 1, cheap, observable in training logs) and **outcome layer** (checked at end of pilot via eval).

### Kill rules (apply during training)


| Trigger                                                                                                                                      | Action                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `mechanism_signal_per_variant < 0.9` at step 1 for any variant                                                                               | Stop that run. The implementation is broken. Fix, redeploy, redo. **Not a research finding.** |
| Training `mean_reward` rolling-mean over last 3 steps drops > 40% below the step-1 baseline **AND** `advantage_var` has also collapsed > 50% | Kill that variant. (Two-signal AND avoids unlucky-sampling false alarms.)                     |
| Mini-eval Pass@1 drops > 5pt below the step-5 baseline for **2 consecutive eval checkpoints**                                                | Kill that variant. (Patient — single bad eval is noise; two is signal.)                       |
| Any single run hits `$50` cap before step 25                                                                                                 | Stop run, log, debug. Do not push through the cap.                                            |
| Matrix burst projected > `$200` based on extrapolated step time                                                                              | Pause launch, reconsider.                                                                     |


### End-of-pilot outcome interpretation

After step ~25 (or $50 cap, whichever first), all surviving runs end with a full AIME-25 16-sample Pass@k + Cover@τ eval.


| Outcome                                                                                                    | Action                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One variant clearly leads on Cover@τ by **> 3pt over baseline GRPO**, mechanism passed, no Pass@1 collapse | **Escalate to Stage 2**: that variant + GRPO baseline, 400 steps × DaPO 17k × 1 epoch, 64-sample eval on AIME-25/26 + Beyond-AIME + HMMT + Minerva                                                                                                                                                                                                                      |
| All three runs within 2pt on all eval metrics, mechanism passed                                            | "No early-stage signal at 25 steps." Decide: (a) Stage 2 on the most-mechanistically-distinct variant anyway, (b) pivot to substrate ablation (next obvious step in the "kill the LM-judge" research program), or (c) reframe pilot writeup as scale-limited negative result. **Default: (a)** unless mechanism diagnostics suggest the method is mechanistically inert |
| `Pass@1` collapsed on `inverse_freq` specifically                                                          | Frequency proxy is misallocating gradient mass. Pivot to substrate ablation (sentence embeddings or n-gram fingerprints) OR to `worst_subset` objective                                                                                                                                                                                                                 |
| `Pass@1` collapsed on baseline GRPO                                                                        | Something broke in the rig that affects all variants. Stop matrix, investigate                                                                                                                                                                                                                                                                                          |
| Any outcome not covered by the rows above (e.g., Pass@1-only lead, only baseline survives, mechanism passed but a variant ties baseline)                  | **Judgment call at review time.** Record actual eval metrics + mechanism signals + qualitative-diversity artifact and decide post-hoc. Default disposition: write up as "pilot completed, no clear escalation signal" and treat Stage 2 escalation as an open question rather than auto-pivoting to substrate ablation. |


### Qualitative artifact

At end of every surviving run, dump 16 generations × 5 random AIME-25 prompts to `qual_diversity_sample.jsonl`. Manual inspection check: do the generations show *different reasoning paths* or are they all the same chain with minor token-level perturbations? Mentor-prescribed signal for CoT diversity (`ifdita_meeting_transcript.md`). This does not gate the Stage 2 decision but informs the writeup.

---

## 6. Smoke spec

Cost target: **≤$10**. Implement as a new `pilot/configs/smoke.yaml` (copy of `run1_grpo.yaml` with `budget_cap_usd: 10`, `debug_max_prompts: 32`, `max_steps: 3`); launch with the same Modal entrypoint as matrix runs. Goal: validate Branches A, B, C land together cleanly before any matrix run. Scope is intentionally narrow — `inverse_freq` and `f_grpo` mechanism validation is deferred to matrix step 1 (the kill rule in §5 catches it there and only costs one bad step per broken variant, not a whole smoke run).

**Configuration:**

- 1× A100, single run, **GRPO objective only**.
- 32 prompts (random DaPO slice), 8 rollouts/prompt, max_new_tokens=1536.
- **3 training steps.**
- Mid-run preemption test: at step 2 rollout-build phase, the orchestrator kills the container externally. Restart should resume from step 1's checkpoint and complete steps 2 and 3.

**Pass criteria (all must hold):**

1. Step 1 completes in < 60 min (perf bundle delivered).
2. If grad-checkpointing-off OOMs, smoke auto-falls-back and reports the fallback. Step 1 then must complete in < 80 min.
3. After step 1, `training_state.json` exists, `checkpoint_step1/` exists, `step_diagnostics.jsonl` has step-1 row.
4. After forced preempt at step 2 and restart, training resumes at step 2 (NOT step 1). `raw_predictions.jsonl` retains step-1 completions (256 lines) and final file has 3 × 256 = 768 lines. No `write_text("")` wipe.
5. `mechanism_signal_per_variant >= 0.9` for GRPO at step 1. (Identity check: `weighted_advantage == reward - group_mean_reward`.)
6. `parser_clean_rate >= 0.9` at step 1.
7. wandb dashboard renders (online or offline-then-synced) with per-step scalars + at least one histogram + at least one sample-completion table.
8. Heartbeat logs appear at < 120s intervals during the rollout phase.
9. B1 parity validation passes (`|Δmean_reward| < 0.01`, `|Δadvantage_l2|/advantage_l2 < 0.05` against a sequential-path reference slice).
10. B6 verification: invoke `run_tier1_eval` on the smoke checkpoint with `debug_max_prompts: 4`; grep the rollout log for `max_new_tokens=1536` (not just yaml — must show in the actual generate call).
11. Final commit of artifacts on smoke completion, including all diagnostic files.

**Fail handling:** if any of 1-11 fail, do not launch the matrix. Surface the failure, fix the offending branch, re-smoke.

**Mechanism deferral note.** `inverse_freq` and `f_grpo` mechanism correlations (per §4.C6) are first validated at matrix step 1, not smoke. If either fires `mechanism_signal_per_variant < 0.9`, the kill rule in §5 stops that one run and the operator fixes the formula before relaunching that variant. The other variants continue.

---

## 7. Implementation order

```
T+0   Orchestrator spawns Cursor agents as needed (often one per branch, parallel OK)
        ├── Branch A — checkpoint/resume + dead-code deletion
        ├── Branch B — perf bundle (config flips + FA2 + fused AdamW + param alignment)
        └── Branch C — substrate fix + logging + mechanism diagnostics

T+1d  Branch merge — operator-owned. Conflicts to resolve manually before smoke:
        - shared_train.yaml (all three branches touch it)
        - hf_grpo_train.py (Branches A and C both touch the training loop)
        - modal_app.py (Branches B and C both add image deps)
        - per-run yamls (Branch A bumps caps, B may delete allow_seeded=false lines)

T+1d  Provision Modal secrets
        - modal secret create wandb-api-key WANDB_API_KEY=<token>
        - attach the wandb-api-key secret to the training Modal function

T+1d  Run audit prompt against this doc + the merged code

T+2d  Smoke run on team workspace
        - Pass criteria from §6
        - If fail: fix offending branch, re-smoke. Do not skip.

T+2d  Pre-matrix checks
        - modal profile current → "team"
        - grep budget_cap enforcement is wired into the train loop, not only between steps
        - confirm wandb secret bundle attached to the training function
        - confirm all 4 run yamls have budget_cap_usd: 50 (per-run wins over shared — verify the merged config in a dry run)
        - confirm execute.py:63 and :152 read max_new_tokens=1536, not 1024

T+3d  Matrix launch
        - run0_proxy + run1_grpo + run2_inverse_freq + run3_f_grpo, all 4 detached, single seed 42
        - Operator monitors wandb in real time; volume pulls if any run looks anomalous
        - Matrix burst projection updated every 5 minutes against the $200 cap
```

---

## 8. Open risks

Listed here so the implementing agent does not silently absorb them.

1. **Perf bundle may underdeliver.** If grad-checkpointing-off OOMs *and* seeded batching's reward parity test fails, step time stays near baseline. Pilot scope shrinks to ~10-15 steps. Decision rules v2 still apply but the "early-stage signal" interpretation gets weaker. No mitigation other than vLLM, which is deferred.
2. **Mechanism check spec for F-GRPO depends on the current objective implementation.** If the F-GRPO code path in `objectives.py` differs from the published formulation, the mechanism correlation will fail even on a "correct" implementation. The implementing agent must verify the F-GRPO formula matches the citation before building the mechanism check.
3. **wandb may be inaccessible from Modal.** If outbound HTTPS to `api.wandb.ai` is blocked in the team workspace, fall back to `wandb.init(mode="offline")` + periodic `wandb sync` on the local Modal volume. Diagnostics still land in `step_diagnostics.jsonl` regardless.
4. **Single-seed risk.** Per-run reward at small N (32 prompts) has high variance. A bad seed could starve a variant of minority-correct prompts and produce a false negative. Multi-seed is deferred to Stage 2 per mentor steer, but the implementing agent should record the seed prominently in the writeup so this caveat is explicit.
5. **DaPO 3k subset vs prescribed 17k.** Mentor prescribed 17k for the 1-epoch run. Stage 1 uses 3k for budget. Stage 2 must switch to 17k. The implementing agent should not silently keep 3k for Stage 2.

---

## 9. Deferred decisions (TBD)

These are out of scope for the Stage 1 implementation but listed so they are not lost.

- **Math-Verify grader** for HMMT-Nov / Beyond-AIME / MATH-500 (per `decisions.md` 2026-05-18). Stage 2 prerequisite, not Stage 1.
- `**gate_decision.json` re-evaluation.** Current state is `PIVOT_WORST_SUBSET` based on `minority_correct_rate=0.000` from a broken-parser run0. After Stage 1's `run0_proxy` completes with the fixed parser, re-run the gate; the 0.000 was likely a parser artifact. Decide then whether the gate stays load-bearing or is retired.
- **Stage 2 rollback plan** if the escalated 400-step run also fails to differentiate variants. Documented fallbacks in `final_decision.md` (substrate-ablation paper, head-to-head minority-objectives paper, pre-registered negative-result paper). Choose post-pilot, with eval data in hand.
- **Pass@k k-value bump for Stage 2.** Mentor prescribes 64-sample at eval; Stage 2 should use this; Stage 1 uses 16. Confirm at Stage 2 planning.
- **Multi-seed for Stage 2.** Probably 2 seeds on the Stage 1 winner per mentor practice; not in Stage 1.

---

## 10. Hand-off to implementing agents

Each implementing Cursor agent should receive: this doc, the relevant section (§4.A, §4.B, or §4.C), and a constraint that it may not change anything outside that branch's "files touched" list without an explicit handoff back to the operator. Model and tooling are chosen by the orchestrator from context — this doc does not prescribe one.

Branch C additionally needs `objectives.py` and `nancy_explore/agent_outputs/final_decision.md` in context for mechanism-check derivation (same agent session is fine).

After all three branches are complete, merge + audit + smoke are further Cursor agent tasks (or the same orchestrator session). Matrix launch is operator-driven, not agent-driven (per `MAIN_RUNS_PLAYBOOK.md` — operator owns the launch button).

---

*Doc author: Opus 4.7 (in conversation with operator on 2026-05-19). Verification status: audited 2026-05-19 against codebase (see `pilot/docs/analysis/0519_redesign_doc_audit.md`); fixes applied 2026-05-19 — step target unified to 25, smoke scope cut to single-variant 3-step, mechanism threshold unified to 0.9, per-run yaml caps + `execute.py` clamp + asymmetric-clip surface explicitly enumerated, §10 renumbered. Implementation status: not started.*