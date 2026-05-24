# PILOT_REDESIGN audit v2 (2026-05-19)

Second independent pass against the revised doc. Different lens from v1 — many v1 issues are fixed; this pass catches what the revision introduced or missed.

---

## 1. Internal consistency contradictions (new)

- **§3 pipeline diagram is stale.** Lines 105-106 still show "5 steps, forced preempt mid-step-3" — §6 was cut to "3 steps, preempt mid-step-2." Diagram contradicts §6.
- **Branch A acceptance is stale.** Line 245 says "kill the container mid-step-3." Smoke (§6) is mid-step-2. Direct contradiction.
- **Branch C acceptance is impossible.** Line 448 demands "mechanism check correlations are logged for all three variants" as a smoke acceptance criterion, but §6 explicitly defers inverse_freq and F-GRPO mechanism validation to matrix step 1. Either the smoke runs all three (re-bloats it) or this criterion gets cut.
- **§5 outcome row 2 still references "20 steps".** Line 476 quotes `"No early-stage signal at 20 steps."` — should be 25.
- **§2 paper alignment lists only `ε_high=0.28`.** Line 87 omits `ε_low=0.2`, but §4.B5 specifies both. §2 should match.
- **Smoke `budget_cap_usd: 10` has no home.** §6 says "`budget_cap_usd: 10` on the smoke config" but no file is named, and the matrix yamls all have `50`. Implicit: a separate `smoke.yaml`, or a launch-time override. Unspecified.

## 2. Code-citation accuracy (new)

- **Branch A "Files touched" is wrong on two axes.** Line 131 lists `hf_grpo_train.py, execute.py, shared_train.yaml`. But:
  - `execute.py` — Branch A no longer touches it (B6 owns the only `execute.py` edit).
  - `modal_app.py` — Branch A *must* touch it. `run_pilot_remote` lives at `modal_app.py:91` (verified), and Branch A's "resume logic on boot" pseudocode (line 198) and the `@modal.exit` flush handler (line 183) both belong there. Missing from files-touched.
  - Also missing: `pilot/configs/run0_proxy.yaml`, `run1_grpo.yaml`, `run2_inverse_freq.yaml`, `run3_f_grpo.yaml` — Branch A bumps caps on all four, per lines 236-239.
- **Branch C "Files touched" includes `pilot/infra/execute.py` (line 309).** No C-item edits it. Stale.
- All other file:line citations were already verified accurate in v1.

## 3. Spec ambiguities (new / persisting)

**Branch A:**
- **`load_checkpoint` signature mismatch.** Line 172 defines `load_checkpoint(out_dir, state, policy, optimizer)`. Line 205 invokes `load_checkpoint(run_dir / state["rng_state_path"], ...)` — different argument shape. Implementing agent must pick one.
- **Append-mode seek doesn't truncate.** Line 206-207: `open(preds_path, "ab")` then `seek(state["preds_offset_bytes"])` with the comment "safety truncate to known good." Append mode writes go to EOF regardless of seek; the seek is a no-op. To truncate, use `open(preds_path, "rb+")` then `truncate(offset)` then re-open as append. Misleading pseudocode.
- **`@modal.exit` placement implicit.** Line 183 says "register an exit hook on the training class" — but `run_pilot_remote` (modal_app.py:91) is currently a `@app.function`, not a class. Implementing agent must decide whether to migrate to `@app.cls` or use another grace mechanism.
- **`$200 matrix burst cap` unenforced.** §2 declares it as a locked constraint and §5 kill rules say "Pause launch, reconsider" if projected > $200, but no branch implements the projection logic. Owner unassigned.

**Branch B:**
- **B1 parity edge case.** `|Δadvantage_l2|/advantage_l2_serial < 0.05` divides by `advantage_l2_serial`, which can be near-zero on a smoke slice if rewards are degenerate. No fallback rule.
- **B3 flash-attn pin** still says "matching the torch/CUDA combo" without committing a version. Implementing agent picks blind.

**Branch C:**
- **C2/C3 `canonicalize_answer` contract is incomplete.** C2 returns `(None, False)` when no boxed match — what does the *eval/cluster pipeline* do with `None`? Current `answer_parse.py:24-26` calls `canonicalize_answer(extract_answer(completion)) == canonicalize_answer(str(gold))`. The gold answer is NOT wrapped in `\boxed{}`. C3 doesn't specify how `canonicalize_answer` handles a gold-side raw integer string.
- **C3 "flag for the cluster_id hash" still undefined.** Persists from v1 — what flag, where stored, who reads it.
- **C6 `normalized_inverse_freq` undefined.** Formula given but not the normalization. Implementing agent must read `objectives.py` to derive — flagged in §8 risk #2 but not specified here.
- **C7 `asdict(run_config)` assumes a dataclass.** Implementing agent must verify or wrap.
- **C9 duplicates §7 secret provisioning.** Line 446 says "Add WANDB_API_KEY to the Modal secret bundle" — but §7 (lines 533-535) has the canonical provisioning step. C9 should just say "attach the wandb-api-key secret to the function" and let §7 own creation.

## 4. Decision rule coherence (new)

The catch-all row in §5 (line 479) closes the gap I called out in v1. No new uncovered outcome on this pass.

One residual: the kill rule "$50 cap before step 25" assumes the budget projection is accurate. If the in-train 60s budget check (Branch A) trips at step 24 with $51 spent, is that a "ran to plan" outcome or a "killed by cap" outcome? Doc doesn't say. Probably not worth a rule but worth a sentence.

## 5. Perf-bundle math (no new issues)

The doc never asserts the "55–60 min/step" figure the original prompt mentioned. Acceptance target `<60 min` (B1-B4) and `<80 min` (B2 fallback) remain conservative against `0519_perf_audit.md` (top-3 estimate 42-50 min at 2048 tokens; at 1536 tokens this drops further). Internally consistent.

Budget arithmetic: $50/run ÷ $0.000694/s ≈ 20 GPU-hours = 25 × 48 min steps. If steps land at 55-60 min, runs hit cap at 20-22 steps. Already noted in §2 ("whichever first").

## 6. Smoke spec sufficiency (new gaps)

Not exercised by the cut-down smoke:
- **Branch B5 asymmetric clip path.** Smoke runs GRPO, which exercises the clip surrogate, but doesn't compare against the old symmetric clip behavior. Bug in the asymmetric implementation would land silently.
- **Branch A budget-cap-trip.** Smoke cap is $10 and step 1 might be $2-3; cap never trips. The in-train 60s check (the new code) is not exercised.
- **`@modal.exit` flush handler.** Smoke preempt mid-step-2 is an external kill, not necessarily the Modal preempt signal. Whether the exit hook fires depends on the kill mechanism — doc doesn't specify which signal/path the smoke uses.
- **Checkpoint retention GC.** 3 steps doesn't exceed `retention_keep_last: 2` + step 1 = 3 retained. GC path never fires.
- **Run0 validity-gate path.** Not in smoke at all (smoke is GRPO-only).
- **C10 4-prompt eval (smoke criterion #10).** Calls for "a 4-prompt eval pass after smoke uses `max_new_tokens=1536`" — but `run_tier1_eval` evaluates the full AIME-25 set. Operator must run a custom 4-prompt eval or modify the eval call. Unspecified.

## 7. Categories still missing

- **Determinism flags.** Still no `cudnn.deterministic` / `cudnn.benchmark=False` / cublas workspace config. Resume RNG fidelity uncertain.
- **Operator runbook.** §7 says "operator monitors wandb in real time" with no thresholds or escalation path.
- **Run0 cap of $50 is wasteful.** Run0 has 500 prompts × 1 rollout, no training; realistic cost is well under $5. Setting it at $50 in the uniform-cap approach loses signal that something's wrong if Run0 burns near the cap. Either set per-run caps (Run0=$5, GRPO runs=$50) or document that Run0 is intentionally cap-loose.
- **Smoke config file location.** Not named anywhere.
- **`run1b_grpo.yaml` orphan.** Listed in §1 as a failed-pilot run but not in the matrix (§2 lists run0/1/2/3 only). Branch A cap-bump skips it. If it's truly orphaned, delete the yaml. If it's a kept-around backup, doc should say so.
- **`gate_decision.json` flow.** §9 defers re-evaluation but matrix start doesn't reference it. If Run0's re-evaluated gate says `PIVOT`, does the matrix continue?

---

## Reconciliation with v1 audit

**Fixed in revision (no longer issues):**
- Step 25 vs 20 across §5 kill rule + outcome rule ✓
- Per-run yaml caps explicitly enumerated ✓
- Smoke math reconciled (cap ≤$10, step count 3) ✓
- Mechanism threshold unified at 0.9 ✓
- §10 numbering ✓
- Catch-all decision rule ✓
- `execute.py:63,152` clamp lift assigned to B6 ✓
- `@modal.exit` flush handler added (placement still ambiguous — see §3) ✓
- Asymmetric clip surface enumerated ✓
- Heartbeat config keys ✓
- wandb offline default ✓
- Data hash pinning ✓
- Checkpoint retention GC ✓
- WANDB secret provisioning step in §7 ✓

**Carried forward (still issues):**
- C3 "flag for cluster_id hash" undefined.
- C6 F-GRPO mechanism formula punted to implementing agent.
- Determinism flags absent.
- Operator runbook absent.
- `gate_decision.json` flow unspecified.

**New regressions introduced by the revision:**
- §3 pipeline diagram and Branch A/C acceptance criteria not updated to match new smoke (§1, §3 above).
- Branch A "Files touched" wrong (§2 above).
- §5 row 2 still says "20 steps" (§1 above).
- `load_checkpoint` signature mismatch within Branch A (§3 above).
- Append-mode seek pseudocode is broken (§3 above).
- Smoke criterion #10 "4-prompt eval" mechanism unspecified (§6 above).
