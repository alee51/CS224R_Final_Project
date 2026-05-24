# PILOT_REDESIGN audit (2026-05-19)

Independent verification of `pilot/docs/operations/PILOT_REDESIGN.md` against the codebase and supporting analysis docs. No rewrite — findings only.

---

## 1. Internal consistency contradictions

- **Step target 25 vs 20.** §2 locks "~25 steps/run." §5 kill rule says "$50 cap before step **20**"; §5 outcome rule says "after step ~20"; §2 mini-eval cadence (5/10/15/20) omits step 25.
- **Run YAML caps still $36/$24.** §2 declares "$50/run hard cap." Current `run0_proxy.yaml:9`=$24, `run1*/run2/run3`=$36. Branch A writes `budget_cap_usd: 50.0` only to `shared_train.yaml`; per-run yamls override. §4.A has no step to bump them.
- **Smoke cost math broken.** §6 targets `<$2` with pass-criterion `step 1 < 60 min` × 5 steps. At $0.000694/s, one 60-min step is $2.50. Even optimistic 42–50 min/step gives 5 steps ≈ $7–10. `<$2` unreachable.
- **Smoke step count vs variant coverage.** §6 says "5 training steps" with preempt mid-step-3, but pass criterion 5 says "GRPO + inverse_freq + F-GRPO sequentially, one step each." 5 steps and 3 variants × 1 step are different runs.
- **Mechanism threshold mismatch.** §5 kill at `<0.9`, §6 acceptance demands `>=0.95`. The 0.90–0.95 gap is undefined at scale.
- **§10 missing.** §7 references "§10" but doc jumps §9 → §11.

---

## 2. Code-citation accuracy

All numeric file:line references verified against current code:

- `rollout_engine.py:15-19` PROMPT_TEMPLATE, `:107-147` seeded-batched path — accurate.
- `hf_grpo_train.py:~865` `write_text("")` (actual 866), `:846-847` `gradient_checkpointing_enable`, `:841-845` `from_pretrained`, `:862` vanilla `AdamW`, `:962-966` / `:991-996` duplicate `save_pretrained` — all accurate.
- `modal_app.py:63-71` pip list (no `flash-attn`/`wandb`) — accurate.
- `canonicalize.py:11` strips all `}` — accurate.
- `HFPolicyModel` (`:298`) and `_differentiable_loss` (`:502`) exist and are unreachable from `_train_step_microbatch_backward` (`:926`) — dead-code claim holds.

No stale citations. External `nancy_explore/*` and `ifdita_meeting_transcript.md` references not verified by this audit.

---

## 3. Spec completeness per branch

**Branch A — ambiguities:**
- `save_checkpoint(step, policy, optimizer, rng_state, preds_offset)` and `load_checkpoint(...)` are referenced but undefined; signatures differ between the two pseudocode blocks.
- `step_is_natural_boundary` used in the gating expression, never defined.
- The locked-constraint promise "budget_cap enforced … every 60s during train phase" appears in §2 and §7 pre-matrix checks, but no §4.A code change implements an in-train-loop 60 s budget check. Today's check is between steps only (`hf_grpo_train.py:876`).
- No instruction to bump per-run yaml caps from $36 → $50.
- No mention of `budget_cap_gpu_hours` (currently a no-op per `0519_perf_consolidated.md` §B6).
- Modal `@modal.exit` flush — flagged in `0519_perf_consolidated.md` §B5 as the preempt-grace lever — is absent.

**Branch B — ambiguities:**
- B1 parity validation has no tolerance (e.g., `|Δmean_reward| < ?`).
- B3 `flash-attn` version pin "matching the torch/CUDA combo" — agent must choose.
- B5 introduces new keys `clip_ratio_high` / `clip_ratio_low`; current code reads single `clip_eps` (`hf_grpo_train.py:855`, `objectives.py` `_clip_surrogate_*`). The "update clip path" instruction is one line but the actual asymmetric-clip surface is wider than the doc shows.

**Branch C — ambiguities:**
- C3 "flag for the cluster_id hash" — semantics, storage, and consumer all undefined.
- C6 F-GRPO mechanism formula: punted to implementing agent. Doc itself flags this as open (§8 risk #2) but doesn't supply a fallback.
- C7 `mode="online" if WANDB_API_KEY else "disabled"` directly contradicts §8 risk #3, which proposes `offline` + sync as the fallback.
- C8 heartbeat: `completions_done % 32` is hardcoded; no config.
- **`execute.py:63,152` clamp `max_new_tokens` to 1024.** Branch C touches `execute.py` but does not lift this clamp. §2 locks `max_new_tokens=1536` for both train and eval; without removing the clamp, eval and `run0_proxy` will silently run at 1024 — the exact "silent behavioral asymmetry" called out in `0519_perf_consolidated.md` §B2.
- `run0_proxy` has no training loop. C5/C6 logging spec doesn't say what subset applies to run0 — the validity-gate path is underspecified.

---

## 4. Decision-rule coherence

**Plausible outcome not covered:**

> All three GRPO variants pass mechanism, no Pass@1 collapse, **one variant leads on Pass@1 by >3 pt** but Cover@τ is within 1 pt across the board.

§5 escalation requires `Cover@τ > 3 pt over baseline`. "Within 2 pt on all metrics" requires both. A Pass@1-only lead falls between the two rules — no documented action.

**Secondary gaps:**
- Kill rules need a rolling window of ≥3 steps; a run that dies on step 2 (budget/preempt) has no rolling window. Decision unspecified.
- §5 outcomes assume ≥1 variant survives. If two variants die on mechanism and the third survives but ties baseline, there is no rule.
- Mechanism threshold straddle (0.90 ≤ signal < 0.95 at smoke vs train) — see §1.

---

## 5. Perf-bundle math

The doc does **not** state "55–60 min/step" — §4.B acceptance says `< 60 min` for smoke step 1 (ceiling, not point estimate); §4.B fallback target `< 80 min`.

Against `0519_perf_audit.md` (top-3-applied: ~42–50 min/step at 2048 tokens) and `0519_perf_consolidated.md`: at 1536 tokens (~75 % of horizon) and with B4 fused AdamW added, ~32–45 min/step is plausible. The `< 60 min` target is **conservative and internally consistent** with the audit. The `< 80 min` fallback (grad-checkpointing-on path) is also consistent with the audit's `~73 + B3` math.

The `$50 / run × 25 steps` budget implies an average ≤ 48 min/step. If step time lands at 50–60 min (within the doc's stated tolerance), runs hit the $50 cap at 16–20 steps, not 25 — consistent with §2 ("whichever first") but worth flagging because §5 kill/outcome rules quote `step 20` as if it were the planned stop.

---

## 6. Smoke changes not exercised

- `execute.py:63,152` 1024 clamp — not touched, not tested.
- §2 B5 asymmetric clip + `kl_coef=0` — labelled "correctness, not perf"; smoke acceptance has no objective-correctness check.
- `$50` budget-cap trip — smoke targets $2; never trips.
- `eval at 1536` — eval not in smoke at all.
- `run0_proxy` 500-prompt validity path — not exercised.
- `matrix burst $200 projection` — no smoke analog.
- Mini-eval-every-5-steps — within smoke window (5 steps) but not in §6 pass criteria.
- Per-run yaml budget cap updates ($50) — no smoke step verifies these were applied.

---

## 7. Categories the doc should address but doesn't

- **§10 missing** (see §1).
- **Data pinning** — no hash/version for `dapo_slice_3k.jsonl`.
- **Checkpoint retention** — per-step × 4 runs × N steps × ~3.4 GB will fill the artifacts volume; no GC policy.
- **Merge ownership** — §7 says "manual or agent-driven"; conflict resolvers in `shared_train.yaml` / `hf_grpo_train.py` / `modal_app.py` not assigned.
- **`@modal.exit` preempt-grace** — known lever from `0519_perf_consolidated.md` §B5; absent.
- **Eval spec** — no script, no `max_new_tokens=1536` plumbing through `execute.py:152`.
- **Secret provisioning** — `WANDB_API_KEY` referenced in C7/C9, no Modal `Secret` creation step in §7.
- **Determinism flags** — no `cudnn.deterministic`/`cublas` settings; resume RNG fidelity uncertain.
- **Salvage re-analysis** — no plan to re-canonicalize prior salvage with the fixed parser.
- **`gate_decision.json` flow** — §9 defers re-evaluation but no rule on whether the matrix branches on it.
- **Operator runbook** — "monitor wandb in real time" has no thresholds, escalation, or pull cadence.
