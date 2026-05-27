# B200 migration plan — audit

**Audited:** 2026-05-27  
**Documents:** [`B200_migration_plan.md`](./B200_migration_plan.md), [`B200_migration_analysis_2026-05-26T034425Z_b01999f.md`](./B200_migration_analysis_2026-05-26T034425Z_b01999f.md), [`status_2026-05-27T0510Z.md`](../../efficiency/status_2026-05-27T0510Z.md)  
**Code verified:** `main/infra/modal_image.py`, `main/train/trainer.py`, `main/train/rollout.py`, `main/train/weight_sync.py`, `main/configs/train_real.yaml`, launch scripts, all live `gpu=` decorators.

---

## Verdict: **GO WITH FIXES**

The plan is directionally correct for a **time-first** poster spike: H200 baseline numbers match repo reality, the 180 GB Modal budgeting call is prudent, vLLM 0.9 + Blackwell FA is the real work, and phased smokes are sensible. **Do not start implementation as written** until blockers below are patched into the plan (or the build agent’s first commits). Without fixes, Phase 4–5 will fail silently (config) or burn a day on train smokes (weight sync).

---

## Blockers (must fix before build)

1. **`train_real_b200.yaml` with `extends:` will not load.** `trainer.load_cfg()` only special-cases legacy shims with keys `<= {arm}`; it does **not** merge `extends` (unlike `group_b_step_probe.load_merged_config`). A two-field overlay yaml will **not** inherit `train`, `rollout`, `arm_profiles`, etc. **Fix:** duplicate full `train_real.yaml` on the branch, add `extends` support to `load_cfg`, or merge in `launch_train.sh` before `modal run`.

2. **Weight-sync validation cannot be deferred to the 10-step train smoke alone.** `tests/test_weight_sync.py` is still `pytest.skip`; `weight_sync.py` hard-codes the vLLM 0.8.5 `driver_worker.model_runner.model` + `load_weights` path. vLLM 0.9.x internal moves are the **highest-probability** break. **Fix:** treat Phase 3 `smoke_weight_sync.py` (or unskipped Modal spike) as a **hard gate** before Phase 5; do not use plan “Option B only.”

3. **Image bring-up spec is under-specified for the actual failure mode.** Today: `debian_slim` + `pip vllm==0.8.5` + Hopper FA2 wheel (`cu12` / `torch2.6`). Plan says “CUDA 12.8+” and “vLLM 0.9 cu128” but does not require an explicit install line (e.g. versioned `cu128` wheel from vLLM docs) or a Modal/NVIDIA base image trial. **Fix:** plan appendix must name the exact pip/ wheel spec and a fallback order (official cu128 wheel → curated base → source).

4. **Rollback is not “one launch” on a merged or misconfigured branch.** Modal hardware comes from `@app.function(gpu=...)` in Python, not yaml. `launch_train.sh --config train_real_b200.yaml` on a branch where `trainer.py` still says `B200` **does not** roll back. True rollback = **`main` + default `launch_train.sh`** (H200 decorator + `train_real.yaml`) or revert image + all `gpu=` sites. Plan should say that explicitly.

---

## Recommendations (should fix)

1. **GPU touchpoint inventory — update analysis, complete plan list.** Verified live decorators (2026-05-27):

   | File | `gpu` today | Plan mentions? |
   |------|-------------|----------------|
   | `train/trainer.py` `train_remote` | `H200` L1310 | Yes (~L1310 ✓) |
   | `probes/smoke_flash_attn.py` | `H200` L135 | Yes (~L135 ✓) |
   | `probes/group_b_step_probe.py` `_MODAL_FN_KWARGS` | `H200` L324 | Yes (~L324 ✓; analysis doc L349 is stale) |
   | `probes/checkpoint_rollout_eval.py` | `H200` L54 | Optional — **should be required** if checkpoint eval runs during B200 training |
   | `probes/stress_n_kept_probe.py` | `H200` L280 | Optional ✓ |
   | `probes/group_a_rollout_judge.py` phase1/2 | **`H100`** L282, L585 | Optional — **still wrong SKU** if anyone runs Group A on the B200 image branch |

   Missing from plan §3: none critical beyond tightening checkpoint_eval. **group_a H100 + B200 image = worst case** (wrong GPU or wrong stack).

2. **Phase ordering tweak:** Run a **minimal vLLM generate** (hello or 1-prompt) immediately after image build **before** full `smoke_flash_attn` collocated stage. Collocated FA smoke already constructs `RolloutEngine` (vLLM) but buries vLLM failures under FA diagnostics. Analysis doc’s order (vLLM → sync → FA) is safer than plan Phase 2→3 if Phase 2 fails late in `collocated`.

3. **Feasibility / calendar:** 1–2 focused days is **optimistic** if Phase 1 needs >2 image iterations (common for Blackwell wheels). Budget **2 days** wall + **+4–12 h** Modal B200 queue (per `b200_deep_dive_verdict_2026-05-26.md`, self-spawn legs). Phase 1 “3–8 h” dominates; parallel Phase 0 is fine.

4. **Success criteria — soften or instrument:**
   - **Rollout ≤ 70 s:** timeline already showed H200 rollout only ~+15% vs H100 at `util=0.45`; 90→70 s (~22%) is plausible but not guaranteed. Keep as stretch; primary gate = **total step ≤ 300 s**.
   - **`n_kept_sequences > 0` most steps:** set arms can legitimately filter everything on some steps — risk false no-go. Gate on finite loss + no NaN + at least one step with `n_kept > 0` in the smoke.
   - **§1.3 full epoch ≤ 65 h:** requires ~292 s/step average; stricter than §1.2 median ≤ 300 s — call out as production stretch, not spike gate.

5. **Economics consistency:** Plan uses H200→B200 **+38% $/s** (0.001261→0.001736) and break-even **~27.4%** faster — correct vs `b200_deep_dive_verdict_2026-05-26.md`. Status doc “~+25% $/s” is **stale**; fix to avoid operator confusion. Realistic **280 s/step** is ~**2% more $/epoch** than H200 at 380 s (per economics note) — plan acknowledges; spike go/no-go should be **wall-clock**, not $/step, for time-first.

6. **`build_hf` / FA:** Plan says “only image wheel changes” — **correct** (`trainer.py` already `attn_implementation="flash_attention_2"`). Analysis snapshot claim “FA not enabled” is **obsolete**. Blackwell work is **wheel/SM100**, not enabling FA in code.

7. **Checkpoint eval:** Uses same `modal_image` and collocated stack. If eval runs while training on B200, bump `checkpoint_rollout_eval.py` gpu + yaml `gpu_class` / `modal_price_per_sec` or eval schedules H200 with a **Hopper image** on a Blackwell branch — silent mismatch.

8. **Production leg chain:** §1.2 defers leg spawn to production; add a **short leg-restart smoke** (e.g. 2 legs × few steps) before full 799-step epoch — B200 queue per `train_remote.spawn` is an underestimated calendar risk.

9. **Wandb / config drift:** `gpu_class` is logged in trainer startup string; `modal_price_per_sec` is **not** auto-logged to wandb in trainer (manual dashboard math). Plan §4.4 wandb group names (`train_real_minority_answer`) don’t match yaml (`train-minority-answer`) — cosmetic only.

---

## Nits (optional)

- Analysis doc frozen at `b01999f`: H100 touchpoints, “FA not in build_hf”, `train_grpo_05-25.yaml` as prod config — plan correctly points at `train_real.yaml` + H200; keep analysis as historical only.
- `hello_modal.py` has no GPU — fine for import smoke; add version print fn as plan suggests.
- Legacy configs (`probe_a_*.yaml`, etc.) still `H100` pricing — harmless if unused; update if re-run.
- Readout path: plan uses `docs/efficiency/B200_readout_*`; analysis suggests `docs/probes/` — pick one.
- `train.microbatch: 64` (= `batch_size`) drives peak VRAM; plan hold-constant is right but explains why H200 already peaks ~130–140 GB and B200 gate `< 175 GB` matters.

---

## Correctness scorecard (plan vs repo)

| Claim | Status |
|-------|--------|
| Prod SKU H200, `trainer.py` ~L1310 | ✓ |
| `vllm==0.8.5`, Hopper FA2 wheel in `modal_image.py` | ✓ |
| H200 pricing 0.001261, B200 0.001736 | ✓ (Modal public pricing; re-check at launch) |
| 180 GB B200 / 141 GB H200 budgeting | ✓ (reasonable for Modal; verify `torch.cuda.get_device_properties` on first B200 boot) |
| Baseline ~380 s/step, ~84 h/epoch | ✓ (aligned with status/timeline) |
| `weight_sync` 0.8.5 API path | ✓ — **will change** on vLLM bump |
| `rollout.gpu_memory_utilization: 0.45` in `train_real.yaml` | ✓ |
| Rollback via yaml alone | ✗ — see blocker 4 |
| `extends` b200 yaml | ✗ — see blocker 1 |
| Analysis “six H100 sites” | ✗ stale — train/group_b/smoke on **H200**; group_a still **H100** |

---

## Risks — harsh read

| Risk | Plan treatment | Audit |
|------|----------------|-------|
| vLLM 0.9 `load_weights` / executor layout | Listed | **Underestimated** without mandatory Phase 3 spike |
| FA2/FA-3 on SM100 | Listed | Correct priority; collocated smoke helps but ≠ train backward peak with `n_kept≈512` |
| `VLLM_USE_V1=0` | Keep | Correct for Modal fork issue |
| B200 queue × spawn legs | Noted | **Underestimated** for full epoch; needs leg smoke |
| $/step at ~280 s | “Time-first OK” | Honest; don’t claim $ win at realistic 280 s |
| vLLM sleep / cumem | Out of scope | Correct — prod `vllm_sleep=0` |
| Sequential logprob loop | Out of scope | Correct — caps HF train speedup on B200 (~180–215 s train in status model) |

---

## Success criteria — achievable?

| Gate | Achievable? |
|------|-------------|
| Stack boot + imports | Yes, if image spec fixed |
| FA2 smoke + collocated &lt; 180 GB | Yes, with correct wheel; collocated is necessary but not sufficient for full train peak |
| Weight sync ≤ 2× H200 | Yes if API intact; **must measure in dedicated spike** |
| Median step ≤ 300 s | **Plausible** (245–280 s band); not guaranteed on first day |
| Peak &lt; 175 GB | **Plausible** (+39 GB vs H200); `microbatch=64` + collocated vLLM still tight — monitor |
| Full epoch ≤ 65 h | **Stretch** — needs ~292 s/step + low queue; use as prod goal post-spike |

---

## Checklist for build agent (ordered, if GO)

Execute on branch `b200-bringup` only; do not merge until Phase 5 green.

1. [ ] **Phase 0:** Branch; record H200 baseline wandb URL (set-arm smoke, 10-step, `train_real.yaml`).
2. [ ] **Phase 1 — image:** Bump `modal_image.py`: vLLM **0.9.x cu128** wheel (explicit URL/version), `transformers` pin smoke with Qwen3; replace FA wheel with **SM100/cu128** (FA-3 or sm100 FA2 per Dao release notes); keep `VLLM_USE_V1=0`; log `torch.cuda`, `vllm`, `flash_attn` versions. `modal run main/infra/hello_modal.py`.
3. [ ] **Phase 1b — vLLM only:** Minimal Modal fn: `LLM(Qwen3-1.7B-Base)`, one generate — **gate** before FA collocated.
4. [ ] **Phase 2 — FA:** `gpu="B200"` in `smoke_flash_attn.py`; `launch_smoke_flash_attn.sh`; require `SUMMARY ok=True`, SM 10.x, collocated VRAM &lt; 180 GB.
5. [ ] **Phase 3 — weight sync (mandatory):** Add `probes/smoke_weight_sync.py` (or equivalent); fix `weight_sync._vllm_runner_model` for 0.9.x; log `SyncStats`; verify logprob/generation shift after HF perturbation.
6. [ ] **Phase 4 — plumbing:** `gpu="B200"` on `train_remote`, `group_b_step_probe`, `checkpoint_rollout_eval` (if eval planned), optional `stress_n_kept_probe`; **fix config strategy** (full `train_real_b200.yaml` or `extends` in `load_cfg` / launch merge); set `gpu_class: B200`, `modal_price_per_sec: 0.001736`.
7. [ ] **Phase 5 — train smoke:** `launch_train.sh --mode smoke --arm minority_answer --config <b200 yaml>`; verify §1.2 gates (wandb: `train/t_rollout_s`, `train/t_train_fwd_bwd_s`, `train/weight_sync_s`, `train/vram_peak_gb_step`, `train/vram_headroom_gb_step`).
8. [ ] **Phase 5b — leg smoke (recommended):** Short run with `train_remote.spawn` / resume on B200; log queue pending time.
9. [ ] **Phase 6 — decision:** If green → full set-arm on B200 + `B200_readout_<ts>_<sha>.md`; if red → stay on `main`/H200 launch path; no merge.
10. [ ] **Do not** change `gpu_memory_utilization`, `token_budget`, `gradient_checkpointing`, or `n_kept` policy during Phases 1–5.

---

## Audit metadata

- **Repo state:** H200 in `trainer.py` / probes (except `group_a_rollout_judge.py` H100).
- **No code changes** in this audit commit.
