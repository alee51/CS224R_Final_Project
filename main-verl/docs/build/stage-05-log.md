# Stage 5 run log — `poly_epo_cot`

**Modal profile:** `chicken602` (all Stage 5 smokes and image rebuilds)

---

## S5.1

- **Executor:** S5.1 executor agent
- **Timestamp (UTC):** 2026-05-31T03:26:35Z
- **Verdict:** PASS

### Pre-staged artifacts verified

| File | Status |
|------|--------|
| `main-verl/train/objective_poly_epo.py` | present |
| `main-verl/infra/patches/maxrl_poly_epo_cot_adv_est.patch` | present (regenerated — see hook iteration) |
| `main-verl/infra/patches/maxrl_poly_epo_cot_ray_trainer.patch` | present (unchanged) |

### Patch dry-run (maxrl @ `7197bbb46a2ecd866da52f6b401ff20a34fe9390`)

Applied Stage 3b stack first, then dry-ran/applied Stage 5 patches (poly_epo patches depend on minority_cot hook being present):

| Patch | Dry-run |
|-------|---------|
| `maxrl_polaris_math_reward` | OK |
| `maxrl_minority_cot_adv_est` | OK |
| `maxrl_minority_cot_ray_trainer` | OK (offset +1) |
| `maxrl_expose_data_to_adv_est` | OK |
| `maxrl_poly_epo_cot_adv_est` | OK after regeneration (original hunk targeted wrong context post-3b) |
| `maxrl_poly_epo_cot_ray_trainer` | OK (offset +5) |

**Note:** Sequential `--dry-run` on a clean tree fails for poly_epo patches because earlier dry-runs do not mutate the tree. Correct procedure: dry-run patches 1–4, apply them, then dry-run/apply 5–6.

### Hook iteration count

**1** — Regenerated `maxrl_poly_epo_cot_adv_est.patch` against post-3b `core_algos.py` (original patch inserted duplicate scatter lines at wrong hunk). `maxrl_poly_epo_cot_ray_trainer.patch` unchanged.

### `modal_image.py` edit

Added two `.run_commands` layers after `maxrl_expose_data_to_adv_est.patch`:

- `maxrl_poly_epo_cot_adv_est.patch`
- `maxrl_poly_epo_cot_ray_trainer.patch`

### Local import check

```bash
PYTHONPATH=main-verl python3 -c "from train.objective_poly_epo import compute_advantages_poly_epo_cot"
# local import OK
```

### Image rebuild

- **Prior count (Stage 3b):** 5
- **New count:** **6** (Stage 5 rebuild 1 of budget ≤2)
- **Profile:** `MODAL_PROFILE=chicken602`
- **Trigger:** `PYTHONPATH=main-verl python3 -m modal run main-verl/probes/poly_epo_cot_registry_assert.py`
- **Build log highlights:** both poly_epo patch steps applied cleanly (`patching file verl/trainer/ppo/core_algos.py`, `patching file verl/trainer/ppo/ray_trainer.py` Hunk #1 succeeded at 449 offset +5)
- **Modal run URL:** https://modal.com/apps/chicken602/main/ap-QoxhMrwWIxXyVN7nhSgZp6
- **Status:** completed successfully (~6 min wall time for full image bake)

### In-container registry assert

Probe: `main-verl/probes/poly_epo_cot_registry_assert.py` (CPU-only, no GPU)

```python
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, AdvantageEstimator
assert "poly_epo_cot" in ADV_ESTIMATOR_REGISTRY
assert AdvantageEstimator.POLY_EPO_COT.value == "poly_epo_cot"
```

**Result:** PASS — `registry assert OK: poly_epo_cot in ADV_ESTIMATOR_REGISTRY`

### Files touched (S5.1)

- `main-verl/infra/modal_image.py` — two new patch apply steps
- `main-verl/infra/patches/maxrl_poly_epo_cot_adv_est.patch` — regenerated
- `main-verl/probes/poly_epo_cot_registry_assert.py` — new minimal registry probe

---

## S5.1 — Audit

- **Auditor:** orchestrator + read-only subagent
- **Timestamp (UTC):** 2026-05-30
- **Verdict:** PASS

| Item | Result |
|------|--------|
| Local `objective_poly_epo` import | PASS |
| `modal_image.py` poly_epo patches after `expose_data` | PASS |
| Cumulative patch apply (dry-run caveat documented) | PASS WITH NOTE |
| Image rebuild count **6** | PASS |
| In-container `"poly_epo_cot"` registry | PASS (`pre-flight: poly_epo_cot registered — OK`, Modal app `ap-hkkJAfcjcJsNv9sIsQDLW9`) |
| Hook iteration ≤2 | PASS (count **1**) |

---

## S5.2 — Unit tests (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifact:** `main-verl/tests/test_objective_poly_epo.py`
- **pytest:** 4/4 poly_epo + 8/8 minority regression — all green (exit 0)

**Verdict: PASS**

---

## S5.2 — Audit

- **Auditor:** read-only subagent
- **Timestamp (UTC):** 2026-05-30
- **Verdict:** PASS WITH NOTES (log documentation gap closed above)

---

## S5.3 — Hydra config (executor)

See prior S5.3 block in dispatch — **PASS** (independent audit confirms checklist).

---

## S5.3 — Audit

- **Auditor:** read-only subagent
- **Timestamp (UTC):** 2026-05-30
- **Verdict:** PASS

---

## S5.4 — Modal probe + launch script (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifacts:** `probes/poly_epo_cot_smoke.py`, `scripts/launch_poly_epo_cot_smoke.sh`, README bullet
- **Verdict:** PASS

---

## S5.4 — Audit

- **Auditor:** read-only subagent
- **Timestamp (UTC):** 2026-05-30
- **Verdict:** PASS

---

## S5.5 — Remote smoke (10-step, parity with Stage 3a)

- **Status:** CANCELLED attempt 1; config fixed to 10 steps (2026-05-30)
- **Precondition blocker (cross-arm):** `stage-03a-log.md` has **no** Stage 3a mock `minority_cot` W&B run id. `STATUS.md` records 10-step smoke + ckpt at `global_step_10` only. Cross-arm `train/mean_advantage` vs 3a at steps 10/25/50 is **BLOCKED** until baseline run id is logged (or 50-step minority mock smoke is run and recorded).
- **Mitigation:** Run smoke anyway; unit-test cross-arm diff already PASS (S5.2). Log poly_epo metrics; mark cross-arm table TBD.

**Launch:**

```bash
export CS224R_APP_NAME=cs224r-verl-stage05
export MODAL_PROFILE=chicken602
./main-verl/scripts/launch_poly_epo_cot_smoke.sh 2>&1 | tee /tmp/s5.5_poly_epo_cot_smoke.log
```

### Attempt 1 — cancelled (orchestrator)

- **Timestamp (UTC):** 2026-05-31T03:28:51Z (launch) — stopped per Nancy/orchestrator directive before completion
- **Modal app:** https://modal.com/apps/chicken602/main/ap-fJY7WU4AHVgnXVVpV05wwY
- **Verdict:** **CANCELLED** — run stopped at orchestrator request. Executor had targeted **50** steps; Stage 3a `minority_cot` mock baseline is **10-step only** (`total_training_steps: 10`, ckpt at `global_step_10`).
- **Pre-flight (before stop):** `poly_epo_cot` in `ADV_ESTIMATOR_REGISTRY` — OK; Ray/trainer subprocess started; no final metrics or checkpoint captured.
- **Stage 3a W&B baseline (recovered from volume):** run id `0rapw31x` at `/vol/checkpoints/main-verl/minority_cot_smoke_1p7b/wandb_id.txt` — not yet used for cross-arm table (smoke did not finish).

### Config update (post-cancel)

- **`poly_epo_cot_smoke_1p7b.yaml`:** `trainer.total_training_steps: 10`, `trainer.save_freq: 10` (parity with `minority_cot_smoke_1p7b.yaml`).
- **Cross-arm check:** compare `train/mean_advantage` at **step 10** only vs W&B run `0rapw31x` on next approved run.

### Attempt 2 — cancelled (Nancy, 2026-05-30)

- **Modal app:** `ap-Si0AzhqP3ZWGkVR9MB6NFh` — `modal app stop -y` (mid-run; mock-cluster smoke not useful for judge/response quality review)
- **Rationale:** `cluster_source: mock` does not exercise judge or real clustering; S5.5 mock gate deferred until needed for hook-only bring-up
- **W&B (partial):** [ywojfepp](https://wandb.ai/224r-project/cs224r-minority-voting/runs/ywojfepp) — few steps only
- **S5.5 verdict:** **CANCELLED** (acceptable — unit tests + registry already PASS; live mock smoke optional)

---

## Dispatch log

| Section | Executor | Audit | Verdict |
|---------|----------|-------|---------|
| S5.1 | DONE | PASS | PASS |
| S5.2 | DONE | PASS w/ notes | PASS |
| S5.3 | DONE | PASS | PASS |
| S5.4 | DONE | PASS | PASS |
| S5.5 | CANCELLED (attempts 1–2) | — | **OVERRIDDEN** (Nancy: mock smoke not needed) |
| S5.6 | DONE | PASS w/ notes | **PASS WITH NOTES** (override) |

---

## S5.6 Stage gate verdict

- **Verdict:** PASS WITH NOTES
- **Auditor:** orchestrator (human override — Nancy, 2026-05-30)
- **Timestamp (UTC):** 2026-05-30
- **Stage 6 ready:** yes (with ack — 4B smokes are credit-heavy per plan)

### Notes

- **S5.5 override:** Live mock-cluster smoke (S5.5) **not run to completion**. Nancy accepted bring-up without mock training — mock path does not exercise judge/rollout quality; unit-test cross-arm diff (S5.2) + in-container registry (S5.1) treated as sufficient for Stage 5 hook/scorer gate.
- **Satisfied:** S5.1–S5.4 audits PASS; image rebuild **6**; hook iteration **1**; `poly_epo_cot` in registry; `test_objective_poly_epo.py` green; config + probe + README bullet present.
- **Deferred (non-blocking):** S5.5 cross-arm `train/mean_advantage` vs 3a W&B `0rapw31x`; optional `poly_epo_cot` judge trace before Stage 8 (plan optional follow-up).
- **Handoff:** yaml `poly_epo_cot_smoke_1p7b.yaml` → fork for 4B (`poly_epo_cot_smoke_4b.yaml` in Stage 6). Checkpoint dir unused (no completed smoke).
