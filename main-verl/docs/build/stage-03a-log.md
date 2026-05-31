# Stage 3a log — minority_cot skeleton with mock cluster IDs

**Orchestrator:** claude-sonnet-4-6 (executor agent, 2026-05-30)
**Stage ID:** `stage-03a`
**Image rebuild count:** 4 (inherited from Stage 2; S3a.2 patch NOT yet applied to image — see S3a.2 section below)
**Config-fix count:** 0
**Plan:** [`stage-03a-agent-plan.md`](./stage-03a-agent-plan.md)
**Reward stack (LOCKED):** [`../reward-decision.md`](../reward-decision.md) — MathReward (math.py) via patched router. Unchanged from Stage 2.
**Modal app name:** `cs224r-verl-stage03a`

---

## Dispatch log

| Section | Executor | Audit | Verdict |
|---------|----------|-------|---------|
| S3a.1 | DONE | pending | — |
| S3a.2 | DONE (local artifacts) | pending | patch dry-run OK; image rebuild deferred |
| S3a.3 | DONE | pending | pytest 8/8 PASS |
| S3a.4 | DONE | pending | — |
| S3a.5 | DONE | pending | — |
| S3a.6 | — | pending | blocked: image rebuild needed first |
| S3a.7 | — | pending | blocked on S3a.6 |

---

## S3a.1 — Mock cluster generator (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifact:** `main-verl/train/clusters_mock.py`

### Summary

Created `main-verl/train/clusters_mock.py` implementing:

- `ClusterAssignment` frozen dataclass: `cluster_ids` (`torch.int64`, `[n_prompts, n_rollouts]`) + `diagnostics` dict (`distinct_clusters_mean`, `degenerate_rollouts`).
- `assign_mock_clusters(problem_ids, n_rollouts, n_clusters, *, seed)` — public entrypoint, pure Python + hashlib, no GPU ops.
- `_mock_cluster(seed, problem_id, rollout_idx, K)` — `hashlib.blake2b` digest-size-8, process-stable. Python `hash()` explicitly excluded (PYTHONHASHSEED-salted).

File header documents the Stage 3b → judge swap contract: which fields stay (`cluster_ids` shape, `diagnostics` keys), which change (`degenerate_rollouts` always 0 in mock; real judge fills it; signature gains `rollout_texts` + `judge_client`).

### Sanity checks

- `n_clusters=4` default (upper end of Stage 4 forced-k candidate range).
- `seed=0` default (overridable via `algorithm.minority_cot.seed`).
- No imports from `main.train.*`.

### TODOs left

- `<!-- TODO -->` confirm `n_clusters=4` once Stage 4 forced-k decision lands.
- `<!-- TODO -->` add `pytest.mark.parametrize` smoke on `_mock_cluster` (covered by `test_mock_cluster_reproducibility` in S3a.3; no additional parametrize added yet).

### Audit checklist (blank = pending)

- [ ] File at `main-verl/train/clusters_mock.py`
- [ ] Public surface: `ClusterAssignment` + `assign_mock_clusters`
- [ ] Uses `hashlib.blake2b`, not `hash()`
- [ ] `cluster_ids` is `torch.int64` shape `[n_prompts, n_rollouts]`
- [ ] Diagnostics dict: `distinct_clusters_mean`, `degenerate_rollouts` (0 in mock)
- [ ] No imports from `main.train.*` or `main.probes.*`
- [ ] File header documents Stage 3b swap contract

---

## S3a.2 — Objective math + @register_adv_est hook (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifacts:**
  - `main-verl/train/objective_minority.py`
  - `main-verl/infra/patches/maxrl_minority_cot_adv_est.patch`
- **Hook iteration count:** 0 (patch authored; no Modal rebuild yet)

### objective_minority.py

Ported verbatim from `main/train/objective.py`:

- Constants: `N_ROLLOUTS=8`, `SUBSET_SIZE=4`, `_SIZE4_SUBSETS`, `_SUBSET_ARR`, `_INCL`
- `AdvantageOut` dataclass
- `_marginal_from_fG(fG)`
- `_minority_subset_score(rewards4, clusters4, rng)`
- `set_based_marginal_advantages(rewards, clusters, subset_score_fn, *, needs_rng, global_seed, problem_ids)`
- `_minority_advantages(rewards, clusters, *, global_seed, problem_ids)`

Adapter helpers (new, Stage 5 reuses unchanged):

- `_group_rewards_by_index(token_level_rewards, response_mask, index, n_rollouts)` → `(rewards [n_prompts, n_rollouts], problem_ids)`
- `_scatter_advantages_to_tokens(per_rollout_adv, index, response_mask)` → `[batch, response_length]`

Shim for unit tests:

- `compute_advantages(arm, rewards, clusters, *, global_seed, problem_ids)` — accepts only `"minority_cot"`. Raises `ValueError` for any other arm string (including `"minority_answer"` — out of scope per migration plan §1; `"poly_epo_*"` — Stage 5).

### maxrl_minority_cot_adv_est.patch

Fetched `core_algos.py` from pinned commit `7197bbb46a2ecd866da52f6b401ff20a34fe9390`.

Patch adds two hunks:

1. `MINORITY_COT = "minority_cot"` to `AdvantageEstimator` enum (after `MAXRL = "maxrl"`).
2. `compute_minority_cot_outcome_advantage(...)` decorated with `@register_adv_est(AdvantageEstimator.MINORITY_COT)` — inserted after `compute_maxrl_outcome_advantage`.

Function signature mirrors `compute_maxrl_outcome_advantage` (`token_level_rewards`, `response_mask`, `index`, `epsilon`, `norm_adv_by_std_in_grpo`) plus `config=None, **kwargs` for Hydra knob threading.

Function body:
- Reads `n_clusters`, `seed`, `global_seed` from `config.algorithm.minority_cot.*` (with safe defaults 4/0/0).
- Calls `_group_rewards_by_index` → `assign_mock_clusters` → `_minority_advantages` → `_scatter_advantages_to_tokens`.
- Returns `(token_adv, token_adv)` matching the `(advantages, returns)` tuple convention of `compute_maxrl_outcome_advantage`.

### Patch dry-run result

```
$ mkdir -p /tmp/patchtest_3a/verl/trainer/ppo
$ cp /tmp/core_algos_pinned.py /tmp/patchtest_3a/verl/trainer/ppo/core_algos.py
$ cd /tmp/patchtest_3a
$ patch -p1 --dry-run < main-verl/infra/patches/maxrl_minority_cot_adv_est.patch
checking file verl/trainer/ppo/core_algos.py
[exit 0 — no error output]
```

**Patch dry-run: PASS.**

Post-apply spot check (real apply to `/tmp/patchtest_3a`):
```
129:    MINORITY_COT = "minority_cot"
447:@register_adv_est(AdvantageEstimator.MINORITY_COT)
448:def compute_minority_cot_outcome_advantage(
```

### Image rebuild TODO

**TODO (next rebuild):** Add the following step to `infra/modal_image.py` AFTER the existing `maxrl_polaris_math_reward.patch` apply step:

```python
.run_commands(
    "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_minority_cot_adv_est.patch",
)
```

Do NOT merge into the same `run_commands` call as the math-reward patch — separate layers allow rollback.  Sequence this rebuild with the completion of the Stage 2 attempt-7 smoke (currently awaiting Nancy's launch).  This will be **image rebuild count 5**.

The pre-flight assertion in `probes/minority_cot_smoke.py` will catch any patch-application failure before Ray spins up:
```python
assert "minority_cot" in ADV_ESTIMATOR_REGISTRY
```

### Audit checklist (blank = pending)

- [ ] File at `main-verl/train/objective_minority.py`
- [ ] Patch at `main-verl/infra/patches/maxrl_minority_cot_adv_est.patch`
- [ ] `infra/modal_image.py` applies new patch at build (TODO pending next rebuild)
- [ ] Math functions byte-equivalent to `main/train/objective.py`
- [ ] `AdvantageEstimator.MINORITY_COT = "minority_cot"` in patched `core_algos.py`
- [ ] `compute_minority_cot_outcome_advantage` registered via `@register_adv_est`
- [ ] Signature matches `compute_maxrl_outcome_advantage` shape + `config=None, **kwargs`
- [ ] Returns `(advantages, returns)` tuple — returns same tensor twice
- [ ] No imports of `main.train.{trainer,reward,weight_sync,rollout,loss}`
- [ ] No `algorithm.adv_estimator=maxrl` in code or config
- [ ] Hook iteration count: 0 (pending first Modal rebuild and end-to-end test)

---

## S3a.3 — Unit tests (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifact:** `main-verl/tests/test_objective_minority.py`

### Pytest output

```
$ PYTHONPATH=main-verl:main /path/to/main/.venv/bin/python -m pytest main-verl/tests/test_objective_minority.py -v
============================= test session starts ==============================
platform darwin -- Python 3.13.9, pytest-9.0.3, pluggy-1.6.0
collecting ... collected 8 items

main-verl/tests/test_objective_minority.py::test_collapsed_cluster_filtered PASSED [ 12%]
main-verl/tests/test_objective_minority.py::test_minority_seven_one_split_signs PASSED [ 25%]
main-verl/tests/test_objective_minority.py::test_minority_marginals_match_reference PASSED [ 37%]
main-verl/tests/test_objective_minority.py::test_tiebreak_reproducible_same_seed PASSED [ 50%]
main-verl/tests/test_objective_minority.py::test_minority_arm_requires_clusters_and_seed PASSED [ 62%]
main-verl/tests/test_objective_minority.py::test_subset_constants PASSED [ 75%]
main-verl/tests/test_objective_minority.py::test_mock_cluster_reproducibility PASSED [ 87%]
main-verl/tests/test_objective_minority.py::test_mock_clusters_drive_minority_advantages_end_to_end PASSED [100%]

============================== 8 passed in 4.36s ===============================
```

**All 8 tests PASS.**

### Tests included

| Test | Source |
|------|--------|
| `test_collapsed_cluster_filtered` | Ported from `main/tests/test_objective_minority.py:32-45`; arm → `minority_cot` |
| `test_minority_seven_one_split_signs` | Ported from `main/tests/test_objective_minority.py:48-70`; arm → `minority_cot` |
| `test_minority_marginals_match_reference` | Ported from `main/tests/test_objective_minority.py:73-92`; arm → `minority_cot` |
| `test_tiebreak_reproducible_same_seed` | Ported from `main/tests/test_objective_minority.py:115-128`; arm → `minority_cot` |
| `test_minority_arm_requires_clusters_and_seed` | Ported from `main/tests/test_objective_minority.py:131-140`; arm → `minority_cot` |
| `test_subset_constants` | Ported from `main/tests/test_objective_minority.py:153-155` |
| `test_mock_cluster_reproducibility` | **New** — S3a.1 surface |
| `test_mock_clusters_drive_minority_advantages_end_to_end` | **New** — mock→objective bridge |

Tests NOT ported (per plan): `test_poly_epo_*` (Stage 5), `test_grpo_unchanged` (verl built-in).

### Audit checklist (blank = pending)

- [ ] File at `main-verl/tests/test_objective_minority.py`
- [ ] All 6 fixtures from source ported (minus 2 poly-epo + 1 GRPO)
- [ ] Test arm string is `"minority_cot"` throughout
- [ ] `compute_advantages` shim does NOT accept `"minority_answer"` or `"poly_epo_*"`
- [ ] `test_mock_cluster_reproducibility` present
- [ ] `test_mock_clusters_drive_minority_advantages_end_to_end` present
- [ ] pytest output shows all green (recorded above)
- [ ] No imports from `main.train.*`

---

## S3a.4 — Hydra config (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifact:** `main-verl/configs/minority_cot_smoke_1p7b.yaml`

### Summary

Created by copying `grpo_smoke_1p7b.yaml` verbatim and applying the following delta:

| Key | Old (Stage 2) | New (Stage 3a) |
|-----|--------------|----------------|
| `algorithm.adv_estimator` | `grpo` | `minority_cot` |
| `algorithm.minority_cot` | — | `{n_clusters: 4, seed: 0, global_seed: 0}` |
| `trainer.experiment_name` | `grpo_smoke_1p7b` | `minority_cot_smoke_1p7b` |
| `trainer.default_local_dir` | `.../grpo_smoke_1p7b` | `.../minority_cot_smoke_1p7b` |
| `+trainer.wandb_kwargs.tags` | `[verl, stage-02, grpo, smoke]` | `[verl, stage-03a, minority_cot, smoke, mock_clusters]` |

All other knobs carried verbatim from Stage 2 final config: `ppo_micro_batch_size_per_gpu=4`, `gpu_memory_utilization=0.45`, `max_prompt_length=1024`, `data.truncation=left`, `rollout.n=8`, `enforce_eager=true`, `model_dtype=bfloat16`.

Header comment documents: mock-only nature of `minority_cot.*` knobs, Stage 2 KL=0 + `loss_agg_mode=token-mean` inheritance, no `adv_estimator=maxrl`.

### Audit checklist (blank = pending)

- [ ] File at `main-verl/configs/minority_cot_smoke_1p7b.yaml`
- [ ] `algorithm.adv_estimator: minority_cot`
- [ ] `algorithm.minority_cot` block with `n_clusters`, `seed`, `global_seed`
- [ ] `reward_model.enable: false`, no `custom_reward_function.path`
- [ ] `actor_rollout_ref.model.path: Qwen/Qwen3-1.7B-Base`
- [ ] `actor_rollout_ref.rollout.n: 8`
- [ ] `data.train_files` / `data.val_files` point at Stage 2's uploaded parquet paths
- [ ] `trainer.total_training_steps: 50`, `nnodes: 1`, `n_gpus_per_node: 4`
- [ ] Micro-batch / util / prompt-length / truncation values from Stage 2 final config
- [ ] `enforce_eager: true` + `model_dtype: bfloat16`
- [ ] W&B tags include `verl`, `stage-03a`, `minority_cot`, `mock_clusters`
- [ ] Header comment documents mock-only nature

---

## S3a.5 — Modal probe + launch script (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifacts:**
  - `main-verl/probes/minority_cot_smoke.py`
  - `main-verl/scripts/launch_minority_cot_smoke.sh` (chmod +x)
  - `main-verl/README.md` (one bullet added to Bring-up section)

### Summary

`probes/minority_cot_smoke.py`: copied from `grpo_smoke.py` with:
- Function `grpo_smoke` → `minority_cot_smoke`.
- Config name `grpo_smoke_1p7b` → `minority_cot_smoke_1p7b`.
- Pre-flight registry assertion added after `main_ppo` import:
  ```python
  from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
  assert "minority_cot" in ADV_ESTIMATOR_REGISTRY, "..."
  ```
  Fails fast before Ray, saving ~3 min per failed hook iteration.
- Same volumes (`ARTIFACTS_MOUNT`, `HF_CACHE_MOUNT`), same secrets (`HUGGINGFACE`, `WANDB_API_KEY`), same `gpu="B200:4"`, same `timeout=3*3600`.

`scripts/launch_minority_cot_smoke.sh`: default `CS224R_APP_NAME=cs224r-verl-stage03a`.

`README.md`: added bullet:
> minority_cot smoke (Stage 3a, mock clusters): `export CS224R_APP_NAME=cs224r-verl-stage03a && ./main-verl/scripts/launch_minority_cot_smoke.sh`

### Audit checklist (blank = pending)

- [ ] File at `main-verl/probes/minority_cot_smoke.py`
- [ ] File at `main-verl/scripts/launch_minority_cot_smoke.sh`, executable, `set -euo pipefail`
- [ ] Pre-flight registry assertion present
- [ ] Subprocess invokes `python -m verl.trainer.main_ppo` with `--config-name minority_cot_smoke_1p7b`
- [ ] Default `CS224R_APP_NAME=cs224r-verl-stage03a`
- [ ] Volumes + secrets match Stage 2
- [ ] No imports from `main.train.*`
- [ ] No judge / HTTP client code
- [ ] README bullet added

---

## S3a.6 — Remote 50-step smoke (pending)

**Blocked on:** image rebuild that applies `maxrl_minority_cot_adv_est.patch` to the Modal image.

**Preconditions before launch:**
1. Stage 2 attempt-7 smoke completes (Nancy dispatch).
2. `infra/modal_image.py` updated to apply the S3a.2 patch (add `run_commands` step after math-reward patch).
3. New image build triggered (rebuild count 4 → 5).
4. S3a.1–S3a.5 audits passed.

**Launch command (once unblocked):**
```bash
export CS224R_APP_NAME=cs224r-verl-stage03a
export MODAL_PROFILE=chicken602
./main-verl/scripts/launch_minority_cot_smoke.sh 2>&1 | tee /tmp/s3a.6_minority_cot_smoke.log
```

---

## Stage 3a executor summary

### Sections completed (S3a.1–S3a.5)

| Section | Status | Key artifact |
|---------|--------|-------------|
| S3a.1 | DONE | `main-verl/train/clusters_mock.py` |
| S3a.2 | DONE (local) | `main-verl/train/objective_minority.py`, `main-verl/infra/patches/maxrl_minority_cot_adv_est.patch` |
| S3a.3 | DONE (8/8 PASS) | `main-verl/tests/test_objective_minority.py` |
| S3a.4 | DONE | `main-verl/configs/minority_cot_smoke_1p7b.yaml` |
| S3a.5 | DONE | `main-verl/probes/minority_cot_smoke.py`, `main-verl/scripts/launch_minority_cot_smoke.sh` |
| S3a.6 | BLOCKED | needs image rebuild |
| S3a.7 | BLOCKED | needs S3a.6 |

### Artifacts created under main-verl/

- `train/clusters_mock.py` — mock cluster ID source
- `train/objective_minority.py` — ported math + adapter helpers + compute_advantages shim
- `infra/patches/maxrl_minority_cot_adv_est.patch` — core_algos.py patch (dry-run verified)
- `tests/test_objective_minority.py` — 8 tests, all PASS
- `configs/minority_cot_smoke_1p7b.yaml` — Stage 3a Hydra config
- `probes/minority_cot_smoke.py` — Modal function with pre-flight registry check
- `scripts/launch_minority_cot_smoke.sh` — launch script (chmod +x)
- `README.md` — added Bring-up bullet
- `infra/modal_image.py` — added TODO comment for next rebuild
- `docs/build/stage-03a-log.md` — this file

### What the human needs to do before S3a.6 Modal smoke

1. **Complete Stage 2 attempt-7 smoke** (currently awaiting Nancy's `./main-verl/scripts/launch_grpo_smoke.sh` launch).

2. **Add S3a.2 patch to the image** — edit `infra/modal_image.py`: after the existing `maxrl_polaris_math_reward.patch` run_commands step, add:
   ```python
   .run_commands(
       "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_minority_cot_adv_est.patch",
   )
   ```
   This is **image rebuild 5** (budget: ≤2 new rebuilds for Stage 3a; this is rebuild 1 of that budget).

3. **Run audits for S3a.1–S3a.5** (dispatch audit agent).

4. **Launch the smoke** (once audits pass + image rebuilt):
   ```bash
   export CS224R_APP_NAME=cs224r-verl-stage03a
   export MODAL_PROFILE=chicken602
   ./main-verl/scripts/launch_minority_cot_smoke.sh 2>&1 | tee /tmp/s3a.6_minority_cot_smoke.log
   ```

### Highest-risk items for human to verify before S3a.6

1. **Hook wiring: `config` kwarg threading.** The registered function reads `config.algorithm.minority_cot.*` via `getattr` with safe defaults. This requires verl's trainer to pass `config=` as a kwarg to the registered estimator. If the maxrl fork calls `fn(token_level_rewards, response_mask, index)` positionally without `config=`, the knobs fall back to hardcoded defaults (4/0/0) rather than raising — functionally safe but the Hydra config block would be silently ignored. Mitigation: the pre-flight assertion only checks registration, not kwarg threading. Add a `print(f"minority_cot config read: n_clusters={n_clusters}")` log line inside the function for the first run to confirm.

2. **Patch line-number stability.** The patch was authored against `MAXRL_COMMIT=7197bbb46a2ecd866da52f6b401ff20a34fe9390` and dry-run verified against the same. If Modal's `git checkout` produces a different file (e.g. line-ending differences on the checkout), the patch may fail with "Hunk #N FAILED". Mitigation: `patch` with `--fuzz=3` as a fallback; or inspect the image build log for the patch apply step.
