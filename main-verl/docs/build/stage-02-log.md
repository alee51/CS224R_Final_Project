# Stage 2 log — GRPO bring-up smoke (1.7B, MathReward)

**Orchestrator:** cursor-agent (2026-05-29)  
**Stage ID:** `stage-02`  
**Image rebuild count:** 3 (S2.3 option (a) + S2.5 attempt 4: drop `expandable_segments` + S2.5b: math_reward router patch)  
**Config-fix count:** 3 (`ray_dir`, `log_prob_micro_batch_size_per_gpu`, `data_source=polaris` parquet) — **exceeded plan budget (≤2)**; next fix `max_prompt_length` = 4th  

**Reward stack (LOCKED 2026-05-29):** [`../reward-decision.md`](../reward-decision.md) — MathReward (`math.py`) + maxrl `\boxed{}` prompt. Patch: `infra/patches/maxrl_polaris_math_reward.patch`.

---

## Dispatch log

| Section | Executor | Audit | Verdict |
|---------|----------|-------|---------|
| S2.1 | DONE | PASS WITH NOTES | parquet + upload OK |
| S2.2 | DONE | PASS | Hydra config OK |
| S2.3 | DONE | PASS WITH NOTES | modal_image + grpo_smoke |
| S2.4 | DONE | PASS | launch script + README |
| S2.5b | DONE | — | MathReward patch + prompt + re-upload |
| S2.5 | DONE | pending | **FAIL** 26/50 — `max_prompt_length` (attempt 6) |
| S2.6 | — | pending | blocked on S2.5 PASS |

---

## S2.1 — Manifest → verl parquet (executor)

- **Timestamp (UTC):** 2026-05-29T20:43:58Z
- **Source:** `main/data/polaris_train.jsonl` (via `main/data/paths.py` `POLARIS_TRAIN_JSONL`)
- **Row counts:** total=51,139 · train=50,883 · val=256 (seed 42, val size `<!-- TODO -->` confirm for `trainer.test_freq`)
- **Local outputs:** `main-verl/data/polaris_train.parquet`, `main-verl/data/polaris_val.parquet`
- **Volume paths:** `/vol/data/main-verl/polaris_train.parquet`, `/vol/data/main-verl/polaris_val.parquet` on `main-artifacts`
- **Upload command (repo root, Modal profile `chicken602`):**

```bash
export MODAL_PROFILE=chicken602
PYTHONPATH=main-verl:main python3 -m data.preprocess_polaris_verl --out-dir main-verl/data --upload
```

- **Preprocess only (no upload):**

```bash
PYTHONPATH=main-verl:main python3 -m data.preprocess_polaris_verl --out-dir main-verl/data
```

- **Sanity check:** PASS (`prompt`, `reward_model`, `data_source`; non-empty `ground_truth`; all `polaris`)
- **Upload status:** PASS (`batch_upload` to `main-artifacts`)
- **Executor verdict:** **DONE**
- **TODOs left:** `<!-- TODO -->` val size confirmation in `preprocess_polaris_verl.py` header; chat-template check deferred to S2.5 per plan

---

## S2.1 Audit

- **Verdict:** PASS WITH NOTES
- **Auditor:** audit-agent
- **Timestamp (UTC):** 2026-05-29T20:44:33Z
- **Checklist:** all 8 items pass; non-blocking: `sanity_check()` omits `extra_info` assertion; val-size TODO remains; upload uses `data.preprocess_polaris_verl` with `PYTHONPATH=main-verl:main`.

---

## S2.2 — GRPO Hydra config (executor)

- **Executor:** S2.2 executor agent
- **Timestamp (UTC):** 2026-05-29
- **Artifact:** `main-verl/configs/grpo_smoke_1p7b.yaml`
- **Verdict:** **DONE**

**Summary:**
- Hydra `defaults: [ppo_trainer, _self_]` with `hydra.searchpath` → `/root/maxrl/verl/trainer/config` (vendored maxrl; no full-file copy).
- 50-step GRPO smoke on `Qwen/Qwen3-1.7B-Base`, Polaris parquet paths on artifacts volume, MathReward via `reward_model.enable: false` + parquet `data_source` routing.
- KL=0, `loss_agg_mode: token-mean`, `adv_estimator: grpo` (not maxrl) documented in file header; pattern aligned with maxrl `qwen3_experiments/run_qwen3_training.sh`.
- B200 knobs: `enforce_eager: true`, `model_dtype: bfloat16`, `gpu_memory_utilization: 0.45`, `+ray_kwargs.ray_init.num_gpus: 4`.
- No `custom_reward_function.path`; no `algorithm.adv_estimator=maxrl`.

**Open TODOs (in yaml):**
- `ppo_micro_batch_size_per_gpu` — tune on first OOM (S2.5)
- `trainer.wandb_kwargs.entity` — confirm on Modal
- `trainer.test_freq` — bump if val eval slow

---

## S2.2 Audit

- **Verdict:** PASS
- **Auditor:** audit-agent
- **Timestamp (UTC):** 2026-05-29T21:15:00Z
- **Checklist:** all 11 items pass; no `custom_reward_function`, no Instruct model, no `adv_estimator: maxrl`.

---

## S2.3 — Trainer Modal app (executor)

- **Executor:** S2.3 executor agent
- **Timestamp (UTC):** 2026-05-29
- **Verdict:** **DONE**

### `infra/modal_image.py` (pre-flight option (a))

- Dropped `--no-deps` on editable install: `pip install -e .` (maxrl/setup.py owns deps).
- Removed hand-curated `.pip_install(...)` layer (`tensordict`, `hydra-core`, `omegaconf`, `accelerate`, `codetiming`, `peft`, `pyarrow`, `pandas`, `dill`, `packaging`).
- Kept initial GPU pin layer before editable install (vLLM 0.9.0 + cu128, `transformers<4.54.0`, flash-attn 2.8.3).
- Added post-editable re-pin layer (same three pins) to restore B200 stack if setup.py drifted wheels.
- **Image rebuild count:** 0 → **1** (code change logged; first Modal build will consume this rebuild).

### `probes/grpo_smoke.py`

- Modal app via `app_name()` (`CS224R_APP_NAME`, default stage01 until launch script sets `cs224r-verl-stage02`).
- `gpu="B200:4"`, `timeout=3*3600`, `HUGGINGFACE` + `WANDB_API_KEY` secrets, artifacts + HF cache volumes.
- Body: torch/cuda diagnostics (expect 4 GPUs), `import verl`, `from verl.trainer import main_ppo` pre-flight, subprocess `python -m verl.trainer.main_ppo` with `grpo_smoke_1p7b`, list `/vol/checkpoints/main-verl/grpo_smoke_1p7b/`.
- `@app.local_entrypoint()` → `grpo_smoke.remote()`.
- No `main.train.*` imports; no custom reward/adv; single subprocess run.

---

## S2.3 Audit

- **Verdict:** PASS WITH NOTES
- **Auditor:** S2.3 audit-agent
- **Timestamp (UTC):** 2026-05-29T21:30:00Z
- **Checklist:** all 9 items pass.
- **Notes:** Image rebuild logged (count 1) but first Modal build deferred to S2.5. `app_name()` default stage01 until S2.4 launch script sets stage02.

---

## S2.4 — Launch script + README (executor)

- **Executor:** S2.4 executor agent
- **Timestamp (UTC):** 2026-05-29
- **Verdict:** **DONE**

### `main-verl/scripts/launch_grpo_smoke.sh`

- Created; `chmod +x`; mirrors `launch_hello_verl.sh` (`set -euo pipefail`, `python3 -m modal run`, `"$@"` passthrough).
- Default `CS224R_APP_NAME=cs224r-verl-stage02` (Stage 1 smoke stays on `stage01`).
- Invokes `main-verl/probes/grpo_smoke.py`.

### `main-verl/README.md`

- Bring-up subsection: added GRPO smoke bullet with documented launch command.

---

## S2.4 Audit

- **Verdict:** PASS
- **Auditor:** S2.4 audit-agent
- **Timestamp (UTC):** 2026-05-29
- **Checklist:** all 4 items pass; no blocking notes.

---

## S2.5 — Remote smoke execution (executor)

- **Modal app:** `cs224r-verl-stage02` · https://modal.com/apps/chicken602/main/ap-iK3ARcAWn0FVNeCUYbeqkU
- **Raw log:** `/tmp/s2.5_grpo_smoke.log`
- **Steps completed:** 0 / 50
- **Verdict:** **FAIL** (3 attempts; furthest = FSDP actor up, vLLM rollout init crash)

### Attempt 1 — Hydra missing `ray_init.ray_dir`

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: ray_init.ray_dir
```

- **Fix:** added `ray_init.ray_dir: /tmp/ray` to `grpo_smoke_1p7b.yaml`
- **Type:** config fix #1

### Attempt 2 — Missing rollout log-prob micro-batch

```
ValueError: [actor_rollout_ref.rollout] Please set at least one of
'log_prob_micro_batch_size' or 'log_prob_micro_batch_size_per_gpu'.
```

- Parquet loaded OK (50,883 train / 256 val); Ray up.
- **Fix:** added `log_prob_micro_batch_size_per_gpu: 4` under `rollout` and `ref`
- **Type:** config fix #2

### Attempt 3 — `expandable_segments` incompatible with vLLM memory pool (final blocker)

```
AssertionError: Expandable segments are not compatible with memory pool.
(vllm/device_allocator/cumem.py)
```

- FSDP actor loaded Qwen3 1.72B on 4× B200; crashed when VeRL spawned vLLM rollout worker.
- **Root cause:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `infra/modal_image.py` (copied from Stage 1 / `main/` image). vLLM 0.9.0's `CuMemAllocator` rejects this flag when using its memory pool — required for colocated FSDP actor + vLLM rollout (VeRL path). Stage 1's direct `LLM()` smoke did not hit this code path.
- **Fix (recommended):** remove `expandable_segments:True` from `modal_image.py` (or strip before trainer subprocess in `grpo_smoke.py`; see `main/train/ablation.py::prepare_pytorch_alloc_for_vllm_sleep`).
- **Type:** image env change (not a yaml knob)

### Side notes (attempt 3)

- Image build OK; `pip install -e .` pulled deps; Ray bumped 2.44.1 → 2.55.1 via maxrl setup.py.
- Flash Attention “model not on GPU” warnings (×3) — non-fatal.
- `enforce_eager` disables async output processor — expected.

---

## S2.5 attempt 4 — image env fix + re-run

- **Timestamp (UTC):** 2026-05-29
- **Change:** removed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from `infra/modal_image.py` (vLLM 0.9 CuMemAllocator + VeRL colocated rollout). Documented in `verl-reference.md` §4.3 / §6.2.
- **Image rebuild count:** 1 → **2**
- **Config:** unchanged from attempts 2–3 (`ray_dir`, `log_prob_micro_batch_size_per_gpu` already in yaml)
- **Command:**

```bash
export CS224R_APP_NAME=cs224r-verl-stage02
export MODAL_PROFILE=chicken602
./main-verl/scripts/launch_grpo_smoke.sh 2>&1 | tee /tmp/s2.5_grpo_smoke_attempt4.log
```

- **Verdict:** **FAIL** — see result below (`math_reward` router; superseded by attempt 5)

### Attempt 4 result (2026-05-29 ~21:14–21:21 UTC)

- **Expandable_segments fix:** **PASS** — vLLM rollout init succeeded (past attempt 3 blocker).
- **Progress:** Hydra OK · parquet loaded · FSDP actor + vLLM up · W&B run `wl3zydvf` · validation gen started.
- **Steps completed:** 0 / 50 (crashed on step-0 val reward)
- **Error:**

```
NotImplementedError: Reward function is not implemented for data_source='math_reward'
(verl/utils/reward_score/__init__.py default_compute_score)
```

- **Likely fix:** flip parquet `data_source` to `math_dapo` (plan S2.1 TODO) + re-upload; or register `math_reward` if maxrl fork uses different name.
- **Verdict:** **FAIL** — config fix #3 candidate (data_source)

---

## S2.5 attempt 5 — `data_source=polaris` re-upload + re-run

- **Root cause (attempt 4):** parquet on Modal still had `math_reward` from S2.1 upload; maxrl router has no such key (see below).
- **Fix:** `preprocess_polaris_verl.py` uses `DATA_SOURCE=polaris` (matches maxrl `examples/maxrl_data_preprocess/polaris.py`); re-uploaded 2026-05-29T21:31:43Z.
- **Command:**

```bash
export CS224R_APP_NAME=cs224r-verl-stage02
export MODAL_PROFILE=chicken602
./main-verl/scripts/launch_grpo_smoke.sh 2>&1 | tee /tmp/s2.5_grpo_smoke_attempt5.log
```

- **Verdict:** **PARTIAL** — training reached step 3+ but used unpatched `math_verify` router; superseded by S2.5b.

---

## S2.5b — MathReward lock (patch + prompt + re-upload)

- **Decision:** [`../reward-decision.md`](../reward-decision.md) — mentor-locked upstream `math_reward.py` stack
- **Code changes:**
  - `infra/patches/maxrl_polaris_math_reward.patch` — `polaris` / `math_reward` → `math.compute_score`
  - `infra/modal_image.py` — apply patch at build (**image rebuild count → 3**)
  - `data/preprocess_polaris_verl.py` — maxrl prompt suffix + `data_source=polaris`
- **Docs:** `reward-decision.md`, `verl-reference.md` §3.3, `stage-02-agent-plan.md` S2.1 updated
- **Re-upload:** 2026-05-29T22:01:48Z (50,883 train / 256 val; maxrl prompt suffix verified locally)

---

## S2.5 attempt 6 — MathReward patch + detached run (final)

- **Launch:** `python3 -m modal run -d main-verl/probes/grpo_smoke.py` (tee → `/tmp/s2.5_grpo_smoke.log`)
- **Modal app:** `ap-gF0GG2pNY8KhP0VbJSf6uJ` · https://modal.com/apps/chicken602/main/ap-gF0GG2pNY8KhP0VbJSf6uJ
- **W&B:** [`ua5e16l8`](https://wandb.ai/224r-project/cs224r-minority-voting/runs/ua5e16l8)
- **Timestamp end (UTC):** ~2026-05-29T22:53Z
- **Steps completed:** **26 / 50** · ~133 s/step · ~67 min wall
- **Metrics (step 26):** `critic/score/mean=0.067`, `response_length/mean≈1006`, `actor/pg_loss≈-0.015`
- **Val @ step 0:** `mean_accuracies/polaris/reward/mean@1=0.102`
- **Checkpoint:** none (`save_freq=50`)

**Failure (step 27 train batch load):**

```
NotImplementedError: sequence_length=1730 is larger than max_length=1024
```

- **Fix:** bump `data.max_prompt_length` (e.g. 2048) in `grpo_smoke_1p7b.yaml`
- **Verdict:** **FAIL** — plumbing OK (B200:4, patched MathReward, non-zero rewards); 50-step gate not met

---

## S2.5 — Remote smoke run (executor summary)

- **Timestamp start (UTC):** 2026-05-29T20:51:49Z
- **Log file:** `/tmp/s2.5_grpo_smoke.log` (5.5 MB; authoritative — `attempt6.log` is stale/partial)

### Image rebuild count: **3**
1. S2.3 pre-flight (logged)
2. Remote build after adding `tensordict` + trainer deps layer
3. Remote build after adding `torchdata`

### Config-fix iterations: **3** (budget ≤2 — exceeded)
1. `ray_init.ray_dir: /tmp/ray` in `grpo_smoke_1p7b.yaml`
2. `log_prob_micro_batch_size_per_gpu: 4` on rollout + ref
3. `DATA_SOURCE: polaris` in preprocess + parquet re-upload (`math_reward` not implemented in maxrl @7197bbb)

### Cluster diagnostics (final run)
- `torch.cuda.device_count()` → **4**
- Devices → **NVIDIA B200** ×4
- `ray.init` → local Ray instance started; `+ray_kwargs.ray_init.num_gpus: 4`
- vLLM + FSDP workers initialized; weight sync OK; GRPO training loop ran

### Run metrics
| Metric | Step 0 (val) | Step 25 | Step 26 (last) |
|--------|--------------|---------|----------------|
| `critic/score/mean` | — | — | 0.067 |
| `response_length/mean` | — | — | 1006 |
| `actor/pg_loss` | — | — | -0.015 |
| `actor/grad_norm` | — | — | 0.243 |
| val `mean@1` (step 0) | 0.102 | — | — |

- **Step time:** ~133 s/step
- **Final step reached:** **26 / 50**

### Failure
```
NotImplementedError: sequence_length=1730 is larger than max_length=1024
```
Train dataloader (`rl_dataset.__getitem__`) on step 27 — prompt+template **1730 tokens** > `max_prompt_length=1024`. Not a val-only crash (step 25 val completed).

### Verdict: **FAIL**
Bring-up succeeded (B200:4, Ray, vLLM, FSDP, patched **MathReward**, non-zero metrics). Next: `max_prompt_length` bump (4th config fix).

### Checkpoint
- **`/vol/checkpoints/main-verl/grpo_smoke_1p7b/`:** none (`save_freq=50`; run stopped at step 26)

