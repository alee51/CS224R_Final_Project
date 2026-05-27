# B200 readiness audit (`b200-bringup`)

**Audited:** 2026-05-26  
**Scope:** Static audit only (no live Modal execution in this artifact).

## Verdict

**GO WITH FIXES**

The branch has substantial B200 bring-up work landed (dual train entrypoints, B200 smokes, vLLM/FA image bump), and `--gpu-class` train routing is mostly correct. A few must-fix issues remain before a reliable B200 train smoke.

## What is complete

- `launch_train.sh` now accepts `--gpu-class h200|b200` (default `h200`) and dispatches by Modal function name (`train_remote_h200` / `train_remote_b200`).
- `trainer.py` defines both `train_remote_h200` (`gpu="H200"`) and `train_remote_b200` (`gpu="B200"`), with per-class spawn chaining preserved (each function spawns its own `.spawn`).
- `modal_image.py` is migrated to `vllm==0.9.0`, cu128 torch index, and updated flash-attn wheel, which aligns with B200/Blackwell intent.
- B200 probe entrypoints exist for:
  - `main/probes/smoke_vllm_generate.py` (`gpu="B200"`)
  - `main/probes/smoke_weight_sync.py` (`gpu="B200"`)
- B200 checkpoint-eval configs exist:
  - `main/configs/checkpoint_eval_2k_dapo_b200.yaml`
  - `main/configs/checkpoint_eval_2k_polaris_aime_b200.yaml`

## Blockers (must-fix before B200 train smoke)

1. **`smoke_weight_sync.py` is currently broken against `trainer.build_hf`.**
   - Current call site uses `build_hf(model_id, gradient_checkpointing=False)`, but `build_hf` now requires a `TrainCfg` and returns `(model, optimizer)`.
   - This smoke cannot run as written; weight-sync readiness is therefore unproven.

2. **`smoke_flash_attn.py` still hardcodes `gpu="H200"`.**
   - This prevents a true B200 FlashAttention smoke unless edited or duplicated.
   - Since FA compatibility is a gating item for B200, this is a pre-smoke blocker.

3. **No launch wrapper exists for weight-sync smoke.**
   - `main/scripts/launch_smoke_weight_sync.sh` is missing.
   - You can still invoke Modal directly, but the expected smoke launcher set is incomplete for repeatable operator flow.

## Should-fix recommendations

- Add `main/configs/train_real_b200.yaml` (or equivalent) so B200 metadata (`gpu_class`, `modal_price_per_sec`) is explicit and reproducible for train runs.
- Align `modal_price_per_sec` in B200 eval configs (currently still `0.001261` in both B200 eval YAMLs).
- Consider making FA smoke GPU-selectable (`--gpu-class` or env) instead of hardcoding.
- Optionally add a `launch_smoke_weight_sync.sh` wrapper for consistent operator UX with other smokes.
- Confirm whether Group B probe should support B200 path (`group_b_step_probe.py` still hardcodes `gpu="H200"`).

## `--gpu-class` behavior verification

### H200 default preserved

- `launch_train.sh` sets `GPU_CLASS="h200"` by default.
- Without `--gpu-class`, it launches `main/train/trainer.py::train_remote_h200`.
- `trainer.py` keeps `train_remote = train_remote_h200` alias for backward compatibility.

### B200 routing correctness

- With `--gpu-class b200`, launcher targets `train_remote_b200`.
- `train_remote_b200` is a real Modal function with `gpu="B200"`.
- Leg respawn path remains class-consistent because each remote fn passes its own spawn handle into `_train_remote_impl` (`train_remote_b200.spawn` vs `train_remote_h200.spawn`).

## Missing expected B200 files/scripts/configs

- Missing: `main/configs/train_real_b200.yaml`
- Missing: `main/scripts/launch_smoke_weight_sync.sh`
- Present but still H200 hardcoded where B200 smoke is expected:
  - `main/probes/smoke_flash_attn.py`
  - `main/probes/group_b_step_probe.py` (if intended for B200 bring-up)

## Exact smoke command sequence (post-fixes)

Run from repo root.

1. **vLLM smoke (B200)**
```bash
bash main/scripts/launch_smoke_vllm_generate.sh
```

2. **FlashAttention smoke (must run on B200 after fixing gpu hardcode)**
```bash
bash main/scripts/launch_smoke_flash_attn.sh
```

3. **Weight sync smoke (after fixing `smoke_weight_sync.py`; optional launcher)**
```bash
main/.venv/bin/modal run main/probes/smoke_weight_sync.py
```

4. **GRPO/minority 10-step smoke on B200**
```bash
bash main/scripts/launch_train.sh --mode smoke --arm minority_answer --gpu-class b200 --config main/configs/train_real.yaml
```

5. **Resume check (non-smoke leg, same run path, B200)**
```bash
bash main/scripts/launch_train.sh --mode full --arm minority_answer --gpu-class b200 --config main/configs/train_real.yaml
```

6. **Manual interruption + relaunch to verify resume**
```bash
bash main/scripts/launch_train.sh --mode full --arm minority_answer --gpu-class b200 --config main/configs/train_real.yaml
```

## What still requires live Modal B200 execution (cannot be statically proven)

- FlashAttention kernel compatibility/perf on actual B200 runtime (`cuda_capability`, kernel dispatch, no runtime kernel-image errors).
- vLLM 0.9 + B200 stability in collocated rollout/training lifecycle.
- HF->vLLM weight sync correctness/perf under real B200 execution (tensor path compatibility, sync latency).
- End-to-end 10-step minority smoke timing targets and VRAM headroom metrics.
- Resume behavior under real checkpoints across container restarts and any spawn/queue effects on B200.

## Bottom line

Branch is close and structurally on-track. Fix the smoke blockers first, then execute the command sequence above on live Modal B200 to close the remaining unknowns.

## GPU flag-flow audit (h200/b200)

### Findings

- **[must-fix] `--gpu-class` and `--config` can silently diverge.**
  - In `launch_train.sh`, `--gpu-class` chooses Modal entrypoint (`train_remote_h200` / `train_remote_b200`) while `--config` can still point to either H200 or B200 YAML.
  - Example risk: `--gpu-class b200 --config main/configs/train_real.yaml` runs on B200 hardware but keeps `gpu_class: H200` and H200 cost metadata in config-derived logging (`setup_wandb` tags use `cfg.raw.gpu_class`).
  - This is a correctness/observability mismatch, not just style: runtime path and experiment metadata can disagree.

- **[should-fix] `trainer.py` has duplicated dual-entrypoint wrappers.**
  - `train_remote_h200` and `train_remote_b200` duplicate the full parameter list and `_train_remote_impl(...)` forwarding, differing only by `gpu=` in decorator and spawn function handle.
  - Any future signature change risks asymmetric edits and latent drift.

- **[should-fix] routing is stringly-typed in shell script.**
  - `TRAIN_FN="train_remote_${GPU_CLASS}"` relies on naming convention and late Modal resolution.
  - Allowed values are validated, so this is not currently broken, but it is brittle and harder to lint/refactor than explicit mapping.

- **[optional] mixed-case GPU labels increase normalization burden.**
  - CLI uses lowercase (`h200|b200`), Modal decorators use uppercase (`"H200"`, `"B200"`), config fields are uppercase (`gpu_class: H200/B200`).
  - Works today, but this encourages repeated ad hoc normalization at boundaries.

### Concrete simplification recommendations

- **[must-fix] enforce one source of truth between gpu class and config metadata.**
  - Add a preflight check in launcher: if `--gpu-class b200`, require config `gpu_class: B200` (and similarly for H200), unless an explicit escape flag is passed.
  - Alternative: inject `CS224R_GPU_CLASS` and override `cfg.raw["gpu_class"]` in load/override path so wandb/config metadata always matches execution class.

- **[should-fix] replace shell name concatenation with a small mapping helper.**
  - In `launch_train.sh`, use a `case`/map table: `h200 -> (train_remote_h200, train_real.yaml)`, `b200 -> (train_remote_b200, train_real_b200.yaml)`.
  - Keep the table next to argument validation so function + default config are declared together.

- **[should-fix] collapse duplicate trainer wrappers via tiny indirection.**
  - Keep two decorated Modal functions (required for static GPU binding), but route both through a shared helper that defines/validates common CLI args once.
  - At minimum, centralize the argument list in one typed dataclass/kwargs builder used by both wrappers to prevent signature drift.

- **[optional] add a thin compat alias strategy.**
  - Keep `train_remote` as explicit alias to default class, but document default in one constant (e.g., `DEFAULT_GPU_CLASS = "h200"`) used by both shell + trainer docs/comments to avoid future split-brain defaults.

### Risk assessment: maintaining `train_remote_h200` + `train_remote_b200`

- **[must-fix risk] metadata/runtime skew** if GPU selection and config metadata are allowed to drift independently.
- **[should-fix risk] maintenance drift** from duplicated signatures and forwarding code across two near-identical entrypoints.
- **[optional risk] operator confusion** from multiple class spellings and convention-based function-name construction.
