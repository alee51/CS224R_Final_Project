# Group B probe — implementation guide

**Drafted:** 2026-05-25. **Updated:** 2026-05-25 (hybrid arm C, implementation shipped, full H100 run in flight). **Strategic plan:** [`05-24_probe_plan.md`](./05-24_probe_plan.md) § "Group B". **Prerequisite:** satisfied — trainer skeleton + `run_one_grpo_step(..., instrument=True)` in `main/train/trainer.py`; probe in `main/probes/group_b_step_probe.py`.

**Purpose:** thin scaffolding on the trainer: timing, VRAM headroom, microbatch OOM ladder, wandb readout. Architecture lives in `trainer_skeleton.md`; this doc is probe-specific only.

**Do not duplicate trainer specs here** — reference `trainer_skeleton.md` sections.

---

## 1. What this probe answers

From [`05-24_probe_plan.md`](./05-24_probe_plan.md) § Group B / B1:

- VRAM watermark at **chosen** microbatch (collocated policy vLLM + HF train, same GPU)
- `update_weights` wall-clock per step
- Microbatch OOM ladder (largest microbatch that fits **forward + backward** on cached step tensors)
- Step-time decomposition (rollout / score / advantage / logprob_fwd / backward / optimizer / weight_sync)
- VRAM **headroom** at fixed `gpu_memory_utilization: 0.45` (inform manual util tuning — **no util sweep in this probe**)
- Whether async rollout/train overlap is worth the complexity (PLAN §5 deferred)

**Decisions unlocked:**

| Output | Updates in PLAN.md |
|---|---|
| Max microbatch + VRAM at max microbatch | §7 microbatch, `grad_accum = ceil(n_kept_sequences / microbatch)` |
| `update_weights` wall-clock | §7 sync cadence |
| Phase wall-clock % | §5 async go/no-go (if rollout >50% of step, overlap pays off) |
| Tokens/sec collocated | §7 step-time budget (vs Group A standalone ~4.2k tok/s) |
| VRAM headroom @ 0.45 | §5 collocated vLLM util (manual bump/down after probe, not swept here) |
| $/step | §7 cost-per-arm |

**Out of scope for Group B:** cross-GPU $/throughput (H100 vs H200 vs B200). Group A + prompt probe measured rollout/judge on **H100 only**. This probe gives **collocated GRPO $/step on H100**; a separate thin re-run (same yaml, change `gpu` + `modal_price_per_sec`) is the right place to compare SKUs — not a new architecture agent.

---

## 2. Locked choices

| Knob | Value | Notes |
|---|---|---|
| GPU | H100 (`modal_price_per_sec: 0.001097`) | Collocated train + policy vLLM |
| Training arm | GRPO only | Set-based arms out of scope |
| Model | `Qwen/Qwen3-1.7B-Base` | Per trainer skeleton |
| **Prompt + reward** | **`hybrid_answer_boxed`** (prompt probe arm C) | Verbatim in `train/prompts.py` `HYBRID_ANSWER_BOXED_TEMPLATE`. Locked 2026-05-25 per team decision; Group B full run uses this variant. |
| Toy batch | **32 prompts × 8 rollouts = 256 completions** | Systems slice only — **not** representative of final §2 train mix |
| Toy data | `probes/05-25/group_a_n800/manifest.jsonl`, **`problem_id` 0–31** | Reuse n800 cohort (prompt probe). Not §2 freeze. Band-skewed (≈25× `0/8` + 7× `1/8`). |
| Seeds | `global_seed + problem_id * n_rollouts + rollout_idx` | STANDARDS / Group A; **use manifest `problem_id`**, not batch index. Step = 0. |
| Collocated vLLM util | **0.45 fixed** | No sweep. Log headroom (§6) to decide post-hoc util change. |
| Microbatch unit | **Completion sequences** per HF forward/backward chunk | After `keep_mask`, `n_kept` sequences; `grad_accum = ceil(n_kept / microbatch)` |
| Microbatch sweep | Start **1**, double until OOM, bisect | Forward **and** backward on cached tensors; vLLM stays loaded |
| `update_weights` | Median of **3** syncs in timed step (1 warmup + 2 measured) | Plus one production sync inside timed step if trainer does per-step sync |
| Runtime budget | <1 hr | |

---

## 3. UNDECIDED / inherited (config knobs, not blockers)

| Item | Status | Effect on probe |
|---|---|---|
| Reward parser (Rank 2) | **Resolved** (2026-05-25) | `compute_reward()` / `extract_rank2()`; log `parse_ok_rate` sentinel vs **n800 hybrid (arm C) ~87.6% rank2**, not 200-run 56% Minerva |
| `max_response_length` | 4096 safe (Group A) | Use **4096** for probe |
| §2 training freeze | Undecided | Probe does **not** use `polaris_train.jsonl` |
| Prompt variant | **Locked: `hybrid_answer_boxed`** (arm C) | Group B smoke + full use this variant |
| Judge | Out of scope | No judge GPU in this probe |
| GPU SKU (H100 vs H200 vs B200) | **Not in this probe** | H100 only; see §1 note — optional follow-up re-run of same toy slice on other SKUs |

---

## 4. Trainer contract (implemented)

§4 “no new trainer **architecture**” — export in `main/train/trainer.py`:

```python
# train/trainer.py
def run_one_grpo_step(cfg, rollout_engine, hf_model, opt, batch, *, instrument=False) -> StepResult: ...
```

- `batch`: prompts, golds, manifest `problem_id`s for seeds.
- `instrument=True`: per-phase timers + `torch.cuda.max_memory_allocated()` reset per phase (see §5).
- Production `train()` loops over this; **probe calls this**, does not reimplement the loop.

---

## 5. Files (shipped)

| Path | Role |
|---|---|
| `main/probes/group_b_step_probe.py` | Modal app: `run_phase1` → `run_phase2` → `run_phase1b`, `run_full` |
| `main/configs/probe_step_b_05-25.yaml` | Full run (`smoke: false`) |
| `main/configs/probe_step_b_05-25_smoke.yaml` | Smoke (`extends` full config, `smoke: true`) |
| `main/scripts/launch_probe_step_b.sh` | Detached launch + `CS224R_APP_NAME` tagging |
| `main/docs/probes/artifacts/.gitkeep` | Pointer written post-run to `05-25_group_b.pointer.json` |

---

## 6. Modal app — `main/probes/group_b_step_probe.py`

Mirror `group_a_rollout_judge.py`: `from infra.modal_image import image`, `CS224R_APP_NAME`, secrets `HUGGINGFACE` + `WANDB_API_KEY`, volumes `main-artifacts` → `/vol`, **`hf-cache`** → `/root/.cache/huggingface`, `gpu="H100"`, `timeout` ≤ 3600.

**Wandb:** entity `224r-project`, project `cs224r-minority-voting`, group **`probe-B-05-25`**, run name `probe-B_{operator}_{MM-DD-HHMM}`, tags: `phase=probe`, `operator`, `gpu_class`, `arm=grpo`, `git_sha_short`, `prompt_variant=hybrid_answer_boxed`.

**Pipeline:** `run_full` → `run_phase1.remote` → `run_phase2.remote` → `run_phase1b.remote` (same wandb run throughout).

### Phase 1 — warmup + timed step @ `starting_microbatch` (default 1)

1. Load yaml (merged with base trainer config).
2. Init wandb; log config, git SHA/dirty, dep versions, `prompt_variant`, `gpu_memory_utilization`.
3. Load manifest rows `problem_id` 0..31; build prompts via `format_problem(..., variant=hybrid_answer_boxed)` from config.
4. Build collocated `RolloutEngine` + HF model.
5. **Warmup** (untimed): `run_one_grpo_step(..., instrument=False)` at `microbatch=1`.
6. **Timed step** @ `train.starting_microbatch` (yaml, default 1): `run_one_grpo_step(..., instrument=True)`.
   - Phases: `t_rollout`, `t_score`, `t_advantage`, `t_logprob_fwd`, `t_backward`, `t_optimizer`, `t_weight_sync`.
   - **Weight sync:** one normal post-step sync inside the step **plus** log median of 3 standalone `sync_hf_to_vllm` benchmarks (warmup + 2) — report both `t_weight_sync_step` and `t_weight_sync_bench_median`.
7. Write **`train_step_cache.pt`** to volume: tensors needed for Phase 2 (completion ids, `old_logprobs`, advantages, masks, `keep_mask`, etc.) — **not** full rollout jsonl.
8. Write **`phase1_done.json`**; `volume.commit()`.

### Phase 2 — microbatch OOM sweep (fresh container, **same wandb run**)

1. Read `phase1_done.json`; `_resume_wandb(wandb_run_id)`.
2. Reload collocated vLLM + HF; load **`train_step_cache.pt`** (do **not** read Group A rollouts).
3. **vLLM stays loaded** for entire sweep loop; `empty_cache()` between attempts.
4. For `mb` in {1, 2, 4, …} with bisect on OOM (max `sweep.max_attempts`):
   - Run **HF forward + backward** microbatched at `mb` on cached tensors (match `loss.py` path).
   - On success: log peak VRAM, `t_fwd_bwd_s`; double `mb`.
   - On OOM: log failure; bisect between last ok and fail.
5. Set `max_microbatch_ok`; append lines to **`microbatch_sweep.jsonl`**; `volume.commit()`.

### Phase 1b — timed step @ `max_microbatch_ok` (fresh container ok; **same wandb run**)

1. Resume wandb; reload engines (fresh container).
2. **Timed** full `run_one_grpo_step` at `max_microbatch_ok` with full instrumentation — **regenerates rollouts** (production-shaped timings at max microbatch; Phase 2 sweep only touched cached tensors).
3. Update `phase1_done.json` with `phase1b_times_s`, `vram_peak_gb_at_max_mb`; final `volume.commit()`.

### Smoke mode

`smoke: true` → 4 prompts × 2 rollouts, sweep capped at `smoke_max_microbatch: 4`, all three phases.

---

## 7. VRAM headroom logging (required)

Log every phase and derived scalars (wandb):

| Field | Notes |
|---|---|
| `vram_peak_gb_{phase}` | Per phase after `max_memory_allocated` reset |
| `vram_peak_gb_step` | Max over timed step |
| `device_vram_total_gb` | From `get_device_properties` |
| `vram_headroom_gb_step` | `device_vram_total_gb - vram_peak_gb_step` |
| `vram_headroom_gb_after_rollout` | Snapshot after rollout, before HF train |
| `rollout.gpu_memory_utilization` | Config (0.45) |

**Post-run manual rule:** headroom &lt; ~5–10 GiB → try lower util (0.40); large headroom + rollout-bound step → try higher (0.50+) in a **follow-up**, not in-probe sweep.

---

## 8. Wandb panels (final)

- Step-time stack (% per phase) — Phase 1 @ mb=1 and Phase 1b @ `max_microbatch_ok`
- Microbatch sweep scatter
- Weight-sync distribution (3 benchmarks)
- Tokens/sec collocated = rollout tokens / `t_rollout`
- VRAM watermark + headroom
- $/step = wall clock × `modal_price_per_sec`
- Sentinels: `parse_ok_rate`, `mean_reward`, `fraction_filtered`, `n_kept_sequences`, `grad_accum_at_max_mb`

---

## 9. Artifact schemas

**Volume:** `/vol/probes/05-25/group_b/`

**`phase1_done.json`:**
```json
{
  "wandb_run_id": "...",
  "prompt_variant": "hybrid_answer_boxed",
  "phase_times_s": {},
  "phase1b_times_s": {},
  "vram_peak_gb": 0.0,
  "vram_peak_gb_at_max_mb": 0.0,
  "max_microbatch_ok": 0,
  "n_kept_sequences": 0,
  "completed_at": "ISO8601"
}
```

**`train_step_cache.pt`:** torch save dict (implementer documents keys in probe module docstring).

**`microbatch_sweep.jsonl`** (one line per attempt):
```json
{"microbatch": 4, "ok": true, "vram_peak_gb": 65.1, "t_fwd_bwd_s": 1.42}
{"microbatch": 8, "ok": false, "vram_peak_gb": null, "error": "CUDA out of memory"}
```

**Pointer:** `main/docs/probes/artifacts/05-25_group_b.pointer.json` — STANDARDS schema; operator `modal volume get` after run.

---

## 10. Config — `main/configs/probe_step_b_05-25.yaml`

```yaml
extends: configs/train_grpo_05-25.yaml   # implement merge in probe loader

global_seed: 42
operator: nancy
gpu_class: H100
modal_price_per_sec: 0.001097
prompt_variant: hybrid_answer_boxed

smoke: false
smoke_prompts: 4
smoke_rollouts: 2
smoke_max_microbatch: 4

toy_batch:
  source_manifest: probes/05-25/group_a_n800/manifest.jsonl
  problem_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  n_rollouts: 8

train:
  batch_size: 32
  n_rollouts: 8
  starting_microbatch: 1   # Phase 1 timed step only
  checkpoint_every_steps: 999999

rollout:
  gpu_memory_utilization: 0.45
  max_response_length: 4096

sweep:
  start_microbatch: 1
  max_attempts: 12

artifacts:
  volume_name: main-artifacts
  volume_mount: /vol
  base_path: probes/05-25/group_b/
  pointer_path: docs/probes/artifacts/05-25_group_b.pointer.json

wandb:
  entity: 224r-project
  project: cs224r-minority-voting
  group: probe-B-05-25
```

---

## 11. Launch — `main/scripts/launch_probe_step_b.sh`

Same as `launch_probe_a.sh`: read `smoke` from yaml → phase `smoke`|`full`; set `CS224R_APP_NAME=cs224r-probe-b-{phase}-{operator}-{MM-DD-HHMM}`, `CS224R_GIT_SHA`, `CS224R_GIT_DIRTY`;  
`exec modal run --detach main/probes/group_b_step_probe.py::run_full --config "$CFG"`

---

## 12. Build order

- [x] Trainer skeleton + `run_one_grpo_step(..., instrument=True)`
- [x] `group_b_step_probe.py` Phase 1 → smoke
- [x] Phase 2 sweep + Phase 1b → smoke
- [ ] Full H100 detached run (launched 2026-05-25; Modal app `ap-CDWaOaDdYVLSNyRsqxjxtd` at time of doc update)
- [ ] Readout → PLAN §5 / §7 + `group_b_results.md` (or timeline addendum)

---

## 13. Post-run readout

| Panel | PLAN update |
|---|---|
| Step-time stack (mb=1 vs max mb) | §7 step time; §5 async if rollout >50% |
| `max_microbatch_ok` + `grad_accum_at_max_mb` | §7 microbatch |
| `update_weights` median | §7 sync cadence |
| Tokens/sec collocated | §7 vs Group A 4.2k |
| `vram_headroom_gb_step` @ 0.45 | §5 util (manual follow-up) |
| $/step | §7 affordability |
| `parse_ok_rate` | Sentinel vs n800 hybrid (~87.6% rank2 offline) |

---

## 14. Hand-off prompt

> **Post-run only:** Pull artifacts from `probes/05-25/group_b/`, write readout, update PLAN §5/§7. Implementation is shipped. Prompt: `hybrid_answer_boxed` (arm C). `gpu_memory_utilization: 0.45` fixed; log VRAM headroom (§7). Phase 2: forward+backward sweep on `train_step_cache.pt`, vLLM loaded. Phase 1b: full timed step at `max_microbatch_ok` (fresh rollouts). Out of scope: judge, set-based arms, §2 freeze, util sweep, H100/H200/B200 SKU ladder.
