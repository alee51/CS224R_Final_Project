# Probe plan

**Drafted:** 05-24, 00:26. **Updated:** 05-24 (implementation details, STANDARDS alignment).

**Purpose:** answer the open questions in `[PLAN.md](../PLAN.md)` §5 and §7 that block locking the training matrix, with minimum sequential delay. Wall-clock is the binding constraint (PLAN.md §5), so this plan bundles probes into the fewest possible runs.

All probe scripts follow `[STANDARDS.md](../STANDARDS.md)` — same wandb / Modal / reproducibility rules as the trainer. No "it's just a probe" exceptions.

**Where details live:**

| Doc / file | Contents |
| --- | --- |
| This file | What each probe measures, why, metrics, soft expectations, decisions unlocked |
| `[group_a_results.md](./group_a_results.md)` | Group A full-run readout (metrics → PLAN §2/§5/§7 decisions) |
| `main/configs/probe_a_05-24.yaml` | Numeric knobs (created at implement time; source of truth for scripts) |
| `STANDARDS.md` | Cross-cutting rules: artifacts, checkpoints, seeds, parsing policy, Modal volume note |
| `main/docs/probes/artifacts/*.pointer.json` | Git pointers to volume paths after a run |

Do not duplicate yaml values here once the config exists — reference config keys instead.

## Philosophy

- **Combine where possible.** A single rollout run can produce response-length distribution, parse rate, pass rate by difficulty band, AND raw vLLM throughput. Don't write four scripts when one will do.
- **Mid-training metrics are just logging, not probes.** Anything we want to see during training is a wandb log line — no separate runs.

## Groups

Two probe groups run in parallel. A third "group" is just instrumentation on the first real training run.

### Group A — one combined standalone run, starts now, parallel with skeleton build

Doesn't need our trainer. Needs vLLM + Qwen3-1.7B-Base + Qwen3-4B-Instruct-2507 + Polaris. **One script, two Modal `@app.function`s** (Phase 1 + Phase 2 chained on the volume — separate containers to avoid vLLM engine-swap VRAM leaks; see `group_a_impl.md` § 4). **All metrics → wandb** so nothing is lost; we don't want to re-run.

**Script / config (implement time):** `main/probes/group_a_rollout_judge.py`, `main/configs/probe_a_05-24.yaml`.

**GPU class:** **H100** (locked). Log `gpu_class` tag and Modal $/hr.

**Sampling:** **25 problems per difficulty band × 8 bands = 200 problems.** Phase 1: 8 rollouts each (1600 completions). Phase 2 judge: **all 200 problems** (200 batched judge calls, 8 rollouts per call) — enough for cost/latency distributions without 100/band rollout cost.

**Seeds:** `global_seed` in yaml (default 42). Per-rollout seeds derived per STANDARDS.md.

**Wandb:** project `cs224r-minority-voting`, run group `probe-A-05-24`. Tags per STANDARDS (`phase`, `operator`, `gpu_class`, …). One wandb run for both phases.

**Prompt + reward:** See `[prompt_extraction_research.md](./prompt_extraction_research.md)` — **DAPO `Answer:` template + `math_dapo` default (Minerva) parser**, not pilot `\boxed{}` (prompt–parser mismatch). Port `math_dapo.py` to `main/train/reward.py`. Log `has_boxed` / `strict_parse_ok` as diagnostics only. **Not** Math-Verify (eval-only). Implementation steps: `[group_a_impl.md](./group_a_impl.md)`.

**vLLM knobs (initial — tune in yaml, see config):**

| Phase | Key | Start |
| --- | --- | --- |
| 1 policy | `gpu_memory_utilization` | 0.90 |
| 1 policy | `max_model_len` | 5120 (1024 + 4096) |
| 1 policy | `max_prompt_length` | 1024 |
| 1 policy | `max_response_length` | 4096 |
| 1 policy | `enable_prefix_caching` | true |
| 2 judge | `gpu_memory_utilization` | 0.88 |
| 2 judge | `max_model_len` | 32768 (HF OOM-fallback default; Qwen3-4B-Instruct-2507 native is 262144) |
| 2 judge | batched judge | yes — match production |

**Artifacts:** Phase 1 rollouts jsonl → Modal volume (streaming). On Phase 1 completion, **flush phase artifact** before engine swap (STANDARDS intermediate checkpoints). Git pointer only: `main/docs/probes/artifacts/05-24_group_a.pointer.json`. Phase 2 reads from volume path in pointer.

**Phase 1 — base rollout**

- Load Qwen3-1.7B-Base in vLLM.
- Sample 200 Polaris problems (stratified 25/band), temp=1.
- **Wandb log per-rollout:** `length_tokens`, `prompt_tokens`, `has_boxed`, `has_answer_line`, `parse_ok`, `parsed_answer`, `reward`, `difficulty_band` (see research doc §10).
- **Wandb log per-prompt:** `mixed_reward` (fraction of 8 rollouts correct ∈ (0,1)) — reward-density signal for §2/§7.
- **Wandb log per-batch:** `vllm_tokens_per_sec`, `wall_clock_s`, `vram_gb_used`.
- **Wandb final panels:** length histogram (p50/p90/p95/p99), prompt token histogram, parse rate, pass rate per band, mixed-reward rate per band, tokens/sec.

**Phase 2 — judge cost**

- Unload 1.7B; load Qwen-3-4B-Instruct in vLLM.
- All 200 problems from Phase 1 volume artifact. Judge prompt: Poly-EPO §A.1 instruction block (pilot `poly_epo_paper_a1` / `0519_poly_epo_methodology.md`); few-shots from paper PDF if not in pilot extract.
- **Wandb log per-call:** `wall_clock_s`, `input_tokens`, `output_tokens`, `cluster_count`, `cluster_100_hits`, `truncated` if input > `max_model_len`.
- **Wandb engine-level:** `judge_vram_gb_used`.
- **Wandb final panels:** wall-clock histogram, output-tokens histogram, judge VRAM, `$/call` (= wall_clock × Modal rate for logged `gpu_class`).

**Soft expectations (not hard gates):** primary `parse_ok` rate >~90% under DAPO `Answer:` prompt; pass rate monotone vs difficulty band; judge median wall-clock and $/call inform whether Minority-CoT stays in scope — decide after seeing panels.

**Combined outputs:**

| Output | Updates |
| --- | --- |
| p50/p90/p95/p99 response length | PLAN.md §5 `max_response_length` |
| Parse rate (%) | Reward parser sanity |
| Pass rate + mixed-reward per band | PLAN.md §2 sampling |
| vLLM tokens/sec | PLAN.md §7 step-time + GPU class |
| Judge wall-clock / VRAM / $/call | PLAN.md §5 judge; §7 cost; may cut CoT arms |

**Runtime:** ~2 hr at 200×8 (was ~1.5 hr at old 200-problem plan); longer if GPU is slower than estimate.

**Results (full run, 05-25):** [`group_a_results.md`](./group_a_results.md) — wandb `t33091vc`.

### Group B — first end-to-end probe, after §5 skeleton is built

Needs the trainer skeleton (`rollout.py` + `reward.py` + `objective.py` + `loss.py` + `trainer.py`) up to one full step. GRPO on 10–50 Polaris prompts. Target GPU class: same as Group A if possible.

**Wandb:** project `cs224r-minority-voting`, run group `probe-B-05-25` (see [`group_b_impl.md`](./group_b_impl.md)).

**B1: End-to-end step probe (combined).**

- *Question:* VRAM watermark, `update_weights` timing, microbatch OOM ladder, step-time decomposition, async overlap need.
- *Method:* one full GRPO step, 32-prompt toy batch; per-phase timings; sweep microbatch until OOM.
- *Outputs:* phase wall-clock %, max microbatch, VRAM at chosen microbatch, `update_weights` time.
- *Decisions:* microbatch, grad accum, `gpu_memory_utilization` for **collocated** train (~0.45 start per sizing agent), sync cadence, async overlap go/no-go, $/step.
- *Runtime:* <1 hr once skeleton exists.

### Group C — mid-training instrumentation (not separate runs)

**Not separate Modal jobs** — implement as `wandb.log` lines in `trainer.py` each training step. Required for **Poly-EPO Fig. 2–style** training curves (see PLAN §5 *Training-time reporting*). Checkpoints alone are **not** sufficient (no rollouts saved).

| ID | Wandb key(s) | Poly-EPO analogue | Arms |
| --- | --- | --- | --- |
| **C1** | `train/prompt_coverage`, `train/mixed_reward_rate`, `train/frac_prompts_{0..8}_correct` | Fig. 2 **right** + per-step **k-of-8** histogram (9 fractions summing to 1) | **All** |
| **C1b** | `train/parse_ok_rate`, `train/extract_path_*` | (diagnostic; not in paper) | **All** |
| **C2** | `train/mean_completion_tokens` (+ optional p95) | length / collapse monitor | **All** |
| **C3** | marginal-advantage percentiles @ step 100, 200, … | set-RL advantage shape | **Minority-*, Poly-EPO only** |
| **C4** | `train/mean_unique_strategy_clusters_correct` | Fig. 2 **left** (LM-judge clusters on **correct** rollouts) | **Minority-CoT** (in-loop judge) |
| **C4b** | `train/mean_unique_answer_clusters_correct` | our cheaper analogue (answer-hash) | **Minority-answer, Poly-EPO-answer** |

**GRPO v1 minimum before full arm:** C1 + C1b + C2. Do **not** block launch on C3/C4.

**Status (2026-05-26):** **C1 + C1b + C2 implemented** in `trainer.py` (`aggregate_train_step_wandb_metrics`). C3/C4 still arm-specific / future.

## Sequencing

```
now ─────────────────► skeleton ready ───────────────► GRPO running
  │                       │                              │
  ├── Group A (Phase 1    │                              │
  │   rollouts → Phase 2  │                              │
  │   judge)              │                              │
  │                       ├── Group B                    │
  │                       │                              ├── C1–C3 logs
```

Group A and skeleton build in parallel. Group B after skeleton. Group C on first real run.

## What gets updated from probe outputs

| Probe | Updates in PLAN.md |
| --- | --- |
| Group A | §5 `max_response_length`, judge subsystem; §2 sampling; §7 GPU class, step-time, judge cost; possibly cut CoT arms |
| Group B | §7 microbatch, grad accum, collocated `gpu_memory_utilization`, sync, step-time, async overlap |
| C1–C4b | PLAN §5 + `trainer.py` wandb (not separate probe runs) |

## Open

- Post-RL length probe — defer until C2.
- Escalate to Rank 2 multi-path parser (research doc §8) if primary `parse_ok` <~90%.
- Modal volume name — lock when launcher is written (STANDARDS Open).
