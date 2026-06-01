# Group A probe — implementation guide

**Prerequisites:** Read `[05-24_probe_plan.md](./05-24_probe_plan.md)`, `[prompt_extraction_research.md](./prompt_extraction_research.md)`, `[STANDARDS.md](../STANDARDS.md)`. Full-run readout: `[group_a_results.md](./group_a_results.md)`.

**Locked choices:**

| Knob | Value |
| --- | --- |
| GPU | H100 (`modal_price_per_sec: 0.001097`) |
| Prompt | DAPO `Answer:` template (research doc §8 Rank 1) |
| Parser | `math_dapo` Minerva default (`Answer:` + `normalize_final_answer`) — **not** pilot `\boxed{}` primary |
| Phase 1 model | `Qwen/Qwen3-1.7B-Base` — plain string to vLLM, no chat template |
| Phase 2 judge | `Qwen/Qwen3-4B-Instruct-2507` — Poly-EPO §A.1 prompt (see § Judge) |
| Sample | 25/band × 8 bands = 200 prompts, 8 rollouts each |
| Polaris `difficulty` | `0/8`, `1/8`, …, `7/8` (8 bands); all bands ≥25 rows after integer cleaning. **Agent must verify** by running `df['difficulty'].value_counts()` once at first launch — if actual values differ (e.g., `1/8..7/8`), update `sampling.difficulty_bands` in yaml to match. PLAN.md §2 is stale on this point; the dataset is source of truth. |
| `problem_id` | Manifest index `0..199` (stable across resume; use in seed hash) |
| Config | `main/configs/probe_a_05-24.yaml` |

---

## Pre-flight locks (agent: do not re-litigate)

| Item | Lock |
| --- | --- |
| Modal billing | `$/call = wall_clock_s × modal_price_per_sec` with `0.001097` on H100 |
| Modal secrets | Standardized uppercase names: `HUGGINGFACE`, `WANDB_API_KEY`. Each teammate creates secrets with these exact names on their Modal profile (the pilot used lowercase `huggingface` / `wandb-api-key` — those need to be re-created or renamed under the new convention). The wandb API key inside the secret auto-identifies the operator in wandb; no per-operator config field needed. |
| Modal workspace | Set via active Modal profile at launch (`modal profile activate <slug>`). Not pinned in code, not in yaml — wandb identity comes from the key inside `WANDB_API_KEY`. |
| Judge prompt source | Canonical source: `main-verl/judge/prompts/poly_epo_a1.md` (mirrored to `main/judge/poly_epo_a1.md`). **Paper §A.1 DOES have two few-shot examples** — the prior "no few-shots" claim was a misread of the paper at 0519 extraction time and led to ~6 KB of FS exemplars being dropped from both copies. Restored 2026-05-31; see `main/docs/timeline.md` 2026-05-31 entry + memory `project_judge_prompt_fewshot_gap`. |
| Judge wrapper format | `f"{n}. {completion}"` joined by `\n`, 1-indexed. Port `_build_responses_block` from `pre-milestone/nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py:245`. Matches the JSON output schema (`"1".."N"` keys) — do not invent a new wrapper. |
| Judge JSON parser | Port `_assignment_from_poly_epo_payload` from same file (line ~324). Validates keys 1..N, each entry has `chain_of_thought` + `cluster_id`, cluster_id=100 is the degenerate bucket. |
| Live judging at training time | **Out of scope for Group A.** Group A's Phase 2 is a one-shot offline batched pass. The $/call + wall-clock + VRAM numbers feed PLAN §5's decision on live-judging architecture (sidecar GPU vs co-located vs API). Do not attempt to solve live judging here. |
| Phase 1 / Phase 2 isolation | **Split into two Modal `@app.function`s** chained on the volume — Phase 1 writes `phase1_done.json` + rollouts jsonl, Phase 2 reads them. Fresh container per phase avoids vLLM engine-swap VRAM leaks. Pass `wandb_run_id` from Phase 1 to Phase 2 so both log to the same run. |
| Rollout `finish_reason` | vLLM returns `RequestOutput.outputs[i].finish_reason` as a string (typically `"stop"` or `"length"`; other values possible by version). **Persist verbatim, do not assert a closed set** in every rollout jsonl line. Used in readout to separate "wrong answer" from "hit `max_response_length`" — critical for the PLAN §5 length-cap decision. |
| Polaris sampling | Stratified 25 per band for `difficulty ∈ {0/8,…,7/8}`; `global_seed`; **probe-only:** drop non-integer gold + empty problem (train freeze keeps full gold — PLAN §2). |
| Seeds | Deterministic formula per STANDARDS: `global_seed + problem_id * rollouts_per_prompt + rollout_idx`. **Do NOT use Python's built-in `hash()`** (process-salted, breaks reproducibility). |
| Phase 1 artifact | `manifest.jsonl` then `phase1_rollouts.jsonl` on volume (schemas below) |
| Packaging | Mirror pilot: `add_local_dir(main → /root/main)`, `sys.path` includes `/root/main`; `modal run` from **repo root** with `--config main/configs/probe_a_05-24.yaml` for smoke and full runs |
| HF weights cache | Reuse Modal volume `hf-cache` at `/root/.cache/huggingface` (same as pilot) |
| Wandb | Entity `224r-project`, project `cs224r-minority-voting`, run name `probe-A_{operator}_{MM-DD-HHMM}` |

**Parser authority:** STANDARDS § Reward (DAPO `Answer:` + `math_dapo`) overrides the older table row that mentions `\boxed{}` only.

---

## Files to create

```
main/
  infra/
    modal_image.py          # pinned deps: vllm, torch, wandb, datasets, pyyaml
    modal_volume.py         # volume name constants (artifacts + hf-cache)
  train/
    reward.py               # port math_dapo default path; 0/1 reward
    prompts.py              # DAPO_PROMPT_TEMPLATE, format_problem(problem)
  judge/
    poly_epo_a1.md          # copied from analysis_a_prompt.md (system + user)
    format.py               # build judge messages, responses_block, truncation hook
  probes/
    group_a_rollout_judge.py  # Modal entrypoint, two phases
  configs/
    probe_a_05-24.yaml
  tests/
    test_reward.py          # unit tests from §2 table
main/docs/probes/artifacts/
  .gitkeep                  # pointer json written after run (or post-pull)
```

---

## 1. `main/train/prompts.py`

```python
DAPO_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""
```

- No chat template for Qwen3-1.7B-Base — return a single string.
- Log `prompt_variant: dapo_answer_v1` in wandb.

---

## 2. `main/train/reward.py`

Port from [verl `math_dapo.py`](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py):

- `last_boxed_only_string`, `remove_boxed`, `normalize_final_answer`
- `is_correct_minerva(solution_str, gt)` — primary
- `is_correct_strict_box` — **diagnostic only** for wandb `strict_parse_ok`

**Primary API:**

```python
def compute_reward(completion: str, gold: str) -> dict:
    # clip completion to [-300:] before parse (match DAPO)
    # return { "reward": 0|1, "parse_ok": bool, "parsed_answer": str|None,
    #          "parsed_is_int": bool, "has_boxed": bool, "has_answer_line": bool,
    #          "strict_parse_ok": bool }
```

- `reward=1` iff Minerva path matches normalized gold.
- `parse_ok` iff extraction is not `[INVALID]` / empty after normalize.

**Unit tests** (run locally before Modal):

| Completion tail | Gold | reward | parse_ok |
| --- | --- | --- | --- |
| `...\nAnswer: 42` | `42` | 1 | 1 |
| `...\nAnswer: 41` | `42` | 0 | 1 |
| `...\n\\boxed{42}` only | `42` | 0 | 0 (primary) |
| no answer marker | `42` | 0 | 0 |

---

## 3. `main/configs/probe_a_05-24.yaml`

```yaml
global_seed: 42
operator: nancy  # STANDARDS run name; operator identity in wandb comes from WANDB_API_KEY
gpu_class: H100
modal_price_per_sec: 0.001097

# smoke: 1 problem per band × 2 rollouts (keeps stratification path live)
smoke: false
smoke_per_band: 1
smoke_n_rollouts: 2

# optional: force Phase 2 only (else auto-detect phase1_done.json on volume)
resume_from_phase: null  # or 2

sampling:
  per_band: 25
  difficulty_bands: ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8"]
  rollouts_per_prompt: 8
  temperature: 1.0

phase1:
  model: Qwen/Qwen3-1.7B-Base
  max_prompt_length: 1024
  max_response_length: 4096
  gpu_memory_utilization: 0.90
  max_model_len: 5120
  max_num_seqs: 128
  enable_prefix_caching: true

phase2:
  model: Qwen/Qwen3-4B-Instruct-2507  # confirmed via HF model card 2026-05-25
  gpu_memory_utilization: 0.88
  # Native context is 262144 (no RoPE scaling). HF card recommends 32768 as OOM fallback.
  # Worst-case judge input ≈ 17-18k tokens (8 × 2048 rollouts + instructions); 32768 leaves
  # ~14k headroom. Do NOT cap individual rollouts (Poly-EPO faithful). Log judge_input_tokens.
  max_model_len: 32768
  max_num_seqs: 4
  temperature: 0.0          # deterministic judge
  max_tokens: 1024          # judge JSON for 8 entries is small; bump only if outputs truncate
  apply_chat_template: true # 4B-Instruct has a chat template (unlike 1.7B-Base)

artifacts:
  volume_name: main-artifacts  # auto-created per workspace via create_if_missing=True
  volume_mount: /vol
  hf_cache_volume: hf-cache
  manifest_path: probes/05-24/group_a/manifest.jsonl
  rollouts_path: probes/05-24/group_a/phase1_rollouts.jsonl
  phase1_done_path: probes/05-24/group_a/phase1_done.json
  pointer_path: docs/probes/artifacts/05-24_group_a.pointer.json

wandb:
  entity: 224r-project
  project: cs224r-minority-voting
  group: probe-A-05-24
```

---

## 4. `main/probes/group_a_rollout_judge.py`

### Modal app skeleton

- `from infra.modal_image import image`
- **App name**: per STANDARDS § Modal, construct dynamically:
  ```python
  app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-probe-a-untagged"))
  ```
  Launch script computes `CS224R_APP_NAME = f"cs224r-probe-a-{phase}-{operator}-{MM-DD-HHMM}"` (e.g. `cs224r-probe-a-smoke-nancy-05-25-1830`) before invoking `modal run`. Phase ∈ {smoke, full}.
- Mount artifact volume `main-artifacts` at `/vol`
- Mount `hf-cache` at `/root/.cache/huggingface`
- **Secrets:** `modal.Secret.from_name("HUGGINGFACE")`, `modal.Secret.from_name("WANDB_API_KEY")`.
- `@app.function(gpu="H100", timeout=10800, volumes={...})` — 3 hr; tune if smoke/full runs differ

### Phase 1 — rollouts

1. Load yaml; init wandb (one run for both phases).
2. Log config, git sha, seeds, dep versions.
3. Sample 200 Polaris rows: 25 per `difficulty` in `sampling.difficulty_bands`, `seed=global_seed`. Stream HF `POLARIS-Project/Polaris-Dataset-53K` or read cached `manifest.jsonl` on volume if resuming.
4. Write `manifest.jsonl` (one row per prompt) before rollouts.
5. Build vLLM engine (1.7B), generate 8 completions per prompt (batched); per-rollout seed from `global_seed + problem_id * rollouts_per_prompt + rollout_idx` (deterministic — not Python `hash()`).
6. Per completion: `compute_reward()` + token lengths; append **rollout** jsonl lines incrementally.
7. Log wandb scalars/histograms per research doc §10.
8. **Phase checkpoint:** `volume.commit()`; write `phase1_done.json` (schema below).

### Phase 2 — judge (separate Modal `@app.function`)

Runs as its own function so the Phase 1 container exits and CUDA state is fresh before the 4B engine loads. Phase 2 auto-runs after Phase 1 in the full pipeline; can also be invoked standalone for resume.

9. Read `phase1_done.json` from volume (error if missing). Reattach to Phase 1's `wandb_run_id`. Read `manifest.jsonl` + `phase1_rollouts.jsonl`.
10. Load `Qwen/Qwen3-4B-Instruct-2507` in a fresh vLLM engine. No engine swap, no teardown ritual needed — different container.
11. Per prompt: load 8 completions; build judge messages from `main/judge/poly_epo_a1.md` via `judge/format.py`. `_build_responses_block` is the pilot's `f"{n}. {completion}"` joiner (port verbatim from `analysis_a_llm_clusters.py:245`). **Do not** pass gold or parsed answers to the judge.
12. Apply chat template (Instruct model). One vLLM request per prompt (batch up to `max_num_seqs` concurrent). Log `judge_input_tokens`; if >`max_model_len`, log `truncated: true` and skip (do not silently chop — we want to know the rate).
13. Parse JSON via ported `_assignment_from_poly_epo_payload`. Log `cluster_count`, `cluster_100_hits`, input/output tokens, `wall_clock_s`, `$/call = wall_clock_s × modal_price_per_sec`, `json_parse_ok`.
14. Final wandb panels; write pointer json on volume; operator pulls to `main/docs/probes/artifacts/05-24_group_a.pointer.json` via `modal volume get`.

### Smoke mode

- `smoke: true` → 2 problems, 2 rollouts, both phases, verify wandb + volume + parser unit behavior.

### Artifact schemas

**`manifest.jsonl`** (200 lines, written once per run):

```json
{"problem_id": 0, "problem": "...", "gold": "42", "difficulty_band": "3/8", "hf_index": 12345}
```

- `problem_id`: int `0..199` in band-major order (band `0/8` ids 0–24, then `1/8`, …).
- `hf_index`: optional HF dataset row index for provenance.

**`phase1_rollouts.jsonl`** (1600 lines = 200 × 8):

```json
{"problem_id": 0, "rollout_idx": 0, "completion": "...", "reward": 1, "parse_ok": true, "parsed_answer": "42", "parsed_is_int": true, "has_boxed": false, "has_answer_line": true, "strict_parse_ok": false, "length_tokens": 1200, "prompt_tokens": 180, "finish_reason": "stop"}
```

- `finish_reason`: verbatim string from vLLM (typically `"stop"` or `"length"`). Used in readout to distinguish "wrong answer" from "hit `max_response_length`" when picking PLAN §5 length cap. Do not assert a closed value set.

**`phase1_done.json`:**

```json
{"n_prompts": 200, "n_rollouts": 1600, "wandb_run_id": "...", "completed_at": "ISO8601"}
```

**Pointer** (`05-24_group_a.pointer.json`, STANDARDS):

```json
{"modal_volume": "main-artifacts", "path": "probes/05-24/group_a/", "wandb_run_id": "...", "created_at": "ISO8601"}
```

---

## 5. Polaris sampling

```python
# 25 rows per difficulty in ["0/8", "1/8", ..., "7/8"], seed=global_seed
# Fields: problem, answer, difficulty
# Probe cleaning only: drop non-integer gold, empty problem (NOT train freeze — see PLAN §2 / decisions.md)
# Post-clean: each band has >= 25 eligible rows (verified; no oversample / merge)
```

Persist sampled manifest to volume: `probes/05-24/group_a/manifest.jsonl` before Phase 1 rollouts.

---

## 6. Launch commands

Use the launch wrapper so `CS224R_APP_NAME` gets set per STANDARDS § Modal. Sketch:

```bash
# main/scripts/launch_probe_a.sh
#!/bin/bash
CFG="${1:-main/configs/probe_a_05-24.yaml}"
PHASE=$(yq '.smoke' "$CFG" | grep -q true && echo smoke || echo full)
OP=$(yq '.operator' "$CFG")
TS=$(date +%m-%d-%H%M)
export CS224R_APP_NAME="cs224r-probe-a-${PHASE}-${OP}-${TS}"
exec modal run --detach main/probes/group_a_rollout_judge.py --config "$CFG"
```

```bash
# Smoke (set smoke: true in yaml first):
bash main/scripts/launch_probe_a.sh

# Full run:
bash main/scripts/launch_probe_a.sh main/configs/probe_a_05-24.yaml
```

After run:

```bash
# Pull pointer or inspect on volume
cat main/docs/probes/artifacts/05-24_group_a.pointer.json
```

---

## 7. Build order (checklist)

- [ ] `reward.py` + local unit tests
- [ ] `prompts.py`
- [ ] `modal_image.py` (pin vllm/torch; document versions in wandb)
- [ ] `modal_image.py` + `modal_volume.py` (volumes auto-create via `create_if_missing=True`)
- [ ] `group_a_rollout_judge.py` Phase 1 `@app.function` → smoke
- [ ] Add Phase 2 as a **separate** `@app.function` (own container); chain after Phase 1 in pipeline mode, callable standalone for resume
- [ ] Full run H100 detached
- [ ] Readout: update PLAN §2/§5/§7 from [`group_a_results.md`](./group_a_results.md)

---

## 8. Post-run readout

**Full run (05-25):** [`group_a_results.md`](./group_a_results.md) — wandb `t33091vc`.

| Panel | PLAN update |
| --- | --- |
| Length p50/p95/p99 | §5 `max_response_length` |
| `parse_ok` rate | Lock `reward.py` or escalate Rank 2 parser |
| Pass rate / mixed-reward per band | §2 sampling |
| tokens/sec | §7 step time on H100 |
| Judge $/call, wall-clock, VRAM | §5 judge hosting; §3 CoT arm go/no-go |

---

## 9. Still open (operator decisions only — agent should not block on these)

| Item | Status |
| --- | --- |
| `pyproject.toml` vs image-only pins | Defer to Group B per STANDARDS. Pin in `modal_image.py` for now. |

### Resolved this round (do not re-litigate)

- **Judge length policy:** no per-rollout cap. Qwen3-4B-Instruct-2507 native context is 262144; we use `max_model_len: 32768` (HF's own OOM fallback) which gives ~14k headroom over worst-case 17-18k input. Log `judge_input_tokens` + `truncated`; only revisit if overflow rate >5%. (Poly-EPO faithful — they don't cap either.)
- **Modal secrets:** `HUGGINGFACE`, `WANDB_API_KEY` (uppercase). Each teammate creates these on their own profile. Operator identity in wandb is automatic via the key.
- **Modal workspace:** chosen at launch via `modal profile activate <slug>`. Not in code, not in yaml.
- **Phase 2 sampling knobs:** `temperature: 0`, `max_tokens: 1024`, `apply_chat_template: true`. In yaml.
- **JSON parse failures:** single attempt, log `json_parse_ok=false`, skip cluster metrics for that prompt. The failure *rate* is the probe signal; retry policy is a training-time decision.
- **Engine swap:** moot — Phase 1 and Phase 2 are separate Modal functions.
- **Truncation vs wrong answer:** `finish_reason` from vLLM persisted per rollout. No new policy needed.
- **Smoke mode:** `smoke_per_band: 1` keeps stratification path live.
- **Live judging during training:** out of scope; deferred to PLAN §5 / Group B.

### Optional (log if cheap, not required)

- `reward_would_be_boxed`, `strict_box_pred` from research §10 diagnostics.

---

## 10. Agent handoff prompt (copy-paste)

> Implement Group A per `main/docs/probes/group_a_impl.md` (including Pre-flight locks and artifact schemas). Port `math_dapo` default parser to `main/train/reward.py` + `tests/test_reward.py`. DAPO Answer prompt in `prompts.py`. Copy judge prompt to `main/judge/poly_epo_a1.md` from `pre-milestone/nancy_explore/run0_analysis/config/analysis_a_prompt.md`; port `_build_responses_block` and `_assignment_from_poly_epo_payload` from `pre-milestone/nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py`. Modal H100 (`modal_price_per_sec: 0.001097`), smoke mode, two `@app.function`s (Phase 1 + Phase 2 split, fresh containers — see § 4), phase artifact flush, wandb entity `224r-project` / project `cs224r-minority-voting`. Polaris bands per § Pre-flight (verify dataset values match before sampling). Use Modal secrets `HUGGINGFACE` and `WANDB_API_KEY` (uppercase). App name constructed dynamically from `CS224R_APP_NAME` env var per STANDARDS § Modal. Do not use pilot `\boxed{}` as primary reward. Reference `prompt_extraction_research.md` for logging fields.
