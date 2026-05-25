# Group A — workflow

Build the Group A probe in 4 sequential phases. Each phase has a **Build** spec (what to create) and an **Audit** spec (what to check). Use a fresh agent for each phase and for each audit — don't reuse context.

**To run a phase:** point an agent at this file and tell it which phase + which mode (build or audit). E.g. *"Read `main/docs/probes/group_a_workflow.md` § Phase A Build and execute it."* or *"Read § Phase C Audit and execute it."*

**Source-of-truth docs all phases must read:**
- `main/docs/probes/group_a_impl.md` — the implementation spec
- `main/docs/STANDARDS.md` — cross-cutting rules
- `main/docs/probes/prompt_extraction_research.md` — parser/prompt rationale

**Dependency graph:**
```
A (pure Python) ──┐
                  ├──► C (Phase 1 rollout) ──► D (Phase 2 judge) ──► full run (operator)
B (Modal infra) ──┘
```

A and B can run in parallel. C and D are strictly sequential.

---

## Phase A — pure Python (reward, prompts, judge format)

**Creates:** `main/train/reward.py`, `main/train/prompts.py`, `main/judge/format.py`, `main/judge/poly_epo_a1.md`, `main/tests/test_reward.py`

**Dependencies:** none.

### Build

Read these first:
- `group_a_impl.md` §§ 1, 2; Pre-flight locks rows on Judge prompt source / wrapper format / JSON parser
- `STANDARDS.md` §§ Reproducibility, Code structure, Reward / answer parsing
- `prompt_extraction_research.md` §§ 3.2 and 8 Rank 1

Then create:

- **`main/train/prompts.py`** — `DAPO_PROMPT_TEMPLATE` literal from `group_a_impl.md` § 1, plus `format_problem(problem: str) -> str` that substitutes via `.format(problem=problem)`.

- **`main/train/reward.py`** — port `last_boxed_only_string`, `remove_boxed`, `normalize_final_answer`, `is_correct_minerva`, `is_correct_strict_box` from `verl/utils/reward_score/math_dapo.py` (upstream). Expose `compute_reward(completion: str, gold: str) -> dict` with the exact schema in `group_a_impl.md` § 2. **Reward sign mapping**: VeRL returns +1/−1; we want `1 if r > 0 else 0`. Clip completion to `[-300:]` once before primary `is_correct_minerva` call. The diagnostic `strict_parse_ok` field comes from `is_correct_strict_box` on the **raw completion** (which internally re-clips to `[-100:]` — let it do its own clip; do not pre-clip the input to it). Python `logging`, not `print`.

- **`main/judge/poly_epo_a1.md`** — byte-identical copy of `pre-milestone/nancy_explore/run0_analysis/config/analysis_a_prompt.md`. No edits.

- **`main/judge/format.py`** — port these from `pre-milestone/nancy_explore/run0_analysis/analysis_a/analysis_a_llm_clusters.py`:
  - `_build_responses_block` (~line 245)
  - `_assignment_from_poly_epo_payload` (~line 324)
  - `_normalize_cluster_id` (called by `_assignment_from_poly_epo_payload`)
  - `_normalize_clusters` (called transitively)
  - `_strip_json_fences` (used by the JSON parser)
  - `_load_prompt_templates` pattern (~line 220) — adapt path to load `main/judge/poly_epo_a1.md`
  - Plus `build_judge_messages(problem: str, rollouts: list[dict]) -> tuple[str, str]` returning `(system, user)` by loading `poly_epo_a1.md` and substituting `{n_responses}`, `{problem}`, `{responses_block}`.
  
  **`n_responses` must be derived from `len(rollouts)`**, not a module-level constant — smoke runs use 2 rollouts/prompt while full runs use 8. Pilot hardcoded `N_RESPONSES = 8`; do not copy that.
  
  Test by porting verbatim then running a quick `python -c "from main.judge.format import _assignment_from_poly_epo_payload; ..."` to confirm no `NameError` on transitive helper calls.

- **`main/tests/test_reward.py`** — pytest cases for the 4-row table in `group_a_impl.md` § 2 (assert on dict values, not truthiness). Plus tests for `build_judge_messages`:
  - 8 dummy rollouts → system contains `"8"` substituted in `{n_responses}`, user contains lines starting `1.` through `8.`
  - **2 dummy rollouts → system contains `"2"`, user contains `1.` and `2.` only** (smoke-mode behavior; catches accidental `N_RESPONSES = 8` constant)
  - One test that `_assignment_from_poly_epo_payload` correctly maps `cluster_id: 100` to the degenerate bucket per pilot semantics

Do not create `__init__.py` unless an import requires it. Do not add dependencies beyond stdlib + pytest. Do not invent helpers — port verbatim where the spec says to.

Finish by running `pytest main/tests/ -v`. Report which files were created and the pytest output.

### Operator verification

```bash
pytest main/tests/ -v
diff main/judge/poly_epo_a1.md pre-milestone/nancy_explore/run0_analysis/config/analysis_a_prompt.md
```

5 tests pass; diff is empty.

### Audit

Read the 5 files created above plus `group_a_impl.md` §§ 1, 2 + Pre-flight locks (Judge rows) + the pilot source `analysis_a_llm_clusters.py:220-340`.

Check:
- `compute_reward` returns exactly the keys in `group_a_impl.md` § 2 — wrong key names break downstream wandb logging.
- Reward sign mapping is literally `1 if r > 0 else 0`, not e.g. `int(r)` (which would give -1 for VeRL's -1).
- 300-char clip applied to the raw completion *before* the primary `is_correct_minerva` call, not after.
- `is_correct_strict_box` is wired as the diagnostic `strict_parse_ok` field only, never as primary `reward`.
- `_build_responses_block` produces `f"{n}. {completion}"` joined by `\n`, 1-indexed (n starts at 1).
- `n_responses` in `build_judge_messages` is `len(rollouts)`, not a hardcoded module constant. Verify by checking the 2-rollout test passes.
- All transitive helpers ported (`_normalize_cluster_id`, `_normalize_clusters`, `_strip_json_fences`) — `python -c "from main.judge.format import *"` should not `NameError`.
- `_assignment_from_poly_epo_payload` correctly handles `cluster_id: 100` → degenerate bucket per pilot.
- `poly_epo_a1.md` byte-identical to pilot source.
- Tests assert on dict values, not just truthiness.
- No silent `except: pass`; errors propagate per STANDARDS.

Report verdict (ready / needs fixes) + bullet list of issues with `file:line`. Under 200 words. Do not propose fixes.

### Exit criteria
All pytest tests pass; audit returns ready.

---

## Phase B — Modal infra

**Creates:** `main/infra/modal_image.py`, `main/infra/modal_volume.py`, `main/infra/hello_modal.py`, `main/scripts/launch_probe_a.sh`

**Dependencies:** none (can run in parallel with A).

### Build

Read first:
- `STANDARDS.md` § Modal (image, secrets, app name, volumes — all required)
- `group_a_impl.md` Pre-flight locks (packaging, HF cache, app name) and § 4 Modal app skeleton
- `pre-milestone/pilot/infra/modal_app.py` — pattern to mirror (image construction, `add_local_dir`, volume mounting). Lift the pattern; do not copy wholesale.
- `pre-milestone/pilot/infra/modal_volumes.py` — pattern for volume name constants.

Create:

- **`main/infra/modal_volume.py`** — module-level constants only:
  ```python
  ARTIFACTS_VOLUME_NAME = "main-artifacts"
  HF_CACHE_VOLUME_NAME = "hf-cache"
  ARTIFACTS_MOUNT = "/vol"
  HF_CACHE_MOUNT = "/root/.cache/huggingface"
  ```
  No `Volume.from_name` calls here — names only.

- **`main/infra/modal_image.py`** — exports `image` (a `modal.Image`) and `app_name()` helper.
  - Image: `modal.Image.debian_slim(python_version="3.11").pip_install([...])`. Pin pilot's deps verbatim where overlapping (`torch==2.5.1` etc. — read `pre-milestone/pilot/infra/modal_image.py` directly). **Pilot does not have `vllm` or `datasets`** — add them: `vllm==0.6.3` and `datasets>=2.20`. Plus `wandb`, `pyyaml`, `pytest`. Match pilot's `extra_index_url` for torch.
  - `add_local_dir("main", "/root/main", ignore=["docs", "data", "*.md", "__pycache__", ".pytest_cache"])` on the **image** (not the function). Path relative to repo root. The `ignore=` is critical — without it, you ship `main/docs/` (which includes this workflow doc) into the container, bloating the image and slowing cold-starts.
  - After `add_local_dir`, append `/root/main` to `sys.path` via `image.run_commands("echo '/root/main' > /usr/local/lib/python3.11/site-packages/main.pth")` OR set `PYTHONPATH=/root/main` in the function decorator. Imports inside the probe file are `from train.reward import compute_reward` (NOT `from main.train.reward`).
  - `app_name()` returns `os.environ.get("CS224R_APP_NAME", "cs224r-untagged")`.

- **`main/scripts/launch_probe_a.sh`** — bash launcher per `group_a_impl.md` § 6. `set -euo pipefail`; compute `CS224R_APP_NAME` from yaml fields + `date +%m-%d-%H%M`; **`exec modal run --detach main/probes/group_a_rollout_judge.py::run_full --config "$CFG"`**. The `::run_full` is required — once Phase D adds a second `@app.function`, bare `modal run <file>` is ambiguous and errors. Phase C will define `run_full` as a thin wrapper around `run_phase1`; Phase D extends it to call both. `chmod +x` the script.

- **`main/infra/hello_modal.py`** — ~10-line verification function. Uses the image, mounts both volumes, lists `/vol`, prints HF cache path. Used only for Phase B verification. `modal run main/infra/hello_modal.py` should succeed in <60s on any operator's profile.

Do not create `group_a_rollout_judge.py` yet — that's Phase C.

Report files created and the verification command.

### Operator verification

```bash
modal profile activate <slug>
modal secret list                       # must show HUGGINGFACE and WANDB_API_KEY
CS224R_APP_NAME=cs224r-hello-test modal run main/infra/hello_modal.py
```

Prints `/vol` listing (empty first run) and HF cache path. ~$0.02.

### Audit

Read the 4 files created plus `STANDARDS.md` § Modal + `group_a_impl.md` Pre-flight locks (packaging/HF cache/app name/secrets) + `pre-milestone/pilot/infra/modal_app.py` (pattern source).

Check:
- App name is dynamic (`os.environ.get("CS224R_APP_NAME", ...)`), not hardcoded.
- `add_local_dir` is on the **image**, not the function. Wrong placement → import failures inside the container.
- `add_local_dir` has an `ignore=` arg excluding at least `docs`, `data`, `*.md`, `__pycache__`. Without it the image ships docs and cold-start time balloons.
- `sys.path` includes `/root/main` (via `.pth` file or `PYTHONPATH` env var). Verify by checking the import convention used: `from train.reward import ...` (correct) not `from main.train.reward import ...` (would fail).
- vLLM and `datasets` are pinned (pilot didn't have them; agent must add). `vllm==0.6.x` and `datasets>=2.20` are sensible defaults.
- Secrets referenced as `Secret.from_name("HUGGINGFACE")` and `Secret.from_name("WANDB_API_KEY")` — uppercase, no typos.
- `Volume.from_name(..., create_if_missing=True)` is the pattern — no manual creation assumed.
- Volume mounts at `/vol` and `/root/.cache/huggingface` (HF tooling expects the latter exactly).
- Launcher invokes `::run_full` explicitly. Bare `modal run <file>` will break in Phase D.
- Launcher `CS224R_APP_NAME` format matches STANDARDS: `cs224r-<purpose>-<phase>-<operator>-<MM-DD-HHMM>`.
- No hardcoded absolute paths that would break on a teammate's machine.

Verdict + issues with `file:line`. Under 200 words.

### Exit criteria
`hello_modal.py` runs successfully on Modal; audit returns ready.

---

## Phase C — Phase 1 rollout function

**Creates:** `main/probes/group_a_rollout_judge.py` (Phase 1 `@app.function` only), `main/configs/probe_a_05-24.yaml`

**Dependencies:** Phases A and B complete and verified.

### Build

Phase 2 (judge) is a separate phase — **do not implement it now**.

Read first:
- `group_a_impl.md` — whole doc, especially §§ 3, 4 Phase 1 steps 1–8, 5, Artifact schemas, Pre-flight locks
- `STANDARDS.md` §§ Reproducibility, Wandb, Modal, Artifact persistence
- `prompt_extraction_research.md` § 10 (logging fields)
- Phase A and B outputs: `main/train/{reward,prompts}.py`, `main/infra/{modal_image,modal_volume}.py`

Create:

- **`main/configs/probe_a_05-24.yaml`** — exactly the yaml block in `group_a_impl.md` § 3. No invented fields. Default `smoke: true` for first run.

- **`main/probes/group_a_rollout_judge.py`** — Phase 1 `@app.function` only:
  - App: `app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-probe-a-untagged"))`.
  - One function `run_phase1`, `@app.function(gpu="H100", timeout=10800, volumes={...}, secrets=[...], image=image)`.
  - Entrypoint reads `--config`, loads yaml.
  - Init wandb (entity `224r-project`, project `cs224r-minority-voting`, run name per STANDARDS format). Log config dict, git SHA, dep versions per STANDARDS § Reproducibility.
  - Load Polaris (`POLARIS-Project/Polaris-Dataset-53K`) via `datasets`. Cache jsonl to `/vol/probes/05-24/group_a/polaris_cache.jsonl` on first run, read cache thereafter.
  - **Before sampling**, run `df['difficulty'].value_counts()` and compare to `sampling.difficulty_bands` — if mismatch, log a warning and use the actual values present. PLAN.md is stale on this point.
  - Stratified sample `per_band` rows per band, `seed=global_seed`. Drop non-integer gold + empty problems first. Smoke mode uses `smoke_per_band` per band.
  - Write `manifest.jsonl` (schema in `group_a_impl.md` § 4) before rollouts.
  - Build vLLM `LLM` with `phase1` yaml block. **Plain string prompts** (no chat template — Qwen3-1.7B-Base has none).
  - Generate `rollouts_per_prompt` completions per prompt via `SamplingParams(n=..., seed=..., ...)`. **Per-rollout seed: `global_seed + problem_id * rollouts_per_prompt + rollout_idx`** — deterministic formula per STANDARDS. **Do NOT use Python's built-in `hash()`** (it's salted per process via `PYTHONHASHSEED` → seeds change every run → no reproducibility).
  - Per completion: `compute_reward`, capture `finish_reason` from vLLM verbatim (do not assert closed set), token counts. Append to `phase1_rollouts.jsonl` on volume **incrementally** (no buffering 1600 in RAM).
  - Per-rollout fields → wandb `Table` (NOT per-step scalars; 1600 rows would blow up the dashboard). Per-batch scalars: `vllm_tokens_per_sec`, `wall_clock_s`, `vram_gb_used`.
  - After all rollouts: log final histograms (length p50/p90/p95/p99, prompt tokens, parse rate, pass rate per band, mixed-reward per band, tokens/sec).
  - `volume.commit()` then write `phase1_done.json` (schema in `group_a_impl.md` § 4).
  - Return wandb run ID (Phase D will reattach to it).

Also define **`run_full` as a thin wrapper** in this same file:
```python
@app.local_entrypoint()
def run_full(config: str):
    run_phase1.remote(config=config)
    # Phase D will extend this to call run_phase2 after Phase 1.
```
The launcher calls `::run_full`. In Phase C, `run_full` just calls Phase 1. Phase D extends it to chain into `run_phase2`.

Python `logging`, not `print`. Errors propagate. Artifact schemas exact — Phase D depends on the contract.

Run smoke. Report wandb URL and volume contents.

### Operator verification

```bash
# smoke: true in yaml (default), then:
bash main/scripts/launch_probe_a.sh

# After ~3-5 min:
modal volume ls main-artifacts probes/05-24/group_a/
# Should show: manifest.jsonl, phase1_rollouts.jsonl, phase1_done.json, polaris_cache.jsonl
```

Wandb run shows config, rollouts Table (~16 rows in smoke), final histograms.

### Audit

Read the 2 files created plus `group_a_impl.md` §§ 3, 4 Phase 1, 5, Artifact schemas, Pre-flight locks; `STANDARDS.md` §§ Reproducibility, Wandb, Artifact persistence.

Check:
- Artifact schemas match exactly. `manifest.jsonl`, `phase1_rollouts.jsonl`, `phase1_done.json` field names + types per § 4. Phase D depends on this contract.
- `finish_reason` persisted verbatim with no `assert` on its value set.
- Per-rollout seeding uses the **deterministic formula** `global_seed + problem_id * rollouts_per_prompt + rollout_idx`. **Reject any use of Python `hash()`** — silently breaks reproducibility because `PYTHONHASHSEED` randomizes per process.
- `run_full` `local_entrypoint` defined as a thin wrapper around `run_phase1.remote(...)`. Launcher must work via `::run_full`.
- Difficulty band verification actually called before sampling, not skipped.
- Polaris caching writes once, reads on subsequent runs.
- Incremental jsonl writes — not buffered in memory.
- Wandb per-rollout logging uses a Table, not per-step scalars.
- `volume.commit()` called before writing `phase1_done.json`.
- App name dynamic (matches Phase B pattern).
- **No chat template** applied to Qwen3-1.7B-Base.
- vLLM `SamplingParams` includes `seed` per rollout.
- Smoke mode uses `smoke_per_band` and `smoke_n_rollouts`, not the full sampling.

Verdict + issues with `file:line`. Under 200 words.

### Exit criteria
Smoke run completes, all 3 artifacts on volume, wandb panels populated; audit returns ready.

---

## Phase D — Phase 2 judge function

**Creates:** Adds a second `@app.function` to `main/probes/group_a_rollout_judge.py`. **Does not modify** the Phase 1 function.

**Dependencies:** Phase C verified (smoke artifacts on volume).

### Build

Read first:
- `group_a_impl.md` § 4 Phase 2 steps 9–14, § 3 `phase2:` yaml block, Pre-flight locks (Phase 1/2 isolation, judge prompt/wrapper/parser rows)
- `05-24_probe_plan.md` — the Group A § "Phase 2 — judge cost" subsection (under "Groups → Group A")
- `main/judge/format.py` (Phase A output) — helpers to use
- The existing Phase 1 function — mirror its pattern for app instance, secrets, volume mounts

Add `run_phase2` `@app.function` to the same file:

- Same `gpu="H100"`, same volumes, same secrets, same image. New container = fresh CUDA state (the whole reason it's a separate function).
- Accepts `--config` and optional `--wandb-run-id` (passed from Phase 1 in pipeline mode; required when running standalone for resume).
- Reads `/vol/probes/05-24/group_a/phase1_done.json` — error clearly if missing. Reads `manifest.jsonl` and `phase1_rollouts.jsonl`.
- `wandb.init(id=wandb_run_id, resume="must")` — reattach to Phase 1's run, do not create a new one.
- Build vLLM `LLM` with `phase2` yaml block: `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len: 32768`, `gpu_memory_utilization: 0.88`, `max_num_seqs: 4`, `temperature: 0`, `max_tokens: 1024`. **Apply chat template** via the vLLM engine's tokenizer — do NOT instantiate a separate `AutoTokenizer`:
  ```python
  tokenizer = llm.get_tokenizer()
  prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
  ```
  Then pass `prompt_str` to `llm.generate(...)`. Using a separate `AutoTokenizer.from_pretrained` would double-download weights and waste cold-start time.
- Group rollouts by `problem_id`. For each prompt:
  - Load 8 completions.
  - `build_judge_messages(problem, rollouts)` → `(system, user)`.
  - Chat: `[{"role": "system", "content": system}, {"role": "user", "content": user}]`. Apply template.
  - Log `judge_input_tokens` (token count after template). If > `max_model_len`: log `truncated: true` and skip (do not chop or summarize — the rate IS the signal).
  - Generate. Log `wall_clock_s`, `output_tokens`, `vram_gb_used`.
  - Parse via `_assignment_from_poly_epo_payload`. On failure: log `json_parse_ok: false`, skip cluster metrics, **do not retry**.
  - On success: log `cluster_count`, `cluster_100_hits`, `json_parse_ok: true`, `$/call = wall_clock_s × modal_price_per_sec`.
- Per-prompt judge results → wandb Table.
- Final panels: wall-clock histogram, output-tokens histogram, judge VRAM, `$/call` histogram, cluster count distribution, `json_parse_ok` rate, `truncated` rate.
- Write `/vol/docs/probes/artifacts/05-24_group_a.pointer.json` (schema in `group_a_impl.md` § 4) so operator can `modal volume get` it.

**Extend the existing `run_full` `local_entrypoint`** (created in Phase C) to chain Phase 2 after Phase 1:
```python
@app.local_entrypoint()
def run_full(config: str):
    wandb_run_id = run_phase1.remote(config=config)
    run_phase2.remote(config=config, wandb_run_id=wandb_run_id)
```
This is the entrypoint the launcher already invokes (`::run_full` from Phase B). Standalone Phase 2 invocation for resume is via `modal run main/probes/group_a_rollout_judge.py::run_phase2 --config ... --wandb-run-id ...`.

Match Phase 1 style. Errors propagate. `logging`, not `print`.

Run Phase 2 alone against the existing smoke artifacts. Report wandb URL and pointer json path.

### Operator verification

```bash
modal run main/probes/group_a_rollout_judge.py::run_phase2 \
  --config main/configs/probe_a_05-24.yaml \
  --wandb-run-id <phase1-smoke-run-id>

modal volume get main-artifacts docs/probes/artifacts/05-24_group_a.pointer.json main/docs/probes/artifacts/
```

Judge Table in wandb (~8 rows in smoke), `$/call` panel, pointer json materializes locally.

### Audit

Read the new `run_phase2` function + pipeline entrypoint, `main/judge/format.py`, `group_a_impl.md` § 4 Phase 2 + § 3 phase2 yaml + Pre-flight locks (Phase 1/2 isolation, judge prompt source).

Check:
- Separate `@app.function` — must be its own decorator, not added to Phase 1's function. Critical for the engine-swap-VRAM fix.
- Reads `phase1_done.json` and errors if missing; does not silently re-run Phase 1.
- `wandb.init(id=..., resume="must")` — reattaches to Phase 1's run.
- Chat template applied via `llm.get_tokenizer().apply_chat_template(...)`, NOT a separate `AutoTokenizer.from_pretrained` (which would double model download).
- No per-rollout truncation — if total input > `max_model_len`, log `truncated` and skip; do not chop or summarize.
- JSON parse failure: single attempt, no retry loop.
- `build_judge_messages` and `_assignment_from_poly_epo_payload` imported from `main/judge/format.py` — not reimplemented inline.
- Pointer json schema matches § 4 (`modal_volume`, `path`, `wandb_run_id`, `created_at`); volume name `main-artifacts`.
- `run_full` is **extended** (not replaced) from Phase C's stub; chains `run_phase1.remote()` → `run_phase2.remote(wandb_run_id=...)`. Launcher path unchanged.
- `$/call` uses `modal_price_per_sec` from yaml.

Verdict + issues with `file:line`. Under 200 words.

### Exit criteria
Phase 2 standalone smoke run completes; pointer json materializes locally; audit returns ready.

---

## Full run (operator, not an agent)

After all 4 phases pass:

1. Set `smoke: false` in `main/configs/probe_a_05-24.yaml`.
2. Sanity-check budget: 200 prompts × 8 rollouts at ~2hr H100 ≈ $8.
3. `bash main/scripts/launch_probe_a.sh`
4. Monitor wandb. Kill if obviously broken (e.g., parse rate 0% after first batch).
5. On completion: pull pointer, update `main/docs/PLAN.md` §2/§5/§7 from readout panels per `group_a_impl.md` § 8.

---

## When an agent gets stuck

- **Do not let it invent.** Tell it to read the specific file/section and report what's ambiguous.
- **Common stuck points:** vLLM API shape (versions drift — point at the pilot's `rollout_engine.py`); Modal `add_local_dir` placement (point at `pre-milestone/pilot/infra/modal_app.py`); wandb `Table` API (link to wandb docs).
- **If genuinely ambiguous in the spec:** update `group_a_impl.md` to disambiguate, then re-run the build agent. Agents must not resolve ambiguity by guessing.

If an audit flags something: fix it, re-run the build agent for the affected file only if needed, re-run the audit. If audit and build disagree, the operator arbitrates.
