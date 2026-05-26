# Engineering standards

Rules that apply to *all* code we write for the main experiment: probe scripts, training, eval, Modal launchers. Probes are not exempt.

**Drafted:** 05-24.

## Reproducibility

- Every script takes a yaml config. No hardcoded magic numbers. Configs live in `main/configs/`.
- **Config naming:** `<purpose>_<MM-DD>.yaml` — e.g. `probe_a_05-24.yaml`, `train_grpo_05-25.yaml`. The yaml is the source of truth for numeric knobs; docs reference keys, not duplicate values.
- **Seeds:** one `global_seed` in config, logged at run start. Set `random`, `numpy`, `torch`, and vLLM from it. For reproducible per-rollout seeds at temp=1, use a **deterministic formula** — `global_seed + problem_index * N_rollouts + rollout_idx` is the canonical choice. **Do NOT use Python's built-in `hash()`** — it's salted per process via `PYTHONHASHSEED` and produces different values across runs, silently breaking reproducibility. If you need a hash, use `hashlib.sha256` and convert to int. Poly-EPO does not report a seed; this matches VeRL/DAPO practice.
- At run start, log to wandb: full config dict, git SHA (`git rev-parse HEAD`), git dirty flag, Python version, and the pinned versions of vllm/torch/transformers/bitsandbytes.
- Dependencies pinned in `pyproject.toml`. No "latest" anything.

## Wandb (required for everything, including probes)

- Team: `224r-project` (`https://wandb.ai/224r-project`).
- Project: `cs224r-minority-voting`.
- Run name format: `<phase>_<operator>_<MM-DD-HHMM>` — e.g. `probe-A_nancy_05-24-0030`, `train-grpo_anastasia_05-25-1400`.
- Required tags on every run: `phase` (`probe` / `train` / `eval`), `operator`, `gpu_class`, `arm` (if applicable), `git_sha_short`.
- Minimum logged per run: everything in Reproducibility above + whatever the script's purpose calls for.

## Modal

- **Shared image.** One Python file (`main/infra/modal_image.py`) exports a single `image` object with all deps pinned. Every Modal script does `from infra.modal_image import image`. One place to bump vLLM, no version drift between probes and training.
- All jobs use `modal run --detach`. No interactive runs that lose state on disconnect.
- **Modal secrets:** standardized uppercase names — `HUGGINGFACE`, `WANDB_API_KEY`. Each teammate creates these on their own profile. The wandb API key inside auto-identifies the operator; no per-operator config field.
- Each person on personal Modal profile (per PLAN.md §6). Active profile chosen at launch via `modal profile activate <slug>` — not pinned in code.
- **Modal app name (REQUIRED, per-launch).** Pilot pinned one app name (`cs224r-pilot`) for every run, which collapses the Modal dashboard into one indistinguishable bucket. Fix: every script constructs its app name dynamically from an env var:
  ```python
  app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-untagged"))
  ```
  Launch script computes `CS224R_APP_NAME = f"cs224r-{purpose}-{phase}-{operator}-{MM-DD-HHMM}"` from yaml + clock before `modal run`. Example: `cs224r-probe-a-smoke-nancy-05-25-1830`, `cs224r-train-grpo-full-anastasia-05-27-0900`. Mirrors the wandb run name format above — one mental model, two dashboards.
- **Modal volumes.** Auto-created per workspace via `modal.Volume.from_name("<name>", create_if_missing=True)`. No manual creation step. Standard names:
  - `main-artifacts` — all run outputs (probes + training)
  - `hf-cache` — HF weights cache (shared with pilot convention)
  Mount at `/vol` and `/root/.cache/huggingface` respectively. Path layout on `main-artifacts`: `/vol/{probes,checkpoints,data}/<MM-DD>_<name>/`.

## Artifact persistence

Goal: **never lose more than an hour of work to a crashed job.**

### Streaming writes (all jobs)

Anything produced during a run (rollouts jsonl, metrics dumps, checkpoints) writes to a **mounted Modal volume**, not laptop disk. Wandb gets scalars/histograms; the volume gets bytes.

### Training checkpoints (jobs expected to run >1 hr)

- Checkpoint **at least every hour** wall-clock (every ~30 min is better).
- Each checkpoint directory includes: model weights, optimizer state, RNG state, step number, wandb run ID (resume must reattach to the same run).
- Checkpoint I/O must not block the train loop for long — async write or background flush.
- **Resume-from-checkpoint is mandatory and must be tested** before matrix launch.
- On completion: promote final checkpoint to HF Hub if the team needs cross-profile access (see routing table).

### Intermediate / phase checkpoints (probes and multi-phase jobs)

Full trainer checkpoints are **not** required for probes under ~1 hr. Still checkpoint **artifacts** at phase boundaries so a crash does not rerun expensive upstream work:

| Job type | What to flush mid-run | When |
| --- | --- | --- |
| Multi-phase probe (e.g. Group A) | Phase 1 rollouts jsonl + wandb run ID | Immediately after Phase 1 completes, before unloading vLLM |
| Long probe (>1 hr) | Same as above + partial metrics jsonl if applicable | Every ~30 min or after each logical batch |
| Training | Full checkpoint (weights + optim + RNG) | § Training checkpoints above |

A **phase artifact checkpoint** is not a model checkpoint — it is a durable copy on the volume (and optional wandb artifact for summaries) so Phase 2 or a resume can read inputs without redoing Phase 1.

Probes <1 hr: phase-boundary artifact flush is enough; wandb + volume final dump is enough if single-phase.

### Resume semantics

- **Training:** resume loads latest checkpoint from volume, continues same wandb run, same config.
- **Multi-phase probe:** if Phase 1 artifact exists on volume for this run ID, skip Phase 1 and start Phase 2 (config flag `resume_from_phase: 2` or auto-detect).

## Artifact routing

| Thing | Where | Notes |
| --- | --- | --- |
| Metrics (scalars, histograms) | Wandb | |
| Small summaries (pass-rate table, parse-rate, probe conclusions) | Wandb artifacts | |
| Big jsonl (rollouts), phase artifacts, training checkpoints | Modal volume | Never commit full rollouts to git |
| Git pointer to volume artifact | `main/docs/probes/artifacts/<MM-DD>_<name>.pointer.json` | `{ "modal_volume", "path", "wandb_run_id", "created_at" }` |
| Frozen datasets, final checkpoints shared across team | HF Hub | When another Modal profile needs the bytes |
| Code | Git | |

**Never local-only.** If it only exists on your laptop, it doesn't exist.

**HF Hub for probe jsonl:** optional, not default — use only when a teammate without volume access needs the file.

## Reward / answer parsing (train vs eval)

Decided scope (see `pre-milestone/nancy_explore/narrative/decisions.md`):

| Context | Method | Rationale |
| --- | --- | --- |
| **Training reward** (Polaris) | Rank-2 extract (`extract_rank2`, arm C) then **`grade_parsed_answer`** = mathd OR sympy (DeepScaleR/rLLM, `math_grade_deepscaler.py`) | Matches upstream rLLM `grade_answer_verl`; rescues strict/format false negatives |
| **Group A probe parse rate** | Same Rank-2 stack as training when rescoring; live 200-run `parse_ok` was Minerva-only headline | Use `parse_ok_rank2` for train expectations (~85–88% on hybrid) |
| **OOD eval** (HMMT, MATH-500, etc.) | **Math-Verify** (`math_verify` package) or equivalent | Integer match under-reports on LaTeX/symbolic gold; not for Polaris train reward |

**MathReward / RewardMATH:** those names refer to **reward-model benchmarks** (LLM-as-judge, PRM), not our train-time parser. Do not use for Polaris 0/1 reward unless we explicitly pivot to learned RMs.

**Prompt template (train, default 2026-05-25):** **`hybrid_answer_boxed`** (arm C) — see `HYBRID_ANSWER_BOXED_TEMPLATE` in `main/train/prompts.py` and [`probes/prompt_probe.md`](./probes/prompt_probe.md). Unvalidated recipe; fallbacks: `dapo_answer_v1` (DAPO-Math-17k verbatim), `verl_math_boxed` (VeRL MATH).

**Parser:** Rank-2 in `main/train/reward.py` (`extract_rank2`, `extract_path`). **Train correctness:** `grade_parsed_answer` = mathd OR sympy (`math_grade_deepscaler.py`, DeepScaleR/rLLM). Minerva strict remains diagnostic only. Research / escalation rules: [`probes/prompt_extraction_research.md`](./probes/prompt_extraction_research.md).

**Inference:** Qwen3-1.7B-Base has no HF chat template — plain string prompt to vLLM, not `apply_chat_template`.

## Code structure

- One file = one logical responsibility (matches `main/train/` layout in PLAN.md §5).
- Python `logging`, not `print`. Logger name = module name.
- Errors propagate. No silent `except: pass`. Wrap with context (`raise X from e`).
- Type hints on public functions encouraged, not required.
- No dead code, no commented-out blocks. Git history is the archive.

## Probe addendum

Probes are one-off but still follow everything above. Specifically:

- Probe scripts live in `main/probes/<topic>.py`.
- Configs live in `main/configs/probe_<topic>_<MM-DD>.yaml`.
- Large outputs → Modal volume; commit only a **pointer json** under `main/docs/probes/artifacts/`.
- Strategic plan (what/why/metrics) lives in `main/docs/probes/<MM-DD>_probe_plan.md`; numeric knobs live in the yaml, not duplicated in the plan.
- "It's just a probe" is not an excuse to skip wandb, configs, seeds, phase artifact flush, or the shared Modal image.

## Open

- Exact pinned versions for vllm / torch / transformers / bitsandbytes — set after Group B probes.
- HF Hub org/account for cross-team checkpoint sharing.
- Whether checkpoints push to HF Hub on completion automatically or only on demand.
- Async-checkpoint mechanism (background thread vs separate Modal function vs torch's built-in).
- GPU class default (H100 vs H200) — Group A logs throughput on whichever is available; lock after comparing $/throughput if it matters.
