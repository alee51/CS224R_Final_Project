# main-verl/

VeRL-based reimplementation of the set-RL arms (via the **[maxrl repo](https://github.com/tajwarfahim/maxrl)** fork — not upstream `pip install verl`, and **not** the MaxRL training algorithm). Sibling to `main/` (the original custom trainer), not a replacement: `main/` stays as the archive that the paper draws from (diagnosis chart, prompt/parser/grader ablations, late-checkpoint eval pull commands, `timeline.md` history). New work happens here.

**Docs:** [`docs/STATUS.md`](docs/STATUS.md) · [`docs/human-remaining-work.md`](docs/human-remaining-work.md) · [`docs/verl_migration_plan.md`](docs/verl_migration_plan.md) (runbook) · [`docs/verl-reference.md`](docs/verl-reference.md) (reference)

## Why this exists

Decided after the 2026-05-28 TA discussion ([`../main/docs/verl_move_ta_meeting.md`](../main/docs/verl_move_ta_meeting.md)):

- **Coding-component bar.** TA: "to satisfy the coding component for the project you could implement set RL with minority voting objective on verl." Reimplementing on the maxrl repo’s VeRL fork is what the project gets graded on.
- **Drops the engineering we've been paying for.** Answer extraction, batching across GPUs (bs=128 splits natively), FA2 plumbing, weight sync — handled by the fork’s trainer loop.
- **Unlocks the experiments we actually want to run.** 4B base (Path C from `ta_discussion.md`), CoT clustering (Path D), async LLM judge — all easier on verl than on our trainer.

## Layout

| Dir | Purpose |
|---|---|
| `configs/` | Verl/Hydra configs (training runs, eval runs, model + data overrides). One config per launch. |
| `train/` | Custom objective code that plugs into verl: `minority_answer`, `poly_epo_answer`, CoT-clustering arm. Pure logic — verl owns the trainer loop. |
| `judge/` | Modal-hosted LLM-judge service (OpenAI-compatible API) + thin client for async batched calls from the trainer. Pattern: host once, semaphore-fan-out 32–128 calls in parallel. |
| `infra/` | Modal app + image: clone [tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl), `pip install -e .`, Ray + pin overrides for B200. |
| `scripts/` | Launch wrappers (`launch_train.sh`, `launch_eval.sh`) — thin shells around `python -m verl.trainer.main_ppo` with our config paths. |
| `probes/` | Smoke tests for the verl move: end-to-end 50-step run, judge bring-up, 4B fit check, weight-sync sanity. |
| `tests/` | Unit tests for the custom objectives — minority-vote scoring, CoT-cluster assignment. Mirrors `main/tests/test_objective_minority.py`. |
| `data/` | Manifest paths + any new filtered manifests (4B-calibrated rollout-pass filter). **Preprocess pipeline is reused from `main/data/`** (Polaris-51K, prompt filter) — no need to fork it. |

## What stays in `main/`

- `main/docs/` — paper writeup, TA discussions, plans, timeline, eval history
- `docs/` — migration runbook + VeRL reference
- `main/data/preprocess_polaris.py`, `main/data/prompt_heuristics.py`, the labeled Polaris manifests — reused as-is.
- The custom trainer (`main/train/`, `main/probes/`, `main/configs/`) — frozen, kept for paper provenance and any final ablation re-pulls.

## Status

Stage 1 bring-up complete (see [`docs/build/stage-01-log.md`](docs/build/stage-01-log.md)). Sprint: [`docs/STATUS.md`](docs/STATUS.md), [`docs/human-remaining-work.md`](docs/human-remaining-work.md).


## Bring-up

From repo root: `./main-verl/scripts/launch_hello_verl.sh` (sets `MODAL_PROFILE=chicken602` via `scripts/modal_bringup_env.sh`).

- GRPO smoke: `export CS224R_APP_NAME=cs224r-verl-stage02 && ./main-verl/scripts/launch_grpo_smoke.sh`
- minority_cot smoke (Stage 3a, mock clusters): `export CS224R_APP_NAME=cs224r-verl-stage03a && ./main-verl/scripts/launch_minority_cot_smoke.sh`
- poly_epo_cot smoke (Stage 5, mock clusters): `export CS224R_APP_NAME=cs224r-verl-stage05 && ./main-verl/scripts/launch_poly_epo_cot_smoke.sh`
- Judge trace smoke (1 step, verbose judge I/O for prompt 0): `./main-verl/scripts/launch_minority_cot_judge_trace.sh`
- Judge service (Stage 4): `export CS224R_APP_NAME=cs224r-verl-stage04-judge && ./main-verl/scripts/launch_judge_service.sh`
- Judge agreement smoke: `./main-verl/scripts/launch_judge_agreement.sh` (config: `configs/judge_agreement_smoke.yaml`)
- Judge latency smoke: `./main-verl/scripts/launch_judge_latency.sh` (config: `configs/judge_latency_smoke.yaml`)
- Serial parse diagnostic: `./main-verl/scripts/launch_judge_latency.sh judge_latency_smoke_serial`

## Next steps (rough order)

1. `infra/` — Modal image with maxrl repo + Ray; `hello_verl.py` smoke.
2. `configs/` — minimal **GRPO** config on Qwen3-1.7B-Base + Polaris (fork preprocess template); stable bring-up, not `main/` parity.
3. `train/` — port `minority_answer` objective; unit test against fixtures from `main/tests/test_objective_minority.py`.
4. `judge/` — Modal app exposing OpenAI-compatible `/v1/chat/completions`; async client with `asyncio.Semaphore`.
5. `configs/` (4B) — Qwen3-4B-Base fit check, then filtered-manifest retrain (Path C).
6. `train/` — CoT-clustering arm (Path D) once judge is up.
