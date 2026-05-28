# main-verl/

Verl-based reimplementation of the set-RL arms. Sibling to `main/` (the original custom trainer), not a replacement: `main/` stays as the archive that the paper draws from (diagnosis chart, prompt/parser/grader ablations, late-checkpoint eval pull commands, `timeline.md` history). New work happens here.

**Full VeRL planning guide:** [`docs/verl.md`](../docs/verl.md) (repo root).

## Why this exists

Decided after the 2026-05-28 TA discussion ([`../main/docs/verl_move_ta_meeting.md`](../main/docs/verl_move_ta_meeting.md)):

- **Coding-component bar.** TA: "to satisfy the coding component for the project you could implement set RL with minority voting objective on verl." Reimplementing on verl is what the project gets graded on.
- **Drops the engineering we've been paying for.** Answer extraction (verl ships `MathReward`), batching across GPUs (bs=128 splits natively), FA2 plumbing, weight sync — all handled.
- **Unlocks the experiments we actually want to run.** 4B base (Path C from `ta_discussion.md`), CoT clustering (Path D), async LLM judge — all easier on verl than on our trainer.

## Layout

| Dir | Purpose |
|---|---|
| `configs/` | Verl/Hydra configs (training runs, eval runs, model + data overrides). One config per launch. |
| `train/` | Custom objective code that plugs into verl: `minority_answer`, `poly_epo_answer`, CoT-clustering arm. Pure logic — verl owns the trainer loop. |
| `judge/` | Modal-hosted LLM-judge service (OpenAI-compatible API) + thin client for async batched calls from the trainer. Pattern: host once, semaphore-fan-out 32–128 calls in parallel. |
| `infra/` | Modal app + image definitions (verl + flash-attn + vllm pins), GPU class selection (B200 default; 4B fit check). |
| `scripts/` | Launch wrappers (`launch_train.sh`, `launch_eval.sh`) — thin shells around `python -m verl.trainer.main_ppo` with our config paths. |
| `probes/` | Smoke tests for the verl move: end-to-end 50-step run, judge bring-up, 4B fit check, weight-sync sanity. |
| `tests/` | Unit tests for the custom objectives — minority-vote scoring, CoT-cluster assignment. Mirrors `main/tests/test_objective_minority.py`. |
| `data/` | Manifest paths + any new filtered manifests (4B-calibrated rollout-pass filter). **Preprocess pipeline is reused from `main/data/`** (Polaris-51K, prompt filter) — no need to fork it. |

## What stays in `main/`

- `main/docs/` — paper writeup, TA discussions, plans, timeline, eval history
- `docs/verl.md` — VeRL migration guide (repo root; spans both codebases)
- `main/data/preprocess_polaris.py`, `main/data/prompt_heuristics.py`, the labeled Polaris manifests — reused as-is.
- The custom trainer (`main/train/`, `main/probes/`, `main/configs/`) — frozen, kept for paper provenance and any final ablation re-pulls.

## Status

Skeleton only — directories created 2026-05-28, no code yet.

## Next steps (rough order)

1. `infra/` — Modal image with verl + pins; `hello_verl.py` smoke.
2. `configs/` — minimal GRPO config on Qwen3-1.7B-Base + Polaris-51K to confirm parity with the `main/` baseline before adding arms.
3. `train/` — port `minority_answer` objective; unit test against fixtures from `main/tests/test_objective_minority.py`.
4. `judge/` — Modal app exposing OpenAI-compatible `/v1/chat/completions`; async client with `asyncio.Semaphore`.
5. `configs/` (4B) — Qwen3-4B-Base fit check, then filtered-manifest retrain (Path C).
6. `train/` — CoT-clustering arm (Path D) once judge is up.
