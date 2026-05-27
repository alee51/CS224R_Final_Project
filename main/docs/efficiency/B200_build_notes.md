# B200 build notes (practical runbook)

Last updated: 2026-05-26  
Branch: `b200-bringup`

This runbook is the practical, low-risk bring-up sequence for B200 migration work that is safe to do without launching long production runs.

## Scope and guardrails

- Keep training semantics unchanged (no objective/loss/`n_kept` policy changes).
- Do not retune performance knobs in this pass (`token_budget`, `gradient_checkpointing`, `gpu_memory_utilization`).
- Keep H200 default launch path working.
- Use B200 only when explicitly requested via `--gpu-class b200` or B200 config files.

## What is wired now

- Train launcher chooses Modal function by SKU:
  - `--gpu-class h200` -> `train_remote_h200`
  - `--gpu-class b200` -> `train_remote_b200`
- `main/configs/train_real_b200.yaml` exists as an overlay:
  - `extends: configs/train_real.yaml`
  - `gpu_class: B200`
  - `modal_price_per_sec: 0.001736`
- `trainer.load_cfg()` now supports recursive `extends` merges.
- Smoke probes/launchers support explicit SKU selection:
  - `launch_smoke_flash_attn.sh --gpu-class h200|b200`
  - `launch_smoke_vllm_generate.sh --gpu-class h200|b200`
  - `launch_smoke_weight_sync.sh --gpu-class h200|b200`
- Checkpoint eval launcher reads `gpu_class` from the chosen config; B200 eval configs now use B200 pricing.

## Exact B200 smoke pipeline (recommended order)

From repo root:

1) vLLM boot smoke (minimal generate)

```bash
bash main/scripts/launch_smoke_vllm_generate.sh --gpu-class b200
```

2) FlashAttention smoke (import/load/forward/collocated)

```bash
bash main/scripts/launch_smoke_flash_attn.sh --gpu-class b200
# optional exhaustive run:
# bash main/scripts/launch_smoke_flash_attn.sh --gpu-class b200 --all
```

3) HF -> vLLM weight sync smoke

```bash
bash main/scripts/launch_smoke_weight_sync.sh --gpu-class b200
```

4) 10-step train smoke on set arm (no long production run)

```bash
bash main/scripts/launch_train.sh --mode smoke --gpu-class b200 --arm minority_answer
```

5) Checkpoint/resume smoke (fresh 10-step then resume-to-11)

```bash
# phase 1: create checkpoint at step 9
bash main/scripts/launch_smoke_ckpt_resume.sh --arm minority_answer --gpu-class b200 --phase fresh

# phase 2: resume from latest checkpoint to at least step 10
bash main/scripts/launch_smoke_ckpt_resume.sh --arm minority_answer --gpu-class b200 --phase resume
```

## Optional B200 checkpoint eval smoke

```bash
bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_2k_polaris_aime_b200.yaml --detach
```

## Rollback instructions (fast path back to H200)

Use these if any B200 smoke gate fails.

1) Train path rollback (default H200):

```bash
bash main/scripts/launch_train.sh --mode smoke --arm minority_answer
# or full:
# bash main/scripts/launch_train.sh --mode full --arm minority_answer
```

2) Explicitly force H200 if needed:

```bash
bash main/scripts/launch_train.sh --mode smoke --gpu-class h200 --arm minority_answer --config main/configs/train_real.yaml
```

3) Smoke rollback commands:

```bash
bash main/scripts/launch_smoke_vllm_generate.sh --gpu-class h200
bash main/scripts/launch_smoke_flash_attn.sh --gpu-class h200
bash main/scripts/launch_smoke_weight_sync.sh --gpu-class h200
```

4) Checkpoint eval rollback config:

```bash
bash main/scripts/launch_checkpoint_eval.sh --config main/configs/checkpoint_eval_2k.yaml
```

## Notes for operators

- `train_real_b200.yaml` only changes SKU/pricing metadata; all core train/rollout defaults come from `train_real.yaml`.
- If `train_real_b200.yaml` is missing in another branch, pass `--config main/configs/train_real.yaml --gpu-class b200` explicitly.
- For cost reporting, ensure eval/train configs with `gpu_class: B200` also use `modal_price_per_sec: 0.001736`.
