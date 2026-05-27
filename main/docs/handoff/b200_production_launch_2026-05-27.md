# B200 production launch — 2026-05-27

Fresh **GRPO** and **minority_answer** full training on Modal workspace **alee72** (profile `anastasia`), isolated from the ongoing H200 GRPO line at `/vol/checkpoints/train_real/`.

---

## What we launched

| Arm | Modal app | W&B run | Checkpoint dir |
|-----|-----------|---------|----------------|
| GRPO (fresh) | `ap-VBmgTVFefkECyZa0r52RMb` | [t11jct0t](https://wandb.ai/224r-project/cs224r-minority-voting/runs/t11jct0t) | `/vol/checkpoints/train_real_b200/` |
| minority_answer (fresh) | `ap-3Acz8FrtQY4D4ubqkzJ4jB` | [o5ypkzja](https://wandb.ai/224r-project/cs224r-minority-voting/runs/o5ypkzja) | `/vol/checkpoints/train_minority_answer_b200/` |

Operator in yaml: **`nancy`** (W&B run names use merged config; Modal app name may show `unknown` if launcher reads overlay yaml without `extends`).

---

## Config stack (known-good B200 bring-up)

### Shared base: `main/configs/train_real.yaml`

| Setting | Value |
|---------|--------|
| `token_budget` | **130000** (B200 minority smoke `au96bwh1`; ~8% faster vs 105k, ~161 GB peak) |
| `gradient_checkpointing` | **true** |
| `checkpoint_every_steps` | **20** (+ hourly wall-clock backstop in trainer) |
| `batch_size` / `n_rollouts` | 64 / 8 |
| `total_steps` | 799 |
| `clustering.sympy_mode` | allowlist |
| `resume` | auto (overridden by `--no-resume` at launch) |
| `vllm_sleep` | **0** (default; do not enable for prod) |
| `rollout.gpu_memory_utilization` | 0.45 |
| `weight_sync.every_n_steps` | 1 |

### B200 overlay: `main/configs/train_real_b200.yaml`

```yaml
extends: train_real.yaml
gpu_class: B200
modal_price_per_sec: 0.001736
```

### Fresh checkpoint isolation (per arm)

**GRPO** — `main/configs/train_real_b200_fresh_grpo.yaml`:

```yaml
extends: train_real_b200.yaml
arm_profiles:
  grpo:
    train:
      checkpoint_dir: /vol/checkpoints/train_real_b200/
```

**minority_answer** — `main/configs/train_real_b200_fresh_minority.yaml`:

```yaml
extends: train_real_b200.yaml
arm_profiles:
  minority_answer:
    train:
      checkpoint_dir: /vol/checkpoints/train_minority_answer_b200/
```

H200 GRPO continues writing **`/vol/checkpoints/train_real/`** until stopped; B200 lines do not resume from those ckpts when launched with `--no-resume`.

---

## Launch commands (replay)

From repo root, Modal profile **anastasia** (alee72):

```bash
# Fresh B200 GRPO
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm grpo \
  --config main/configs/train_real_b200_fresh_grpo.yaml --no-resume --fresh-wandb

# Fresh B200 minority_answer
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm minority_answer \
  --config main/configs/train_real_b200_fresh_minority.yaml --no-resume --fresh-wandb
```

`launch_train.sh` uses `modal run --detach -q` so the CLI returns after submit (safe to chain two launches).

---

## Log streaming

```bash
main/.venv/bin/modal app logs ap-VBmgTVFefkECyZa0r52RMb -f      # GRPO
main/.venv/bin/modal app logs ap-3Acz8FrtQY4D4ubqkzJ4jB -f    # minority

main/.venv/bin/modal app logs ap-VBmgTVFefkECyZa0r52RMb --tail 200
```

---

## Monitoring

```bash
bash main/scripts/monitor_b200_prod.sh
```

Tracks W&B step, Modal app state, and log errors; writes state to `main/docs/probes/artifacts/b200_prod_monitor/state.json`.

---

## After B200 GRPO is healthy

Stop the H200 GRPO app so only one writer uses `/vol/checkpoints/train_real/` (B200 uses `train_real_b200/` regardless, but avoids double spend).

---

## Not used (failed efficiency track)

- `vllm_sleep=1` + `gradient_checkpointing=false` — OOM; see `main/docs/efficiency/B200_sleep_gc_off_give_up_2026-05-27.md`
- Default prod path: **sleep off, gc on, 130k token budget**

---

## Validation reference (pre-prod smokes)

| Run | W&B | Notes |
|-----|-----|--------|
| B200 minority 105k baseline | `wdl3fczm` | 10-step smoke |
| B200 minority 130k | `au96bwh1` | 10-step smoke |
| B200 GRPO smoke | `1hg8fs5u` | 10-step + resume ladder |
