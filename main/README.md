# Main experiment

Code and docs for the post-milestone minority-voting training run. Run Modal jobs from the **repo root** (`cs224r_finalproject/`), not from this folder.

## Where to find what

| Doc | Path | Read when you need… |
| --- | --- | --- |
| **Launch training** | [`docs/launch_training.md`](docs/launch_training.md) | **How to launch smokes and full runs** — copy-paste commands for agents (`--gpu-class`, `--arm`, no bad config pairings). |
| **Context** | [`docs/context.md`](docs/context.md) | Team facts, deadlines, budget, reading list. Start here for orientation. |
| **Plan** | [`docs/PLAN.md`](docs/PLAN.md) | What we're building and why: dataset, training arms, eval, ops, cost sizing. Updated from probe readouts. |
| **Standards** | [`docs/STANDARDS.md`](docs/STANDARDS.md) | Cross-cutting engineering rules: configs, seeds, wandb, Modal, artifacts. All code must follow this. |
| **Probe plan** | [`docs/probes/05-24_probe_plan.md`](docs/probes/05-24_probe_plan.md) | Why we're running Group A/B probes and what decisions they unlock. |
| **Probe impl** | [`docs/probes/group_a_impl.md`](docs/probes/group_a_impl.md) | Locked knobs, file list, artifact schemas, launch commands for Group A. |
| **Probe workflow** | [`docs/probes/group_a_workflow.md`](docs/probes/group_a_workflow.md) | Phased build/audit checklist (A→D). For agents or structured implementation. |

**Rule of thumb:** `context` → orientation · `PLAN` → experiment design · `STANDARDS` → how to write code · `probes/*` → pre-training validation runs.

### Code layout (high level)

```
main/
  train/          reward + prompts (shared with training later)
  judge/          Poly-EPO judge prompt + format helpers
  infra/          Modal image, volumes, hello verification
  probes/         Modal entrypoints (e.g. group_a_rollout_judge.py)
  configs/        yaml configs (source of truth for knobs)
  scripts/        launch wrappers
  tests/          local unit tests (no GPU)
```

GPU deps (torch, vllm, etc.) are pinned in `infra/modal_image.py` and run on Modal H100 — not in local `requirements.txt`.

---

## Quickstart

### 1. Local venv

From repo root:

```bash
cd main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v          # 7 tests, no GPU
```

### 2. Modal auth

```bash
modal setup               # or: pip install modal && modal token new
modal profile activate <your-slug>
```

### 3. Secrets (uppercase names required)

Create on your Modal profile (same tokens as pilot if you have them):

```bash
modal secret create HUGGINGFACE HF_TOKEN=<hf-token>
modal secret create WANDB_API_KEY WANDB_API_KEY=<wandb-key>
```

Verify: `modal secret list` should show `HUGGINGFACE` and `WANDB_API_KEY`.

Wandb team/project are fixed in configs: entity `224r-project`, project `cs224r-minority-voting`. Operator identity comes from the wandb key.

### 4. Infra smoke test (~$0.02, no GPU)

From repo root:

```bash
CS224R_APP_NAME=cs224r-hello-test modal run main/infra/hello_modal.py
```

Expect: `/vol listing: (empty)` and `HF cache path: /root/.cache/huggingface`. First run builds the shared image (~few min); later runs are fast.

### 5. Group A probe smoke (H100)

Config defaults to smoke mode (`smoke: true` in `configs/probe_a_05-24.yaml`):

```bash
bash main/scripts/launch_probe_a.sh
```

Monitor on [Modal dashboard](https://modal.com) and [wandb](https://wandb.ai/224r-project/cs224r-minority-voting). After completion:

```bash
modal volume ls main-artifacts probes/05-24/group_a/
```

Full run: set `smoke: false` in the yaml, then launch again (~2 hr H100).

### 6. Production training (B200)

See **[`docs/launch_training.md`](docs/launch_training.md)** for canonical commands. Example:

```bash
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm grpo
```
