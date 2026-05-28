# Main experiment

Code and docs for the post-milestone minority-voting training run. Run Modal jobs from the **repo root** (`cs224r_finalproject/`), not from this folder.

## Where to find what

| Doc | Read when you need… |
| --- | --- |
| [`docs/context.md`](docs/context.md) | Orientation: team, budget, deadlines, reading list. **Start here.** |
| [`docs/PLAN.md`](docs/PLAN.md) | Strategy: dataset, arms, eval, ops, cost sizing. |
| [`docs/timeline.md`](docs/timeline.md) | Chronology: what we tried, in order. |
| [`docs/decisions.md`](docs/decisions.md) | Indexed log of locked decisions. |
| [`docs/STANDARDS.md`](docs/STANDARDS.md) | Engineering rules: seeds, wandb, Modal, artifacts. |
| [`docs/launch_training.md`](docs/launch_training.md) | How to launch smokes / full runs. Pair with `docs/handoff/`. |
| [`docs/handoff/`](docs/handoff/) | Active production configs + relaunch / resume commands. |
| [`docs/monitoring/`](docs/monitoring/) | W&B dashboard quickstart + full diagnostics. |
| [`docs/build_spec/`](docs/build_spec/) | Trainer arch + per-arm implementation specs (minority, poly-EPO, clustering). |
| [`docs/efficiency/`](docs/efficiency/) | B200 tuning levers (token_budget, sleep, gc); operational notes. |
| [`docs/probes/`](docs/probes/) | Frozen probe findings — prompt arm C, mathd∨sympy grader, Polaris-vs-DAPO. Cite from paper. |
| [`docs/paper/`](docs/paper/) | Paper-bound `method.md` + `results/` (final numbers only). |
| [`docs/ta_discussion.md`](docs/ta_discussion.md) | Office-hours agenda: status, learnings, open questions. |
| [`docs/reference/`](docs/reference/) | Completed audits and superseded designs. Consult, don't edit. |

**Rule of thumb:** `context` → orientation · `PLAN` → strategy · `timeline` → history · `STANDARDS` → engineering rules · `handoff` → launch ops · `paper/` → writeup.

### Code layout (high level)

```
main/
  train/          GRPO trainer, objective, loss, reward, clustering
  judge/          Poly-EPO judge prompt + format helpers
  infra/          Modal image, volumes, hello verification
  probes/         Modal probe entrypoints (rollout eval, smokes)
  configs/        yaml configs (source of truth for knobs); _archive/ for stale
  scripts/        launch wrappers
  tests/          local unit tests (no GPU)
```

GPU deps (torch, vllm, etc.) are pinned in `infra/modal_image.py` and run on Modal B200 (default) or H200 — not in local `requirements.txt`.

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

### 5. Production training (B200)

See **[`docs/launch_training.md`](docs/launch_training.md)** for canonical commands. Example:

```bash
bash main/scripts/launch_train.sh --mode full --gpu-class b200 --arm grpo
```

Monitor on [Modal dashboard](https://modal.com) and [wandb](https://wandb.ai/224r-project/cs224r-minority-voting).
