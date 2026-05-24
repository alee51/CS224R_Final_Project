# Model & dataset setup (HuggingFace + Modal)

Required **before** any GPU run. Preflight blocks on placeholder JSONL or unset SHA256 hashes.

## 0. Activate the project venv (required for Modal CLI)

```bash
cd /path/to/cs224r_finalproject
source .venv/bin/activate
pip install -r pilot/requirements.txt
```

## 1. Local HuggingFace access

```bash
huggingface-cli login   # or: export HF_TOKEN=...
```

## 2. Materialize frozen JSONL slices

Pulls datasets and updates `pilot/preflight_lock.json` SHA256 fields:

```bash
PYTHONPATH=. python pilot/scripts/materialize_data_slices.py
```

| Local file | HuggingFace source | N |
|------------|-------------------|---|
| `dapo_slice_3k.jsonl` | `open-r1/DAPO-Math-17k-Processed` (config **`en`**) | 3000 |
| `aime25_eval_30.jsonl` | `MathArena/aime_2025` | 30 |
| `hmmt_nov25_eval_30.jsonl` | `MathArena/hmmt_nov_2025` | 30 |
| `math500_sanity_100.jsonl` | `HuggingFaceH4/MATH-500` | 100 (proportional level × subject, seed 42) |
| `beyond_aime_eval_100.jsonl` | `ByteDance-Seed/BeyondAIME` (test) | 100 (paper tier) |
| `hmmt_feb25_eval_30.jsonl` | `MathArena/hmmt_feb_2025` | 30 (paper tier) |
| `math500_eval_500.jsonl` | `HuggingFaceH4/MATH-500` | 500 (paper tier) |

If a MathArena repo 404s, check the exact dataset name on [huggingface.co/MathArena](https://huggingface.co/MathArena) (competition vs `*_outputs`).

## 3. Model weights — `Qwen/Qwen3-1.7B-Base`

**Local dev / smoke tests:** weights download on first `transformers` load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
```

**Modal (enabled in `modal_app.py`):** two volumes:

| Volume | Mount | Purpose |
|--------|-------|---------|
| `hf-cache` | `/root/.cache/huggingface` | Skip re-downloading ~3GB weights |
| `pilot-artifacts` | `/root/pilot/artifacts` | Persist run outputs; pulled locally after each job |

Manual pull if needed:

```bash
# Manual pull into a timestamped dir (modal_app.py does this automatically):
mkdir -p pilot/artifacts/run0_proxy/20260519T120000Z
modal volume get --force pilot-artifacts run0_proxy/ /tmp/modal-staging/
cp -r /tmp/modal-staging/run0_proxy/* pilot/artifacts/run0_proxy/20260519T120000Z/
```

Volume wiring lives in `pilot/infra/modal_app.py` (not `modal_launch.py`).

```python
# Reference — already applied in modal_app.py
@app.function(
    gpu="A100-80GB",
    volumes={
        "/root/pilot/artifacts": modal.Volume.from_name("pilot-artifacts", create_if_missing=True),
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_run(config: dict):
    import os
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    # load Qwen/Qwen3-1.7B-Base from cache ...
```

Create Modal secret:

```bash
modal secret create huggingface HF_TOKEN=hf_...
```

## 4. Verify preflight

```bash
PYTHONPATH=. python pilot/scripts/preflight_check.py
```

Must exit 0 (real JSONL + non-placeholder SHA256).

## 5. Launch order

1. Materialize data (step 2)  
2. Preflight (step 4)  
3. `python pilot/scripts/launch_run.py --run-id run0_proxy` (after trainer registers `train_fn`)  
4. Run1 / Run1b / Run2 / (conditional Run3)  
5. `python pilot/eval/gate.py`

## What Modal needs per run

| Asset | How |
|-------|-----|
| Base model | Volume cache or `from_pretrained` in container |
| Train JSONL | Bake into image, or `modal.Mount` of `pilot/data/` |
| Eval JSONL | Same mount; eval writes `raw_predictions.jsonl` with `eval_split` ∈ `{aime25_eval_30, hmmt_nov25_eval_30, math500_sanity_100}` |
| HF hub access | `HF_TOKEN` secret for gated models (if any) |

Paper-tier JSONL (`beyond_aime_eval_100`, etc.) are **not** used in pilot gate — only after `ESCALATE`.

## Local Mac vs Modal for the pilot

| Phase | Local M5 32GB | Modal GPU |
|-------|----------------|-----------|
| Data materialization | Yes | Optional |
| Run0 proxy (500 × 8 rollouts, 1.7B) | Possible but **very slow** via MLX/MPS; not wired in trainer yet | **Recommended** (~1 GPU-h) |
| Run1–3 (100 steps × 3k prompts, GRPO) | **Not reasonable** — no CUDA, unified memory bandwidth limits throughput; days of wall-clock | **Required** (~3×4 GPU-h) |

`Qwen3-1.7B-Base` fits in 32GB RAM for **inference**, but this pilot is on-policy RL with 8 rollouts/prompt and 100 training steps. The orchestrator plan assumes **Modal A100-80GB** for all GPU runs. Use the Mac for scripts, preflight, and eval post-processing only unless you explicitly port the trainer to MLX (out of pilot scope).
