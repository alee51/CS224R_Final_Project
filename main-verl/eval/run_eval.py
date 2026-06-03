"""Eval 4B verl-trained ckpt on math benchmarks (AIME-25, AIME-26, MATH-500, HMMT, BeyondAIME, etc.).

Steps inside container:
  1. Merge verl FSDP shards at CKPT_DIR/actor → HF format at /tmp/merged_hf.
     (Skipped when CS224R_EVAL_BASE=1; loads Qwen/Qwen3-4B-Base from HF directly.)
  2. Load merged HF (or HF base) with vLLM on B200:1 (enforce_eager=True).
  3. For each requested dataset, sample N rollouts/prompt at temp=1.0, optionally
     with top-N logprobs per token if CS224R_EVAL_LOGPROBS>0.
  4. Grade with verl's math reward (matches training grader, mathd∨sympy fallback).
  5. Compute pass@k for k ∈ {1, 2, 4, 8, 16, 32, 64} (skipping k > n_rollouts).
  6. Write JSON result to /vol/probes/eval_4b/<arm>_<step>_<datasets>.json.

Env vars (set by launch shell):
  - CS224R_EVAL_CKPT_PATH   absolute path inside container, e.g.
      /vol/checkpoints/main-verl/grpo_train_4b_1epoch_lr3e6/global_step_400/actor
      (informational only when CS224R_EVAL_BASE=1)
  - CS224R_EVAL_LABEL       short label, e.g. "grpo_step400"
  - CS224R_EVAL_DATASETS    comma list: aime25,aime26,hmmt_feb25,hmmt_nov25,beyondaime,math500
  - CS224R_EVAL_N_ROLLOUTS  default 16 (production: 64)
  - CS224R_EVAL_LOGPROBS    default 0; if > 0 save top-N logprobs per token in per_prompt[i].logprobs
  - CS224R_EVAL_BASE        default 0; if =1 skip FSDP merge and load Qwen/Qwen3-4B-Base from HF
  - CS224R_EVAL_OUTPUT_DIR  default /vol/probes/eval_4b
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Probe lives at main-verl/eval/run_eval.py; parents[1] = main-verl/
_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_VERL_ROOT))

import modal

from infra.modal_image import app_name, image as _base_image
from infra.modal_volume import (
    ARTIFACTS_MOUNT,
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
)

_CKPT_PATH = os.environ.get("CS224R_EVAL_CKPT_PATH", "").strip()
_BASE_MODE = int(os.environ.get("CS224R_EVAL_BASE", "0"))
_DEFAULT_LABEL = "base_step400" if _BASE_MODE else "eval"
_LABEL = os.environ.get("CS224R_EVAL_LABEL", _DEFAULT_LABEL).strip()
_DATASETS = os.environ.get("CS224R_EVAL_DATASETS", "aime25").strip()
_N_ROLLOUTS = int(os.environ.get("CS224R_EVAL_N_ROLLOUTS", "16"))
_LOGPROBS = int(os.environ.get("CS224R_EVAL_LOGPROBS", "0"))
_OUTPUT_DIR = os.environ.get("CS224R_EVAL_OUTPUT_DIR", "/vol/probes/eval_4b").strip()
# Default to B200:1 for eval — 4B fits easily on a single B200 (180GB) and
# TP=1 avoids NCCL/multiproc overhead. Override via env var if needed.
_GPU_COUNT = int(os.environ.get("CS224R_EVAL_GPU_COUNT", "1"))

_RUNTIME_SECRET = modal.Secret.from_dict({
    "CS224R_EVAL_CKPT_PATH": _CKPT_PATH,
    "CS224R_EVAL_LABEL": _LABEL,
    "CS224R_EVAL_DATASETS": _DATASETS,
    "CS224R_EVAL_N_ROLLOUTS": str(_N_ROLLOUTS),
    "CS224R_EVAL_LOGPROBS": str(_LOGPROBS),
    "CS224R_EVAL_BASE": str(_BASE_MODE),
    "CS224R_EVAL_OUTPUT_DIR": _OUTPUT_DIR,
    "CS224R_EVAL_GPU_COUNT": str(_GPU_COUNT),
})

app = modal.App(app_name())

image = _base_image.add_local_dir(
    str(_MAIN_VERL_ROOT / "data"),
    remote_path="/root/main-verl/data",
    ignore=["*.parquet.tmp", "__pycache__", ".DS_Store"],
)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=f"B200:{_GPU_COUNT}",
    timeout=3 * 3600,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
        _RUNTIME_SECRET,
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def eval_4b() -> None:
    import json
    import time
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch

    ckpt_path = os.environ["CS224R_EVAL_CKPT_PATH"]
    label = os.environ["CS224R_EVAL_LABEL"]
    datasets = os.environ["CS224R_EVAL_DATASETS"].split(",")
    n_rollouts = int(os.environ["CS224R_EVAL_N_ROLLOUTS"])
    logprobs_topn = int(os.environ.get("CS224R_EVAL_LOGPROBS", "0"))
    base_mode = int(os.environ.get("CS224R_EVAL_BASE", "0")) == 1
    output_dir = Path(os.environ["CS224R_EVAL_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_4b] label={label} ckpt={ckpt_path} datasets={datasets} "
          f"n={n_rollouts} logprobs={logprobs_topn} base_mode={base_mode}")
    print(f"[eval_4b] cuda: device_count={torch.cuda.device_count()}")

    # ---- 1. Resolve model dir (merge FSDP -> HF, or use HF base directly) ----
    if base_mode:
        # Base arm: skip merge, load Qwen3-4B-Base directly from HF.
        model_id = "Qwen/Qwen3-4B-Base"
        print(f"[eval_4b] BASE mode: loading {model_id} from HF (CKPT_PATH ignored: {ckpt_path!r})")
    else:
        merged_dir = Path("/tmp/merged_hf")
        if merged_dir.exists():
            import shutil
            shutil.rmtree(merged_dir)
        merged_dir.mkdir(parents=True)

        t0 = time.time()
        print(f"[eval_4b] merging FSDP shards from {ckpt_path} -> {merged_dir}")
        merge_cmd = [
            sys.executable,
            "/root/maxrl/scripts/model_merger.py",
            "merge",
            "--backend", "fsdp",
            "--local_dir", ckpt_path,
            "--target_dir", str(merged_dir),
        ]
        subprocess.run(merge_cmd, check=True)
        print(f"[eval_4b] merge done in {time.time() - t0:.1f}s")
        print(f"[eval_4b] merged files: {sorted(p.name for p in merged_dir.iterdir())[:10]}")
        model_id = str(merged_dir)

    # ---- 2. Load vLLM ----
    print(f"[eval_4b] loading vLLM model={model_id}")
    from vllm import LLM, SamplingParams

    gpu_count = int(os.environ.get("CS224R_EVAL_GPU_COUNT", "1"))
    llm = LLM(
        model=model_id,
        tensor_parallel_size=gpu_count,
        gpu_memory_utilization=0.95,         # bumped from 0.85; single vLLM proc per app, nothing else on GPU
        max_model_len=5120,
        enforce_eager=True,                   # B200 requirement (memory: project_b200_eager_required)
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=4096,                    # was vLLM default 256 → GPU idle waiting for batch fills.
                                              # 4096 is a soft ceiling; vLLM auto-caps to whatever the KV cache fits.
        max_num_batched_tokens=32768,         # was default 2048 → tiny batches at decode. 32K matches B200 compute throughput.
    )

    sampling_kwargs = dict(
        n=n_rollouts,
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
    )
    if logprobs_topn > 0:
        sampling_kwargs["logprobs"] = logprobs_topn
    sampling_params = SamplingParams(**sampling_kwargs)

    # ---- 3. Dataset registry ----
    DATASET_PATHS = {
        "aime25": "/root/main-verl/data/aime_val.parquet",
        "aime26": "/root/main-verl/data/aime26.parquet",
        "polaris_val": "/root/main-verl/data/polaris_val.parquet",
        "math500": "/root/main-verl/data/math500.parquet",
        "hmmt_feb25": "/root/main-verl/data/hmmt_feb25.parquet",
        "hmmt_nov25": "/root/main-verl/data/hmmt_nov25.parquet",
        "beyondaime": "/root/main-verl/data/beyondaime.parquet",
        "dapo_slice_3k": "/root/main-verl/data/dapo_slice_3k.parquet",
    }

    results = {
        "label": label,
        "ckpt_path": ckpt_path,
        "n_rollouts": n_rollouts,
        "datasets": {},
    }

    for ds_name in datasets:
        if ds_name not in DATASET_PATHS:
            print(f"[eval_4b] WARN dataset {ds_name} not in registry, skipping")
            continue
        ds_path = DATASET_PATHS[ds_name]
        print(f"[eval_4b] === dataset={ds_name} path={ds_path} ===")
        df = pd.read_parquet(ds_path)
        print(f"[eval_4b] {ds_name}: {len(df)} prompts")

        # Build chat-template prompts; verl parquets have prompt = [{role, content}, ...]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        rendered = []
        gts = []
        problem_ids = []
        for i, row in df.iterrows():
            chat = list(row["prompt"])
            chat = [{"role": m["role"], "content": m["content"]} for m in chat]
            txt = tok.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            rendered.append(txt)
            gts.append(row["reward_model"]["ground_truth"])
            ei = row["extra_info"] if isinstance(row["extra_info"], dict) else {}
            problem_ids.append(ei.get("problem_id", f"{ds_name}_{i}"))

        # Generate
        gen_t0 = time.time()
        outs = llm.generate(rendered, sampling_params)
        print(f"[eval_4b] {ds_name}: generated in {time.time() - gen_t0:.1f}s")

        # Score: use verl's math.compute_score — same path the training reward router
        # uses for data_source=polaris / aime / math_dapo. Hendrycks is_equiv handles
        # latex normalization (\\frac{1}{2} vs 0.5, integer formats, etc.). math_dapo
        # with strict_box_verify is exact-string match — strictly stricter and biases
        # pass@k low by 5–10 pp on polaris/math500.
        from verl.utils.reward_score.math import (
            compute_score as math_compute_score,
            last_boxed_only_string,
            remove_boxed,
        )

        per_prompt = []
        for prompt_id, gt, out, rendered_prompt in zip(problem_ids, gts, outs, rendered):
            rollouts = [comp.text for comp in out.outputs]
            rewards = []
            preds = []
            for r in rollouts:
                try:
                    rewards.append(float(math_compute_score(r, str(gt))))
                    # also extract the boxed answer for offline diversity analysis
                    boxed = last_boxed_only_string(r)
                    preds.append(remove_boxed(boxed) if boxed is not None else "")
                except Exception:
                    rewards.append(0.0)
                    preds.append("")

            # Optional: serialize per-token top-N logprobs.
            # vLLM yields comp.logprobs as a list (one element per generated token),
            # each element a dict {token_id: Logprob(logprob=..., decoded_token=..., rank=...)}.
            # We store only {token_id: logprob} to keep JSON small (we just need the
            # distribution for entropy / KL — decoded_token and rank are recoverable
            # from the tokenizer + sort order).
            rollout_logprobs = None
            if logprobs_topn > 0:
                rollout_logprobs = []
                for comp in out.outputs:
                    token_logprobs = []
                    if comp.logprobs is not None:
                        for step in comp.logprobs:
                            if step is None:
                                token_logprobs.append({})
                                continue
                            # step is a dict {int_token_id: Logprob}
                            entry = {}
                            for tok_id, lp in step.items():
                                # lp may be a vllm Logprob object or a float; handle both.
                                lp_val = getattr(lp, "logprob", lp)
                                try:
                                    entry[int(tok_id)] = float(lp_val)
                                except (TypeError, ValueError):
                                    continue
                            token_logprobs.append(entry)
                    rollout_logprobs.append(token_logprobs)

            entry = {
                "problem_id": prompt_id,
                "ground_truth": str(gt),
                # Save the chat-templated prompt verbatim. Phase 3 (kl_from_base)
                # needs this to teacher-force the base model with the SAME left
                # context the policy saw — otherwise base distributions are
                # conditioned on the wrong tokens and the KL is meaningless.
                "rendered_prompt": rendered_prompt,
                "n_correct": int(sum(1 for r in rewards if r > 0.5)),
                "rewards": rewards,
                "preds": preds,
                "rollouts": rollouts,  # keep for downstream analysis
            }
            if rollout_logprobs is not None:
                entry["logprobs"] = rollout_logprobs
            per_prompt.append(entry)

        # Pass@k metrics — extended ladder per eval.md (cap k=64).
        n = n_rollouts
        K_VALUES = [1, 2, 4, 8, 16, 32, 64]
        passk = {}
        for k in K_VALUES:
            if k > n:
                continue
            vals = []
            for p in per_prompt:
                c = p["n_correct"]
                # Unbiased pass@k = 1 - C(n-c,k)/C(n,k)
                from math import comb
                if c == 0:
                    vals.append(0.0)
                else:
                    vals.append(1.0 - comb(n - c, k) / comb(n, k))
            passk[f"pass@{k}"] = float(np.mean(vals))

        # Majority@k for k in {1,4,8,16}: take k random rollouts, pick mode of non-empty parsed_answer-equivalent, score 1 if matches gold.
        # Simplification: use string match on a normalized rollout extract... skip for v1.

        ds_result = {
            "n_prompts": len(per_prompt),
            "pass_at_k": passk,
            "mean_reward_at_1": float(np.mean([p["rewards"][0] for p in per_prompt])),
            "per_prompt": per_prompt,
        }
        results["datasets"][ds_name] = ds_result
        # Incremental write: dump per-dataset JSON immediately so we don't lose
        # results if a later dataset gets cancelled mid-run.
        per_ds_path = output_dir / f"{label}_{ds_name}.json"
        per_ds_payload = {
            "label": label,
            "ckpt_path": ckpt_path,
            "n_rollouts": n_rollouts,
            "datasets": {ds_name: ds_result},
        }
        with per_ds_path.open("w") as f:
            json.dump(per_ds_payload, f, indent=2)
        artifacts_volume.commit()
        print(f"[eval_4b] wrote per-dataset {per_ds_path}")
        # Only print pass@k for k values that actually exist in the dict (i.e.,
        # k <= n_rollouts). Schema probe at n=8 originally surfaced bogus
        # "pass@16=0.000" / "pass@32=0.000" because .get(..., 0) defaulted
        # missing keys to 0.0 — and downstream readers could mistake that for
        # a real zero pass rate.
        _passk_summary = " ".join(f"{k}={v:.3f}" for k, v in passk.items())
        print(f"[eval_4b] {ds_name} done: {_passk_summary}")

    # ---- 4. Write JSON ----
    output_path = output_dir / f"{label}_{'-'.join(datasets)}.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval_4b] wrote {output_path}")
    artifacts_volume.commit()


@app.local_entrypoint()
def main() -> None:
    if not _BASE_MODE and not _CKPT_PATH:
        raise SystemExit("CS224R_EVAL_CKPT_PATH required (or set CS224R_EVAL_BASE=1)")
    mode = "BASE" if _BASE_MODE else "TRAINED"
    print(f"[launch] mode={mode} label={_LABEL} ckpt={_CKPT_PATH} "
          f"datasets={_DATASETS} n={_N_ROLLOUTS} logprobs={_LOGPROBS}")
    # .spawn() is fire-and-forget: the remote function call gets enqueued and
    # the local entrypoint returns immediately. With `modal run --detach` the
    # remote job survives any local-side disconnect (wifi drop, laptop close,
    # `modal run` ctrl-C). Use .remote() only when you want the local caller
    # to block on the result.
    call = eval_4b.spawn()
    print(f"[launch] spawned eval_4b call_id={call.object_id}; "
          f"output JSON will land on the artifacts volume at "
          f"{_OUTPUT_DIR}/{_LABEL}_<dataset>.json")
