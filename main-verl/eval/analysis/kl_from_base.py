"""Per-token KL(π_arm ‖ π_base) on saved policy rollouts (Phase 3).

For each (trained arm × dataset) JSON we have:
  - per_prompt[i].rendered_prompt — chat-templated prompt the policy was conditioned on
  - per_prompt[i].rollouts        — generation text from the trained policy
  - per_prompt[i].logprobs        — policy's saved top-K (token_id → logprob)

This script loads Qwen3-4B-Base via vLLM with `logprobs=20`, teacher-forces
the base model with `rendered_prompt + rollout_text` (prompt_logprobs=K),
slices the base distributions for the rollout positions, and computes
per-token KL(p_policy_topK ‖ p_base_topK).

### Alignment math (do NOT change without re-deriving)

Policy: `comp.logprobs[t]` is the distribution that generated the t-th rollout
token, given (rendered_prompt + rollout_tokens[0:t]). One entry per generated
token; index 0 is the dist for the first generated token. The policy JSON
serializes this verbatim as `per_prompt[i].logprobs[r][t]`.

Base (teacher-forced): vLLM `prompt_logprobs` is one entry per INPUT token.
At input position p, the entry is the distribution PREDICTING input token p,
conditioned on input tokens [0, p). Index 0 is None (no left context).

Therefore for the base to match the policy's conditioning we must feed it
`rendered_prompt + rollout_text` and SLICE the rollout positions. If
P = len(tokenize(rendered_prompt)) (no special tokens, since the prompt is
already chat-templated text), then `base_prompt_lp[P + t]` is the base's
distribution over rollout token t given (rendered_prompt + rollout_tokens[0:t])
— exactly aligned with `pol_steps[t]`.

We renormalize each distribution over the union of token_ids that appear in
either side's top-K dict so the divergence is well-defined.

Skipped arms: base (KL(base ‖ base) = 0).

Output per (arm, dataset): a JSON dump at
    /vol/probes/kl/<arm>_<dataset>.json
with per-prompt token-position KL curves and aggregate statistics. Plus a
single summary markdown at main-verl/writeup/results/kl_summary.md.

This module exposes a Modal app — run with:
    modal run main-verl/eval/analysis/kl_from_base.py

Env vars (set by launcher):
  - CS224R_KL_INPUT_GLOB     glob of saved trained-arm eval JSONs to process
                             default: /vol/probes/eval_4b/*_step400_*.json
  - CS224R_KL_OUTPUT_DIR     default /vol/probes/kl
  - CS224R_KL_MAX_PROMPTS    cap problems per (arm,dataset); 0 = no cap
  - CS224R_KL_MAX_ROLLOUTS   cap rollouts per problem; default 8
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# kl_from_base.py lives at main-verl/eval/analysis/kl_from_base.py;
# parents[2] = main-verl/
_MAIN_VERL_ROOT = Path(__file__).resolve().parents[2]
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

_INPUT_GLOB = os.environ.get("CS224R_KL_INPUT_GLOB",
                             "/vol/probes/eval_4b/*_step400_*.json").strip()
_OUTPUT_DIR = os.environ.get("CS224R_KL_OUTPUT_DIR", "/vol/probes/kl").strip()
_MAX_PROMPTS = int(os.environ.get("CS224R_KL_MAX_PROMPTS", "0"))
_MAX_ROLLOUTS = int(os.environ.get("CS224R_KL_MAX_ROLLOUTS", "8"))
_TOPN = int(os.environ.get("CS224R_KL_LOGPROBS", "20"))

_RUNTIME_SECRET = modal.Secret.from_dict({
    "CS224R_KL_INPUT_GLOB": _INPUT_GLOB,
    "CS224R_KL_OUTPUT_DIR": _OUTPUT_DIR,
    "CS224R_KL_MAX_PROMPTS": str(_MAX_PROMPTS),
    "CS224R_KL_MAX_ROLLOUTS": str(_MAX_ROLLOUTS),
    "CS224R_KL_LOGPROBS": str(_TOPN),
})

app = modal.App(app_name())
image = _base_image
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def arm_from_label(label: str) -> str:
    if "_step" in label:
        return label.rsplit("_step", 1)[0]
    return label


def per_token_kl(policy_topk: dict, base_topk: dict) -> float:
    """KL(policy ‖ base) using each model's top-K dict.

    Renormalize both over the union of token_ids; tokens missing from a side
    get a tiny mass so we never divide by 0. KL is in bits.
    """
    import math
    if not policy_topk or not base_topk:
        return float("nan")
    keys = set(policy_topk.keys()) | set(base_topk.keys())
    eps = 1e-12

    def renorm(d: dict) -> dict[int, float]:
        masses = {}
        for k in keys:
            lp = d.get(k)
            masses[k] = math.exp(lp) if lp is not None else eps
        s = sum(masses.values())
        if s <= 0:
            return {k: 1.0 / len(keys) for k in keys}
        return {k: v / s for k, v in masses.items()}

    p = renorm(policy_topk)
    b = renorm(base_topk)
    kl = 0.0
    for k in keys:
        pk = p[k]
        bk = b[k]
        if pk > 0 and bk > 0:
            kl += pk * (math.log2(pk) - math.log2(bk))
    return kl


@app.function(
    image=image,
    gpu="B200:1",
    timeout=4 * 3600,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        _RUNTIME_SECRET,
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def kl_pass() -> None:
    import glob
    import json
    import time

    import numpy as np
    import torch

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    input_glob = os.environ["CS224R_KL_INPUT_GLOB"]
    output_dir = Path(os.environ["CS224R_KL_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_prompts = int(os.environ["CS224R_KL_MAX_PROMPTS"])
    max_rollouts = int(os.environ["CS224R_KL_MAX_ROLLOUTS"])
    topn = int(os.environ["CS224R_KL_LOGPROBS"])

    print(f"[kl] cuda: device_count={torch.cuda.device_count()}")
    print(f"[kl] loading Qwen/Qwen3-4B-Base with logprobs={topn}")

    base_model_id = "Qwen/Qwen3-4B-Base"
    llm = LLM(
        model=base_model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=5120,
        enforce_eager=True,  # B200 requirement
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # 1 generated token, but request prompt_logprobs so we get top-K at every
    # position of the teacher-forced prompt.
    tf_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=topn,
    )

    summary_rows = []  # for the markdown

    inputs = sorted(glob.glob(input_glob))
    if not inputs:
        print(f"[kl] no inputs matched {input_glob}")
        return

    for fp in inputs:
        top = json.loads(Path(fp).read_text())
        label = top.get("label", Path(fp).stem)
        arm = arm_from_label(label)
        if arm == "base":
            print(f"[kl] skipping base arm ({fp}) — KL(base ‖ base) = 0")
            continue

        for ds_name, ds in top.get("datasets", {}).items():
            t0 = time.time()
            print(f"[kl] === {arm} / {ds_name} ===")
            per_prompt = ds["per_prompt"]
            if max_prompts:
                per_prompt = per_prompt[:max_prompts]

            # Build (rendered_prompt + rollout_text, prompt_token_count, policy_logprobs) batch.
            # Per the docstring alignment math: we MUST teacher-force the base
            # model on the same left context the policy saw. That's the chat-
            # templated prompt text plus the rollout. Then we slice the base
            # `prompt_logprobs` starting at position P=len(tokenize(prompt))
            # so each base distribution shares conditioning with the matching
            # policy step.
            tf_inputs = []
            meta = []  # (prompt_idx, rollout_idx, policy_token_steps, prompt_token_count)
            missing_prompt_warned = False
            for i_p, p in enumerate(per_prompt):
                pol_lp_list = p.get("logprobs") or []
                rollouts = p["rollouts"][:max_rollouts]
                pol_lp_list = pol_lp_list[:max_rollouts]
                rendered_prompt = p.get("rendered_prompt")
                if not rendered_prompt:
                    if not missing_prompt_warned:
                        print(f"[kl] WARN: per_prompt entries missing 'rendered_prompt' "
                              f"in {fp}; can't align base TF without it. Re-run run_eval.py "
                              f"on this arm/dataset to repopulate.")
                        missing_prompt_warned = True
                    continue
                # Count prompt tokens with the same tokenizer the base uses.
                # add_special_tokens=False because the chat-templated string
                # already contains every special-token marker; we don't want
                # the tokenizer to prepend another BOS.
                prompt_token_count = len(tok(rendered_prompt, add_special_tokens=False).input_ids)
                for i_r, (text, pol_steps) in enumerate(zip(rollouts, pol_lp_list)):
                    if not text or not pol_steps:
                        continue
                    tf_inputs.append(rendered_prompt + text)
                    meta.append((i_p, i_r, pol_steps, prompt_token_count))

            if not tf_inputs:
                print(f"[kl] no rollouts with logprobs for {arm}/{ds_name}")
                continue

            print(f"[kl]   teacher-forcing {len(tf_inputs)} rollouts...")
            outs = llm.generate(tf_inputs, tf_params)

            per_prompt_kls: list[dict] = []
            current_prompt_idx = -1
            current_entry = None
            for (i_p, i_r, pol_steps, prompt_token_count), out in zip(meta, outs):
                # prompt_logprobs: list aligned to tokenized (prompt+rollout) input;
                # index 0 is None (no left context for the first token). For the
                # base distribution that PREDICTED rollout token t given
                # (prompt + rollout_tokens[0:t]), we want
                # prompt_logprobs[prompt_token_count + t].
                base_prompt_lp = out.prompt_logprobs or []
                base_steps = []
                # Slice rollout positions; this gives base distributions for
                # rollout-token positions 0..R-1, exactly aligned with pol_steps.
                rollout_slice = base_prompt_lp[prompt_token_count:]
                for slot in rollout_slice:
                    if slot is None:
                        base_steps.append({})
                        continue
                    entry = {}
                    for tok_id, lp_obj in slot.items():
                        lp_val = getattr(lp_obj, "logprob", lp_obj)
                        try:
                            entry[int(tok_id)] = float(lp_val)
                        except (TypeError, ValueError):
                            continue
                    base_steps.append(entry)

                # Defensive: pol_steps and base_steps should now match length;
                # trim to min in case of tokenizer edge cases (e.g. trailing
                # whitespace re-tokenization). A few-token mismatch is fine;
                # a large mismatch indicates the slicing is off.
                n = min(len(pol_steps), len(base_steps))
                kls = []
                for t in range(n):
                    kl = per_token_kl(pol_steps[t], base_steps[t])
                    if not np.isnan(kl):
                        kls.append(kl)

                if i_p != current_prompt_idx:
                    if current_entry is not None:
                        per_prompt_kls.append(current_entry)
                    current_prompt_idx = i_p
                    current_entry = {
                        "problem_id": per_prompt[i_p]["problem_id"],
                        "rollout_mean_kls": [],
                        "n_rollouts_scored": 0,
                    }
                if kls:
                    current_entry["rollout_mean_kls"].append(float(np.mean(kls)))
                    current_entry["n_rollouts_scored"] += 1

            if current_entry is not None:
                per_prompt_kls.append(current_entry)

            # Aggregate
            all_mean_kls = [
                v for e in per_prompt_kls for v in e["rollout_mean_kls"]
            ]
            agg = {
                "arm": arm,
                "dataset": ds_name,
                "n_prompts": len(per_prompt_kls),
                "n_rollouts": len(all_mean_kls),
                "kl_mean_bits": float(np.mean(all_mean_kls)) if all_mean_kls else float("nan"),
                "kl_median_bits": float(np.median(all_mean_kls)) if all_mean_kls else float("nan"),
                "kl_p90_bits": float(np.percentile(all_mean_kls, 90)) if all_mean_kls else float("nan"),
                "per_prompt": per_prompt_kls,
            }

            out_path = output_dir / f"{arm}_{ds_name}.json"
            out_path.write_text(json.dumps(agg, indent=2))
            artifacts_volume.commit()
            print(f"[kl]   wrote {out_path} in {time.time() - t0:.1f}s "
                  f"(mean KL={agg['kl_mean_bits']:.4f} bits)")
            summary_rows.append(agg)

    # Summary markdown
    md_lines = ["# KL(π_arm ‖ π_base) per-token summary", "",
                "Mean over rollouts of (mean over token positions of per-token",
                "KL between policy top-K and base top-K, both renormalized).",
                "Skips base arm (KL=0).",
                "",
                "| arm | dataset | n_prompts | n_rollouts | mean KL (bits) | median | p90 |",
                "|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        md_lines.append(
            f"| {r['arm']} | {r['dataset']} | {r['n_prompts']} | "
            f"{r['n_rollouts']} | {r['kl_mean_bits']:.4f} | "
            f"{r['kl_median_bits']:.4f} | {r['kl_p90_bits']:.4f} |"
        )

    # Write the markdown to the artifacts volume; analysts can sync locally.
    md_path = output_dir / "kl_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    artifacts_volume.commit()
    print(f"[kl] wrote {md_path}")


@app.local_entrypoint()
def main() -> None:
    print(f"[launch] input_glob={_INPUT_GLOB} out={_OUTPUT_DIR} "
          f"max_prompts={_MAX_PROMPTS} max_rollouts={_MAX_ROLLOUTS} topn={_TOPN}")
    # .spawn() = fire-and-forget; survives local-side disconnect when paired
    # with `modal run --detach`. Output JSONs land on the artifacts volume.
    call = kl_pass.spawn()
    print(f"[launch] spawned kl_pass call_id={call.object_id}; "
          f"per-(arm,dataset) JSONs will land at {_OUTPUT_DIR}/<arm>_<dataset>.json")
