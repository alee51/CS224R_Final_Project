"""Shared Modal image for main-verl Stage 1+ jobs.

MAXRL_COMMIT: 7197bbb46a2ecd866da52f6b401ff20a34fe9390 (maxrl main @ 2026-05-29)
GPU pins: vLLM 0.9.0 + cu128 torch index, transformers<4.54, flash-attn 2.8.3 (Blackwell)
  — aligned with main/infra/modal_image.py; not maxRL README defaults (torch 2.6 / vLLM 0.8.4).
Stage 2 (S2.3): editable install uses maxrl/setup.py deps; GPU wheels re-pinned after install.
Stage 2 (S2.5b): patch maxrl router so `polaris` / `math_reward` → math.py (upstream math_reward.py).
Stage 3a (S3a.2): patch adds MINORITY_COT enum + compute_minority_cot_outcome_advantage to core_algos.py.
  <!-- TODO (S3a next rebuild): add .run_commands("cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_minority_cot_adv_est.patch") AFTER the existing math_reward patch step. Do NOT merge into the same run_commands call — keep patches as separate layers for easier rollback. Sequence this with Stage 2 smoke completion before triggering a new image build. -->
Runtime: ray installed explicitly for smoke; torch/vllm versions come from vLLM 0.9.0 pin.
PYTORCH_CUDA_ALLOC_CONF: intentionally omits expandable_segments:True — vLLM 0.9 CuMemAllocator
  (VeRL colocated rollout) hard-fails if set; see verl-reference §4.3 / stage-02-log S2.5 attempt 3.
  main/ keeps expandable_segments for the custom trainer; fragmentation fallback = micro-batch / util.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MAIN_VERL_DIR = _LOCAL_REPO_ROOT / "main-verl"

MAXRL_COMMIT = "7197bbb46a2ecd866da52f6b401ff20a34fe9390"
_VLLM_VERSION = "0.9.0"
_RAY_VERSION = "2.44.1"  # <!-- TODO: bump if smoke Ray/GPU init fails -->

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        "git clone https://github.com/tajwarfahim/maxrl.git /root/maxrl",
        f"cd /root/maxrl && git checkout {MAXRL_COMMIT}",
    )
    .add_local_dir(
        str(_LOCAL_MAIN_VERL_DIR / "infra" / "patches"),
        remote_path="/root/main-verl/infra/patches",
        copy=True,  # required by Modal >=1.x when build steps follow add_local_dir
    )
    .run_commands(
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_polaris_math_reward.patch",
    )
    .run_commands(
        # Stage 3a (S3a.2): additive patch — registers AdvantageEstimator.MINORITY_COT and
        # compute_minority_cot_outcome_advantage in core_algos.py. Does NOT modify GRPO path;
        # only activates when algorithm.adv_estimator=minority_cot is set. Stage 2 GRPO results
        # remain bit-identical; the new key is unreachable from the Stage 2 config.
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_minority_cot_adv_est.patch",
    )
    .run_commands(
        # Stage 3a (S3a.2 follow-up 2026-05-30): ray_trainer.py has a hardcoded allowlist of
        # advantage estimators that disable the critic (lines 437–457 in pinned source). The
        # @register_adv_est registry in core_algos.py is extensible, but RayPPOTrainer.__init__
        # also gates on this separate list — unknown estimators fall to `else: raise
        # NotImplementedError` (caught on Stage 3a smoke first launch). This patch adds
        # MINORITY_COT to that allowlist. Same additive guarantee: Stage 2 GRPO path is
        # untouched (GRPO is already in the list).
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_minority_cot_ray_trainer.patch",
    )
    .run_commands(
        # Stage 3b (2026-05-30): expose DataProto to registered adv_estimator hooks so the
        # minority_cot/poly_epo_cot judge variants can read data.batch["responses"] +
        # data.non_tensor_batch["raw_prompt"]. Adds one key ("data": data) to adv_kwargs
        # in the compute_advantage dispatch else-branch (ray_trainer.py:361). Existing
        # estimators that don't take **kwargs are not in that dispatch path; only the
        # @register_adv_est-decorated hooks go through it, and our minority_cot hook
        # accepts **kwargs. Backward-compatible additive change. Image rebuild count 5.
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_expose_data_to_adv_est.patch",
    )
    .run_commands(
        # Stage 5 (S5.1): register AdvantageEstimator.POLY_EPO_COT +
        # compute_poly_epo_cot_outcome_advantage. Additive — Stage 2 GRPO and
        # Stage 3a/3b minority_cot paths untouched unless adv_estimator switched.
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_poly_epo_cot_adv_est.patch",
    )
    .run_commands(
        # Stage 5 (S5.1): ray_trainer critic-disabled allowlist — mirrors S3a ray_trainer patch.
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_poly_epo_cot_ray_trainer.patch",
    )
    .run_commands(
        # Stage 7: forward per-step diagnostics (distinct_clusters, pass_at_8, prompts_unlocked,
        # fraction_filtered, judge_parse_ok_rate) from the @register_adv_est hooks to W&B.
        # Adds one line after compute_data_metrics() in the training loop: reads
        # batch.meta_info["cs224r_metrics"] written by the minority_cot / poly_epo_cot hooks.
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_cs224r_metrics_ray_trainer.patch",
    )
    .run_commands(
        # Stage 8: add permanent_ckpt_freq support to ray_trainer._save_checkpoint.
        # Saves every save_freq steps; keeps only the latest temp ckpt between
        # permanent_ckpt_freq boundaries (multiples of permanent_ckpt_freq are never deleted).
        "cd /root/maxrl && patch -p1 < /root/main-verl/infra/patches/maxrl_permanent_ckpt.patch",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/main-verl",
            "CS224R_MAIN_VERL_ROOT": "/root/main-verl",
            "VLLM_USE_V1": "0",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .pip_install(
        f"vllm=={_VLLM_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers<4.54.0")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install(
        f"ray[default]=={_RAY_VERSION}",
        "wandb",
        "math-verify",
        "datasets>=2.20",
        "pyyaml>=6.0",
    )
    .run_commands("cd /root/maxrl && pip install -e .")
    .pip_install(
        "tensordict",
        "hydra-core",
        "omegaconf",
        "accelerate",
        "codetiming",
        "peft",
        "pyarrow",
        "pandas",
        "dill",
        "torchdata",
    )
    .pip_install(
        f"vllm=={_VLLM_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers<4.54.0")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(
        str(_LOCAL_MAIN_VERL_DIR),
        remote_path="/root/main-verl",
        ignore=[
            "docs",
            "*.md",
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
        ],
    )
)


def app_name() -> str:
    return os.environ.get("CS224R_APP_NAME", "cs224r-verl-stage01")
