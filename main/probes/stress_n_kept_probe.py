"""Stress test: HF train with n_kept=512 (all rollouts kept) under train_real.yaml settings."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

def package_root() -> Path:
    if env := os.environ.get("CS224R_MAIN_ROOT"):
        return Path(env)
    probe = Path(__file__).resolve()
    for candidate in (probe.parents[1], probe.parents[2] / "main"):
        if (candidate / "train").is_dir():
            return candidate
    return probe.parents[1]


_MAIN_ROOT = package_root()
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from train.ablation import apply_ablation_env, vllm_sleep_enabled
from train.rollout import RolloutEngine, RolloutResult
from train.trainer import (
    TrainCfg,
    _train_step_microbatched,
    build_hf,
    load_cfg,
    set_seeds,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_stress_config(path: str | Path) -> tuple[dict[str, Any], TrainCfg]:
    raw_path = Path(path)
    if not raw_path.is_file():
        rel = _MAIN_ROOT / "configs" / raw_path.name
        if rel.is_file():
            raw_path = rel
    data = yaml.safe_load(raw_path.read_text())
    extends = data.pop("extends", None)
    if extends:
        base_path = raw_path.parent / extends
        if not base_path.is_file():
            base_path = _MAIN_ROOT / "configs" / extends
        base = yaml.safe_load(base_path.read_text())
        data = _deep_merge(base, data)
    train_cfg = load_cfg_from_merged(data)
    return data, train_cfg


def load_cfg_from_merged(data: dict[str, Any]) -> TrainCfg:
    """Build TrainCfg without extends handling in load_cfg."""
    tmp = _MAIN_ROOT / "configs" / ".stress_runtime.yaml"
    tmp.write_text(yaml.dump({k: v for k, v in data.items() if k != "stress"}))
    try:
        return load_cfg(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _synthetic_rollouts(
    *,
    n_kept: int,
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str,
) -> list[RolloutResult]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab = int(tok.vocab_size)
    rollouts: list[RolloutResult] = []
    for i in range(n_kept):
        # Deterministic valid token streams (avoid pad-only).
        p_ids = [(1000 + i * 13 + j) % (vocab - 100) + 100 for j in range(prompt_tokens)]
        c_ids = [(2000 + i * 17 + j) % (vocab - 100) + 100 for j in range(completion_tokens)]
        old_lps = [0.0] * len(c_ids)
        rollouts.append(
            RolloutResult(
                prompt_idx=i // 8,
                rollout_idx=i % 8,
                completion_ids=c_ids,
                completion_text="",
                prompt_ids=p_ids,
                old_logprobs=old_lps,
                finish_reason="stop",
            )
        )
    return rollouts


def _vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _run_hf_microbatch(
    cfg: TrainCfg,
    hf_model: Any,
    rollouts: list[RolloutResult],
    microbatch: int,
) -> dict[str, Any]:
    device = next(hf_model.parameters()).device
    opt = torch.optim.AdamW(hf_model.parameters(), lr=float(cfg.train["lr"]))
    adv = [1.0 if i % 2 == 0 else -1.0 for i in range(len(rollouts))]
    hf_model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    t0 = time.monotonic()
    try:
        loss, eff_mb = _train_step_microbatched(
            cfg,
            hf_model,
            opt,
            [r.prompt_ids for r in rollouts],
            [r.completion_ids for r in rollouts],
            [torch.tensor(r.old_logprobs, dtype=torch.float32) for r in rollouts],
            adv,
            microbatch,
            device,
            optimizer_step=True,
            instrument=True,
            phase_times={},
            vram_peak={},
        )
        ok = True
        err = None
    except torch.cuda.OutOfMemoryError as e:
        ok = False
        loss, eff_mb = 0.0, 0
        err = str(e)[:500]
    elapsed = time.monotonic() - t0
    grad_accum = max(1, (len(rollouts) + eff_mb - 1) // eff_mb) if ok else None
    return {
        "ok": ok,
        "loss": float(loss) if ok else None,
        "effective_microbatch": eff_mb,
        "grad_accum": grad_accum,
        "t_fwd_bwd_s": elapsed,
        "vram_peak_gb": _vram_gb(),
        "error": err,
    }


def run_stress(config_path: str) -> dict[str, Any]:
    raw, cfg = load_stress_config(config_path)
    stress = raw["stress"]
    n_kept = int(stress["n_kept"])
    prompt_tokens = int(stress["prompt_tokens"])
    completion_tokens = int(stress["completion_tokens"])
    microbatch = int(stress.get("microbatch", cfg.train["microbatch"]))
    scenarios = list(stress.get("scenarios", ["hf_only", "vllm_awake", "vllm_sleep"]))

    set_seeds(cfg.global_seed)
    apply_ablation_env(vllm_sleep=1 if vllm_sleep_enabled() else 0)

    rollouts = _synthetic_rollouts(
        n_kept=n_kept,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model_name=cfg.rollout.model,
    )
    results: dict[str, Any] = {
        "n_kept": n_kept,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "microbatch": microbatch,
        "model": cfg.rollout.model,
        "gpu_memory_utilization": cfg.rollout.gpu_memory_utilization,
        "vllm_sleep_env": vllm_sleep_enabled(),
        "scenarios": {},
        "materialized_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Stress n_kept=%s prompt=%s completion=%s microbatch=%s",
        n_kept,
        prompt_tokens,
        completion_tokens,
        microbatch,
    )

    if "hf_only" in scenarios:
        logger.info("=== scenario: hf_only (no vLLM loaded) ===")
        hf_model, _ = build_hf(cfg)
        results["scenarios"]["hf_only"] = _run_hf_microbatch(
            cfg, hf_model, rollouts, microbatch
        )
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rollout_engine = None
    if "vllm_awake" in scenarios or "vllm_sleep" in scenarios:
        logger.info("=== init RolloutEngine (vLLM collocated footprint) ===")
        rollout_engine = RolloutEngine(cfg.rollout)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        vllm_resident_gb = _vram_gb()
        results["vllm_resident_gb_after_init"] = vllm_resident_gb

    if "vllm_awake" in scenarios and rollout_engine is not None:
        logger.info("=== scenario: vllm_awake (vLLM resident + HF microbatch) ===")
        hf_model, _ = build_hf(cfg)
        results["scenarios"]["vllm_awake"] = _run_hf_microbatch(
            cfg, hf_model, rollouts, microbatch
        )
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "vllm_sleep" in scenarios and rollout_engine is not None:
        logger.info("=== scenario: vllm_sleep (sleep then HF microbatch) ===")
        rollout_engine.sleep_for_train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results["vram_gb_after_vllm_sleep"] = _vram_gb()
        hf_model, _ = build_hf(cfg)
        results["scenarios"]["vllm_sleep"] = _run_hf_microbatch(
            cfg, hf_model, rollouts, microbatch
        )
        del hf_model

    if rollout_engine is not None:
        rollout_engine.shutdown()

    return results


def _write_results(raw: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stress_n_kept_results.json"
    out_path.write_text(json.dumps(raw, indent=2) + "\n")
    return out_path


# --- Modal --------------------------------------------------------------------

try:
    import modal

    from infra.modal_image import image
    from infra.modal_volume import (
        ARTIFACTS_MOUNT,
        ARTIFACTS_VOLUME_NAME,
        HF_CACHE_MOUNT,
        HF_CACHE_VOLUME_NAME,
    )

    app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-stress-n-kept"))
    _artifacts_vol = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
    _hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu="H200",
        timeout=60 * 60,
        volumes={
            ARTIFACTS_MOUNT: _artifacts_vol,
            HF_CACHE_MOUNT: _hf_vol,
        },
        secrets=[modal.Secret.from_name("HUGGINGFACE")],
    )
    def stress_remote(
        config_path: str = "configs/stress_n_kept_512.yaml",
    ) -> dict[str, Any]:
        # Match production train (vLLM sleep before HF backward).
        apply_ablation_env(vllm_sleep=1)
        cfg_path = str(_MAIN_ROOT / config_path)
        results = run_stress(cfg_path)
        out_dir = Path(ARTIFACTS_MOUNT) / "probes/stress_n_kept_512"
        path = _write_results(results, out_dir)
        _artifacts_vol.commit()
        logger.info("Wrote %s", path)
        return results

except ImportError:
    app = None  # type: ignore[assignment, misc]
    stress_remote = None  # type: ignore[assignment, misc]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(_MAIN_ROOT / "configs/stress_n_kept_512.yaml"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_MAIN_ROOT / "data/probes/stress_n_kept_512",
    )
    args = parser.parse_args()
    apply_ablation_env(vllm_sleep=int(os.environ.get("CS224R_VLLM_SLEEP", "1")))
    out = run_stress(args.config)
    path = _write_results(out, args.out_dir)
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")
