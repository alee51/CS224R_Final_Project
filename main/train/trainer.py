"""GRPO trainer loop and Modal entrypoint."""

from __future__ import annotations

import logging
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from data.dataset import JsonlPromptDataset
from train.loss import grpo_loss
from train.objective import compute_advantages
from train.prompts import format_problem
from train.reward import compute_reward
from train.rollout import RolloutCfg, RolloutEngine, RolloutResult
from train.weight_sync import SyncStats

logger = logging.getLogger(__name__)


@dataclass
class StepBatch:
    prompts: list[str]
    golds: list[str]
    problem_ids: list[int]


@dataclass
class StepResult:
    loss: float
    mean_reward: float
    fraction_filtered: float
    mean_advantage: float
    n_kept_sequences: int
    sync_stats: SyncStats | None = None
    phase_times_s: dict[str, float] = field(default_factory=dict)
    vram_peak_gb: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    step_cache: dict[str, Any] | None = None


@dataclass
class TrainCfg:
    raw: dict[str, Any]
    global_seed: int
    arm: str
    train: dict[str, Any]
    rollout: RolloutCfg
    loss: dict[str, Any]
    weight_sync: dict[str, Any]
    wandb: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainCfg:
        return cls(
            raw=d,
            global_seed=int(d["global_seed"]),
            arm=str(d.get("arm", "grpo")),
            train=d["train"],
            rollout=RolloutCfg.from_dict(d["rollout"]),
            loss=d["loss"],
            weight_sync=d.get("weight_sync", {"every_n_steps": 1}),
            wandb=d["wandb"],
        )


def load_cfg(path: str | Path) -> TrainCfg:
    with open(path) as f:
        return TrainCfg.from_dict(yaml.safe_load(f))


def train_cfg_from_dict(d: dict[str, Any]) -> TrainCfg:
    return TrainCfg.from_dict(d)


def _device_vram_total_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def _backward_microbatched(
    cfg: TrainCfg,
    hf_model: Any,
    opt: torch.optim.Optimizer,
    new_t: torch.Tensor,
    old_t: torch.Tensor,
    mask: torch.Tensor,
    adv_t: torch.Tensor,
    keep_t: torch.Tensor,
    microbatch: int,
    *,
    optimizer_step: bool = True,
    instrument: bool = False,
    phase_times: dict[str, float] | None = None,
    vram_peak: dict[str, float] | None = None,
) -> float:
    length_norm = str(cfg.loss.get("length_norm", "per_seq"))
    clip_low = float(cfg.loss.get("clip_low", 0.20))
    clip_high = float(cfg.loss.get("clip_high", 0.28))
    n_kept = new_t.shape[0]
    microbatch = max(1, min(microbatch, n_kept))
    grad_accum = max(1, math.ceil(n_kept / microbatch)) if n_kept else 1
    device = new_t.device

    opt.zero_grad(set_to_none=True)
    phase_times = phase_times if phase_times is not None else {}
    vram_peak = vram_peak if vram_peak is not None else {}

    with _phase_timer("t_backward", instrument, phase_times, vram_peak):
        if n_kept == 0:
            return 0.0
        total_loss = torch.tensor(0.0, device=device)
        for start in range(0, n_kept, microbatch):
            end = min(start + microbatch, n_kept)
            chunk_loss = grpo_loss(
                new_t[start:end],
                old_t[start:end],
                adv_t[start:end],
                mask[start:end],
                keep_t[start:end],
                clip_low=clip_low,
                clip_high=clip_high,
                length_norm=length_norm,
            )
            (chunk_loss / grad_accum).backward()
            total_loss = total_loss + chunk_loss.detach()
        loss_val = float(
            (total_loss / max(1, (n_kept + microbatch - 1) // microbatch)).item()
        )

    if optimizer_step:
        grad_clip = float(cfg.train.get("grad_clip", 1.0))
        with _phase_timer("t_optimizer", instrument, phase_times, vram_peak):
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), grad_clip)
            opt.step()
    return loss_val


def run_microbatch_forward_backward(
    cfg: TrainCfg,
    hf_model: Any,
    step_cache: dict[str, Any],
    microbatch: int,
) -> tuple[bool, float, float]:
    """Phase 2 sweep: forward+backward on cached logprob tensors (no optim step)."""
    device = next(hf_model.parameters()).device
    tensors = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in step_cache["tensors"].items()
    }
    opt = torch.optim.AdamW(hf_model.parameters(), lr=1e-6)
    hf_model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    try:
        _backward_microbatched(
            cfg,
            hf_model,
            opt,
            tensors["new_logprobs"],
            tensors["old_logprobs"],
            tensors["mask"],
            tensors["advantages"],
            tensors["keep_mask"],
            microbatch,
            optimizer_step=False,
            instrument=False,
        )
        elapsed = time.monotonic() - t0
        peak = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        torch.cuda.empty_cache()
        return True, peak, elapsed
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, 0.0, 0.0


def rollout_seed(global_seed: int, problem_id: int, n_rollouts: int, rollout_idx: int) -> int:
    return global_seed + problem_id * n_rollouts + rollout_idx


def set_seeds(global_seed: int) -> None:
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)


def _git_metadata() -> dict[str, Any]:
    env_sha = os.environ.get("CS224R_GIT_SHA")
    if env_sha:
        dirty_raw = os.environ.get("CS224R_GIT_DIRTY", "false").lower()
        return {
            "git_sha": env_sha,
            "git_dirty": dirty_raw in ("true", "1", "yes"),
            "git_sha_short": os.environ.get("CS224R_GIT_SHA_SHORT", env_sha[:7]),
        }

    def _run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]))
        short = _run(["git", "rev-parse", "--short", "HEAD"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty, short = "unknown", False, "unknown"
    return {"git_sha": sha, "git_dirty": dirty, "git_sha_short": short}


def _dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("vllm", "torch", "transformers", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


@contextmanager
def _phase_timer(
    name: str,
    instrument: bool,
    times: dict[str, float],
    vram: dict[str, float],
) -> Iterator[None]:
    if instrument and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    yield
    times[name] = time.monotonic() - t0
    if instrument and torch.cuda.is_available():
        vram[name] = torch.cuda.max_memory_allocated() / (1024**3)


def _pad_sequences(
    seqs: list[list[float]], pad: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        raise ValueError("empty sequence list")
    t_max = max(len(s) for s in seqs)
    b = len(seqs)
    out = torch.full((b, t_max), pad, dtype=torch.float32)
    mask = torch.zeros((b, t_max), dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
        mask[i, : len(s)] = 1.0
    return out, mask


def _completion_logprobs_hf(
    model: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: torch.device,
) -> list[float]:
    """Teacher-forcing logprobs on completion tokens only."""
    if not completion_ids:
        return []
    input_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=device
    )
    attn = torch.ones_like(input_ids)
    with torch.set_grad_enabled(True):
        out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    lps: list[float] = []
    for j, tid in enumerate(completion_ids):
        pos = start + j
        lps.append(float(log_probs[pos, tid].detach().cpu()))
    return lps


def build_hf(cfg: TrainCfg) -> tuple[Any, torch.optim.Optimizer]:
    from transformers import AutoModelForCausalLM

    model_name = cfg.rollout.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if cfg.train.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train["lr"]),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    return model, opt


def run_one_grpo_step(
    cfg: TrainCfg,
    rollout_engine: RolloutEngine,
    hf_model: Any,
    opt: torch.optim.Optimizer,
    batch: StepBatch,
    *,
    instrument: bool = False,
    step: int = 0,
    microbatch: int | None = None,
    step_cache: dict[str, Any] | None = None,
) -> StepResult:
    """One GRPO step: rollout → reward → advantage → loss → optim → sync."""
    n_rollouts = int(cfg.train["n_rollouts"])
    mb = int(microbatch if microbatch is not None else cfg.train["microbatch"])
    prompt_variant = str(cfg.raw.get("prompt_variant", "dapo_answer_v1"))

    phase_times: dict[str, float] = {}
    vram_peak: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}

    device = next(hf_model.parameters()).device

    if step_cache is not None:
        tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in step_cache["tensors"].items()
        }
        new_t = tensors["new_logprobs"]
        old_t = tensors["old_logprobs"]
        mask = tensors["mask"]
        adv_t = tensors["advantages"]
        keep_t = tensors["keep_mask"]
        n_kept = int(tensors.get("n_kept", new_t.shape[0]))
        rewards_t = None
        adv_out = None
        rollout_tokens = 0
    else:
        formatted = [
            format_problem(p, variant=prompt_variant) for p in batch.prompts
        ]
        seeds = [
            rollout_seed(cfg.global_seed, pid, n_rollouts, r)
            for pid in batch.problem_ids
            for r in range(n_rollouts)
        ]

        with _phase_timer("t_rollout", instrument, phase_times, vram_peak):
            rollouts = rollout_engine.generate(formatted, n_rollouts, seeds)
        rollout_tokens = sum(len(r.completion_ids) for r in rollouts)
        if instrument:
            diagnostics["rollout_output_tokens"] = rollout_tokens
            diagnostics["vram_headroom_gb_after_rollout"] = (
                _device_vram_total_gb() - vram_peak.get("t_rollout", 0.0)
            )

        rewards_grid: list[list[float]] = []
        parse_ok = 0
        total_rw = 0
        with _phase_timer("t_score", instrument, phase_times, vram_peak):
            for p_idx in range(len(batch.prompts)):
                row: list[float] = []
                for r_idx in range(n_rollouts):
                    flat_idx = p_idx * n_rollouts + r_idx
                    rr = rollouts[flat_idx]
                    rw = compute_reward(rr.completion_text, batch.golds[p_idx])
                    row.append(float(rw["reward"]))
                    parse_ok += int(rw["parse_ok"])
                    total_rw += 1
                rewards_grid.append(row)
        if instrument and total_rw:
            diagnostics["parse_ok_rate"] = parse_ok / total_rw

        rewards_t = torch.tensor(rewards_grid, dtype=torch.float32)

        with _phase_timer("t_advantage", instrument, phase_times, vram_peak):
            adv_out = compute_advantages(cfg.arm, rewards_t)

        kept_rollouts: list[RolloutResult] = []
        kept_adv: list[float] = []
        for p_idx in range(len(batch.prompts)):
            if not bool(adv_out.keep_mask[p_idx].item()):
                continue
            for r_idx in range(n_rollouts):
                flat_idx = p_idx * n_rollouts + r_idx
                kept_rollouts.append(rollouts[flat_idx])
                kept_adv.append(float(adv_out.advantages[p_idx, r_idx].item()))

        new_lps_list: list[list[float]] = []
        with _phase_timer("t_logprob_fwd", instrument, phase_times, vram_peak):
            for rr in kept_rollouts:
                new_lps_list.append(
                    _completion_logprobs_hf(
                        hf_model,
                        rr.prompt_ids,
                        rr.completion_ids,
                        device,
                    )
                )

        old_lps_list = [rr.old_logprobs for rr in kept_rollouts]
        new_t, mask = _pad_sequences(new_lps_list)
        old_t, _ = _pad_sequences(old_lps_list)
        adv_t = torch.tensor(kept_adv, dtype=torch.float32)
        keep_t = torch.ones(len(kept_rollouts), dtype=torch.bool)
        new_t = new_t.to(device)
        old_t = old_t.to(device)
        mask = mask.to(device)
        adv_t = adv_t.to(device)
        keep_t = keep_t.to(device)
        n_kept = len(kept_rollouts)

    grad_accum = max(1, math.ceil(n_kept / mb)) if n_kept else 1
    diagnostics["grad_accum_at_mb"] = grad_accum

    loss_val = _backward_microbatched(
        cfg,
        hf_model,
        opt,
        new_t,
        old_t,
        mask,
        adv_t,
        keep_t,
        mb,
        optimizer_step=True,
        instrument=instrument,
        phase_times=phase_times,
        vram_peak=vram_peak,
    )

    sync_stats: SyncStats | None = None
    every_n = int(cfg.weight_sync.get("every_n_steps", 1))
    if every_n > 0 and (step + 1) % every_n == 0:
        with _phase_timer("t_weight_sync", instrument, phase_times, vram_peak):
            sync_stats = rollout_engine.update_weights(hf_model)
        if instrument and sync_stats is not None:
            diagnostics["t_weight_sync_step_s"] = sync_stats.wall_clock_s

    if instrument:
        from train.weight_sync import sync_hf_to_vllm

        bench_measured: list[float] = []
        for i in range(3):
            t_b = time.monotonic()
            sync_hf_to_vllm(hf_model, rollout_engine.llm)
            if i > 0:
                bench_measured.append(time.monotonic() - t_b)
        if bench_measured:
            diagnostics["t_weight_sync_bench_median"] = sorted(bench_measured)[
                len(bench_measured) // 2
            ]

    vram_peak_step = max(vram_peak.values()) if vram_peak else 0.0
    diagnostics["vram_peak_gb_step"] = vram_peak_step
    diagnostics["device_vram_total_gb"] = _device_vram_total_gb()
    diagnostics["vram_headroom_gb_step"] = (
        diagnostics["device_vram_total_gb"] - vram_peak_step
    )

    cache_out = None
    if step_cache is None and n_kept > 0:
        cache_out = {
            "tensors": {
                "new_logprobs": new_t.detach().cpu(),
                "old_logprobs": old_t.detach().cpu(),
                "mask": mask.detach().cpu(),
                "advantages": adv_t.detach().cpu(),
                "keep_mask": keep_t.detach().cpu(),
                "n_kept": n_kept,
            }
        }

    mean_reward = float(rewards_t.mean().item()) if rewards_t is not None else 0.0
    mean_adv = (
        float(adv_out.advantages[adv_out.keep_mask].mean().item())
        if adv_out is not None and adv_out.keep_mask.any()
        else 0.0
    )
    frac_filt = (
        float(adv_out.diagnostics.get("fraction_filtered", 0.0))
        if adv_out is not None
        else 0.0
    )

    return StepResult(
        loss=loss_val,
        mean_reward=mean_reward,
        fraction_filtered=frac_filt,
        mean_advantage=mean_adv,
        n_kept_sequences=n_kept,
        sync_stats=sync_stats,
        phase_times_s=phase_times,
        vram_peak_gb=vram_peak,
        diagnostics=diagnostics,
        step_cache=cache_out,
    )


def setup_wandb(cfg: TrainCfg, repro: dict[str, Any]) -> Any:
    import wandb

    run_name = cfg.raw.get("wandb_run_name")
    if not run_name:
        op = cfg.raw.get("operator", "unknown")
        run_name = f"train-{cfg.arm}_{op}"
    tags = [
        "phase=train",
        f"operator={cfg.raw.get('operator', 'unknown')}",
        f"gpu_class={cfg.raw.get('gpu_class', 'unknown')}",
        f"arm={cfg.arm}",
        f"git_sha_short={repro.get('git_sha_short', 'unknown')}",
    ]
    return wandb.init(
        entity=cfg.wandb["entity"],
        project=cfg.wandb["project"],
        group=cfg.wandb.get("group"),
        name=run_name,
        config={**cfg.raw, **repro, "dep_versions": _dep_versions()},
        tags=tags,
    )


def log_repro(cfg: TrainCfg, repro: dict[str, Any]) -> None:
    logger.info(
        "Repro: seed=%s git=%s dirty=%s deps=%s",
        cfg.global_seed,
        repro.get("git_sha_short"),
        repro.get("git_dirty"),
        _dep_versions(),
    )


def should_checkpoint(step: int, cfg: TrainCfg, last_ckpt_time: float) -> bool:
    every = int(cfg.train.get("checkpoint_every_steps", 50))
    if (step + 1) % every == 0:
        return True
    return (time.monotonic() - last_ckpt_time) >= 3600


def save_ckpt(
    path: Path,
    hf_model: Any,
    opt: torch.optim.Optimizer,
    step: int,
    cfg: TrainCfg,
    wandb_run_id: str,
) -> None:
    """STANDARDS checkpoint: weights, optim, RNG, step, wandb run id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
    payload = {
        "step": step,
        "wandb_run_id": wandb_run_id,
        "config": cfg.raw,
        "rng": rng,
        "model": hf_model.state_dict(),
        "optimizer": opt.state_dict(),
    }
    torch.save(payload, path)
    logger.info("Wrote checkpoint to %s (step %s)", path, step)


def train(cfg: TrainCfg) -> None:
    repro = _git_metadata()
    set_seeds(cfg.global_seed)
    run = setup_wandb(cfg, repro)
    log_repro(cfg, repro)

    rollout_engine = RolloutEngine(cfg.rollout)
    hf_model, opt = build_hf(cfg)
    dataset = JsonlPromptDataset(cfg.train["data_path"], cfg.global_seed)

    total_steps = int(cfg.train["total_steps"])
    batch_size = int(cfg.train["batch_size"])
    last_ckpt = time.monotonic()
    ckpt_dir = Path(cfg.train.get("checkpoint_dir", "/vol/checkpoints/train/"))

    try:
        for step in range(total_steps):
            problems, golds, pids = dataset.next_batch_with_ids(batch_size)
            batch = StepBatch(
                prompts=problems,
                golds=golds,
                problem_ids=pids,
            )
            result = run_one_grpo_step(
                cfg,
                rollout_engine,
                hf_model,
                opt,
                batch,
                instrument=False,
                step=step,
            )
            import wandb

            log_dict: dict[str, Any] = {
                "train/loss": result.loss,
                "train/mean_reward": result.mean_reward,
                "train/fraction_filtered": result.fraction_filtered,
                "train/mean_advantage": result.mean_advantage,
                "train/n_kept_sequences": result.n_kept_sequences,
            }
            if result.sync_stats is not None:
                log_dict["train/weight_sync_s"] = result.sync_stats.wall_clock_s
            wandb.log(log_dict, step=step)

            if should_checkpoint(step, cfg, last_ckpt):
                save_ckpt(
                    ckpt_dir / f"step_{step:06d}.pt",
                    hf_model,
                    opt,
                    step,
                    cfg,
                    run.id,
                )
                last_ckpt = time.monotonic()
    finally:
        rollout_engine.shutdown()
        run.finish()


# --- Modal entrypoint ---------------------------------------------------------

try:
    import modal

    from infra.modal_image import image
    from infra.modal_volume import (
        ARTIFACTS_MOUNT,
        ARTIFACTS_VOLUME_NAME,
        HF_CACHE_MOUNT,
        HF_CACHE_VOLUME_NAME,
    )

    app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-train-untagged"))
    _artifacts_vol = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
    _hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu="H100",
        timeout=60 * 60 * 8,
        volumes={
            ARTIFACTS_MOUNT: _artifacts_vol,
            HF_CACHE_MOUNT: _hf_vol,
        },
        secrets=[
            modal.Secret.from_name("HUGGINGFACE"),
            modal.Secret.from_name("WANDB_API_KEY"),
        ],
    )
    def train_remote(config_path: str) -> None:
        train(load_cfg(config_path))

except ImportError:
    app = None  # type: ignore[assignment, misc]
    train_remote = None  # type: ignore[assignment, misc]
