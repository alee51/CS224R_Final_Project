"""GRPO trainer loop and Modal entrypoint."""

from __future__ import annotations

import html
import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

# Local `modal run` resolves imports from repo `main/` package root.
_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from data.dataset import JsonlPromptDataset
from train.loss import grpo_loss
from train.ablation import ablation_label, logprob_chunk_size, vllm_sleep_enabled
from train.objective import compute_advantages
from train.prompts import format_problem
from train.repro import dep_versions as _dep_versions
from train.repro import git_metadata as _git_metadata
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
    train_wandb: dict[str, float] = field(default_factory=dict)
    step_cache: dict[str, Any] | None = None
    sample_completions: list[str] = field(default_factory=list)


_EXTRACT_PATH_WANDB_KEYS = (
    "hybrid",
    "boxed",
    "answer_line",
    "none",
)


def aggregate_train_step_wandb_metrics(
    rewards_grid: list[list[float]],
    reward_meta: list[list[dict[str, Any]]],
    completion_token_lens: list[int],
) -> dict[str, float]:
    """Poly-EPO Fig. 2–style training dynamics (PLAN §5 / probe plan Group C1–C2)."""
    n_prompts = len(rewards_grid)
    if n_prompts == 0:
        return {}

    n_rollouts = len(rewards_grid[0])
    n_coverage = 0
    n_mixed = 0
    parse_ok = 0
    n_scored = 0
    path_counts = {k: 0 for k in _EXTRACT_PATH_WANDB_KEYS}
    path_reward_sums = {k: 0.0 for k in _EXTRACT_PATH_WANDB_KEYS}
    correct_count_hist = [0] * (n_rollouts + 1)

    for p_idx in range(n_prompts):
        row = rewards_grid[p_idx]
        n_correct = sum(1 for r in row if r > 0.0)
        k = min(n_correct, n_rollouts)
        correct_count_hist[k] += 1
        if n_correct > 0:
            n_coverage += 1
        if 0 < n_correct < len(row):
            n_mixed += 1
        for r_idx in range(len(row)):
            meta = reward_meta[p_idx][r_idx]
            parse_ok += int(bool(meta.get("parse_ok")))
            n_scored += 1
            path = str(meta.get("extract_path", "none"))
            if path not in path_counts:
                path = "none"
            path_counts[path] += 1
            path_reward_sums[path] += float(row[r_idx])

    out: dict[str, float] = {
        "train/prompt_coverage": n_coverage / n_prompts,
        "train/mixed_reward_rate": n_mixed / n_prompts,
    }
    for k, count in enumerate(correct_count_hist):
        out[f"train/frac_prompts_{k}_correct"] = count / n_prompts
    if n_scored:
        out["train/parse_ok_rate"] = parse_ok / n_scored
        for key, count in path_counts.items():
            out[f"train/extract_path_{key}"] = count / n_scored
            if count > 0:
                out[f"train/mean_reward_extract_{key}"] = (
                    path_reward_sums[key] / count
                )

    if completion_token_lens:
        lens_sorted = sorted(completion_token_lens)
        out["train/mean_completion_tokens"] = sum(lens_sorted) / len(lens_sorted)
        p95_idx = min(len(lens_sorted) - 1, int(0.95 * (len(lens_sorted) - 1)))
        out["train/p95_completion_tokens"] = float(lens_sorted[p95_idx])

    return out


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


def apply_launch_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Env overrides from launch_train.sh — smoke/full share one yaml."""
    out = dict(raw)
    train = dict(out.get("train", {}))
    steps_env = os.environ.get("CS224R_TOTAL_STEPS")
    if steps_env:
        train["total_steps"] = int(steps_env)
        logger.info("CS224R_TOTAL_STEPS override → total_steps=%s", train["total_steps"])
    out["train"] = train
    mode = os.environ.get("CS224R_TRAIN_MODE")
    if mode:
        out["launch_mode"] = mode
    abl = ablation_label()
    if abl:
        out["ablation"] = abl
        out["vllm_sleep"] = os.environ.get("CS224R_VLLM_SLEEP", "0") == "1"
        out["logprob_chunk"] = logprob_chunk_size()
        train = dict(out.get("train", {}))
        train["checkpoint_dir"] = f"/vol/checkpoints/train_real_ablate_{abl}/"
        out["train"] = train
    return out


def load_cfg(path: str | Path) -> TrainCfg:
    with open(path) as f:
        return TrainCfg.from_dict(apply_launch_overrides(yaml.safe_load(f) or {}))


def train_cfg_from_dict(d: dict[str, Any]) -> TrainCfg:
    return TrainCfg.from_dict(d)


def _device_vram_total_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def _pack_chunks_by_tokens(
    completion_lens: list[int], token_budget: int
) -> list[list[int]]:
    """Greedy first-fit-decreasing packing: indices grouped so sum(lens) ≤ budget per chunk.

    Returns list of index lists. Each chunk holds at least 1 sequence even if its length
    exceeds the budget (avoids infinite loop on pathologically long completions; the
    chunk just runs at higher peak — caller's responsibility to size budget appropriately).
    """
    order = sorted(range(len(completion_lens)), key=lambda i: -completion_lens[i])
    chunks: list[tuple[int, list[int]]] = []  # (running_tokens, indices)
    for i in order:
        L = completion_lens[i]
        placed = False
        for c_idx, (tok, idxs) in enumerate(chunks):
            if tok + L <= token_budget:
                chunks[c_idx] = (tok + L, idxs + [i])
                placed = True
                break
        if not placed:
            chunks.append((L, [i]))
    return [idxs for _, idxs in chunks]


def _train_step_microbatched(
    cfg: TrainCfg,
    hf_model: Any,
    opt: torch.optim.Optimizer,
    prompt_ids_list: list[list[int]],
    completion_ids_list: list[list[int]],
    old_logprobs_list: list[torch.Tensor],
    adv_list: list[float],
    microbatch: int,
    device: torch.device,
    *,
    token_budget: int | None = None,
    optimizer_step: bool = True,
    instrument: bool = False,
    phase_times: dict[str, float] | None = None,
    vram_peak: dict[str, float] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[float, int, int]:
    """Interleaved per-microbatch forward+backward; gradients accumulate.

    Two packing modes:
      * `token_budget=None` (default): fixed `microbatch` sequences per chunk.
        Peak VRAM scales linearly with completion length × microbatch.
      * `token_budget=int`: greedy-pack sequences so each chunk has ≤ token_budget total
        completion tokens. Peak VRAM is bounded by token_budget (not by sequence count),
        which auto-handles long completions safely and packs short completions densely
        for throughput.

    Loss is weighted per-chunk by chunk_size / n_kept so the result equals the full-batch
    mean regardless of chunk-size variation.

    Returns (mean_loss, max_chunk_size, num_chunks).
    """
    length_norm = str(cfg.loss.get("length_norm", "per_seq"))
    clip_low = float(cfg.loss.get("clip_low", 0.20))
    clip_high = float(cfg.loss.get("clip_high", 0.28))
    n_kept = len(prompt_ids_list)

    phase_times = phase_times if phase_times is not None else {}
    vram_peak = vram_peak if vram_peak is not None else {}

    opt.zero_grad(set_to_none=True)
    if n_kept == 0:
        phase_times.setdefault("t_logprob_fwd", 0.0)
        phase_times.setdefault("t_backward", 0.0)
        return 0.0, 0, 0

    if token_budget is not None and token_budget > 0:
        completion_lens = [len(c) for c in completion_ids_list]
        chunk_index_groups = _pack_chunks_by_tokens(completion_lens, token_budget)
    else:
        eff_mb = max(1, min(microbatch, n_kept))
        chunk_index_groups = [
            list(range(start, min(start + eff_mb, n_kept)))
            for start in range(0, n_kept, eff_mb)
        ]

    fwd_time = 0.0
    bwd_time = 0.0
    loss_accum = 0.0
    max_chunk_size = 0
    total_tokens = 0
    ratio_mean_w = 0.0
    ratio_p95_w = 0.0
    clip_low_w = 0.0
    clip_high_w = 0.0
    neg_lp_w = 0.0
    ratio_max_global = 0.0

    if instrument and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_total0 = time.monotonic()

    for idxs in chunk_index_groups:
        chunk_prompts = [prompt_ids_list[i] for i in idxs]
        chunk_completions = [completion_ids_list[i] for i in idxs]
        chunk_old = [old_logprobs_list[i] for i in idxs]
        chunk_adv = [adv_list[i] for i in idxs]
        chunk_size = len(idxs)
        max_chunk_size = max(max_chunk_size, chunk_size)

        t_fwd0 = time.monotonic()
        new_lps = [
            _completion_logprobs_hf(hf_model, pid, cid, device)
            for pid, cid in zip(chunk_prompts, chunk_completions)
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_time += time.monotonic() - t_fwd0

        new_t, mask = _pad_sequences(new_lps)
        old_t, _ = _pad_sequences(chunk_old)
        new_t = new_t.to(device)
        old_t = old_t.to(device)
        mask = mask.to(device)
        adv_t = torch.tensor(chunk_adv, dtype=torch.float32, device=device)
        keep_t = torch.ones(chunk_size, dtype=torch.bool, device=device)

        t_bwd0 = time.monotonic()
        # Per-chunk grpo_loss returns mean over `chunk_size` sequences. To recover
        # the full-batch mean we weight by chunk_size / n_kept; sum of weights
        # over all chunks equals 1.
        weight = chunk_size / n_kept
        chunk_loss_raw, chunk_stats = grpo_loss(
            new_t,
            old_t,
            adv_t,
            mask,
            keep_t,
            clip_low=clip_low,
            clip_high=clip_high,
            length_norm=length_norm,
            return_stats=True,
        )
        chunk_loss = chunk_loss_raw * weight
        chunk_loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bwd_time += time.monotonic() - t_bwd0

        loss_accum += float(chunk_loss.detach().item())

        n_tok = int(chunk_stats["n_tokens"])
        if n_tok > 0:
            with torch.no_grad():
                neg_lp_chunk = float(
                    (-(new_t.detach() * mask).sum() / mask.sum().clamp(min=1)).item()
                )
            total_tokens += n_tok
            ratio_mean_w += chunk_stats["ratio_mean"] * n_tok
            ratio_p95_w += chunk_stats["ratio_p95"] * n_tok
            clip_low_w += chunk_stats["clipped_low_frac"] * n_tok
            clip_high_w += chunk_stats["clipped_high_frac"] * n_tok
            neg_lp_w += neg_lp_chunk * n_tok
            if chunk_stats["ratio_max"] > ratio_max_global:
                ratio_max_global = chunk_stats["ratio_max"]

        del new_lps, new_t, old_t, mask, adv_t, keep_t, chunk_loss, chunk_loss_raw

    total_time = time.monotonic() - t_total0
    phase_times["t_logprob_fwd"] = fwd_time
    phase_times["t_backward"] = bwd_time
    phase_times["t_train_fwd_bwd"] = total_time
    if instrument and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        vram_peak["t_train_fwd_bwd"] = peak_gb
        vram_peak["t_logprob_fwd"] = peak_gb
        vram_peak["t_backward"] = peak_gb

    grad_norm_preclip: float | None = None
    if optimizer_step:
        grad_clip = float(cfg.train.get("grad_clip", 1.0))
        with _phase_timer("t_optimizer", instrument, phase_times, vram_peak):
            gn = torch.nn.utils.clip_grad_norm_(hf_model.parameters(), grad_clip)
            grad_norm_preclip = float(gn.item()) if torch.is_tensor(gn) else float(gn)
            opt.step()

    if diagnostics is not None:
        if total_tokens > 0:
            diagnostics["ratio_mean"] = ratio_mean_w / total_tokens
            diagnostics["ratio_max"] = ratio_max_global
            diagnostics["ratio_p95"] = ratio_p95_w / total_tokens
            diagnostics["clipped_low_frac"] = clip_low_w / total_tokens
            diagnostics["clipped_high_frac"] = clip_high_w / total_tokens
            diagnostics["mean_neg_logprob"] = neg_lp_w / total_tokens
        if grad_norm_preclip is not None:
            diagnostics["grad_norm_preclip"] = grad_norm_preclip
    return loss_accum, max_chunk_size, len(chunk_index_groups)


def run_microbatch_forward_backward(
    cfg: TrainCfg,
    hf_model: Any,
    step_cache: dict[str, Any],
    microbatch: int,
) -> tuple[bool, float, float]:
    """Phase 2 sweep: HF forward + backward on cached rollouts (no optim step).

    `new_logprobs` must be recomputed each attempt — cached logprobs are detached and
    cannot be backproped through. Recomputing also gives a faithful peak-VRAM reading.
    """
    device = next(hf_model.parameters()).device
    rollouts = step_cache["rollouts"]
    adv_list = step_cache["advantages"]
    opt = torch.optim.AdamW(hf_model.parameters(), lr=1e-6)
    hf_model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    try:
        _train_step_microbatched(
            cfg,
            hf_model,
            opt,
            [r["prompt_ids"] for r in rollouts],
            [r["completion_ids"] for r in rollouts],
            [r["old_logprobs"] for r in rollouts],
            list(adv_list),
            microbatch,
            device,
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
        logger.warning(
            "CUDA OOM in run_microbatch_forward_backward at microbatch=%d", microbatch
        )
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
    seqs: list[list[float] | torch.Tensor], pad: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        raise ValueError("empty sequence list")
    if isinstance(seqs[0], torch.Tensor):
        from torch.nn.utils.rnn import pad_sequence

        rows = [s.float() for s in seqs]
        out = pad_sequence(rows, batch_first=True, padding_value=pad)
        mask = torch.zeros_like(out)
        for i, s in enumerate(seqs):
            mask[i, : len(s)] = 1.0
        return out, mask
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
    *,
    logprob_chunk: int | None = None,
) -> torch.Tensor:
    """Teacher-forcing logprobs on completion tokens; differentiable w.r.t. model."""
    if not completion_ids:
        return torch.zeros(0, device=device, dtype=torch.float32)
    chunk = logprob_chunk_size() if logprob_chunk is None else logprob_chunk
    input_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=device
    )
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[0]
    start = len(prompt_ids) - 1
    if chunk <= 0:
        log_probs = torch.log_softmax(logits, dim=-1)
        token_lps = [
            log_probs[start + j, tid] for j, tid in enumerate(completion_ids)
        ]
        return torch.stack(token_lps)
    token_lps: list[torch.Tensor] = []
    for i in range(0, len(completion_ids), chunk):
        chunk_ids = completion_ids[i : i + chunk]
        sl = logits[start + i : start + i + len(chunk_ids)]
        targets = torch.tensor(chunk_ids, dtype=torch.long, device=device)
        lp = torch.log_softmax(sl, dim=-1)
        token_lps.append(lp.gather(1, targets.unsqueeze(1)).squeeze(1))
    return torch.cat(token_lps)


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
    train_wandb: dict[str, float] = {}

    device = next(hf_model.parameters()).device

    if step_cache is not None:
        raise NotImplementedError(
            "run_one_grpo_step does not consume step_cache; phase 2 uses "
            "run_microbatch_forward_backward to recompute new_logprobs under grad."
        )
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
        if rollouts:
            n_roll = len(rollouts)
            n_stop = sum(1 for r in rollouts if r.finish_reason == "stop")
            n_length = sum(1 for r in rollouts if r.finish_reason == "length")
            diagnostics["frac_finish_stop"] = n_stop / n_roll
            diagnostics["frac_finish_length"] = n_length / n_roll
            diagnostics["frac_finish_other"] = 1.0 - (n_stop + n_length) / n_roll
        if instrument:
            diagnostics["rollout_output_tokens"] = rollout_tokens
            diagnostics["vram_headroom_gb_after_rollout"] = (
                _device_vram_total_gb() - vram_peak.get("t_rollout", 0.0)
            )

        rewards_grid: list[list[float]] = []
        reward_meta: list[list[dict[str, Any]]] = []
        completion_token_lens: list[int] = []
        parse_ok = 0
        total_rw = 0
        with _phase_timer("t_score", instrument, phase_times, vram_peak):
            for p_idx in range(len(batch.prompts)):
                row: list[float] = []
                meta_row: list[dict[str, Any]] = []
                for r_idx in range(n_rollouts):
                    flat_idx = p_idx * n_rollouts + r_idx
                    rr = rollouts[flat_idx]
                    rw = compute_reward(
                        rr.completion_text,
                        batch.golds[p_idx],
                        prompt_variant=prompt_variant,
                    )
                    row.append(float(rw["reward"]))
                    meta_row.append(rw)
                    completion_token_lens.append(len(rr.completion_ids))
                    parse_ok += int(rw["parse_ok"])
                    total_rw += 1
                rewards_grid.append(row)
                reward_meta.append(meta_row)
        if instrument and total_rw:
            diagnostics["parse_ok_rate"] = parse_ok / total_rw
        train_wandb = aggregate_train_step_wandb_metrics(
            rewards_grid, reward_meta, completion_token_lens
        )

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

        if not kept_rollouts:
            raise RuntimeError(
                "All GRPO groups had zero advantage (every prompt's rollouts agreed). "
                "Raise batch_size / smoke_prompts so some groups have mixed rewards "
                f"(this batch: {len(batch.prompts)} prompts x {n_rollouts} rollouts)."
            )

        rollout_engine.sleep_for_train()

        n_kept = len(kept_rollouts)

    token_budget = cfg.train.get("token_budget")
    loss_val, max_chunk_size, num_chunks = _train_step_microbatched(
        cfg,
        hf_model,
        opt,
        [rr.prompt_ids for rr in kept_rollouts],
        [rr.completion_ids for rr in kept_rollouts],
        [rr.old_logprobs for rr in kept_rollouts],
        kept_adv,
        mb,
        device,
        token_budget=int(token_budget) if token_budget else None,
        optimizer_step=True,
        instrument=instrument,
        phase_times=phase_times,
        vram_peak=vram_peak,
        diagnostics=diagnostics,
    )
    diagnostics["grad_accum_at_mb"] = num_chunks
    diagnostics["effective_microbatch"] = max_chunk_size
    diagnostics["num_chunks"] = num_chunks
    diagnostics["max_chunk_size"] = max_chunk_size

    if vllm_sleep_enabled():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    sync_stats: SyncStats | None = None
    every_n = int(cfg.weight_sync.get("every_n_steps", 1))
    if every_n > 0 and (step + 1) % every_n == 0:
        with _phase_timer("t_weight_sync", instrument, phase_times, vram_peak):
            rollout_engine.wake_weights_only()
            sync_stats = rollout_engine.update_weights(hf_model)
        if instrument and sync_stats is not None:
            diagnostics["t_weight_sync_step_s"] = sync_stats.wall_clock_s
    rollout_engine.wake_for_rollout()

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
            "rollouts": [
                {
                    "prompt_ids": rr.prompt_ids,
                    "completion_ids": rr.completion_ids,
                    "old_logprobs": rr.old_logprobs,
                }
                for rr in kept_rollouts
            ],
            "advantages": list(kept_adv),
            "n_kept": n_kept,
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

    sample_completions = [
        rr.completion_text for rr in kept_rollouts[: min(3, len(kept_rollouts))]
    ]

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
        train_wandb=train_wandb,
        step_cache=cache_out,
        sample_completions=sample_completions,
    )


def setup_wandb(
    cfg: TrainCfg,
    repro: dict[str, Any],
    *,
    wandb_run_id: str | None = None,
) -> Any:
    import wandb

    run_name = cfg.raw.get("wandb_run_name")
    if not run_name:
        op = cfg.raw.get("operator", "unknown")
        abl = cfg.raw.get("ablation")
        run_name = f"train-{cfg.arm}_{op}" + (f"_ablate-{abl}" if abl else "")
    tags = [
        "phase=train",
        f"operator={cfg.raw.get('operator', 'unknown')}",
        f"gpu_class={cfg.raw.get('gpu_class', 'unknown')}",
        f"arm={cfg.arm}",
        f"git_sha_short={repro.get('git_sha_short', 'unknown')}",
    ]
    launch_mode = cfg.raw.get("launch_mode")
    if launch_mode:
        tags.append(f"launch_mode={launch_mode}")
    abl = cfg.raw.get("ablation")
    if abl:
        tags.append(f"ablation={abl}")
        tags.append(f"vllm_sleep={cfg.raw.get('vllm_sleep', False)}")
        tags.append(f"logprob_chunk={cfg.raw.get('logprob_chunk', 0)}")
    if wandb_run_id:
        tags.append("resumed=true")
    init_kw: dict[str, Any] = {
        "entity": cfg.wandb["entity"],
        "project": cfg.wandb["project"],
        "group": cfg.wandb.get("group"),
        "name": run_name,
        "config": {**cfg.raw, **repro, "dep_versions": _dep_versions()},
        "tags": tags,
    }
    if wandb_run_id:
        init_kw["id"] = wandb_run_id
        init_kw["resume"] = "must"
    return wandb.init(**init_kw)


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def load_ckpt(
    path: Path,
    hf_model: Any,
    opt: torch.optim.Optimizer,
    dataset: JsonlPromptDataset,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hf_model.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    rng = payload.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])
    if "cuda" in rng and torch.cuda.is_available():
        for d, state in enumerate(rng["cuda"]):
            torch.cuda.set_rng_state(state, d)
    if "dataset" in payload:
        dataset.load_state_dict(payload["dataset"])
    return payload


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
    dataset: JsonlPromptDataset,
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
        "dataset": dataset.state_dict(),
        "model": hf_model.state_dict(),
        "optimizer": opt.state_dict(),
    }
    torch.save(payload, path)
    logger.info("Wrote checkpoint to %s (step %s)", path, step)


def _resume_enabled(cfg: TrainCfg) -> bool:
    mode = str(cfg.train.get("resume", "auto")).lower()
    if mode in ("false", "0", "no", "off"):
        return False
    if os.environ.get("CS224R_NO_RESUME", "").lower() in ("1", "true", "yes"):
        return False
    return True


def train(
    cfg: TrainCfg,
    *,
    after_checkpoint: Any | None = None,
) -> None:
    repro = _git_metadata()
    ckpt_dir = Path(cfg.train.get("checkpoint_dir", "/vol/checkpoints/train/"))
    resume_path: Path | None = None
    if _resume_enabled(cfg):
        explicit = cfg.train.get("resume_from")
        if explicit:
            resume_path = Path(str(explicit))
        else:
            resume_path = find_latest_checkpoint(ckpt_dir)

    start_step = 0
    wandb_run_id: str | None = None
    resumed_from_step: int | None = None
    if resume_path is not None and resume_path.is_file():
        logger.info("Resuming from checkpoint %s", resume_path)
        head = torch.load(resume_path, map_location="cpu", weights_only=False)
        resumed_from_step = int(head["step"])
        start_step = resumed_from_step + 1
        wandb_run_id = str(head["wandb_run_id"])
    else:
        set_seeds(cfg.global_seed)

    run = setup_wandb(cfg, repro, wandb_run_id=wandb_run_id)
    log_repro(cfg, repro)

    rollout_engine = RolloutEngine(cfg.rollout)
    hf_model, opt = build_hf(cfg)
    dataset = JsonlPromptDataset(cfg.train["data_path"], cfg.global_seed)
    if resumed_from_step is not None and resume_path is not None:
        load_ckpt(resume_path, hf_model, opt, dataset)
        import wandb

        wandb.log({"train/resumed_from_step": resumed_from_step}, step=start_step)

    total_steps = int(cfg.train["total_steps"])
    if start_step >= total_steps:
        logger.info(
            "Checkpoint step %s >= total_steps %s; nothing to run",
            start_step - 1,
            total_steps,
        )
        rollout_engine.shutdown()
        run.finish()
        return

    batch_size = int(cfg.train["batch_size"])
    last_ckpt = time.monotonic()

    try:
        for step in range(start_step, total_steps):
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
                instrument=True,
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
            log_dict.update(result.train_wandb)
            if result.sync_stats is not None:
                log_dict["train/weight_sync_s"] = result.sync_stats.wall_clock_s
            if result.vram_peak_gb:
                peak_step = max(result.vram_peak_gb.values())
                log_dict["train/vram_peak_gb_step"] = peak_step
                log_dict["train/vram_headroom_gb_step"] = (
                    _device_vram_total_gb() - peak_step
                )
                for phase, v in result.vram_peak_gb.items():
                    log_dict[f"train/vram_peak_gb_{phase}"] = v
            if result.phase_times_s:
                for phase, t in result.phase_times_s.items():
                    log_dict[f"train/{phase}_s"] = t
            for k, v in result.diagnostics.items():
                if isinstance(v, (int, float)):
                    log_dict[f"train/{k}"] = v
            wandb.log(log_dict, step=step)

            if step % 50 == 0 and result.sample_completions:
                wandb.log(
                    {
                        f"sample/completion_{i}": wandb.Html(
                            "<pre>" + html.escape(text) + "</pre>"
                        )
                        for i, text in enumerate(result.sample_completions)
                    },
                    step=step,
                )

            if should_checkpoint(step, cfg, last_ckpt):
                save_ckpt(
                    ckpt_dir / f"step_{step:06d}.pt",
                    hf_model,
                    opt,
                    step,
                    cfg,
                    run.id,
                    dataset,
                )
                last_ckpt = time.monotonic()
                if after_checkpoint is not None:
                    after_checkpoint()
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
        gpu="H200",
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
    def train_remote(
        config_path: str,
        ablation: str = "",
        vllm_sleep: int = 0,
        logprob_chunk: int = 0,
    ) -> None:
        from train.ablation import apply_ablation_env

        apply_ablation_env(
            ablation=ablation,
            vllm_sleep=vllm_sleep,
            logprob_chunk=logprob_chunk,
        )

        def _commit_volume() -> None:
            _artifacts_vol.commit()

        train(load_cfg(config_path), after_checkpoint=_commit_volume)
        _commit_volume()

except ImportError:
    app = None  # type: ignore[assignment, misc]
    train_remote = None  # type: ignore[assignment, misc]
