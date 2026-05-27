"""GRPO trainer loop and Modal entrypoint."""

from __future__ import annotations

import html
import json
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
from train.ablation import (
    ablation_label,
    logprob_chunk_size,
    logprob_seq_batch_size,
    vllm_sleep_enabled,
)
from train.clustering import answer_hash_clusters, sympy_equiv, sympy_equiv_allowlist
from train.objective import SET_ARMS, compute_advantages
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
    skipped: bool = False
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
    clusters_grid: list[list[int]] | None = None,
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

    if clusters_grid is not None:
        # C4b: distinct answer-hash cluster ids among rollouts with reward > 0,
        # averaged over the batch (prompts with zero correct contribute 0).
        unique_correct = []
        for p_idx in range(n_prompts):
            cids = clusters_grid[p_idx]
            row = rewards_grid[p_idx]
            distinct = {c for c, r in zip(cids, row) if r > 0.0}
            unique_correct.append(len(distinct))
        out["train/mean_unique_answer_clusters_correct"] = (
            sum(unique_correct) / n_prompts
        )

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
        arm = str(d.get("arm", "grpo"))
        loss = dict(d["loss"])
        # Arm-driven length_norm: set-based arms (minority_*, poly_epo_*) require
        # Dr.GRPO / Poly-EPO batch_max normalization. Enforce here so a YAML that
        # forgets to flip it can't silently train with per-seq norm.
        if arm in SET_ARMS:
            want = "batch_max"
            cur = loss.get("length_norm")
            if cur and cur != want:
                logger.warning(
                    "arm %s overriding loss.length_norm: %s -> %s", arm, cur, want
                )
            loss["length_norm"] = want
        return cls(
            raw=d,
            global_seed=int(d["global_seed"]),
            arm=arm,
            train=d["train"],
            rollout=RolloutCfg.from_dict(d["rollout"]),
            loss=loss,
            weight_sync=d.get("weight_sync", {"every_n_steps": 1}),
            wandb=d["wandb"],
        )


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _resolve_cfg_path(base_path: Path, extends_path: str) -> Path:
    """Resolve `extends` targets relative to config, then main root fallbacks."""
    cand = Path(extends_path)
    if cand.is_absolute():
        return cand
    rel = base_path.parent / cand
    if rel.is_file():
        return rel
    main_rel = _MAIN_ROOT / cand
    if main_rel.is_file():
        return main_rel
    if cand.parts[:1] == ("main",):
        trimmed = _MAIN_ROOT / Path(*cand.parts[1:])
        if trimmed.is_file():
            return trimmed
    return rel


def _load_cfg_dict(path: Path, _visited: tuple[str, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    key = str(resolved)
    if key in _visited:
        chain = " -> ".join((*_visited, key))
        raise ValueError(f"Cycle in config extends chain: {chain}")
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    extends = data.pop("extends", None)
    if not extends:
        return data
    base_path = _resolve_cfg_path(resolved, str(extends))
    base = _load_cfg_dict(base_path, _visited=(*_visited, key))
    return _deep_merge_dict(base, data)


def apply_arm_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge arm_profiles[arm] into the shared config (train_real.yaml layout)."""
    out = dict(raw)
    arm = os.environ.get("CS224R_ARM") or out.get("arm", "grpo")
    profiles = out.pop("arm_profiles", None)
    if profiles and arm in profiles:
        out = _deep_merge_dict(out, profiles[arm])
        if profiles[arm].get("arm"):
            out["arm"] = str(profiles[arm]["arm"])
        else:
            out["arm"] = arm
    else:
        out["arm"] = arm
    return out


def apply_launch_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Env overrides from launch_train.sh — smoke/full share one yaml."""
    out = dict(raw)
    train = dict(out.get("train", {}))
    steps_env = os.environ.get("CS224R_TOTAL_STEPS")
    if steps_env:
        train["total_steps"] = int(steps_env)
        logger.info("CS224R_TOTAL_STEPS override → total_steps=%s", train["total_steps"])
    mode = os.environ.get("CS224R_TRAIN_MODE")
    if mode:
        out["launch_mode"] = mode
    if mode == "smoke":
        probes = raw.get("smoke_probes") or {}
        tpl = probes.get("rollouts_jsonl_path")
        if tpl:
            train["rollouts_jsonl_path"] = str(tpl).format(arm=out.get("arm", "grpo"))
            logger.info(
                "smoke_probes → train.rollouts_jsonl_path=%s",
                train["rollouts_jsonl_path"],
            )
    ckpt_override = os.environ.get("CS224R_CHECKPOINT_DIR", "").strip()
    if ckpt_override:
        train["checkpoint_dir"] = (
            ckpt_override if ckpt_override.endswith("/") else f"{ckpt_override}/"
        )
        logger.info("CS224R_CHECKPOINT_DIR override → checkpoint_dir=%s", train["checkpoint_dir"])
    abl = ablation_label()
    if abl:
        out["ablation"] = abl
        out["vllm_sleep"] = os.environ.get("CS224R_VLLM_SLEEP", "0") == "1"
        out["logprob_chunk"] = logprob_chunk_size()
        out["logprob_seq_batch"] = logprob_seq_batch_size()
        train["checkpoint_dir"] = f"/vol/checkpoints/train_real_ablate_{abl}/"
    out["train"] = train
    return out


def load_cfg(path: str | Path) -> TrainCfg:
    path = Path(path)
    data = _load_cfg_dict(path)
    # Shim: legacy configs that only set `arm:` merge into train_real.yaml.
    if set(data.keys()) <= {"arm"}:
        arm = str(data["arm"])
        os.environ.setdefault("CS224R_ARM", arm)
        path = _MAIN_ROOT / "configs" / "train_real.yaml"
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data = apply_arm_profile(data)
    return TrainCfg.from_dict(apply_launch_overrides(data))


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
        new_lps = _completion_logprobs_for_chunk(
            hf_model, chunk_prompts, chunk_completions, device
        )
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


def _append_train_rollout_records(
    path: Path,
    *,
    step: int,
    batch: StepBatch,
    rollouts: list[RolloutResult],
    reward_meta: list[list[dict[str, Any]]],
    clusters_grid: list[list[int]] | None,
    n_rollouts: int,
    prompt_variant: str,
) -> None:
    """Append one jsonl record per rollout (probe-compatible; no gold in row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for p_idx in range(len(batch.prompts)):
            for r_idx in range(n_rollouts):
                flat_idx = p_idx * n_rollouts + r_idx
                rr = rollouts[flat_idx]
                rw = reward_meta[p_idx][r_idx]
                rec: dict[str, Any] = {
                    "step": step,
                    "problem_id": batch.problem_ids[p_idx],
                    "rollout_idx": r_idx,
                    "prompt_variant": prompt_variant,
                    "completion": rr.completion_text,
                    "reward": rw["reward"],
                    "parse_ok": rw["parse_ok"],
                    "parsed_answer": rw.get("parsed_answer"),
                    "parsed_is_int": rw.get("parsed_is_int"),
                    "has_boxed": rw.get("has_boxed"),
                    "has_answer_line": rw.get("has_answer_line"),
                    "strict_parse_ok": rw.get("strict_parse_ok"),
                    "extract_path": rw.get("extract_path"),
                    "length_tokens": len(rr.completion_ids),
                    "finish_reason": rr.finish_reason,
                }
                if clusters_grid is not None:
                    rec["cluster_id"] = clusters_grid[p_idx][r_idx]
                f.write(json.dumps(rec) + "\n")


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


def _completion_logprobs_hf_batch(
    model: Any,
    prompt_ids_list: list[list[int]],
    completion_ids_list: list[list[int]],
    device: torch.device,
    *,
    logprob_chunk: int | None = None,
) -> list[torch.Tensor]:
    """Batched teacher-forcing logprobs; one forward for len(prompt_ids_list) sequences."""
    n = len(prompt_ids_list)
    if n == 0:
        return []
    if n == 1:
        return [
            _completion_logprobs_hf(
                model,
                prompt_ids_list[0],
                completion_ids_list[0],
                device,
                logprob_chunk=logprob_chunk,
            )
        ]
    chunk = logprob_chunk_size() if logprob_chunk is None else logprob_chunk
    seqs = [p + c for p, c in zip(prompt_ids_list, completion_ids_list)]
    lengths = [len(s) for s in seqs]
    t_max = max(lengths)
    b = len(seqs)
    input_ids = torch.zeros((b, t_max), dtype=torch.long, device=device)
    attn = torch.zeros((b, t_max), dtype=torch.long, device=device)
    prompt_lens = [len(p) for p in prompt_ids_list]
    for i, s in enumerate(seqs):
        input_ids[i, : lengths[i]] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, : lengths[i]] = 1
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits
    results: list[torch.Tensor] = []
    for i in range(b):
        start = prompt_lens[i] - 1
        comp_ids = completion_ids_list[i]
        if not comp_ids:
            results.append(torch.zeros(0, device=device, dtype=torch.float32))
            continue
        if chunk <= 0:
            log_probs = torch.log_softmax(logits[i], dim=-1)
            token_lps = [
                log_probs[start + j, tid] for j, tid in enumerate(comp_ids)
            ]
            results.append(torch.stack(token_lps))
            continue
        token_lps: list[torch.Tensor] = []
        for j in range(0, len(comp_ids), chunk):
            cids = comp_ids[j : j + chunk]
            sl = logits[i, start + j : start + j + len(cids)]
            targets = torch.tensor(cids, dtype=torch.long, device=device)
            lp = torch.log_softmax(sl, dim=-1)
            token_lps.append(lp.gather(1, targets.unsqueeze(1)).squeeze(1))
        results.append(torch.cat(token_lps))
    return results


def _completion_logprobs_for_chunk(
    model: Any,
    chunk_prompts: list[list[int]],
    chunk_completions: list[list[int]],
    device: torch.device,
) -> list[torch.Tensor]:
    """Forward logprobs for one token-budget chunk (batched or sequential)."""
    seq_batch = logprob_seq_batch_size()
    if seq_batch <= 1:
        return [
            _completion_logprobs_hf(model, pid, cid, device)
            for pid, cid in zip(chunk_prompts, chunk_completions)
        ]
    out: list[torch.Tensor] = []
    for b0 in range(0, len(chunk_prompts), seq_batch):
        out.extend(
            _completion_logprobs_hf_batch(
                model,
                chunk_prompts[b0 : b0 + seq_batch],
                chunk_completions[b0 : b0 + seq_batch],
                device,
            )
        )
    return out


def build_hf(cfg: TrainCfg) -> tuple[Any, torch.optim.Optimizer]:
    from transformers import AutoModelForCausalLM

    model_name = cfg.rollout.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # FA2 re-enabled 2026-05-27 after smoke_flash_attn.py passed on Modal H200
    # (import, load, forward, collocated vLLM+HF). See docs/efficiency_wins_2026-05-26.md.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
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

        # Set-based arms (minority_answer, poly_epo_answer) need per-rollout
        # cluster ids derived from parsed-answer identity. minority_cot uses
        # the LLM judge instead and is not wired here yet.
        clusters_grid: list[list[int]] | None = None
        if cfg.arm in SET_ARMS:
            if cfg.arm == "minority_cot":
                raise NotImplementedError(
                    "minority_cot judge integration pending; see "
                    "docs/build_spec/remaining_arms.md §4"
                )
            clusters_grid = []
            clustering_cfg = cfg.raw.get("clustering", {})
            sympy_mode = clustering_cfg.get("sympy_mode", "allowlist")
            if sympy_mode == "off":
                use_sympy = False
                sympy_fn = None
            elif sympy_mode == "blocklist":
                use_sympy = True
                sympy_fn = sympy_equiv
            elif sympy_mode == "allowlist":
                use_sympy = True
                sympy_fn = sympy_equiv_allowlist
            else:
                raise ValueError(
                    f"clustering.sympy_mode must be one of allowlist|blocklist|off, "
                    f"got {sympy_mode!r}"
                )
            for p_idx in range(len(batch.prompts)):
                parsed = [
                    reward_meta[p_idx][r].get("parsed_answer")
                    for r in range(n_rollouts)
                ]
                ok = [
                    bool(reward_meta[p_idx][r].get("parse_ok"))
                    for r in range(n_rollouts)
                ]
                clusters_grid.append(
                    answer_hash_clusters(
                        parsed, ok, use_sympy=use_sympy, sympy_equiv_fn=sympy_fn
                    )
                )

        train_wandb = aggregate_train_step_wandb_metrics(
            rewards_grid,
            reward_meta,
            completion_token_lens,
            clusters_grid=clusters_grid,
        )

        rewards_t = torch.tensor(rewards_grid, dtype=torch.float32)

        with _phase_timer("t_advantage", instrument, phase_times, vram_peak):
            if cfg.arm in SET_ARMS:
                assert clusters_grid is not None
                clusters_t = torch.tensor(clusters_grid, dtype=torch.long)
                adv_out = compute_advantages(
                    cfg.arm,
                    rewards_t,
                    clusters_t,
                    global_seed=cfg.global_seed,
                    problem_ids=batch.problem_ids,
                )
            else:
                adv_out = compute_advantages(cfg.arm, rewards_t)

        # C3: surface set-arm marginal-advantage percentiles to wandb if present.
        for k in ("adv_marginal_p05", "adv_marginal_p50", "adv_marginal_p95"):
            if k in adv_out.diagnostics:
                train_wandb[f"train/{k}"] = float(adv_out.diagnostics[k])

        rollouts_path = cfg.train.get("rollouts_jsonl_path")
        if rollouts_path:
            _append_train_rollout_records(
                Path(str(rollouts_path)),
                step=step,
                batch=batch,
                rollouts=rollouts,
                reward_meta=reward_meta,
                clusters_grid=clusters_grid,
                n_rollouts=n_rollouts,
                prompt_variant=prompt_variant,
            )

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
            frac_filt = float(adv_out.diagnostics.get("fraction_filtered", 1.0))
            logger.warning(
                "No kept rollouts after filtering (arm=%s); skipping train and "
                "weight sync for this step (%s prompts x %s rollouts, "
                "fraction_filtered=%.3f).",
                cfg.arm,
                len(batch.prompts),
                n_rollouts,
                frac_filt,
            )
            return StepResult(
                loss=float("nan"),
                mean_reward=float(rewards_t.mean().item()),
                fraction_filtered=frac_filt,
                mean_advantage=0.0,
                n_kept_sequences=0,
                skipped=True,
                phase_times_s=phase_times,
                vram_peak_gb=vram_peak,
                diagnostics={**diagnostics, "skipped_no_kept": True},
                train_wandb={**train_wandb, "train/skipped_no_kept": 1.0},
                sample_completions=[
                    rollouts[i].completion_text
                    for i in range(min(3, len(rollouts)))
                ],
            )

        if vllm_sleep_enabled():
            with _phase_timer("t_vllm_sleep", instrument, phase_times, vram_peak):
                rollout_engine.sleep_for_train()
        else:
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
            if vllm_sleep_enabled():
                with _phase_timer(
                    "t_vllm_wake_weights", instrument, phase_times, vram_peak
                ):
                    rollout_engine.wake_weights_only()
            else:
                rollout_engine.wake_weights_only()
            sync_stats = rollout_engine.update_weights(hf_model)
        if instrument and sync_stats is not None:
            diagnostics["t_weight_sync_step_s"] = sync_stats.wall_clock_s
    if vllm_sleep_enabled():
        with _phase_timer("t_vllm_wake_kv", instrument, phase_times, vram_peak):
            rollout_engine.wake_for_rollout()
    else:
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
    leg_number = os.environ.get("CS224R_LEG_NUMBER")
    if leg_number:
        tags.append(f"leg_number={leg_number}")
    abl = cfg.raw.get("ablation")
    if abl:
        tags.append(f"ablation={abl}")
        tags.append("smoke")
        tags.append(f"vllm_sleep={cfg.raw.get('vllm_sleep', False)}")
        tags.append(f"logprob_chunk={cfg.raw.get('logprob_chunk', 0)}")
    elif launch_mode == "smoke":
        tags.append("smoke")
    else:
        tags.append("production")
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


def _checkpoint_family_slug(cfg: TrainCfg) -> str:
    """e.g. ``train_real`` from yaml ``/vol/checkpoints/train_real/``."""
    return Path(str(cfg.train.get("checkpoint_dir", "/vol/checkpoints/train/"))).name


def resolve_checkpoint_dir(cfg: TrainCfg, *, checkpoint_run_id: str = "") -> Path:
    """Per-run dir: ``/vol/checkpoints/{family}_{run_id}/``.

    ``run_id`` comes from launch (``CS224R_APP_NAME``). Without it, use yaml
    ``checkpoint_dir`` as-is (legacy flat ``step_*.pt`` layout).
    """
    yaml_dir = Path(str(cfg.train.get("checkpoint_dir", "/vol/checkpoints/train/")))
    run_id = (
        checkpoint_run_id or os.environ.get("CS224R_CHECKPOINT_RUN_ID", "")
    ).strip()
    if run_id:
        return Path(f"/vol/checkpoints/{_checkpoint_family_slug(cfg)}_{run_id}")
    return yaml_dir


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
    *,
    restore_dataset: bool = True,
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
    if restore_dataset and "dataset" in payload:
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
    checkpoint_run_id: str = "",
    after_checkpoint: Any | None = None,
    leg_budget_s: float | None = None,
    on_leg_exhausted: Any | None = None,
) -> None:
    """Run the GRPO train loop.

    Auto-relaunch hooks:
      * `leg_budget_s` — if set, exit cleanly the first step after this many
        seconds have elapsed since the start of this call. A checkpoint at the
        last completed step is guaranteed before exit.
      * `on_leg_exhausted(final_step)` — called immediately before return when
        the leg budget triggers exit. Used by `train_remote` to spawn the next
        Modal container, which will pick up via `resume: auto`.
    """
    repro = _git_metadata()
    init_t0 = time.monotonic()

    def _init_elapsed(label: str) -> None:
        logger.info("train init [%s] +%.1fs since start", label, time.monotonic() - init_t0)

    ckpt_dir = resolve_checkpoint_dir(cfg, checkpoint_run_id=checkpoint_run_id)
    logger.info("Checkpoint dir=%s", ckpt_dir)

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
        _init_elapsed("before checkpoint head load")
        head = torch.load(resume_path, map_location="cpu", weights_only=False)
        _init_elapsed("after checkpoint head load")
        resumed_from_step = int(head["step"])
        start_step = resumed_from_step + 1
        # CS224R_FRESH_WANDB: one-shot escape hatch when the live wandb run has
        # logged past the resume checkpoint and rewind/fork is not available.
        # Starts a fresh wandb run; subsequent legs spawn without this env var
        # and chain onto the new run via the next checkpoint's wandb_run_id.
        if os.environ.get("CS224R_FRESH_WANDB", "").lower() in ("1", "true", "yes"):
            logger.info("CS224R_FRESH_WANDB set; starting fresh wandb run")
            wandb_run_id = None
        else:
            wandb_run_id = str(head["wandb_run_id"])
    else:
        set_seeds(cfg.global_seed)

    run = setup_wandb(cfg, repro, wandb_run_id=wandb_run_id)
    log_repro(cfg, repro)
    _init_elapsed("after wandb setup")

    logger.info("Initializing vLLM RolloutEngine (expect ~60-120s before HF load)")
    rollout_engine = RolloutEngine(cfg.rollout)
    _init_elapsed("after vLLM RolloutEngine")

    logger.info("Loading HF model (attn_implementation=flash_attention_2 when enabled)")
    hf_model, opt = build_hf(cfg)
    _init_elapsed("after build_hf")

    dataset = JsonlPromptDataset(cfg.train["data_path"], cfg.global_seed)
    if resumed_from_step is not None and resume_path is not None:
        logger.info("Loading full checkpoint state into HF model + optimizer")
        load_ckpt(resume_path, hf_model, opt, dataset)
        _init_elapsed("after load_ckpt")
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
    leg_start = time.monotonic()
    _init_elapsed("entering train loop")

    try:
        for step in range(start_step, total_steps):
            if step == start_step:
                logger.info("Starting step %s (init complete)", step)
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
                "train/mean_reward": result.mean_reward,
                "train/fraction_filtered": result.fraction_filtered,
                "train/n_kept_sequences": result.n_kept_sequences,
            }
            if result.skipped:
                log_dict["train/skipped_no_kept"] = 1.0
            else:
                log_dict["train/loss"] = result.loss
                log_dict["train/mean_advantage"] = result.mean_advantage
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

            # Auto-relaunch: spawn a successor leg before the 24h Modal timeout.
            if (
                leg_budget_s is not None
                and (time.monotonic() - leg_start) >= leg_budget_s
            ):
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                if not ckpt_path.exists():
                    save_ckpt(
                        ckpt_path, hf_model, opt, step, cfg, run.id, dataset
                    )
                    last_ckpt = time.monotonic()
                    if after_checkpoint is not None:
                        after_checkpoint()
                elapsed_h = (time.monotonic() - leg_start) / 3600.0
                logger.info(
                    "Leg budget %.2fh reached at step %d; spawning successor and exiting",
                    elapsed_h,
                    step,
                )
                if on_leg_exhausted is not None:
                    on_leg_exhausted(step)
                return
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

    def _train_remote_impl(
        *,
        spawn_fn: Any,
        config_path: str,
        ablation: str = "",
        vllm_sleep: int = 0,
        logprob_chunk: int = 0,
        logprob_seq_batch: int = 1,
        leg_number: int = 1,
        max_legs: int = 10,
        fresh_wandb: bool = False,
        launch_mode: str = "",
        total_steps_override: int = 0,
        no_resume: bool = False,
        arm_override: str = "",
        checkpoint_run_id: str = "",
    ) -> None:
        """Modal entrypoint.

        Auto-relaunch: when the leg budget (CS224R_LEG_HOURS, default 23.0h) is
        hit, `train()` exits cleanly after a final checkpoint and this function
        spawns a successor container via `train_remote.spawn(...)`. The new
        container resumes from the latest checkpoint. `leg_number` increments
        per chained leg; `max_legs` is a runaway guard.

        Launch overrides (`launch_mode`, `total_steps_override`, `no_resume`,
        `arm_override`, `checkpoint_run_id`) are Modal CLI args — host-shell
        exports are not forwarded into the container.
        """
        from train.ablation import apply_ablation_env

        if checkpoint_run_id:
            os.environ["CS224R_CHECKPOINT_RUN_ID"] = checkpoint_run_id
        if arm_override:
            os.environ["CS224R_ARM"] = arm_override
        if launch_mode:
            os.environ["CS224R_TRAIN_MODE"] = launch_mode
        if total_steps_override > 0:
            os.environ["CS224R_TOTAL_STEPS"] = str(total_steps_override)
        if no_resume:
            os.environ["CS224R_NO_RESUME"] = "1"

        apply_ablation_env(
            ablation=ablation,
            vllm_sleep=vllm_sleep,
            logprob_chunk=logprob_chunk,
            logprob_seq_batch=logprob_seq_batch,
        )
        os.environ["CS224R_LEG_NUMBER"] = str(leg_number)
        os.environ["CS224R_MAX_LEGS"] = str(max_legs)
        if fresh_wandb:
            os.environ["CS224R_FRESH_WANDB"] = "1"

        leg_budget_h = float(os.environ.get("CS224R_LEG_HOURS", "23.0"))
        leg_budget_s = leg_budget_h * 3600.0

        def _commit_volume() -> None:
            _artifacts_vol.commit()

        def _on_leg_exhausted(final_step: int) -> None:
            _commit_volume()
            if leg_number >= max_legs:
                logger.warning(
                    "Leg %d hit max_legs=%d; not spawning successor (final step=%d)",
                    leg_number,
                    max_legs,
                    final_step,
                )
                return
            next_leg = leg_number + 1
            logger.info(
                "Spawning leg %d (last completed step=%d, config=%s)",
                next_leg,
                final_step,
                config_path,
            )
            spawn_fn(
                config_path=config_path,
                ablation=ablation,
                vllm_sleep=vllm_sleep,
                logprob_chunk=logprob_chunk,
                logprob_seq_batch=logprob_seq_batch,
                leg_number=next_leg,
                max_legs=max_legs,
                arm_override=arm_override,
                checkpoint_run_id=checkpoint_run_id,
            )

        train(
            load_cfg(config_path),
            checkpoint_run_id=checkpoint_run_id,
            after_checkpoint=_commit_volume,
            leg_budget_s=leg_budget_s,
            on_leg_exhausted=_on_leg_exhausted,
        )
        _commit_volume()

    @app.function(
        image=image,
        gpu="H200",
        timeout=60 * 60 * 24,
        volumes={
            ARTIFACTS_MOUNT: _artifacts_vol,
            HF_CACHE_MOUNT: _hf_vol,
        },
        secrets=[
            modal.Secret.from_name("HUGGINGFACE"),
            modal.Secret.from_name("WANDB_API_KEY"),
        ],
    )
    def train_remote_h200(
        config_path: str,
        ablation: str = "",
        vllm_sleep: int = 0,
        logprob_chunk: int = 0,
        logprob_seq_batch: int = 1,
        leg_number: int = 1,
        max_legs: int = 10,
        fresh_wandb: bool = False,
        launch_mode: str = "",
        total_steps_override: int = 0,
        no_resume: bool = False,
        arm_override: str = "",
        checkpoint_run_id: str = "",
    ) -> None:
        _train_remote_impl(
            spawn_fn=train_remote_h200.spawn,
            config_path=config_path,
            ablation=ablation,
            vllm_sleep=vllm_sleep,
            logprob_chunk=logprob_chunk,
            logprob_seq_batch=logprob_seq_batch,
            leg_number=leg_number,
            max_legs=max_legs,
            fresh_wandb=fresh_wandb,
            launch_mode=launch_mode,
            total_steps_override=total_steps_override,
            no_resume=no_resume,
            arm_override=arm_override,
            checkpoint_run_id=checkpoint_run_id,
        )

    @app.function(
        image=image,
        gpu="B200",
        timeout=60 * 60 * 24,
        volumes={
            ARTIFACTS_MOUNT: _artifacts_vol,
            HF_CACHE_MOUNT: _hf_vol,
        },
        secrets=[
            modal.Secret.from_name("HUGGINGFACE"),
            modal.Secret.from_name("WANDB_API_KEY"),
        ],
    )
    def train_remote_b200(
        config_path: str,
        ablation: str = "",
        vllm_sleep: int = 0,
        logprob_chunk: int = 0,
        logprob_seq_batch: int = 1,
        leg_number: int = 1,
        max_legs: int = 10,
        fresh_wandb: bool = False,
        launch_mode: str = "",
        total_steps_override: int = 0,
        no_resume: bool = False,
        arm_override: str = "",
        checkpoint_run_id: str = "",
    ) -> None:
        _train_remote_impl(
            spawn_fn=train_remote_b200.spawn,
            config_path=config_path,
            ablation=ablation,
            vllm_sleep=vllm_sleep,
            logprob_chunk=logprob_chunk,
            logprob_seq_batch=logprob_seq_batch,
            leg_number=leg_number,
            max_legs=max_legs,
            fresh_wandb=fresh_wandb,
            launch_mode=launch_mode,
            total_steps_override=total_steps_override,
            no_resume=no_resume,
            arm_override=arm_override,
            checkpoint_run_id=checkpoint_run_id,
        )

    # Backward-compatible alias: default target stays H200.
    train_remote = train_remote_h200

except ImportError:
    app = None  # type: ignore[assignment, misc]
    train_remote = None  # type: ignore[assignment, misc]
    train_remote_h200 = None  # type: ignore[assignment, misc]
    train_remote_b200 = None  # type: ignore[assignment, misc]
